"""SQLite FTS5 关键词索引。"""

import json
import re
import sqlite3
from collections import Counter
from pathlib import Path

from langchain_core.documents import Document

from utils.logger_util import logger


DEFAULT_DATABASE_PATH = Path(__file__).resolve().parent / "data" / "keyword_index.db"
# 查询侧剪枝阈值：文档频率超过语料文档数该比例的二元词元视为噪声，不参与召回。
# 用比例而非绝对条数表达，是为了与语料规模无关；实测 2% 与 5% 指标完全一致，
# 说明该参数不敏感。
DEFAULT_MAX_DOCUMENT_RATIO = 0.02
# 比例阈值在小语料上会退化成个位数，此时“2%”不具备统计意义，容易把仅出现两三次
# 的真实信号词元一并剪掉。因此给阈值一个绝对下限：小语料上只剪真正无处不在的词元。
MINIMUM_PRUNABLE_DOCUMENT_FREQUENCY = 10
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+")
_keyword_index = None


def tokenize(text: str) -> list[str]:
    """生成英文、数字、下划线词元和中文二元词。"""
    tokens = []
    for match in _TOKEN_PATTERN.findall(str(text).lower()):
        if all("\u4e00" <= character <= "\u9fff" for character in match):
            if len(match) == 1:
                tokens.append(match)
            else:
                tokens.extend(match[index : index + 2] for index in range(len(match) - 1))
        else:
            tokens.append(match)
    return tokens


class KeywordIndex:
    """管理单个 SQLite FTS5 关键词索引文件。"""

    def __init__(self, database_path=DEFAULT_DATABASE_PATH, tokenizer=tokenize,
                 query_tokenizer=None, max_document_ratio=DEFAULT_MAX_DOCUMENT_RATIO):
        self.database_path = Path(database_path)
        # 分词器可替换，便于在同一套 BM25 下对比不同切分策略。
        self.tokenize = tokenizer
        # 查询侧可独立配置：文档侧放宽召回，查询侧可另行抑制噪声词元。
        self.query_tokenize = query_tokenizer or tokenizer
        # 查询侧按文档频率剪枝的阈值；置 None 关闭剪枝，退回朴素二元切分。
        self.max_document_ratio = max_document_ratio
        self._statistics_available = False
        self._statistics_warning_emitted = False
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self):
        return sqlite3.connect(self.database_path)

    def _initialize(self):
        connection = self._connect()
        try:
            with connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS keyword_records (
                        id INTEGER PRIMARY KEY,
                        text TEXT NOT NULL,
                        source TEXT,
                        section TEXT,
                        document_id TEXT NOT NULL,
                        parent_id TEXT NOT NULL,
                        chunk_index INTEGER NOT NULL,
                        chunk_count INTEGER,
                        metadata TEXT NOT NULL,
                        UNIQUE(document_id, parent_id, chunk_index)
                    )
                    """
                )
                connection.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS keyword_fts USING fts5(tokens)"
                )
                # 词元文档频率：查询侧剪枝的统计依据。按“出现在多少个子块里”计数，
                # 同一词元在单个子块内出现多次只记一次。
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS keyword_token_stats (
                        token TEXT PRIMARY KEY,
                        document_frequency INTEGER NOT NULL
                    )
                    """
                )
        finally:
            connection.close()

    def upsert_documents(self, documents):
        """按 document_id、parent_id、chunk_index 覆盖写入子块，并同步词元文档频率。"""
        connection = self._connect()
        frequency_deltas = Counter()
        try:
            with connection:
                for document in documents:
                    metadata = document.metadata
                    stable_key = (
                        str(metadata.get("document_id", "")),
                        str(metadata.get("parent_id", "")),
                        metadata.get("chunk_index"),
                    )
                    if not stable_key[0] or not stable_key[1] or stable_key[2] is None:
                        raise ValueError("关键词索引子块必须包含 document_id、parent_id 和 chunk_index")

                    row = connection.execute(
                        """
                        SELECT id, text FROM keyword_records
                        WHERE document_id = ? AND parent_id = ? AND chunk_index = ?
                        """,
                        stable_key,
                    ).fetchone()
                    if row is not None:
                        connection.execute("DELETE FROM keyword_fts WHERE rowid = ?", (row[0],))
                        connection.execute("DELETE FROM keyword_records WHERE id = ?", (row[0],))
                        # 覆盖写入前先撤销旧文本的贡献，否则反复覆盖同一子块会让
                        # 文档频率单调上涨，剪枝阈值随之失真。
                        frequency_deltas.subtract(set(self.tokenize(row[1])))

                    tokens = self.tokenize(document.page_content)
                    cursor = connection.execute(
                        """
                        INSERT INTO keyword_records (
                            text, source, section, document_id, parent_id,
                            chunk_index, chunk_count, metadata
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            document.page_content,
                            metadata.get("source"),
                            metadata.get("section"),
                            stable_key[0],
                            stable_key[1],
                            stable_key[2],
                            metadata.get("chunk_count"),
                            json.dumps(metadata.get("metadata", {}), ensure_ascii=False),
                        ),
                    )
                    connection.execute(
                        "INSERT INTO keyword_fts(rowid, tokens) VALUES (?, ?)",
                        (cursor.lastrowid, " ".join(tokens)),
                    )
                    frequency_deltas.update(set(tokens))

                self._apply_document_frequency(connection, frequency_deltas)
        finally:
            connection.close()

    def _apply_document_frequency(self, connection, frequency_deltas):
        """把文档频率的增减写入统计表，归零的词元直接删除。"""
        if not frequency_deltas:
            return
        connection.executemany(
            """
            INSERT INTO keyword_token_stats(token, document_frequency) VALUES (?, ?)
            ON CONFLICT(token) DO UPDATE SET
                document_frequency = document_frequency + excluded.document_frequency
            """,
            frequency_deltas.items(),
        )
        connection.execute("DELETE FROM keyword_token_stats WHERE document_frequency <= 0")

    def rebuild_token_statistics(self):
        """从已存储的子块文本重建词元文档频率统计，返回统计到的词元数量。

        统计量完全由 keyword_records.text 推导，因此给历史索引补齐统计无需访问 Milvus，
        秒级即可完成；全量重建索引时统计也会随写入一并生成。
        """
        connection = self._connect()
        try:
            with connection:
                connection.execute("DELETE FROM keyword_token_stats")
                document_frequency = Counter()
                for (text,) in connection.execute(
                    "SELECT text FROM keyword_records ORDER BY id"
                ):
                    document_frequency.update(set(self.tokenize(text)))
                connection.executemany(
                    """
                    INSERT INTO keyword_token_stats(token, document_frequency)
                    VALUES (?, ?)
                    """,
                    document_frequency.items(),
                )
        finally:
            connection.close()
        # 统计已就位，无需再探测；统计为空说明索引本身没有子块。
        self._statistics_available = bool(document_frequency)
        return len(document_frequency)

    def _statistics_are_available(self, connection):
        """判断词元文档频率统计是否可用。

        统计表为空说明索引是在引入剪枝之前建立的，此时阈值无从判定，剪枝会静默失效。
        “静默退化”正是本项目已经踩过的坑，因此这里显式告警而不是默默退回朴素切分。
        索引重建后本方法会自动转为可用，无需重启判定的缓存。
        """
        if self._statistics_available:
            return True
        if connection.execute("SELECT 1 FROM keyword_token_stats LIMIT 1").fetchone() is None:
            if not self._statistics_warning_emitted:
                logger.warning(
                    "关键词索引缺少词元文档频率统计，查询侧剪枝已失效；"
                    "请执行 scripts/rebuild_bm25_index.py 重建索引"
                )
                self._statistics_warning_emitted = True
            return False
        self._statistics_available = True
        return True

    def _prune_tokens(self, connection, tokens):
        """按文档频率丢弃查询侧噪声词元，返回剪枝后的词元列表。

        二元切分对文档侧是好事（不漏词），对查询侧是坏事：长查询会切出大量跨词边界的
        噪声词元（“情况”“具体”），它们几乎出现在每个子块里、区分度极低，却因为数量
        众多而在 BM25 的累加打分中淹没少数真正稀有的信号词元。这里按语料自身的文档
        频率剪枝，不依赖任何领域词表。
        """
        if self.max_document_ratio is None:
            return tokens

        unique_tokens = list(dict.fromkeys(tokens))
        document_count = connection.execute(
            "SELECT COUNT(*) FROM keyword_records"
        ).fetchone()[0]
        if document_count <= 0:
            return unique_tokens

        if not self._statistics_are_available(connection):
            return unique_tokens

        placeholders = ", ".join("?" for _ in unique_tokens)
        document_frequency = dict(
            connection.execute(
                f"""
                SELECT token, document_frequency FROM keyword_token_stats
                WHERE token IN ({placeholders})
                """,
                unique_tokens,
            ).fetchall()
        )
        threshold = max(
            document_count * self.max_document_ratio,
            MINIMUM_PRUNABLE_DOCUMENT_FREQUENCY,
        )
        pruned = [
            token
            for token in unique_tokens
            if document_frequency.get(token, 0) <= threshold
        ]
        # 整条查询都由高频词元组成时会把词元剪空，召回随之恒为零——那正是 OR 语义
        # 修复前的故障形态。这里退回未剪枝的词元，保证剪枝不会比朴素二元切分更差。
        return pruned or unique_tokens

    def search(self, query, source=None, filters=None, top_k=40):
        """按通用词元查询并恢复完整 LangChain Document。"""
        if top_k <= 0:
            return []
        tokens = self.query_tokenize(query)
        if not tokens:
            return []
        if filters is not None and not isinstance(filters, dict):
            raise ValueError("filters 必须是字典")

        # 过滤条件必须下推到 SQL，否则 LIMIT 会在过滤前截断，导致召回结果缺失。
        conditions = []
        parameters = []
        if source is not None:
            conditions.append("records.source = ?")
            parameters.append(source)
        for key, value in (filters or {}).items():
            if not isinstance(key, str) or not key.isidentifier():
                raise ValueError("filters 的字段名必须是 Python 标识符")
            conditions.append("json_extract(records.metadata, ?) = ?")
            parameters.extend([f"$.{key}", value])

        connection = self._connect()
        try:
            tokens = self._prune_tokens(connection, tokens)
            # 二元切分必然切出跨词边界的噪声词元（如“的营”“入是”），要求全部命中会让
            # 自然语言长查询恒定落空，因此改用 OR 召回，由 BM25 打分决定排序：命中越多、
            # 命中的词元越稀有，得分越高。
            fts_query = " OR ".join(f'"{token}"' for token in tokens)
            rows = connection.execute(
                f"""
                SELECT records.text, records.source, records.section,
                       records.document_id, records.parent_id, records.chunk_index,
                       records.chunk_count, records.metadata
                FROM keyword_fts
                JOIN keyword_records AS records ON records.id = keyword_fts.rowid
                WHERE {" AND ".join(["keyword_fts MATCH ?", *conditions])}
                ORDER BY bm25(keyword_fts)
                LIMIT ?
                """,
                [fts_query, *parameters, top_k],
            ).fetchall()
        finally:
            connection.close()

        return [
            Document(
                page_content=row[0],
                metadata={
                    "source": row[1],
                    "section": row[2],
                    "document_id": row[3],
                    "parent_id": row[4],
                    "chunk_index": row[5],
                    "chunk_count": row[6],
                    "metadata": json.loads(row[7]),
                },
            )
            for row in rows
        ]

def get_keyword_index():
    """获取关键词索引单例。"""
    global _keyword_index
    if _keyword_index is None:
        _keyword_index = KeywordIndex()
    return _keyword_index
