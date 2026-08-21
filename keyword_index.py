"""SQLite FTS5 关键词索引。"""

import json
import re
import sqlite3
from pathlib import Path

from langchain_core.documents import Document


DEFAULT_DATABASE_PATH = Path(__file__).resolve().parent / "data" / "keyword_index.db"
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

    def __init__(self, database_path=DEFAULT_DATABASE_PATH):
        self.database_path = Path(database_path)
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
        finally:
            connection.close()

    def upsert_documents(self, documents):
        """按 document_id、parent_id、chunk_index 覆盖写入子块。"""
        connection = self._connect()
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
                        SELECT id FROM keyword_records
                        WHERE document_id = ? AND parent_id = ? AND chunk_index = ?
                        """,
                        stable_key,
                    ).fetchone()
                    if row is not None:
                        connection.execute("DELETE FROM keyword_fts WHERE rowid = ?", (row[0],))
                        connection.execute("DELETE FROM keyword_records WHERE id = ?", (row[0],))

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
                        (cursor.lastrowid, " ".join(tokenize(document.page_content))),
                    )
        finally:
            connection.close()

    def search(self, query, source=None, filters=None, top_k=40):
        """按通用词元查询并恢复完整 LangChain Document。"""
        if top_k <= 0:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        if filters is not None and not isinstance(filters, dict):
            raise ValueError("filters 必须是字典")

        fts_query = " AND ".join(f'"{token}"' for token in tokens)
        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT records.text, records.source, records.section,
                       records.document_id, records.parent_id, records.chunk_index,
                       records.chunk_count, records.metadata
                FROM keyword_fts
                JOIN keyword_records AS records ON records.id = keyword_fts.rowid
                WHERE keyword_fts MATCH ?
                ORDER BY bm25(keyword_fts)
                """,
                (fts_query,),
            ).fetchall()
        finally:
            connection.close()

        documents = []
        for row in rows:
            metadata = json.loads(row[7])
            if source is not None and row[1] != source:
                continue
            if filters and any(metadata.get(key) != value for key, value in filters.items()):
                continue
            documents.append(
                Document(
                    page_content=row[0],
                    metadata={
                        "source": row[1],
                        "section": row[2],
                        "document_id": row[3],
                        "parent_id": row[4],
                        "chunk_index": row[5],
                        "chunk_count": row[6],
                        "metadata": metadata,
                    },
                )
            )
            if len(documents) == top_k:
                break
        return documents

def get_keyword_index():
    """获取关键词索引单例。"""
    global _keyword_index
    if _keyword_index is None:
        _keyword_index = KeywordIndex()
    return _keyword_index
