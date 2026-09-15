# 独立混合检索 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 FinRAG 实现独立的 SQLite FTS5 关键词召回，与 Milvus 向量召回通过 RRF 融合后统一重排，并可重建历史索引。

**Architecture:** `keyword_index.py` 独立负责 SQLite FTS5 的通用分词、增量写入、过滤查询及原子重建。`vector_store.py` 在 Milvus 入库成功后同步写入关键词索引；`retriever.py` 独立取得两路 Top 40 候选，按稳定子块键用 RRF 融合，再沿用 rerank、章节多样化和 `parent_id` 连续记录恢复。重建脚本分页读取 Milvus 记录，以临时 SQLite 文件重建并原子替换正式索引。

**Tech Stack:** Python 3.13、标准库 `sqlite3`、SQLite FTS5、Milvus、LangChain `Document`、pytest。

---

## 文件结构

- Create: `keyword_index.py`：关键词索引的唯一读写边界。
- Create: `scripts/rebuild_bm25_index.py`：从 Milvus 重建历史关键词索引的运维入口。
- Create: `tests/test_keyword_index.py`：SQLite FTS5 词项、写入、过滤、更新与重建测试。
- Modify: `vector_store.py`：有效子块写入 Milvus 后同步写入关键词索引。
- Modify: `retriever.py`：独立向量/关键词召回和 RRF 融合。
- Modify: `tests/test_vector_store.py`：验证有效子块同步进入关键词索引。
- Modify: `tests/test_retriever.py`：验证独立关键词召回、RRF 和故障降级。
- Create: `tests/test_rebuild_bm25_index.py`：验证分页重建与原子发布。

### Task 1: 实现通用关键词索引

**Files:**
- Create: `keyword_index.py`
- Test: `tests/test_keyword_index.py`

- [ ] **Step 1: 写入关键词索引失败测试**

```python
from langchain_core.documents import Document

from keyword_index import KeywordIndex


def _doc(text, parent_id, chunk_index, source="manual.md", metadata=None):
    return Document(
        page_content=text,
        metadata={
            "source": source,
            "section": "安装",
            "document_id": "document-1",
            "parent_id": parent_id,
            "chunk_index": chunk_index,
            "chunk_count": 1,
            "metadata": metadata or {},
        },
    )


def test_search_returns_exact_match_not_dependent_on_vector_candidates(tmp_path):
    index = KeywordIndex(tmp_path / "keyword.db")
    target = _doc("身份验证令牌配置", "parent-1", 0)
    index.upsert_documents([target])

    assert index.search("身份验证令牌", top_k=10) == [target]


def test_search_applies_source_and_metadata_filters(tmp_path):
    index = KeywordIndex(tmp_path / "keyword.db")
    guide = _doc("安装令牌", "parent-guide", 0, metadata={"category": "guide"})
    other = _doc("安装令牌", "parent-other", 0, source="other.md")
    index.upsert_documents([guide, other])

    assert index.search("安装令牌", source="manual.md", filters={"category": "guide"}) == [guide]


def test_upsert_replaces_document_with_same_stable_key(tmp_path):
    index = KeywordIndex(tmp_path / "keyword.db")
    index.upsert_documents([_doc("旧令牌", "parent-1", 0)])
    replacement = _doc("新令牌", "parent-1", 0)
    index.upsert_documents([replacement])

    assert index.search("旧令牌", top_k=10) == []
    assert index.search("新令牌", top_k=10) == [replacement]
```

- [ ] **Step 2: 运行失败测试确认接口尚不存在**

Run: `pytest tests/test_keyword_index.py -v`

Expected: FAIL，提示 `ModuleNotFoundError: No module named 'keyword_index'`。

- [ ] **Step 3: 实现最小 SQLite FTS5 索引**

```python
class KeywordIndex:
    def __init__(self, database_path):
        self.database_path = Path(database_path)

    def upsert_documents(self, documents):
        """按 document_id、parent_id、chunk_index 覆盖写入子块。"""

    def search(self, query, source=None, filters=None, top_k=40):
        """以通用词元查询 FTS5，并返回排序后的 Document。"""


def tokenize(text: str) -> list[str]:
    """生成英文、数字、下划线词元和中文二元词。"""
```

实现要求：

- 建立普通记录表，使用 `(document_id, parent_id, chunk_index)` 唯一键保存原文及 JSON 元数据。
- 建立以分词后文本为内容的 FTS5 表，并通过 `rowid` 关联普通记录表。
- `upsert_documents` 在单个 SQLite 事务中先删除同稳定键旧记录，再插入新记录与 FTS5 内容。
- `search` 以查询词元构造 AND FTS5 查询，使用 `bm25` 升序返回结果；对 `source` 和 JSON 元数据在 Python 中做精确等值过滤。
- 对空查询或没有词元的查询返回空列表；所有返回 `Document` 都恢复完整元数据。

- [ ] **Step 4: 运行关键词索引测试确认通过**

Run: `pytest tests/test_keyword_index.py -v`

Expected: PASS。

- [ ] **Step 5: 提交关键词索引实现**

```bash
git add keyword_index.py tests/test_keyword_index.py
git commit -m "feat: 添加持久化关键词索引"
```

### Task 2: 同步维护关键词索引

**Files:**
- Modify: `vector_store.py:160-195`
- Modify: `tests/test_vector_store.py`

- [ ] **Step 1: 写入同步入库失败测试**

```python
def test_add_documents_writes_only_valid_milvus_chunks_to_keyword_index(monkeypatch):
    valid_chunk = SimpleNamespace(metadata={"source": "guide.md", "section": "安装", "document_id": "doc", "parent_id": "parent"})
    invalid_chunk = SimpleNamespace(metadata={"source": "x" * 1025, "section": "安装", "document_id": "doc", "parent_id": "parent"})
    indexed = []

    monkeypatch.setattr(vector_store.connections, "connect", lambda **kwargs: None)
    monkeypatch.setattr(vector_store.utility, "has_collection", lambda name: True)
    monkeypatch.setattr(vector_store, "get_vector_store", lambda: SimpleNamespace(add_documents=lambda chunks: None))
    monkeypatch.setattr(vector_store, "get_keyword_index", lambda: SimpleNamespace(upsert_documents=lambda chunks: indexed.extend(chunks)))

    vector_store.add_documents_to_milvus([valid_chunk, invalid_chunk])

    assert indexed == [valid_chunk]
```

- [ ] **Step 2: 运行失败测试确认尚未写入关键词索引**

Run: `pytest tests/test_vector_store.py::test_add_documents_writes_only_valid_milvus_chunks_to_keyword_index -v`

Expected: FAIL，`vector_store` 中没有 `get_keyword_index` 或 `indexed` 为空。

- [ ] **Step 3: 在 Milvus 成功批次后写入关键词索引**

```python
from keyword_index import get_keyword_index


@time_consume
def add_documents_to_milvus(chunks, batch_size=256):
    # 保留现有字段校验、Milvus 写入与单条回退逻辑。
    # 全部有效子块的 Milvus 写入完成后执行：
    get_keyword_index().upsert_documents(valid_chunks)
```

要求：若关键词索引写入失败，记录错误后重新抛出异常，避免上层将该次混合索引写入记为成功。

- [ ] **Step 4: 运行向量库测试确认通过**

Run: `pytest tests/test_vector_store.py -v`

Expected: PASS。

- [ ] **Step 5: 提交同步入库改动**

```bash
git add vector_store.py tests/test_vector_store.py
git commit -m "feat: 同步维护关键词索引"
```

### Task 3: 实现 RRF 双路候选融合

**Files:**
- Modify: `retriever.py:17-150,241-256`
- Modify: `tests/test_retriever.py`

- [ ] **Step 1: 写入独立关键词召回和 RRF 失败测试**

```python
def test_search_reranks_keyword_hit_absent_from_vector_results(retriever_module, monkeypatch):
    semantic = _document("语义命中", parent_id="semantic", chunk_index=0)
    keyword_only = _document("精确编号 ABC-123", parent_id="keyword", chunk_index=0)
    store = FakeVectorStore([semantic])
    monkeypatch.setattr(retriever_module, "get_vector_store", lambda: store)
    monkeypatch.setattr(
        retriever_module,
        "get_keyword_index",
        lambda: types.SimpleNamespace(search=lambda **kwargs: [keyword_only]),
    )
    retriever = retriever_module.AdvancedRetriever()
    rerank_inputs = []
    retriever.rerank = lambda query, docs, top_k: rerank_inputs.extend(docs) or docs

    asyncio.run(retriever.search("ABC-123", top_k=2))

    assert keyword_only in rerank_inputs


def test_fuse_candidates_uses_rrf_and_deduplicates_stable_chunk_key(retriever_module):
    shared = _document("共同命中", parent_id="shared", chunk_index=0)
    vector_only = _document("向量命中", parent_id="vector", chunk_index=0)
    keyword_only = _document("词项命中", parent_id="keyword", chunk_index=0)

    fused = retriever_module.AdvancedRetriever._fuse_candidates(
        [vector_only, shared], [shared, keyword_only]
    )

    assert fused == [shared, vector_only, keyword_only]
```

- [ ] **Step 2: 运行失败测试确认缺少独立融合接口**

Run: `pytest tests/test_retriever.py::test_search_reranks_keyword_hit_absent_from_vector_results tests/test_retriever.py::test_fuse_candidates_uses_rrf_and_deduplicates_stable_chunk_key -v`

Expected: FAIL，`get_keyword_index` 或 `_fuse_candidates` 不存在。

- [ ] **Step 3: 用独立检索和 RRF 替换候选集内词项筛选**

```python
VECTOR_CANDIDATE_COUNT = 40
KEYWORD_CANDIDATE_COUNT = 40
RRF_K = 60


@classmethod
def _fuse_candidates(cls, vector_docs, keyword_docs):
    scores = {}
    documents = {}
    for ranked_docs in (vector_docs, keyword_docs):
        for rank, document in enumerate(ranked_docs, start=1):
            key = cls._document_key(document)
            documents.setdefault(key, document)
            scores[key] = scores.get(key, 0.0) + 1 / (RRF_K + rank)
    return [documents[key] for key in sorted(scores, key=lambda key: -scores[key])]
```

修改 `search`：

- Milvus 使用 `k=VECTOR_CANDIDATE_COUNT` 独立召回。
- `get_keyword_index().search(query=query, source=source, filters=filters, top_k=KEYWORD_CANDIDATE_COUNT)` 独立召回。
- 关键词索引初始化或查询失败时记录 warning 并使用空关键词候选，保留向量结果。
- 删除 `_lexical_terms`、`_lexical_score`、`_lexical_scores`、`_select_rerank_candidates` 与关联常量。
- RRF 结果作为 `rerank` 输入；其余章节分组、`parent_id` 展开代码保持不变。

- [ ] **Step 4: 运行检索器测试确认通过**

Run: `pytest tests/test_retriever.py -v`

Expected: PASS。

- [ ] **Step 5: 提交双路融合改动**

```bash
git add retriever.py tests/test_retriever.py
git commit -m "feat: 实现独立混合召回"
```

### Task 4: 重建历史关键词索引

**Files:**
- Create: `scripts/rebuild_bm25_index.py`
- Create: `tests/test_rebuild_bm25_index.py`

- [ ] **Step 1: 写入分页重建失败测试**

```python
def test_rebuild_reads_all_milvus_pages_and_publishes_index(monkeypatch, tmp_path):
    pages = [
        [{"text": "第一条", "source": "a.md", "section": "正文", "document_id": "doc", "parent_id": "p1", "chunk_index": 0, "chunk_count": 1, "metadata": {}}],
        [{"text": "第二条", "source": "a.md", "section": "正文", "document_id": "doc", "parent_id": "p2", "chunk_index": 0, "chunk_count": 1, "metadata": {}}],
        [],
    ]
    monkeypatch.setattr(rebuild_bm25_index, "iter_milvus_records", lambda page_size: iter(pages))
    target = tmp_path / "keyword.db"

    assert rebuild_bm25_index.rebuild_index(target, page_size=1) == 2
    assert KeywordIndex(target).search("第一条", top_k=10)[0].page_content == "第一条"
    assert KeywordIndex(target).search("第二条", top_k=10)[0].page_content == "第二条"
```

- [ ] **Step 2: 运行失败测试确认重建脚本不存在**

Run: `pytest tests/test_rebuild_bm25_index.py -v`

Expected: FAIL，提示无法导入 `scripts.rebuild_bm25_index`。

- [ ] **Step 3: 实现临时索引、分页读取和原子发布**

```python
def iter_milvus_records(page_size):
    """使用 pk > last_pk 的游标分页读取完整子块字段。"""


def rebuild_index(target_path, page_size=500):
    """写入同目录临时数据库，全部成功后以 os.replace 发布并返回记录数。"""
```

实现要求：

- 使用 `vector_store` 的 Milvus 连接参数和 collection 名称。
- 每页将字典记录转换为 `Document` 后写入临时 `KeywordIndex`。
- 重建过程中任一异常都删除临时文件并重新抛出，正式索引保持不变。
- CLI 参数支持 `--database-path` 和 `--page-size`，默认使用 `keyword_index.DEFAULT_DATABASE_PATH` 与 500。
- 成功后打印 `重建完成：<count> 条子块`。

- [ ] **Step 4: 运行重建测试确认通过**

Run: `pytest tests/test_rebuild_bm25_index.py -v`

Expected: PASS。

- [ ] **Step 5: 提交重建脚本**

```bash
git add scripts/rebuild_bm25_index.py tests/test_rebuild_bm25_index.py
git commit -m "feat: 添加关键词索引重建脚本"
```

### Task 5: 全量验证

**Files:**
- Verify: `keyword_index.py`
- Verify: `vector_store.py`
- Verify: `retriever.py`
- Verify: `scripts/rebuild_bm25_index.py`
- Verify: `tests/test_keyword_index.py`
- Verify: `tests/test_vector_store.py`
- Verify: `tests/test_retriever.py`
- Verify: `tests/test_rebuild_bm25_index.py`

- [ ] **Step 1: 运行混合检索相关测试**

Run: `pytest tests/test_keyword_index.py tests/test_vector_store.py tests/test_retriever.py tests/test_rebuild_bm25_index.py -q`

Expected: PASS。

- [ ] **Step 2: 运行全量测试**

Run: `pytest -q`

Expected: PASS，允许已有第三方弃用警告。

- [ ] **Step 3: 检查工作区和提交历史**

Run: `git status --short && git log --oneline -5`

Expected: 实现相关文件均已提交；未提交文件仅为预先存在的缓存、日志、IDE 文件或无关草案。
