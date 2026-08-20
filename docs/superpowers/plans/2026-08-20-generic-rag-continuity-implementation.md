# 通用 RAG 连续记录 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 Milvus collection 从年报专用 metadata 迁移为通用文档 schema，并在命中任一子块后恢复完整逻辑记录。

**Architecture:** `chunker.py` 为每个逻辑记录生成稳定的 `document_id`、`parent_id` 和顺序元数据；普通文本块与恢复后的表格行使用同一子块模型。`retriever.py` 先 rerank 子块，再按 `parent_id` 取回完整记录。Milvus 只保留通用来源和结构字段，业务属性放入 JSON metadata。

**Tech Stack:** Python 3.13、LangChain Document、langchain-milvus、pymilvus、pytest、FastAPI。

---

### Task 1: 通用切分契约

**Files:**
- Modify: `chunker.py`
- Test: `tests/test_chunker.py`

- [ ] **Step 1: 写入失败测试，定义普通文本的通用 metadata**

```python
def test_split_md_content_assigns_generic_record_metadata():
    chunks = split_md_content("# 安装\n请连接网络。", "guide.md")

    assert chunks[0].metadata["source"] == "guide.md"
    assert chunks[0].metadata["document_id"]
    assert chunks[0].metadata["parent_id"]
    assert chunks[0].metadata["chunk_index"] == 0
    assert chunks[0].metadata["chunk_count"] == 1
    assert chunks[0].metadata["metadata"] == {}
```

- [ ] **Step 2: 运行失败测试**

Run: `pytest tests/test_chunker.py::test_split_md_content_assigns_generic_record_metadata -v`

Expected: FAIL，因为当前函数要求 `year` 且不提供逻辑记录 metadata。

- [ ] **Step 3: 实现最小通用切分模型**

```python
def split_md_content(md_text, source_filename, metadata=None, document_id=None):
    # 生成或接收 document_id；每个逻辑记录都带 parent_id、顺序和 metadata。
```

- [ ] **Step 4: 运行测试验证通过**

Run: `pytest tests/test_chunker.py::test_split_md_content_assigns_generic_record_metadata -v`

Expected: PASS。

### Task 2: 表格续行恢复

**Files:**
- Modify: `chunker.py`
- Test: `tests/test_chunker.py`

- [ ] **Step 1: 写入失败测试，定义跨页表格续行合并**

```python
def test_split_md_content_merges_table_continuation_into_one_parent_record():
    chunks = split_md_content(TABLE_WITH_BLANK_FIRST_CELL_CONTINUATION, "report.md")

    row_chunks = [chunk for chunk in chunks if "承诺 A" in chunk.page_content]
    assert len({chunk.metadata["parent_id"] for chunk in row_chunks}) == 1
    assert "续行条款" in "".join(chunk.page_content for chunk in row_chunks)
```

- [ ] **Step 2: 运行失败测试**

Run: `pytest tests/test_chunker.py::test_split_md_content_merges_table_continuation_into_one_parent_record -v`

Expected: FAIL，因为当前递归切分不识别 Markdown 表格续行。

- [ ] **Step 3: 实现表格逻辑记录恢复和子块编号**

```python
def _table_records(lines, heading_path):
    # 空首列的续行追加到上一数据行；返回包含表头和完整行的逻辑记录。
```

- [ ] **Step 4: 运行测试验证通过**

Run: `pytest tests/test_chunker.py -v`

Expected: PASS。

### Task 3: 通用 Milvus schema 和入库验证

**Files:**
- Modify: `vector_store.py`
- Modify: `tests/test_vector_store.py`

- [ ] **Step 1: 写入失败测试，定义新 collection 的固定字段**

```python
assert {field["name"] for field in field_definitions} >= {
    "pk", "text", "vector", "document_id", "source", "section",
    "parent_id", "chunk_index", "chunk_count", "metadata",
}
assert "year" not in {field["name"] for field in field_definitions}
```

- [ ] **Step 2: 运行失败测试**

Run: `pytest tests/test_vector_store.py::test_init_collection_uses_generic_continuity_schema -v`

Expected: FAIL，因为现有 schema 固定包含 `year`，且缺少父记录字段。

- [ ] **Step 3: 新建通用 collection schema 并校验 metadata 长度**

```python
COLLECTION_NAME = "general_rag"
# `metadata` 使用 DataType.JSON；所有固定 VARCHAR 字段在写入前按 UTF-8 字节数校验。
```

- [ ] **Step 4: 运行测试验证通过**

Run: `pytest tests/test_vector_store.py -v`

Expected: PASS。

### Task 4: 父记录上下文恢复与通用筛选

**Files:**
- Modify: `retriever.py`
- Modify: `tests/test_retriever.py`

- [ ] **Step 1: 写入失败测试，定义从命中子块恢复完整父记录**

```python
async def test_search_expands_reranked_hits_to_complete_parent_records():
    docs = await retriever.search("查询")

    assert [doc.metadata["chunk_index"] for doc in docs] == [0, 1]
```

- [ ] **Step 2: 运行失败测试**

Run: `pytest tests/test_retriever.py::test_search_expands_reranked_hits_to_complete_parent_records -v`

Expected: FAIL，因为当前检索器只返回 rerank 命中的子块。

- [ ] **Step 3: 实现静态来源筛选和父记录精确查询**

```python
async def search(self, query, source=None, filters=None, top_k=5):
    # rerank 后按 parent_id 去重，再以 parent_id 查询并按 chunk_index 排序。
```

- [ ] **Step 4: 运行测试验证通过**

Run: `pytest tests/test_retriever.py -v`

Expected: PASS。

### Task 5: API 去年报化并更新设计稿

**Files:**
- Modify: `request/insert_request.py`
- Modify: `request/search_request.py`
- Modify: `request/doc_result.py`
- Modify: `main.py`
- Modify: `service/main_service.py`
- Modify: `rag_graph.py`
- Modify: `service/chat_service.py`
- Modify: `docs/superpowers/specs/2026-08-19-table-continuation-rag-design.md`
- Test: `tests/test_api_contracts.py`
- Test: `tests/test_rag_graph.py`

- [ ] **Step 1: 写入失败测试，定义无固定 `year` 的请求和图状态**

```python
def test_insert_request_accepts_generic_metadata_without_year():
    request = InsertRequest(text="内容", source="guide.md", metadata={"category": "guide"})
    assert request.metadata == {"category": "guide"}
```

- [ ] **Step 2: 运行失败测试**

Run: `pytest tests/test_api_contracts.py::test_insert_request_accepts_generic_metadata_without_year -v`

Expected: FAIL，因为 `InsertRequest.year` 目前是必填字段。

- [ ] **Step 3: 迁移 API 和工作流调用方**

```python
class InsertRequest(BaseModel):
    text: str
    source: str
    metadata: dict[str, object] = Field(default_factory=dict)
```

- [ ] **Step 4: 运行目标测试和回归测试**

Run: `pytest tests/test_api_contracts.py tests/test_rag_graph.py -v`

Expected: PASS。

### Task 6: 完整验证

**Files:**
- Verify: 全部已修改文件

- [ ] **Step 1: 运行全量测试**

Run: `python -m pytest`

Expected: PASS；若缺少 requirements 中声明的依赖，记录为环境阻塞而非实现失败。

- [ ] **Step 2: 检查空白错误与工作区差异**

Run: `git diff --check` and `git status --short`

Expected: 没有空白错误；仅包含本功能所需的源码、测试和文档改动。
