# Milvus IP 索引初始化实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新建 `financial_rag` collection 时在首次写入前创建并加载 IP HNSW 索引，消除检索 IP 与自动 L2 索引不匹配的问题。

**Architecture:** collection schema 创建成功后，由 `init_collection()` 立即创建 `vector` 字段的 HNSW/IP 索引并加载 collection。Embedding 归一化和检索端的 IP 参数保持不变，因此写入与搜索的度量统一。单元测试通过模拟 PyMilvus collection 验证索引创建及加载调用。

**Tech Stack:** Python 3.12、PyMilvus、langchain-milvus、pytest。

---

### Task 1: 为 collection 初始化补充 IP 索引测试

**Files:**

- Modify: `tests/test_vector_store.py`
- Test: `tests/test_vector_store.py`

- [ ] **Step 1: 写入失败测试**

在现有 `test_init_collection_allows_long_utf8_section_metadata` 后新增：

```python
def test_init_collection_creates_and_loads_ip_hnsw_index(monkeypatch):
    calls = []

    class FakeCollection:
        def __init__(self, **kwargs):
            calls.append(("create_collection", kwargs))

        def create_index(self, field_name, index_params):
            calls.append(("create_index", field_name, index_params))

        def load(self):
            calls.append(("load",))

    monkeypatch.setattr(vector_store.connections, "connect", lambda **kwargs: None)
    monkeypatch.setattr(vector_store, "FieldSchema", lambda **kwargs: kwargs)
    monkeypatch.setattr(vector_store, "CollectionSchema", lambda fields, description: fields)
    monkeypatch.setattr(vector_store, "Collection", FakeCollection)

    vector_store.init_collection()

    assert ("create_index", "vector", {
        "index_type": "HNSW",
        "metric_type": "IP",
        "params": {"M": 8, "efConstruction": 64},
    }) in calls
    assert ("load",) in calls
```

- [ ] **Step 2: 运行测试确认失败**

运行：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_vector_store.py::test_init_collection_creates_and_loads_ip_hnsw_index -v
```

预期：测试失败，因为当前 `init_collection()` 尚未调用 `create_index()` 和 `load()`。

- [ ] **Step 3: 实现最小索引初始化逻辑**

在 `vector_store.py` 的 `init_collection()` 中，在 `Collection(...)` 创建后、记录成功日志前增加：

```python
    collection.create_index(
        field_name="vector",
        index_params={
            "index_type": "HNSW",
            "metric_type": "IP",
            "params": {"M": 8, "efConstruction": 64},
        },
    )
    collection.load()
```

不修改 `get_embedding_model()` 的归一化设置、不修改 `AdvancedRetriever.search()` 的 IP 搜索参数，也不自动操作已存在的 collection。

- [ ] **Step 4: 运行目标测试确认通过**

运行：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_vector_store.py -v
```

预期：`test_init_collection_creates_and_loads_ip_hnsw_index` 与该文件其他测试均通过。

- [ ] **Step 5: 提交实现**

运行：

```powershell
git add vector_store.py tests/test_vector_store.py
git commit -m "fix: 初始化 Milvus IP 索引"
```

### Task 2: 验证现有 collection 的真实索引配置

**Files:**

- Modify: 无
- Test: 本地 Milvus collection 元数据

- [ ] **Step 1: 上传任意小文档或调用现有入库接口触发首次 collection 创建**

在应用重启后，通过现有 `/insert` 或 `/upload_file` 流程完成一次入库；不直接执行删除操作。

- [ ] **Step 2: 查询索引元数据**

运行：

```powershell
.\.venv\Scripts\python.exe -c "from pymilvus import connections, Collection; connections.connect(host='127.0.0.1', port='19530'); c=Collection('financial_rag'); print([(index.field_name, index.params) for index in c.indexes])"
```

预期：输出包含 `vector` 字段，并且参数中有 `index_type: HNSW` 和 `metric_type: IP`。

- [ ] **Step 3: 验证检索调用**

通过现有 `/assistant` 或 `/search` 请求执行一次查询。

预期：不再出现 `expected=L2][actual=IP`，并且请求获得正常响应。

## 自检

- 规格中的“首次建库即创建 IP 索引”由任务 1 实现。
- 规格中的“测试锁定 HNSW + IP”由任务 1 的失败测试与通过测试覆盖。
- 规格中“不改动已有 collection”由任务 1 的实现范围与任务 2 的仅查询验证覆盖。
- 文档中不存在 TBD、TODO 或未定义的实现步骤。
