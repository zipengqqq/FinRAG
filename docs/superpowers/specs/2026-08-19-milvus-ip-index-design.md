# Milvus IP 索引初始化设计

## 目标

确保新建的 `financial_rag` collection 在写入文档前使用 IP 向量索引，避免
归一化的 BGE-M3 embedding 在检索时与 LangChain 自动创建的 L2 索引冲突。

## 范围

- 保持 `normalize_embeddings=True` 和检索端的 `metric_type="IP"`。
- 在首次创建 collection 时创建并加载 IP 类型的 HNSW 索引。
- 正常启动和入库时，索引创建应可重复执行而不破坏已有索引。
- 新增单元测试，验证 collection 初始化会向 Milvus 传入 `HNSW` 和 `IP`。

## 数据流程

首次入库时，`add_documents_to_milvus()` 发现 collection 不存在，调用
`init_collection()`。该函数创建 schema 后，创建并加载
`metric_type="IP"` 的 HNSW 索引；随后 LangChain vector store 写入已归一化的
embedding。检索仍传入 `metric_type="IP"`，因此与索引保持一致。

## 异常处理

创建 collection 或索引时的 Milvus 异常沿用现有文件上传失败处理流程。此次修改
不会自动删除、重建或修改已存在的 collection/index。

## 测试

在 `tests/test_vector_store.py` 中使用模拟的 `Collection`，断言新 collection
初始化时会为 `vector` 字段创建 HNSW + IP 索引并加载 collection。现有
vector-store 测试保持不变。
