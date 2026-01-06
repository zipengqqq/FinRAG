# FinRAG
## 文件作用
- test.py: 测试文件
- main.py: fastapi 启动程序
- download_model.py: 下载重排序和嵌入模型
- parser.py: md 解析器，将 pdf 转换为 markdown 格式文件
- chunker.py: 文本分割器，将 markdown 文本分割为 List[Document]
- vector_store.py: 可以将 List[Document] 插入到 milvus中
- rag_graph.py: rag 工作流，可以实现 rag 搜索

## 文件