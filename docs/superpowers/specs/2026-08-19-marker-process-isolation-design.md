# Marker 解析进程隔离设计

## 目标

保留 Marker 作为 PDF 转 Markdown 的解析器，同时将其执行放到独立 Python 进程中。若
`pypdfium2` 的 `pdfium.dll` 崩溃，只终止解析进程，FastAPI/Uvicorn 主进程继续运行。

## 方案

新增轻量 worker 模块，仅导入并调用 `parse_pdf_marker()`。主服务用
`subprocess.run()` 启动该 worker，并传递 PDF 路径、输出目录和唯一结果文件路径。
worker 成功时将 Markdown 路径写入结果 JSON；可捕获的 Python 异常也写入结果 JSON
并以非零码退出。若 DLL 直接崩溃，结果文件不会出现，主服务根据非零返回码抛出明确的
解析失败异常。

不使用 `multiprocessing.spawn`，因为 Windows spawn 会重新导入 `main.py`，从而重复
初始化 RAG 和 reranker；独立 worker 只加载 Marker 依赖。

## 数据流程

`Service._step_parse_pdf()` 在本地 PDF 目录下创建唯一结果文件，启动 worker 并等待其
结束。成功时返回 worker 报告的 Markdown 路径，后续切块与 Milvus 入库流程不变。失败
时，现有 `upload_file_async()` 捕获异常并将文件状态标记为失败。

## 测试

测试主进程在 worker 成功时返回 Markdown 路径；在非零退出码或缺少结果文件时抛出
`RuntimeError`。测试通过模拟 `subprocess.run()`，不调用 Marker 模型或 PDFium DLL。
