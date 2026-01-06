from fastapi import FastAPI, Depends
from starlette.responses import JSONResponse, Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from request.file_download_request import FileDownloadRequest
from request.file_preview_request import FilePreviewRequest
from utils.logger_util import logger
from utils.response_util import build_response
from chunker import split_md_content
from rag_graph import app as graph_workflow, retriever as graph_retriever
from request.ask_request import AskRequest
from request.doc_result import DocResult
from request.document_request import DocumentRequest
from request.insert_request import InsertRequest
from request.search_request import SearchRequest
from service.service import Service
from vector_store import add_documents_to_milvus

app = FastAPI(title="FinRAG API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

service = Service()
@app.post("/document", summary="文档查询")
async def document(req: DocumentRequest):
    data = service.list(req)
    return build_response(data)

@app.post("/file_download", summary="文件下载")
async def file_download(req: FileDownloadRequest):
    resp = service.file_download(req)
    if not resp:
        return JSONResponse(status_code=404, content={'message': '文件不存在'})
    return resp

@app.get("/file_preview", summary="文件预览")
async def file_preview(req: FilePreviewRequest):
    resp = service.file_preview(req)
    if not resp:
        return JSONResponse(status_code=404, content={'message': '文件不存在'})
    return resp


@app.post("/assistant", summary="RAG调用")
async def assistant(req: AskRequest):
    async def event_generator():
        state = {"query": req.query, "year": req.year}

        # 核心：使用 astream_events (version="v2")
        # 它可以捕获图内部所有发生的事件，包括 LLM 生成的 token
        collectd_messages = []
        async for event in graph_workflow.astream_events(state, version="v2"):

            # 过滤事件类型：我们只关心 Chat Model 的流式输出
            if event["event"] == "on_chat_model_stream":
                # 获取 chunk 内容
                chunk = event["data"]["chunk"]
                if hasattr(chunk, "content") and chunk.content:
                    # 输出 SSE 格式数据
                    collectd_messages.append(chunk.content)
                    yield f"data: {chunk.content}\n\n"
        logger.info(f"问题：{req.query}\nLLM响应内容：{''.join(collectd_messages)}")

        # 结束信号
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )

@app.post("/insert", summary="文本插入到向量数据库")
async def ingest(req: InsertRequest):
    chunks = split_md_content(req.text, source_filename=req.source, year=req.year)
    add_documents_to_milvus(chunks)
    return JSONResponse(status_code=200, content={'message': 'success'})

@app.post("/search", summary="向量数据库检索")
async def search(req: SearchRequest):
    docs = graph_retriever.search(req.query, year=req.year, source=req.source, top_k=req.top_k)
    out = []
    for d in docs:
        out.append(DocResult(
            content=d.page_content,
            rerank_score=d.metadata.get("rerank_score"),
            source=d.metadata.get("source"),
            section=d.metadata.get("section"),
            year=d.metadata.get("year"),
        ))
    return {"documents": out}


if __name__ == "__main__":
    import uvicorn
    # uvicorn.run("main:app", host="127.0.0.1", port=8288, reload=True)
    uvicorn.run("main:app", host="127.0.0.1", port=8288)
