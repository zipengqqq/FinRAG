from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from starlette.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from request.file_download_request import FileDownloadRequest
from request.file_preview_request import FilePreviewRequest
from service.chat_service import ChatService
from utils.logger_util import logger
from utils.response_util import build_response
from chunker import split_md_content
from rag_graph import app as graph_workflow, retriever as graph_retriever
from request.ask_request import AskRequest
from request.doc_result import DocResult
from request.document_request import DocumentRequest
from request.insert_request import InsertRequest
from request.search_request import SearchRequest
from service.main_service import Service
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
chat_service = ChatService()

@app.post("/upload_file", summary="上传文件，文件解析")
async def upload_file(
    file: UploadFile = File(..., description='文件'),
    background_tasks: BackgroundTasks = None
):
    file_content = await file.read()
    background_tasks.add_task(
        service.upload_file_async,
        filename=file.filename,
        file_content=file_content,
        content_type=file.content_type
    )
    return build_response({"message": "文件上传成功，正在后台处理", "filename": file.filename})



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

@app.post("/file_preview", summary="文件预览")
async def file_preview(req: FilePreviewRequest):
    resp = service.file_preview(req)
    if not resp:
        return JSONResponse(status_code=404, content={'message': '文件不存在'})
    return resp


@app.post("/assistant", summary="RAG调用")
async def assistant(req: AskRequest):
    return await chat_service.sse_response(req)

@app.post("/insert", summary="文本插入到向量数据库")
async def ingest(req: InsertRequest):
    chunks = split_md_content(req.text, source_filename=req.source, year=req.year)
    add_documents_to_milvus(chunks)
    return JSONResponse(status_code=200, content={'message': 'success'})

@app.post("/search", summary="向量数据库检索")
async def search(req: SearchRequest):
    docs = await graph_retriever.search(req.query, year=req.year, source=req.source, top_k=req.top_k)
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
