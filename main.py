from fastapi import FastAPI
from starlette.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware

from chunker import split_md_content
from rag_graph import app as graph_workflow, retriever as graph_retriever
from request.ask_request import AskRequest
from request.doc_result import DocResult
from request.insert_request import InsertRequest
from request.search_request import SearchRequest
from vector_store import add_documents_to_milvus

app = FastAPI(title="FinRAG API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/assistant", summary="RAG调用")
async def assistant(req: AskRequest):
    state = {"query": req.query, "year": req.year}
    result = graph_workflow.invoke(state)
    return JSONResponse(status_code=200, content={'data': result.get('answer'), 'message': 'success'})

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
    uvicorn.run(app, host="127.0.0.1", port=8288)