import asyncio
import importlib
import sys
import types
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from langchain_core.documents import Document

from request.doc_result import DocResult
from request.insert_request import InsertRequest
from request.search_request import SearchRequest


def load_main(monkeypatch):
    fake_chunker = types.ModuleType("chunker")
    fake_chunker.split_md_content = lambda *args, **kwargs: []
    fake_graph = types.ModuleType("rag_graph")
    fake_graph.app = object()
    fake_graph.retriever = object()
    fake_vector_store = types.ModuleType("vector_store")
    fake_vector_store.add_documents_to_milvus = lambda chunks: None
    fake_main_service = types.ModuleType("service.main_service")
    fake_main_service.Service = type("Service", (), {})
    fake_chat_service = types.ModuleType("service.chat_service")
    fake_chat_service.ChatService = type("ChatService", (), {})
    monkeypatch.setitem(sys.modules, "chunker", fake_chunker)
    monkeypatch.setitem(sys.modules, "rag_graph", fake_graph)
    monkeypatch.setitem(sys.modules, "vector_store", fake_vector_store)
    monkeypatch.setitem(sys.modules, "service.main_service", fake_main_service)
    monkeypatch.setitem(sys.modules, "service.chat_service", fake_chat_service)
    sys.modules.pop("main", None)
    return importlib.import_module("main")


def load_main_service(monkeypatch):
    fake_db_util = types.ModuleType("utils.db_util")
    fake_db_util.create_session = lambda: None
    fake_file_model = types.ModuleType("entity.file_model")
    fake_file_model.FileModel = object
    fake_minio_util = types.ModuleType("utils.minio_util")
    fake_minio_util.BUCKET_NAME = "documents"
    fake_minio_util.minio_client = object()
    fake_vector_store = types.ModuleType("vector_store")
    fake_vector_store.add_documents_to_milvus = lambda chunks: None
    monkeypatch.setitem(sys.modules, "utils.db_util", fake_db_util)
    monkeypatch.setitem(sys.modules, "entity.file_model", fake_file_model)
    monkeypatch.setitem(sys.modules, "utils.minio_util", fake_minio_util)
    monkeypatch.setitem(sys.modules, "vector_store", fake_vector_store)
    sys.modules.pop("service.main_service", None)
    return importlib.import_module("service.main_service")


def test_insert_request_uses_generic_metadata_without_year():
    request = InsertRequest(text="知识内容", source="manual.md")

    assert request.metadata == {}
    assert "year" not in InsertRequest.model_fields


def test_search_request_uses_optional_source_and_filters_without_year():
    request = SearchRequest(query="如何安装", source="manual.md", filters={"category": "guide"})

    assert request.source == "manual.md"
    assert request.filters == {"category": "guide"}
    assert request.top_k == 5
    assert "year" not in SearchRequest.model_fields


def test_doc_result_exposes_metadata_without_year():
    result = DocResult(content="知识内容")

    assert result.metadata == {}
    assert "year" not in DocResult.model_fields


def test_ingest_passes_metadata_to_chunker(monkeypatch):
    main = load_main(monkeypatch)
    captured = {}

    def fake_split(text, source_filename, metadata):
        captured.update(text=text, source_filename=source_filename, metadata=metadata)
        return [Document(page_content="知识内容")]

    monkeypatch.setattr(main, "split_md_content", fake_split)
    monkeypatch.setattr(main, "add_documents_to_milvus", lambda chunks: captured.update(chunks=chunks))

    asyncio.run(
        main.ingest(
            InsertRequest(text="知识内容", source="manual.md", metadata={"category": "guide"})
        )
    )

    assert captured == {
        "text": "知识内容",
        "source_filename": "manual.md",
        "metadata": {"category": "guide"},
        "chunks": [Document(page_content="知识内容")],
    }


def test_search_passes_source_filters_and_top_k_to_retriever(monkeypatch):
    main = load_main(monkeypatch)
    captured = {}

    class FakeRetriever:
        async def search(self, query, source=None, filters=None, top_k=5):
            captured.update(query=query, source=source, filters=filters, top_k=top_k)
            return [
                Document(
                    page_content="知识内容",
                    metadata={
                        "source": "manual.md",
                        "section": "安装",
                        "metadata": {"category": "guide"},
                    },
                )
            ]

    monkeypatch.setattr(main, "graph_retriever", FakeRetriever())

    response = asyncio.run(
        main.search(
            SearchRequest(
                query="如何安装",
                source="manual.md",
                filters={"category": "guide"},
                top_k=3,
            )
        )
    )

    assert captured == {
        "query": "如何安装",
        "source": "manual.md",
        "filters": {"category": "guide"},
        "top_k": 3,
    }
    assert response["documents"][0].metadata == {"category": "guide"}


def test_step_ingest_md_passes_empty_generic_metadata(monkeypatch, tmp_path):
    markdown_path = tmp_path / "manual.md"
    markdown_path.write_text("# 安装", encoding="utf-8")
    captured = {}

    def fake_split(text, source_filename, metadata):
        captured.update(text=text, source_filename=source_filename, metadata=metadata)
        return []

    main_service = load_main_service(monkeypatch)
    monkeypatch.setattr(main_service, "split_md_content", fake_split)
    monkeypatch.setattr(main_service, "add_documents_to_milvus", lambda chunks: None)

    main_service.Service()._step_ingest_md(str(markdown_path), "annual-report-2024.pdf")

    assert captured == {
        "text": "# 安装",
        "source_filename": "annual-report-2024.pdf",
        "metadata": {},
    }


def test_chat_service_does_not_put_year_in_graph_state(monkeypatch):
    captured = {}

    class FakeWorkflow:
        async def astream_events(self, state, version):
            captured.update(state=state, version=version)
            if False:
                yield None

    fake_graph = types.ModuleType("rag_graph")
    fake_graph.app = FakeWorkflow()
    monkeypatch.setitem(sys.modules, "rag_graph", fake_graph)
    sys.modules.pop("service.chat_service", None)
    chat_service_module = importlib.import_module("service.chat_service")

    response = asyncio.run(
        chat_service_module.ChatService().sse_response(
            types.SimpleNamespace(query="如何安装", conversation_id="c1")
        )
    )

    async def consume():
        return [part async for part in response.body_iterator]

    asyncio.run(consume())

    assert captured["state"] == {"query": "如何安装", "history_str": ""}
    assert captured["version"] == "v2"
