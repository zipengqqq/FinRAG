import asyncio
import importlib.util
import sys
import types
from pathlib import Path

import pytest
from langchain_core.documents import Document


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def retriever_module(monkeypatch):
    """隔离导入检索器，避免测试加载真实模型和连接 Milvus。"""
    fake_torch = types.ModuleType("torch")
    fake_torch.no_grad = lambda: _NoopContext()

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    class FakeModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def eval(self):
            pass

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = FakeTokenizer
    fake_transformers.AutoModelForSequenceClassification = FakeModel

    fake_decorator = types.ModuleType("decorator.time_consume")
    fake_decorator.time_consume = lambda func: func
    fake_logger = types.SimpleNamespace(
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
    )
    fake_logger_util = types.ModuleType("utils.logger_util")
    fake_logger_util.logger = fake_logger
    fake_model_paths = types.ModuleType("utils.model_paths")
    fake_model_paths.resolve_model_path = lambda *args: "fake-reranker"
    fake_vector_store = types.ModuleType("vector_store")
    fake_vector_store.get_vector_store = lambda: None
    fake_keyword_index = types.ModuleType("keyword_index")
    fake_keyword_index.get_keyword_index = lambda: None

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "decorator.time_consume", fake_decorator)
    monkeypatch.setitem(sys.modules, "utils.logger_util", fake_logger_util)
    monkeypatch.setitem(sys.modules, "utils.model_paths", fake_model_paths)
    monkeypatch.setitem(sys.modules, "vector_store", fake_vector_store)
    monkeypatch.setitem(sys.modules, "keyword_index", fake_keyword_index)

    module_name = "retriever_under_test"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, PROJECT_ROOT / "retriever.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class _NoopContext:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class FakeCollection:
    def __init__(self, records_by_parent):
        self.records_by_parent = records_by_parent
        self.queries = []

    def query(self, expr, output_fields):
        self.queries.append((expr, output_fields))
        parent_id = expr.split('"')[1]
        return self.records_by_parent.get(parent_id, [])


class FakeVectorStore:
    def __init__(self, initial_docs, records_by_parent=None):
        self.initial_docs = initial_docs
        self.col = FakeCollection(records_by_parent or {})
        self.searches = []

    def similarity_search(self, query, **kwargs):
        self.searches.append((query, kwargs))
        return self.initial_docs[: kwargs["k"]]


def _document(
    content, *, parent_id=None, chunk_index=None, source="report.md", section="章节"
):
    metadata = {"source": source, "section": section}
    if parent_id is not None:
        metadata.update(
            document_id="document-1",
            parent_id=parent_id,
            chunk_index=chunk_index,
            chunk_count=2,
            metadata={"report_year": 2024},
        )
    return Document(page_content=content, metadata=metadata)


def _record(content, parent_id, chunk_index, section="章节"):
    return {
        "text": content,
        "source": "report.md",
        "section": section,
        "document_id": "document-1",
        "parent_id": parent_id,
        "chunk_index": chunk_index,
        "chunk_count": 2,
        "metadata": {"report_year": 2024},
    }


def _set_rerank_scores(retriever, scores):
    def rerank(query, docs, top_k):
        ranked = []
        for doc in docs:
            score = scores[doc.page_content]
            doc.metadata["rerank_score"] = score
            ranked.append(doc)
        return sorted(ranked, key=lambda doc: doc.metadata["rerank_score"], reverse=True)

    retriever.rerank = rerank


def test_search_returns_all_chunks_of_the_matched_parent_in_chunk_order(
    retriever_module, monkeypatch
):
    hit = _document("第二段", parent_id="parent-a", chunk_index=1)
    store = FakeVectorStore(
        [hit],
        {"parent-a": [_record("第二段", "parent-a", 1), _record("第一段", "parent-a", 0)]},
    )
    monkeypatch.setattr(retriever_module, "get_vector_store", lambda: store)
    retriever = retriever_module.AdvancedRetriever()
    _set_rerank_scores(retriever, {"第二段": 0.9})

    results = asyncio.run(retriever.search("承诺内容", top_k=1))

    assert [document.page_content for document in results] == ["第一段", "第二段"]
    assert [document.metadata["chunk_index"] for document in results] == [0, 1]
    assert {document.metadata["rerank_score"] for document in results} == {0.9}
    assert store.col.queries == [
        (
            'parent_id == "parent-a"',
            [
                "text",
                "source",
                "section",
                "document_id",
                "parent_id",
                "chunk_index",
                "chunk_count",
                "metadata",
            ],
        )
    ]


def test_search_orders_parent_records_by_their_highest_rerank_score(
    retriever_module, monkeypatch
):
    lower_score = _document(
        "父记录 A", parent_id="parent-a", chunk_index=0, section="章节 A"
    )
    higher_score = _document(
        "父记录 B", parent_id="parent-b", chunk_index=0, section="章节 B"
    )
    store = FakeVectorStore(
        [lower_score, higher_score],
        {
            "parent-a": [_record("A 完整内容", "parent-a", 0, "章节 A")],
            "parent-b": [_record("B 完整内容", "parent-b", 0, "章节 B")],
        },
    )
    monkeypatch.setattr(retriever_module, "get_vector_store", lambda: store)
    retriever = retriever_module.AdvancedRetriever()
    _set_rerank_scores(retriever, {"父记录 A": 0.2, "父记录 B": 0.8})

    results = asyncio.run(retriever.search("查询", top_k=2))

    assert [document.page_content for document in results] == ["B 完整内容", "A 完整内容"]
    assert [document.metadata["rerank_score"] for document in results] == [0.8, 0.2]


def test_search_keeps_the_highest_scored_hit_when_a_parent_appears_multiple_times(
    retriever_module, monkeypatch
):
    lower_scored_hit = _document(
        "父记录 A 低分", parent_id="parent-a", chunk_index=0, section="章节 A"
    )
    other_parent_hit = _document(
        "父记录 B", parent_id="parent-b", chunk_index=0, section="章节 B"
    )
    higher_scored_hit = _document(
        "父记录 A 高分", parent_id="parent-a", chunk_index=1, section="章节 A"
    )
    for document, score in (
        (lower_scored_hit, 0.1),
        (other_parent_hit, 0.8),
        (higher_scored_hit, 0.9),
    ):
        document.metadata["rerank_score"] = score

    store = FakeVectorStore(
        [lower_scored_hit, other_parent_hit, higher_scored_hit],
        {
            "parent-a": [_record("A 完整内容", "parent-a", 0, "章节 A")],
            "parent-b": [_record("B 完整内容", "parent-b", 0, "章节 B")],
        },
    )
    monkeypatch.setattr(retriever_module, "get_vector_store", lambda: store)
    retriever = retriever_module.AdvancedRetriever()
    retriever.rerank = lambda query, docs, top_k: docs

    results = asyncio.run(retriever.search("查询", top_k=2))

    assert [document.page_content for document in results] == ["A 完整内容", "B 完整内容"]
    assert [document.metadata["rerank_score"] for document in results] == [0.9, 0.8]


def test_search_builds_source_and_json_metadata_filter_expression(
    retriever_module, monkeypatch
):
    store = FakeVectorStore([])
    monkeypatch.setattr(retriever_module, "get_vector_store", lambda: store)
    retriever = retriever_module.AdvancedRetriever()

    asyncio.run(
        retriever.search(
            "安装说明",
            source="manual.md",
            filters={"category": "guide", "report_year": 2024, "published": True},
        )
    )

    assert store.searches == [
        (
            "安装说明",
            {
                "k": 40,
                "expr": (
                    'source == "manual.md" and metadata["category"] == "guide" '
                    'and metadata["report_year"] == 2024 and metadata["published"] == true'
                ),
                "param": {"metric_type": "IP", "params": {"ef": 64}},
            },
        )
    ]


def test_search_reranks_keyword_hit_absent_from_vector_results(
    retriever_module, monkeypatch
):
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


def test_search_diversifies_results_by_source_and_section(retriever_module, monkeypatch):
    first = _document("同一章节的首个命中")
    duplicate = _document("同一章节的重复命中")
    other_section = Document(
        page_content="另一个章节的命中",
        metadata={"source": "report.md", "section": "另一章节"},
    )
    for document, score in ((first, 0.9), (duplicate, 0.8), (other_section, 0.7)):
        document.metadata["rerank_score"] = score
    store = FakeVectorStore([first, duplicate, other_section])
    monkeypatch.setattr(retriever_module, "get_vector_store", lambda: store)
    retriever = retriever_module.AdvancedRetriever()
    retriever.rerank = lambda query, docs, top_k: docs

    results = asyncio.run(retriever.search("查询", top_k=2))

    assert results == [first, other_section]


@pytest.mark.parametrize("filters", [{"bad-name": "guide"}, {"category": ["guide"]}])
def test_search_rejects_invalid_metadata_filters(retriever_module, filters):
    retriever = retriever_module.AdvancedRetriever()

    with pytest.raises(ValueError):
        asyncio.run(retriever.search("安装说明", filters=filters))


def test_search_keeps_legacy_document_without_parent_id_unexpanded(
    retriever_module, monkeypatch
):
    legacy = _document("旧数据")
    store = FakeVectorStore([legacy])
    monkeypatch.setattr(retriever_module, "get_vector_store", lambda: store)
    retriever = retriever_module.AdvancedRetriever()
    _set_rerank_scores(retriever, {"旧数据": 0.6})

    results = asyncio.run(retriever.search("旧数据", top_k=1))

    assert results == [legacy]
    assert results[0].metadata["rerank_score"] == 0.6
    assert store.col.queries == []
