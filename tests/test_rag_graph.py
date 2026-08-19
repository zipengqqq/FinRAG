import importlib
import sys
import types
from pathlib import Path

import langchain_openai


PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def load_rag_graph(monkeypatch):
    fake_retriever_module = types.ModuleType("retriever")
    fake_retriever_module.retriever = object()
    monkeypatch.setitem(sys.modules, "retriever", fake_retriever_module)
    monkeypatch.setattr(langchain_openai, "ChatOpenAI", lambda **kwargs: object())
    sys.modules.pop("rag_graph", None)

    return importlib.import_module("rag_graph")


def test_generation_messages_include_retrieved_knowledge_base_context(monkeypatch):
    rag_graph = load_rag_graph(monkeypatch)

    messages = rag_graph.build_generation_messages(
        "What does the manual say?", ["The product manual says X."]
    )

    assert "知识库上下文" in messages[0].content
    assert "The product manual says X." in messages[1].content
    assert "依据与来源" in messages[0].content


def test_generation_messages_omit_context_when_no_document_is_retrieved(monkeypatch):
    rag_graph = load_rag_graph(monkeypatch)

    messages = rag_graph.build_generation_messages("What does the manual say?", [])

    assert "通用助手" in messages[0].content
    assert "知识库上下文" not in messages[0].content
    assert "依据与来源" not in messages[0].content
    assert "What does the manual say?" in messages[1].content
