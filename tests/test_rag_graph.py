import asyncio
import importlib
import sys
import types
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def load_rag_graph(monkeypatch):
    fake_retriever_module = types.ModuleType("retriever")
    fake_retriever_module.retriever = object()
    fake_langchain_openai = types.ModuleType("langchain_openai")
    model_kwargs = {}

    def fake_chat_openai(**kwargs):
        model_kwargs.update(kwargs)
        return object()

    fake_langchain_openai.ChatOpenAI = fake_chat_openai
    fake_langgraph = types.ModuleType("langgraph")
    fake_langgraph_graph = types.ModuleType("langgraph.graph")

    class FakeStateGraph:
        def __init__(self, state_type):
            self.state_type = state_type

        def add_node(self, *args):
            pass

        def add_edge(self, *args):
            pass

        def compile(self):
            return object()

    fake_langgraph_graph.StateGraph = FakeStateGraph
    fake_langgraph_graph.START = "start"
    fake_langgraph_graph.END = "end"
    monkeypatch.setitem(sys.modules, "retriever", fake_retriever_module)
    monkeypatch.setitem(sys.modules, "langchain_openai", fake_langchain_openai)
    monkeypatch.setitem(sys.modules, "langgraph", fake_langgraph)
    monkeypatch.setitem(sys.modules, "langgraph.graph", fake_langgraph_graph)
    sys.modules.pop("rag_graph", None)

    module = importlib.import_module("rag_graph")
    module._test_model_kwargs = model_kwargs
    return module


def test_rag_graph_uses_deepseek_v4_flash_by_default(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)

    rag_graph = load_rag_graph(monkeypatch)

    assert rag_graph._test_model_kwargs["model"] == "deepseek-v4-flash"


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


def test_retrieve_node_uses_standard_query_when_available(monkeypatch):
    rag_graph = load_rag_graph(monkeypatch)
    captured = {}

    class FakeRetriever:
        async def search(self, query):
            captured["query"] = query
            return [types.SimpleNamespace(page_content="matched document")]

    monkeypatch.setattr(rag_graph, "retriever", FakeRetriever())

    result = asyncio.run(
        rag_graph.retrieve_node(
            {
                "query": "What commitments were made? Show the original text.",
                "standard_query": "What commitments did Wang Chuanfu make in the non-compete undertaking?",
            }
        )
    )

    assert captured == {
        "query": "What commitments did Wang Chuanfu make in the non-compete undertaking?",
    }
    assert result == {"documents": ["matched document"]}


def test_rewrite_node_uses_a_knowledge_base_retrieval_prompt(monkeypatch):
    rag_graph = load_rag_graph(monkeypatch)
    captured = {}

    class FakeLlm:
        async def ainvoke(self, messages):
            captured["messages"] = messages
            return types.SimpleNamespace(content="rewritten query")

    monkeypatch.setattr(rag_graph, "llm", FakeLlm())

    result = asyncio.run(
        rag_graph.rewrite_node(
            {
                "query": "Show the original text.",
                "history_str": "user: When was the non-compete undertaking signed?",
            }
        )
    )

    system_prompt = captured["messages"][0].content
    assert "知识库检索" in system_prompt
    assert "实体" in system_prompt
    assert "原文" in system_prompt
    assert result == {"standard_query": "rewritten query"}
