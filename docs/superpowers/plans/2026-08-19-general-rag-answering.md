# General RAG Answering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `rag_graph.py` answer from retrieved knowledge-base content when available and answer directly when a query retrieves no documents.

**Architecture:** Keep the graph topology and retriever unchanged. Split generation prompt construction by `state["documents"]`: a populated list creates a context-grounded message pair, while an empty list creates a direct-answer message pair with no source requirement.

**Tech Stack:** Python, pytest, LangChain Core, LangGraph, ChatOpenAI.

---

### Task 1: Characterize and test prompt selection

**Files:**
- Create: `tests/test_rag_graph.py`
- Modify: `rag_graph.py`
- Test: `tests/test_rag_graph.py`

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_generate_node_uses_retrieved_documents_as_context(monkeypatch):
    messages = await invoke_generate_node(monkeypatch, ["A product manual says X."])

    assert "A product manual says X." in messages[1].content
    assert "knowledge-base context" in messages[0].content.lower()


@pytest.mark.asyncio
async def test_generate_node_omits_knowledge_base_context_when_no_documents_found(monkeypatch):
    messages = await invoke_generate_node(monkeypatch, [])

    assert "knowledge-base context" not in messages[0].content.lower()
    assert "source" not in messages[0].content.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_rag_graph.py -v`

Expected: FAIL because the existing generator always uses the same finance-specific, context-required prompt.

- [ ] **Step 3: Write minimal implementation**

```python
def build_generation_messages(query: str, documents: list[str]):
    if documents:
        context_str = "\n\n".join(documents)
        return [
            SystemMessage(content=knowledge_base_system_prompt),
            HumanMessage(content=knowledge_base_user_prompt.format(query=query, context_str=context_str)),
        ]
    return [
        SystemMessage(content=direct_answer_system_prompt),
        HumanMessage(content=direct_answer_user_prompt.format(query=query)),
    ]
```

Use domain-neutral language. In the context path, direct the model to base claims on the retrieved documents and provide source-oriented support. In the empty-result path, do not include document context or require sources.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_rag_graph.py -v`

Expected: PASS with both prompt-selection tests green.

- [ ] **Step 5: Commit**

```bash
git add rag_graph.py tests/test_rag_graph.py
git commit -m "feat: 支持空检索直接回答"
```

### Task 2: Verify the project test suite

**Files:**
- Test: `tests/`

- [ ] **Step 1: Run the complete suite**

Run: `pytest -q`

Expected: exit code 0 with the new RAG prompt tests and existing tests passing.

- [ ] **Step 2: Inspect the final change set**

Run: `git diff --check HEAD^ HEAD` and `git status --short`

Expected: no whitespace errors; unrelated pre-existing untracked files remain untouched.
