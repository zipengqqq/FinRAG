# 通用混合检索 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不加入领域规则的前提下，提升普通文本问答的检索召回，并正确记录异步检索耗时。

**Architecture:** `retriever.py` 从有限的扩大语义候选中补充通用词项候选，去重后由既有 reranker 统一排序；`time_consume.py` 为协程函数提供异步包装器。

**Tech Stack:** Python、LangChain Document、Milvus、pytest。

---

### Task 1: 通用候选融合

**Files:**
- Modify: `retriever.py`
- Test: `tests/test_retriever.py`

- [ ] 写入失败测试，证明第 50 个语义候选中包含更多查询词项的普通文本会被送入重排并返回。
- [ ] 将初始向量候选数量设为 200，提取通用中英文词项，选择词项匹配候选并与语义候选去重。
- [ ] 运行 `pytest tests/test_retriever.py -v`。

### Task 2: 异步计时

**Files:**
- Modify: `decorator/time_consume.py`
- Create: `tests/test_time_consume.py`

- [ ] 写入失败测试，证明异步函数的日志耗时覆盖 `await`。
- [ ] 为协程函数返回异步包装器，保留同步函数行为。
- [ ] 运行 `pytest tests/test_time_consume.py -v`。

### Task 3: 集成验证

**Files:**
- Verify: `retriever.py`、`decorator/time_consume.py`

- [ ] 运行目标测试、全量测试和 `git diff --check`。
- [ ] 调用 `/search`，确认营业额查询包含 `602,315`。
