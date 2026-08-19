import os
from typing import TypedDict, List

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig

from retriever import retriever
from utils.logger_util import logger

load_dotenv()
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv('DEEPSEEK_API_KEY'),
    base_url=os.getenv('DEEPSEEK_BASE_URL'),
    temperature=0.7,
    streaming=True  # 开启流式
)

knowledge_base_system_prompt = """
你是一个严谨、简洁的通用知识库助手。请仅依据已检索到的知识库上下文回答用户问题，
不得编造上下文不支持的事实。

请先直接回答，再给出“依据与来源”部分，列出相关支撑要点。若上下文信息不足，
请明确说明。
"""

knowledge_base_user_prompt = """
问题：
{query}

检索到的知识库上下文：
{context_str}
"""

direct_answer_system_prompt = """
你是一个严谨、简洁的通用助手。当前未检索到可参考的知识库内容，请直接基于你的
通用知识回答用户问题。不要声称参考了文档，也不要提供未给出的来源。
"""

direct_answer_user_prompt = """
问题：
{query}
"""


def build_generation_messages(query: str, documents: List[str]):
    if documents:
        context_str = "\n\n".join(documents)
        return [
            SystemMessage(content=knowledge_base_system_prompt),
            HumanMessage(
                content=knowledge_base_user_prompt.format(
                    query=query, context_str=context_str
                )
            ),
        ]

    return [
        SystemMessage(content=direct_answer_system_prompt),
        HumanMessage(content=direct_answer_user_prompt.format(query=query)),
    ]

class AgentState(TypedDict):
    query: str  # 用户的问题
    documents: List[str]  # 检索到的文档内容
    answer: str  # 最终生成的答案
    year: int  # 年份
    standard_query: str # 标准化 query
    history_str: str # 历史对话

# 节点 0 -- 提示词重写
async def rewrite_node(state: AgentState):
    history_message = state.get('history_str', '当前暂无历史对话')
    query = state['query']
    messages = [
        SystemMessage(content="请基于给定的历史对话和当前问题，生成规范化检索问句，只返回该问句。"),
        HumanMessage(content=f"历史对话：{history_message}\n\n当前问题：{query}")
    ]
    response = await llm.ainvoke(messages)
    standard_query = response.content.strip()
    logger.info(f"提示词重写后的问题是：{standard_query}")
    return {"standard_query": standard_query}


# 节点 1 -- 检索员
async def retrieve_node(state: AgentState):
    logger.info(f"正在检索数据")
    query = state['query']
    year = state.get('year')
    docs = await retriever.search(query, year=year)

    return {"documents": [doc.page_content for doc in docs]}


# 节点 2 -- 写作员
async def generate_node(state: AgentState, config: RunnableConfig):
    logger.info(f"正在生成回答")
    messages = build_generation_messages(state['query'], state.get('documents', []))

    # 直接调用 ainvoke，LangGraph 会自动挂钩处理流
    response = await llm.ainvoke(messages, config=config)
    logger.info("回答生成结束")

    return {"answer": response.content}

# 构建图
workflow = StateGraph(AgentState)
workflow.add_node("rewrite", rewrite_node)
workflow.add_node("retriever", retrieve_node)
workflow.add_node("generator", generate_node)

workflow.add_edge(START, "rewrite")
workflow.add_edge("rewrite", "retriever")
workflow.add_edge("retriever", "generator")
workflow.add_edge("generator", END)

app = workflow.compile()

if __name__ == "__main__":
    app.invoke({"query": "业务咨询有多少条数据？"})
