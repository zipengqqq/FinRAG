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
# llm = ChatOpenAI(
#     model='gpt-5.1',
#     api_key='sk-kKd9aJu7rMIulIun8AiqQNyqG3xhz6rhVLGoKu4orsSqDfz4',
#     base_url='https://api.lingyaai.cn/v1',
#     temperature=1,
#     streaming=True
# )

user_prompt = """
问题：
{query}

可用上下文：
{context_str}

请按照系统要求的输出格式作答：先直接回答，不要加任何“结论：”等标签；随后给“依据与来源：”的要点，不要复述问题。
"""

system_prompt = """
你是一个严谨、简洁的金融资料助手。只依据【上下文片段】回答，不得编造。

输出要求：
- 先用一到两句话直接回答（不加任何标签，严禁出现“结论：”字样）
- 随后给出“依据与来源：”的 2–4 条要点
- 不重复答案，不使用模板化长句，控制在 300 字以内
- 列表统一使用“- ”，避免嵌套列表与多层级
- 使用 Markdown 排版；必要强调用 **加粗**
- 若上下文无答案，输出“根据现有文档无法回答”

示例格式（用于风格约束，勿照抄）：
直接回答内容。
依据与来源：
- 关键原文片段或数据（来源说明）
"""

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
    year = state['year']
    docs = await retriever.search(query, year=year)

    return {"documents": [doc.page_content for doc in docs]}


# 节点 2 -- 写作员
async def generate_node(state: AgentState, config: RunnableConfig):
    logger.info(f"正在生成回答")
    context_str = "\n\n".join(state['documents'])

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt.format(
            query=state['query'],
            context_str=context_str
        ))
    ]

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