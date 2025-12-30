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

user_prompt = """
用户问题：{query}

参考上下文：
{context_str}
"""

system_prompt = """
你是一个专业的金融数据分析助手，请基于提供的【上下文片段】回答用户的提问。
要求：
1. 回答必须基于上下文，不要编造。
2. 如果上下文中没有答案，直接说“根据现有文档无法回答”。
3. 回答时引用关键数据，并说明数据来源。
4. 保持回答条理清晰，可以使用 Markdown 格式。
"""

class AgentState(TypedDict):
    query: str  # 用户的问题
    documents: List[str]  # 检索到的文档内容
    answer: str  # 最终生成的答案
    year: int  # 年份


# 节点 1 -- 检索员
def retrieve_node(state: AgentState):
    logger.info(f"正在检索数据")
    query = state['query']
    year = state['year']
    docs = retriever.search(query, year=year)

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
workflow.add_node("retriever", retrieve_node)
workflow.add_node("generator", generate_node)

workflow.add_edge(START, "retriever")
workflow.add_edge("retriever", "generator")
workflow.add_edge("generator", END)

app = workflow.compile()

if __name__ == "__main__":
    app.invoke({"query": "业务咨询有多少条数据？"})