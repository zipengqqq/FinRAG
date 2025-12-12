from typing import TypedDict

from langgraph.graph import StateGraph, START, END


#  1) 定义 State
class GraphState(TypedDict):
    topic: str
    joke: str

# 2) 构建工人
# 工人 A：负责写初稿
def generate_joke_node(state: GraphState):
    print(f"====正在思考笑话====")
    topic = state['topic']

    # 模拟 AI 生成笑话
    generated_joke = f"为什么{topic}过马路？为了去对面！"
    return {'joke': generated_joke}

# 工人 B：负责润色（加 Emoji）
def polish_joke_node(state: GraphState):
    print(f"====正在润色笑话====")

    # 模拟 AI 润色
    polished_joke = state['joke'] + " 😂😂😂 哈哈哈哈！"
    return {'joke': polished_joke}

# 3) 构建 Graph
workflow = StateGraph(GraphState)

# 把工人加入到图中，并起名字
workflow.add_node("generator", generate_joke_node)
workflow.add_node("polisher", polish_joke_node)

# 定义 Edges
workflow.add_edge(START, "generator")
workflow.add_edge("generator", "polisher")
workflow.add_edge("polisher", END)

# 编译
app = workflow.compile()

# 运行
results = app.invoke({"topic": "小鸡"})

print(results)