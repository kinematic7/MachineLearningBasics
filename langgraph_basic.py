from langgraph.graph import StateGraph, END
from typing import TypedDict, List

def start(state):
    state["step"] = ["start"]
    return state

def mid(state):
    state["step"].append("mid")
    return state

def end(state):
    state["step"].append("end")
    return state


class State(TypedDict):
    step: List[str]

graph = StateGraph(State)
graph.add_node("start", start)
graph.add_node("mid", mid)
graph.add_node("end", end)

graph.set_entry_point("start")
graph.add_edge("start", "mid")
graph.add_edge("mid", "end")
graph.add_edge("end", END)

app = graph.compile()
result = app.invoke({"step": []})

print(result)
