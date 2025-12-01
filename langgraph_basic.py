from langgraph.graph import StateGraph, END

class State(dict):
    step: {"state", ""}

def start(state):
    state["step"].append("start")
    return state

def mid(state):
    state["step"].append("Middle")
    return state

def end(state):
    state["step"].append("End")
    return state

graph = StateGraph(State)

graph.add_node("start", start)
graph.add_node("middle", mid)
graph.add_node("end", end)

graph.set_entry_point("start")
graph.add_edge("start", "middle")
graph.add_edge("middle", "end")
graph.add_edge("end", END)

app = graph.compile()
result = app.invoke({"step": []})

print(result)

