import random
from langgraph.graph import StateGraph, END

# -------------------------------
# Node functions
# -------------------------------
def start(state):
    state["step"].append("start")
    state["value"] = random.randint(1, 20)
    return state

def router(state):
    state["step"].append("router")
    return state

def high(state):
    state["step"].append("high")
    return state

def low(state):
    state["step"].append("low")
    return state

def end(state):
    state["step"].append("end")
    return state

# -------------------------------
# Conditional routing logic
# -------------------------------
def route_logic(state):
    if state["value"] > 10:
        return "go_high"
    else:
        return "go_low"

# -------------------------------
# Build graph
# -------------------------------
graph = StateGraph(dict)

graph.add_node("start", start)
graph.add_node("router", router)
graph.add_node("high", high)
graph.add_node("low", low)
graph.add_node("end", end)

graph.set_entry_point("start")

graph.add_edge("start", "router")

# ⭐ Correct: conditional edges
graph.add_conditional_edges(
    "router",
    route_logic,
    {
        "go_high": "high",
        "go_low": "low"
    }
)

graph.add_edge("high", "end")
graph.add_edge("low", "end")
graph.add_edge("end", END)

# -------------------------------
# Run
# -------------------------------
app = graph.compile()
result = app.invoke({"step": []})

print(result)
