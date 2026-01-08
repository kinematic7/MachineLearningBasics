import httpx
from fastapi import FastAPI
from pydantic import BaseModel
from typing import TypedDict

from langgraph.graph import StateGraph, END

# =========================
# CONFIG
# =========================

OLLAMA_URL = "http://localhost:11434/api/chat"
LLAMA_MODEL = "llama3.1:8b"
GEMMA_MODEL = "gemma2:9b" # gemma2:27b

# =========================
# FASTAPI
# =========================

app = FastAPI(title="LangGraph Local LLM Q&A Bot")

class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    answer: str

# =========================
# LANGGRAPH STATE
# =========================

class GraphState(TypedDict):
    question: str
    base_answer: str
    final_answer: str


# =========================
# OLLAMA CALL
# =========================

async def ask_ollama(model: str, messages: list[dict], timeout: float = 120.0) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        return response.json()["message"]["content"]


# =========================
# GRAPH NODES
# =========================

async def llama_node(state: GraphState) -> GraphState:
    """Step 1: Generate base answer"""
    answer = await ask_ollama(
        LLAMA_MODEL,
        [
            {"role": "system", "content": "You are an expert healthcare assistant. Provide a concise answer."},
            {"role": "user", "content": state["question"]}
        ]
    )

    return {
        **state,
        "base_answer": answer
    }


async def gemma_node(state: GraphState) -> GraphState:
    """Step 2: Convert answer to strict SMART JSON"""

    instructions = f"""
        You are an AI that ONLY outputs raw JSON. No markdown, no commentary.

        Refine or enhance the following answer while keeping its meaning:
        {state["base_answer"]}

        Return JSON in the exact structure:

                
        "Goals": [
            {
            "Name": "Healthcare Software Modernization",
            "Description": "Design and implement a secure, scalable healthcare software platform to improve patient care, data interoperability, and operational efficiency.",
            "Tasks": [
                {
                "Chief Complaint": "Patient complaint"
                "Diagnosis": "Diagnosis details",
                "Procedure": "Procedure details",
                "StartDate": "2026-02-01T09:00:00",
                "EndDate": "2026-05-30T17:00:00",
                "Details": "<p>The core patient management module is the foundation of the healthcare software platform and is responsible for securely handling patient demographics, medical histories, appointments, and care coordination workflows. This task focuses on building a compliant, reliable, and scalable module that integrates seamlessly with existing clinical systems while maintaining strict data privacy standards such as HIPAA. The module will enable healthcare providers to quickly access accurate patient information, reduce administrative overhead, and improve clinical decision-making. Emphasis will be placed on role-based access control, audit logging, and data validation to ensure that only authorized users can view or modify sensitive records. The system will be designed with extensibility in mind, allowing future integration with laboratory systems, billing platforms, and third-party health information exchanges. Performance, usability, and maintainability are key considerations, ensuring that clinicians can efficiently use the system in high-pressure environments without technical friction or delays.</p><br><br><table><tr><th>Key Area</th><th>Focus</th></tr><tr><td><b>Security</b></td><td>Encryption, role-based access, audit logs</td></tr><tr><td><b>Compliance</b></td><td>HIPAA-aligned data handling and storage</td></tr><tr><td><b>Interoperability</b></td><td>FHIR-ready APIs for system integration</td></tr><tr><td><b>Usability</b></td><td>Clinician-friendly workflows and UI</td></tr></table>"
                }
            ]
            }
        ]            
    """
    #4. ~200 word conclusion with a wikipedia.org URL in blue font opening new tab

    enhanced = await ask_ollama(
        GEMMA_MODEL,
        [
            {"role": "system", "content": "You are an expert AI JSON generator."},
            {"role": "user", "content": instructions}
        ],
        timeout=200.0
    )

    return {
        **state,
        "final_answer": enhanced
    }


# =========================
# BUILD GRAPH
# =========================

graph = StateGraph(GraphState)

graph.add_node("llama", llama_node)
graph.add_node("gemma", gemma_node)
graph.set_entry_point("llama")
graph.add_edge("llama", "gemma")
graph.add_edge("gemma", END)
llm_graph = graph.compile()


# =========================
# API ENDPOINT
# =========================

@app.post("/ask", response_model=AnswerResponse)
async def ask(req: QuestionRequest):
    result = await llm_graph.ainvoke(
        {"question": req.question}
    )

    return {"answer": result["final_answer"]}

