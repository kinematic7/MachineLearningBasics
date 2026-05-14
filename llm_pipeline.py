# -*- coding: utf-8 -*-
import json
import asyncio
import re
from dataclasses import dataclass, field
from typing import List, TypedDict, Dict
from datetime import datetime, timedelta
from langgraph.graph import StateGraph, END

# Ensure these utilities are correctly defined in your local environment
from config import OLLAMA_URL, SELECTED_MODEL, GEMMA_12B_MODEL
from .utils import ask_ollama, ask_json_with_validation, sanitize_task_title

# =========================
# LOCALIZATION CONFIG
# =========================

STRINGS = {
    "en": {
        "considerations": "Important Considerations",
        "criteria": "Acceptance Criteria",
        "given": "GIVEN",
        "when": "WHEN",
        "then": "THEN",
        "fallback_desc": "Description not generated.",
        "as_a": "As a... I want... So that...",
        "task_fallback": "Task"
    },
    "es": {
        "considerations": "Consideraciones Importantes",
        "criteria": "Criterios de Aceptación",
        "given": "DADO",
        "when": "CUANDO",
        "then": "ENTONCES",
        "fallback_desc": "Descripción no generada.",
        "as_a": "Como... quiero... para...",
        "task_fallback": "Tarea"
    }
}

# =========================
# DATA MODELS
# =========================

@dataclass
class TaskContent:
    title: str
    task_objective: str
    acceptance_criteria: List[str]
    approach_outcome: str

@dataclass
class GoalContent:
    name: str
    description: str
    tasks: List[TaskContent] = field(default_factory=list)

class GraphState(TypedDict):
    purpose: str
    language: str # 'en' or 'es'
    goal: GoalContent
    task_titles: List[str]
    final_answer: str

# =========================
# NODES
# =========================

async def goal_name_node(state: GraphState) -> GraphState:
    # 1. Robust Language Detection via LLM
    detect_prompt = f"""
    Analyze the language of the text below. 
    Respond ONLY with the ISO language code: 'es' for Spanish, 'en' for English.
    Text: {state['purpose']}
    """
    detected_lang = await ask_ollama(SELECTED_MODEL, detect_prompt)
    state["language"] = "es" if "es" in detected_lang.lower() else "en"

    # 2. Generate Goal Name in detected language
    prompt = f"""
        Return a concise professional goal title (max 30 chars).
        Plain text only.
        LANGUAGE: Respond ONLY in the language of the provided Purpose.

        Purpose:
        {state['purpose']}
        """
    name = await ask_ollama(SELECTED_MODEL, prompt)
    
    # Regex allows alphanumeric plus Spanish characters (áéíóúñ)
    clean_name = re.sub(r'[^A-Za-z0-9 áéíóúÁÉÍÓÚñÑ]+', '', name.strip())
    state["goal"] = GoalContent(name=clean_name, description="")
    return state

async def goal_description_node(state: GraphState) -> GraphState:
    prompt = f"""
        Write a professional goal description (max 50 chars).
        Plain text only.
        Goal: {state['goal'].name}
        LANGUAGE: Respond ONLY in the language of the Goal: {state['goal'].name}
        """
    state["goal"].description = await ask_ollama(SELECTED_MODEL, prompt)
    return state

async def task_titles_node(state: GraphState) -> GraphState:
    prompt = f"""
        Return a JSON array of 4 unique, descriptive task titles for the following goal.
        DO NOT use markdown formatting. Output ONLY the raw JSON array.
        Goal: {state['goal'].name}                
        LANGUAGE: Respond ONLY in the language of the Goal: {state['goal'].name}
        """
    raw = await ask_ollama(GEMMA_12B_MODEL, prompt)
    lang = state["language"]
    
    try:
        raw = raw.strip()
        # Handle cases where LLM might wrap in ```json
        if raw.startswith("```"):
            raw = re.sub(r'^```json\s*|```$', '', raw, flags=re.MULTILINE)
        titles = json.loads(raw)
    except:
        titles = [f"{STRINGS[lang]['task_fallback']} {i+1}" for i in range(4)]

    if not isinstance(titles, list) or len(titles) != 4:
        titles = [f"{STRINGS[lang]['task_fallback']} {i+1}" for i in range(4)]

    state["task_titles"] = [sanitize_task_title(t) for t in titles]
    return state

async def task_generation_node(state: GraphState) -> GraphState:
    tasks: List[TaskContent] = []
    lang = state["language"]
    labels = STRINGS[lang]

    for title in state["task_titles"]:
        prompt = f"""
            Context: {state['purpose']}
            Task: {title}
            Goal: {state['goal'].name}

            Act as a Senior Product Owner. Generate a structured User Story in JSON format. 
    
            CRITICAL CONSTRAINTS:            
            - Use active voice. 
            - Ensure "Task Objective" follows: "{labels['as_a']}".
            - Respond ONLY in the language of the Task: {title}.
    
            Return ONLY a valid JSON object with these EXACT keys:
            {{
              "Task Objective": "...",
              "Acceptance Criteria": ["4-5 specific requirements"],
              "Approach and Outcome": "Technical flow using {labels['given']}/{labels['when']}/{labels['then']}.",
              "Conclusion": "One sentence success state."
            }}
            """

        data = await ask_json_with_validation(prompt, max_retries=7)

        objective = data.get("Task Objective", labels["fallback_desc"])
        outcome = data.get("Approach and Outcome", "")
        conclusion = data.get("Conclusion", "")

        # -------- Bilingual Gherkin Parsing --------        
        given, when, then = "", "", ""
        # Support for GIVEN/DADO, WHEN/CUANDO, THEN/ENTONCES
        pattern = r"(GIVEN|DADO)\s*[:\-]?\s*(.*?)(?=WHEN|CUANDO|THEN|ENTONCES|$)|(WHEN|CUANDO)\s*[:\-]?\s*(.*?)(?=GIVEN|DADO|THEN|ENTONCES|$)|(THEN|ENTONCES)\s*[:\-]?\s*(.*?)(?=GIVEN|DADO|WHEN|CUANDO|$)"
        
        matches = re.findall(pattern, outcome, re.IGNORECASE | re.DOTALL)
        for m in matches:
            if m[1]: given = m[1].replace("▪️", "").strip()
            if m[3]: when = m[3].replace("▪️", "").strip()
            if m[5]: then = m[5].replace("▪️", "").strip()

        # -------- HTML Styling (Localized Labels) --------
        html_objective = f"<p style='font-size:14px; line-height:1.6; margin:0 0 12px 0; color:#2c3e50;'>{objective}</p>"

        key_points = data.get("Acceptance Criteria", [])
        html_considerations = ""
        if key_points:
            html_considerations = f"<table style='width:100%; border-collapse:collapse; margin-top:10px; font-size:14px;'>"
            html_considerations += f"<tr><th style='text-align:left; background-color:#f4f6f8; color:#2c3e50; padding:8px; border-bottom:2px solid #2c3e50;'>{labels['considerations']}</th></tr>"
            for idx, point in enumerate(key_points, 1):
                html_considerations += f"<tr><td style='padding:8px 6px; border-bottom:1px solid #e1e4e8;'><span style='color:#7f8c8d; font-weight:bold;'>{idx}.</span> {point}</td></tr>"
            html_considerations += "</table>"

        html_criteria = f"""
            <table style='width:100%; border-collapse:collapse; margin-top:14px; font-size:14px;'>
            <tr><th colspan='2' style='text-align:left; background-color:#f4f6f8; color:#2c3e50; padding:8px; border-bottom:2px solid #2c3e50;'>{labels['criteria']}</th></tr>
            <tr><td style='width:14%; padding:8px; color:#1f4fd8; font-weight:bold; vertical-align:top;'>{labels['given']}</td><td style='padding:8px; line-height:1.5;'>{given}</td></tr>
            <tr><td style='width:14%; padding:8px; color:#b26a00; font-weight:bold; vertical-align:top;'>{labels['when']}</td><td style='padding:8px; line-height:1.5;'>{when}</td></tr>
            <tr><td style='width:14%; padding:8px; color:#1b7f5a; font-weight:bold; vertical-align:top;'>{labels['then']}</td><td style='padding:8px; line-height:1.5;'>{then}</td></tr>
            </table>"""

        html_conclusion = f"<p style='margin-top:14px; font-size:14px; color:#2c3e50;'>{conclusion.strip()}</p>" if conclusion.strip() else ""

        tasks.append(TaskContent(
            title=title,
            task_objective=objective,
            acceptance_criteria=key_points,
            approach_outcome=html_objective + html_considerations + html_criteria + html_conclusion
        ))

    state["goal"].tasks = tasks
    return state

# =========================
# ASSEMBLY & GRAPH
# =========================

def assemble_final(goal: GoalContent) -> str:
    now = datetime.utcnow()
    output = {"Goals": [{"Name": goal.name, "Description": goal.description, "Tasks": []}]}
    durations = [3, 3, 3, 2]  
    current_start = now + timedelta(days=1)

    for idx, task in enumerate(goal.tasks):
        end_date = current_start + timedelta(days=durations[idx])
        output["Goals"][0]["Tasks"].append({
            "Task": task.title,
            "Description": task.approach_outcome,
            "TaskPriority": 1 if idx < 3 else 2,
            "StartDate": current_start.isoformat() + "Z",
            "EndDate": end_date.isoformat() + "Z"
        })
        current_start = end_date

    return json.dumps(output, ensure_ascii=False)

async def assembly_node(state: GraphState) -> GraphState:
    state["final_answer"] = assemble_final(state["goal"])
    return state

def build_graph():
    graph = StateGraph(GraphState)
    nodes = [("goal_name", goal_name_node), ("goal_description", goal_description_node), 
             ("task_titles", task_titles_node), ("task_generation", task_generation_node), 
             ("assemble", assembly_node)]
    
    for name, func in nodes:
        graph.add_node(name, func)
    
    graph.set_entry_point("goal_name")
    graph.add_edge("goal_name", "goal_description")
    graph.add_edge("goal_description", "task_titles")
    graph.add_edge("task_titles", "task_generation")
    graph.add_edge("task_generation", "assemble")
    graph.add_edge("assemble", END)
    
    return graph.compile()

llm_graph = build_graph()

async def run_pipeline(purpose: str) -> str:
    result = await llm_graph.ainvoke({"purpose": purpose})
    return result["final_answer"]