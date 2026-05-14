# -*- coding: utf-8 -*-
import json
import re
import httpx
import asyncio

from datetime import datetime, timedelta
from config import OLLAMA_URL, SELECTED_MODEL, GEMMA_12B_MODEL

# Localized fallbacks
FALLBACKS = {
    "en": {
        "obj": "Description not generated.",
        "crit": ["Action needed."],
        "out": "Task contributes to goal.",
        "header": "Important Considerations"
    },
    "es": {
        "obj": "Descripción no generada.",
        "crit": ["Acción requerida."],
        "out": "La tarea contribuye al objetivo.",
        "header": "Consideraciones Importantes"
    }
}

async def ask_ollama(model: str, prompt: str, timeout: float = 120.0) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        return response.json()["message"]["content"].strip()

async def ask_json_with_validation(prompt: str, max_retries: int = 7, lang: str = "en") -> dict:
    for attempt in range(max_retries):
        try:
            raw = await asyncio.wait_for(
               ask_ollama(SELECTED_MODEL, prompt),
               timeout=100)  
            
            raw = raw.strip()
            # Clean markdown code blocks
            raw = re.sub(r'^```json\s*|\s*```$', '', raw, flags=re.MULTILINE)

            if not raw:
                await asyncio.sleep(0.5)
                continue

            data = json.loads(raw)
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, asyncio.TimeoutError, Exception) as e:
            print(f"[Warning] LLM Retry {attempt+1}/{max_retries} due to: {str(e)}")
            await asyncio.sleep(0.5)
            continue

    # Return localized fallback if all retries fail
    fb = FALLBACKS.get(lang, FALLBACKS["en"])
    return {
        "Task Objective": fb["obj"], 
        "Acceptance Criteria": fb["crit"], 
        "Approach and Outcome": fb["out"]
    }

def sanitize_task_title(title: str) -> str:
    # Only remove leading numbers and dots (e.g., "1. Task" -> "Task")
    # This preserves digits inside the title (e.g., "Web 3.0")
    title = re.sub(r"^\d+[\.\s\-]+", "", title).strip()
    return title if len(title) >= 3 else "Perform task step"

def build_html(task_objective: str, acceptance_criteria: list[str], approach_outcome: str, lang: str = "en") -> str:
    """
    Builds localized HTML content for the task description.
    """
    header_text = FALLBACKS.get(lang, FALLBACKS["en"])["header"]
    
    html_objective = f"<p style='font-size:14px; color:#2c3e50; margin-bottom:12px;'>{task_objective}</p>"
    
    # Acceptance Criteria Table
    rows = "".join(f"<tr><td style='padding:8px; border-bottom:1px solid #eee;'>{i+1}. {kp}</td></tr>" 
                   for i, kp in enumerate(acceptance_criteria))
    
    table = (
        f"<table style='width:100%; border-collapse:collapse; font-family:sans-serif; font-size:13px;'>"
        f"<tr><th style='text-align:left; background:#f8f9fa; padding:8px; border-bottom:2px solid #2c3e50;'>{header_text}</th></tr>"
        f"{rows}</table>"
    )

    return f"{html_objective}{table}<div style='margin-top:12px;'>{approach_outcome}</div>"

def iso_now_plus(days: int) -> str:
    return (datetime.utcnow() + timedelta(days=days)).isoformat() + "Z"