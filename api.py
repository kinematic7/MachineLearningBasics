from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import shutil
import os
import re
from jsonschema import validate, ValidationError
import json
import uvicorn

from backend.llm_pipeline import run_pipeline
from backend import vector_store, utils

app = FastAPI(title="Taskalytics API with Chroma RAG")

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ----------------------------
# Ping
# ----------------------------
@app.get("/ping")
async def ping():
    return {"status": "UP"}

# ----------------------------
# Upload JSON 
# ----------------------------
@app.post("/upload_json")
async def upload_json(file: UploadFile = File(...)):
    if not file.filename.endswith(".json"):
        return JSONResponse(status_code=400, content={"error": "Only JSON files are allowed."})

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        # Use ingest_json in vector_store
        #count = vector_store.ingest_json(file_path)
        count = vector_store.ingest_json_to_postgres(file_path)
        return {"status": "success", "ingested_rows": count}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ----------------------------
# Ask with RAG
# ----------------------------

@app.post("/smartevaluation")
async def smartevaluation(req: QuestionRequest):
    question = req.question    
    answer = await utils.ask_ollama(utils.GEMMA_12B_MODEL, question)
    return {"answer": answer}


@app.post("/ask", response_model=AnswerResponse)
async def ask(req: QuestionRequest):
    # Extract tags from [brackets]
    tags = re.findall(r'\[([^\]]+)\]', req.question)

    if tags:
        # Search for the tags in your vector database (returns list of dicts)
        context_data = vector_store.semantic_search_postgres(tags)
        
        # 1. Filter: ONLY keep content that actually contains one of the tags
        # We use .lower() for a case-insensitive match
        context_list = []
        for c in context_data:
            content = c['content']
            
            # Check if any tag (e.g., "Time-bound") is inside the text chunk
            if any(tag.lower() in content.lower() for tag in tags):                
                context_list.append(content)
        
        # 2. Join the filtered list into the final text block
        context_text = "\n\n".join(context_list)
    else:
        context_text = ""

    # Check if we actually found any useful context
    if context_text:
        augmented_question = f"""
            You are an AI Assistant.

            ### SOURCE OF TRUTH (MANDATORY)
            The following content is retrieved from the official documentation.
            You MUST base your response primarily on this text.
            Do NOT introduce features, tasks, or goals not explicitly mentioned.

            {context_text}

            ### USER REQUEST
            {req.question}

            ### STRICT REQUIREMENTS
            - At least 90% of the response must be grounded in the SOURCE OF TRUTH.
            - Do NOT include any excluded tasks or goals listed by the user.
            - Avoid generic SaaS language.
        """
    else:
        # If no tags or no context found, use the original question
        augmented_question = req.question

    print("===== FINAL PROMPT =====")
    print(augmented_question)
    print("========================")

    answer = await run_pipeline(augmented_question)

    SCHEMA = {
        "type": "object",
        "required": ["Goals"],
        "properties": {
            "Goals": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["Name", "Description", "Tasks"],
                    "properties": {
                        "Name": {
                            "type": "string",
                            "maxLength": 100
                        },
                        "Description": {
                            "type": "string",
                            "maxLength": 150
                        },
                        "Tasks": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "required": [
                                    "Task",
                                    "Description",
                                    "TaskPriority",
                                    "StartDate",
                                    "EndDate"
                                ],
                                "properties": {
                                    "Task": {
                                        "type": "string"
                                    },
                                    "Description": {
                                        "type": "string"
                                    },
                                    "TaskPriority": {
                                        "type": "integer",
                                        "enum": [1, 2, 3]
                                    },
                                    "StartDate": {
                                        "type": "string"
                                    },
                                    "EndDate": {
                                        "type": "string"
                                    }
                                },
                                "additionalProperties": False
                            }
                        }
                    },
                    "additionalProperties": False
                }
            }
        },
        "additionalProperties": False
        }

    try:
        answer = re.sub(r"^```(?:json)?\s*|\s*```$", "", answer, flags=re.IGNORECASE)
        data = json.loads(answer)        
        #print(data)
        validate(instance=data, schema=SCHEMA)
    except ValidationError as e:
       return JSONResponse(status_code=500, content={"error": f"Response validation error: {e.message}"})
          
    return {"answer": answer}

# This is the ONLY part modified to prevent the 'FastAPI has no attribute run' crash:
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)