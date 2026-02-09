import json
import os
from datetime import datetime
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

app = FastAPI(title="Smart Task Analyzer", version="1.0.0")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class TaskRequest(BaseModel):
    text: str = Field(..., min_length=1, description="User natural language task")


class TaskAnalysis(BaseModel):
    description: str = Field(..., description="Parsed task description")
    priority: Literal["low", "medium", "high"] = Field(
        ..., description="Priority level"
    )
    due_date: str | None = Field(
        None, description="ISO-8601 datetime string for due date"
    )
    category: str = Field(..., description="Task category label")


SYSTEM_PROMPT = """
You are a task parsing assistant. Extract structured fields from the user input.

Return ONLY valid JSON with the following fields:
- description: short task summary
- priority: one of low/medium/high
- due_date: ISO-8601 datetime (e.g., 2026-02-10T15:00:00). If not provided, use null.
- category: short category label (e.g., work, personal, study). If not provided, use "general".

Rules & Defaults:
- If priority is missing, set it to "medium".
- If due_date is missing, set it to null.
- If category is missing, set it to "general".
- Use the user's locale context if time is relative (e.g., "tomorrow at 3pm").
- Do not include any extra keys.

API call example (OpenAI Python client):
client.responses.create(
  model="gpt-4.1-mini",
  input=[{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": text}]
)

Error handling guidance:
- If the model output is not valid JSON, return a 502 with a clear error.
- If required fields are missing, apply defaults or return 422.
""".strip()


@app.post("/analyze-task")
async def analyze_task(payload: TaskRequest):
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY is not set. Configure it to call the OpenAI API.",
        )
    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": payload.text},
            ],
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"OpenAI API error: {exc}") from exc

    try:
        content = response.output_text
        data = json.loads(content)
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail=f"Invalid JSON from model: {exc}"
        ) from exc

    if data.get("priority") is None:
        data["priority"] = "medium"
    if data.get("category") in (None, ""):
        data["category"] = "general"
    if data.get("due_date") is None:
        data["due_date"] = None

    try:
        validated = TaskAnalysis(**data)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=json.loads(exc.json())) from exc

    try:
        if validated.due_date is not None:
            datetime.fromisoformat(validated.due_date)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid due_date: {exc}") from exc

    return JSONResponse(content=validated.model_dump())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
