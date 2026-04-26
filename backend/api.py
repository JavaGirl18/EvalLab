from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from runner import run_eval
from judge import score_all
from output import save_to_csv
from db import init_db, save_preference, get_summary

app = FastAPI(title="EvalLab API")

import os

ALLOWED_ORIGINS = [
    "http://localhost:4200",
    os.getenv("FRONTEND_URL", ""),
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o for o in ALLOWED_ORIGINS if o],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the SQLite database when the server starts.
init_db()


class EvalRequest(BaseModel):
    task: str
    user_prompt: str
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)


class ScoreResult(BaseModel):
    model: str
    task: str
    variant: str
    user_prompt: str
    response_text: str
    temperature: float
    usefulness: int
    clarity: int
    confidence: int
    reliability: int
    judge_reasoning: str
    needs_human_review: bool
    bias_score: int
    bias_flags: list
    bias_assessment: str


class EvalResponse(BaseModel):
    results: list[ScoreResult]
    csv_path: str


class PreferenceRequest(BaseModel):
    model: str
    task: str
    variant: str
    user_prompt: str


@app.post("/eval", response_model=EvalResponse)
def run_evaluation(request: EvalRequest):
    valid_tasks = ["resume", "tax", "career", "budgeting"]
    if request.task not in valid_tasks:
        raise HTTPException(status_code=400, detail=f"Invalid task. Choose from: {valid_tasks}")

    responses = run_eval(
        task=request.task,
        user_prompt=request.user_prompt,
        temperature=request.temperature,
    )
    scored = score_all(responses)
    csv_path = save_to_csv(scored)

    return EvalResponse(
        results=[ScoreResult(**vars(r)) for r in scored],
        csv_path=csv_path,
    )


@app.post("/preference")
def record_preference(request: PreferenceRequest):
    save_preference(
        model=request.model,
        task=request.task,
        variant=request.variant,
        user_prompt=request.user_prompt,
    )
    return {"status": "saved"}


@app.get("/preferences/summary")
def preferences_summary():
    return get_summary()


@app.get("/health")
def health_check():
    return {"status": "ok"}
