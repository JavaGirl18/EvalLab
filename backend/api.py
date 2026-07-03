from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from runner import run_eval
from judge import score_all
from output import save_to_csv
from db import init_db, save_preference, get_summary
from research_runner import run_research_eval, load_dataset, get_dataset_item
from research_judge import score_all_research

app = FastAPI(title="EvalLab API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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


# ── Research Eval ─────────────────────────────────────────────────────────────

class DimensionScoreResult(BaseModel):
    severity: int
    confidence: str
    explanation: str
    source_evidence: str
    output_evidence: str


class ResearchScoredResult(BaseModel):
    model: str
    subject_model_config: dict
    item_id: str
    article_type: str
    source_title: str
    system_prompt: str
    eval_prompt: str
    prompt_variant_name: str
    response_text: str
    temperature: float
    timestamp: str
    judge_subject_model_config: dict
    dimension_scores: dict[str, DimensionScoreResult]
    overall_information_integrity_score: float
    overall_cultural_fidelity_score: float
    executive_summary: str
    most_significant_failures: list[str]
    suggested_improvements: str


class ResearchEvalRequest(BaseModel):
    item_id: str
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    prompt_variant: Optional[str] = None  # variant name; defaults to first if omitted


class ResearchEvalResponse(BaseModel):
    results: list[ResearchScoredResult]
    item_id: str


@app.get("/dataset")
def get_dataset():
    """Returns the pilot dataset item list (including source_text for display)."""
    items = load_dataset()
    return [
        {
            "id": item.id,
            "article_type": item.article_type,
            "source_type": item.source_type,
            "high_context": item.high_context,
            "expected_failure_categories": item.expected_failure_categories,
            "source_title": item.source_title,
            "source_url": item.source_url,
            "source_text": item.source_text,
            "metadata": item.metadata,
            "prompt_variants": item.prompt_variants,
            "benchmark_rationale": item.benchmark_rationale,
            "human_notes": item.human_notes,
            "human_override": item.human_override,
        }
        for item in items
    ]


@app.post("/research-eval", response_model=ResearchEvalResponse)
def run_research_evaluation(request: ResearchEvalRequest):
    try:
        item = get_dataset_item(request.item_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    if not item.source_text.strip():
        raise HTTPException(
            status_code=400,
            detail=f"Dataset item '{request.item_id}' has no source article. "
                   f"Populate source_text in pilot_dataset.json before running an evaluation.",
        )

    responses = run_research_eval(request.item_id, request.temperature, request.prompt_variant)
    scored = score_all_research(item, responses)

    results = [
        ResearchScoredResult(
            model=r.model,
            subject_model_config=r.subject_model_config,
            item_id=r.item_id,
            article_type=r.article_type,
            source_title=r.source_title,
            system_prompt=r.system_prompt,
            eval_prompt=r.eval_prompt,
            prompt_variant_name=r.prompt_variant_name,
            response_text=r.response_text,
            temperature=r.temperature,
            timestamp=r.timestamp,
            judge_subject_model_config=r.judge_subject_model_config,
            dimension_scores={
                cid: DimensionScoreResult(
                    severity=ds.severity,
                    confidence=ds.confidence,
                    explanation=ds.explanation,
                    source_evidence=ds.source_evidence,
                    output_evidence=ds.output_evidence,
                )
                for cid, ds in r.dimension_scores.items()
            },
            overall_information_integrity_score=r.overall_information_integrity_score,
            overall_cultural_fidelity_score=r.overall_cultural_fidelity_score,
            executive_summary=r.executive_summary,
            most_significant_failures=r.most_significant_failures,
            suggested_improvements=r.suggested_improvements,
        )
        for r in scored
    ]

    return ResearchEvalResponse(results=results, item_id=request.item_id)
