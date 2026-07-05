from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from runner import run_eval
from judge import score_all
from output import save_to_csv
from db import init_db, save_preference, get_summary
from research_runner import run_research_eval, load_dataset, get_dataset_item, DatasetItem
from research_judge import score_all_research
from config import CANONICAL_SYSTEM_PROMPT, STANDARD_PROMPT_VARIANTS

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


class FetchArticleRequest(BaseModel):
    url: str


class FetchArticleResult(BaseModel):
    title: str
    text: str
    publisher: str = ""
    published_date: str = ""


class ResearchEvalRequest(BaseModel):
    # Dataset mode: provide item_id
    item_id: Optional[str] = None
    # Inline mode (from URL): provide these directly
    source_url: Optional[str] = None
    source_title: Optional[str] = None
    source_text: Optional[str] = None
    human_notes: Optional[str] = None
    # Common
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    prompt_variant: Optional[str] = None
    task_instructions: Optional[str] = None


class ResearchEvalResponse(BaseModel):
    results: list[ResearchScoredResult]
    item_id: str


@app.post("/fetch-article", response_model=FetchArticleResult)
def fetch_article_endpoint(request: FetchArticleRequest):
    """Fetch and extract article text from a URL using trafilatura."""
    try:
        import trafilatura
        downloaded = trafilatura.fetch_url(request.url)
        if not downloaded:
            raise HTTPException(
                status_code=422,
                detail="Could not fetch that URL — it may be blocked, paywalled, or require a browser. Try the Paste tab instead.",
            )

        result = trafilatura.bare_extraction(
            downloaded,
            include_comments=False,
            include_tables=False,
            with_metadata=True,
        )
        if not result:
            raise HTTPException(status_code=422, detail="Could not extract article text from that page.")

        def _get(key: str) -> str:
            val = getattr(result, key, None)
            return val.strip() if isinstance(val, str) and val.strip() else ""

        text = _get("text")
        if not text:
            raise HTTPException(
                status_code=422,
                detail="Extracted text is empty — page may be JS-rendered or paywalled. Try the Paste tab instead.",
            )

        return FetchArticleResult(
            title=_get("title") or request.url,
            text=text,
            publisher=_get("sitename") or "",
            published_date=_get("date") or "",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fetch failed: {e}")


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
    # ── Resolve the dataset item ──────────────────────────────────────────────
    if request.item_id:
        try:
            item = get_dataset_item(request.item_id)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        if not item.source_text.strip():
            raise HTTPException(
                status_code=400,
                detail=f"Dataset item '{request.item_id}' has no source article. "
                       f"Run python3 ingest_articles.py first.",
            )
    elif request.source_text and request.source_text.strip():
        # Inline mode: build a temporary DatasetItem from the provided article data
        item = DatasetItem(
            id="inline_eval",
            article_type="news",
            source_type="custom",
            high_context=False,
            expected_failure_categories=[],
            source_title=request.source_title or request.source_url or "Untitled",
            source_url=request.source_url or "",
            source_text=request.source_text,
            system_prompt=CANONICAL_SYSTEM_PROMPT,
            metadata={},
            prompt_variants=STANDARD_PROMPT_VARIANTS,
            benchmark_rationale="",
            human_notes=request.human_notes or "",
            human_override=False,
        )
    else:
        raise HTTPException(status_code=400, detail="Provide either item_id or source_text.")

    responses = run_research_eval(item_id=None, temperature=request.temperature,
                                  prompt_variant_name=request.prompt_variant, item=item,
                                  task_instructions=request.task_instructions)
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

    return ResearchEvalResponse(results=results, item_id=item.id)
