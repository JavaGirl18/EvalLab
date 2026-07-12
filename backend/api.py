import json
from pathlib import Path
from typing import Optional
from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field
from runner import run_eval
from judge import score_all
from output import save_to_csv
from db import init_db, save_preference, get_summary
from research_runner import run_research_eval, load_dataset, get_dataset_item, DatasetItem
from research_judge import score_all_research
from config import CANONICAL_SYSTEM_PROMPT, STANDARD_PROMPT_VARIANTS, JUDGE_MODEL
from batch_db import (
    get_batch, get_batch_items, list_batches,
    cancel_batch, get_completed_results,
)
from batch_runner import validate_request, create_batch_from_request, run_batch, compute_batch_summary
from human_review_db import (
    create_human_review, get_human_review, list_human_reviews,
    count_reviews, mark_review_archived,
)
from human_review_selection import select_runs_for_review
from human_review_export import (
    build_packet_snapshot, export_html, export_json_packets,
    export_csv_template, export_csv_context,
)
from human_review_import import import_from_csv, import_from_json
from human_review_db import get_responses_for_review as _get_responses_for_review
from human_review_stats import compute_agreement_summary, compute_emerging_patterns

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
            "characteristics": item.characteristics,
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


# ── Experiment configs ────────────────────────────────────────────────────────

EXPERIMENTS_DIR = Path(__file__).parent / "data" / "experiments"


@app.get("/experiments")
def list_experiments():
    """List all experiment config files as summaries."""
    if not EXPERIMENTS_DIR.exists():
        return []
    results = []
    for path in sorted(EXPERIMENTS_DIR.glob("*.json")):
        with open(path) as f:
            exp = json.load(f)
        results.append({
            "experiment_id":   exp.get("experiment_id", path.stem),
            "experiment_name": exp.get("experiment_name", path.stem),
            "research_phase":  exp.get("research_phase", ""),
            "status_phase":    (exp.get("status") or {}).get("phase", "planned"),
        })
    return results


@app.get("/experiment/{experiment_id}")
def get_experiment_config(experiment_id: str):
    """Return the full experiment config JSON by ID."""
    path = EXPERIMENTS_DIR / f"{experiment_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found.")
    with open(path) as f:
        return json.load(f)


@app.put("/experiment/{experiment_id}")
def update_experiment_config(experiment_id: str, body: dict):
    """Write UI state back to the experiment file. Preserves fields the UI doesn't own."""
    path = EXPERIMENTS_DIR / f"{experiment_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found.")

    with open(path) as f:
        existing = json.load(f)

    # Map ExperimentMeta → file schema, preserving unmanaged fields
    if "experiment_name" in body:
        existing["experiment_name"] = body["experiment_name"]
    if "phase" in body:
        existing["research_phase"] = body["phase"].replace("_", " ").title()
    if "research_objective" in body:
        existing["research_objective"] = body["research_objective"]
    if "research_question" in body:
        existing["research_question"] = body["research_question"]
    if "hypothesis" in body:
        existing["hypothesis"] = body["hypothesis"]
    if "prompt_variant" in body:
        existing["prompt_variant"] = body["prompt_variant"]
    if "models" in body:
        existing["subject_models"] = body["models"]
    if "task_instructions" in body:
        tt = existing.get("transformation_task") or {}
        tt["instructions"] = body["task_instructions"]
        existing["transformation_task"] = tt
    if "status" in body:
        existing["status"] = body["status"]

    with open(path, "w") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
        f.write("\n")

    return existing


# ── Batch Eval ────────────────────────────────────────────────────────────────

class BatchRequest(BaseModel):
    experiment_id:     str
    item_ids:          list[str]
    prompt_variants:   list[str]
    models:            list[str]
    temperature:       float          = Field(default=0.0, ge=0.0, le=1.0)
    task_instructions: Optional[str]  = None
    experiment_meta:   Optional[dict] = None
    max_concurrency:   int            = Field(default=3, ge=1, le=10)
    retry_limit:       int            = Field(default=3, ge=1, le=5)
    resume_existing:   bool           = True
    max_runs:          Optional[int]  = None


class BatchRunItemResponse(BaseModel):
    run_key:        str
    item_id:        str
    prompt_variant: str
    model:          str
    status:         str
    attempt:        int
    error:          Optional[str]
    started_at:     Optional[str]
    completed_at:   Optional[str]


class BatchStatusResponse(BaseModel):
    batch_id:      str
    experiment_id: str
    status:        str
    total:         int
    completed:     int
    failed:        int
    skipped:       int
    pending:       int
    warnings:      list[str]
    created_at:    str
    started_at:    Optional[str]
    completed_at:  Optional[str]
    items:         list[BatchRunItemResponse]


class BatchSummaryResponse(BaseModel):
    batch_id:      str
    experiment_id: str
    status:        str
    total:         int
    completed:     int
    failed:        int
    pending:       int
    created_at:    str
    started_at:    Optional[str]
    completed_at:  Optional[str]


@app.post("/batch", response_model=BatchStatusResponse, status_code=201)
async def create_batch_endpoint(request: BatchRequest, background_tasks: BackgroundTasks):
    req_dict = request.model_dump()
    warnings, errors = validate_request(req_dict)
    if errors:
        raise HTTPException(status_code=422, detail={"errors": errors, "warnings": warnings})

    batch_id = create_batch_from_request(req_dict, warnings)
    background_tasks.add_task(run_batch, batch_id)

    batch = get_batch(batch_id)
    items = get_batch_items(batch_id)
    return BatchStatusResponse(
        **{k: batch[k] for k in ("batch_id","experiment_id","status","total","completed",
                                  "failed","skipped","pending","warnings","created_at",
                                  "started_at","completed_at")},
        items=[BatchRunItemResponse(**{k: it[k] for k in BatchRunItemResponse.model_fields}) for it in items],
    )


@app.get("/batch/{batch_id}", response_model=BatchStatusResponse)
def get_batch_status(batch_id: str):
    batch = get_batch(batch_id)
    if not batch:
        raise HTTPException(status_code=404, detail=f"Batch '{batch_id}' not found.")
    items = get_batch_items(batch_id)
    return BatchStatusResponse(
        **{k: batch[k] for k in ("batch_id","experiment_id","status","total","completed",
                                  "failed","skipped","pending","warnings","created_at",
                                  "started_at","completed_at")},
        items=[BatchRunItemResponse(**{k: it[k] for k in BatchRunItemResponse.model_fields}) for it in items],
    )


@app.get("/batch/{batch_id}/results")
def get_batch_results(batch_id: str):
    batch = get_batch(batch_id)
    if not batch:
        raise HTTPException(status_code=404, detail=f"Batch '{batch_id}' not found.")
    results = get_completed_results(batch_id)
    summary = compute_batch_summary(batch_id)
    config  = json.loads(batch["config"]) if isinstance(batch["config"], str) else batch["config"]
    return {
        "batch_id":        batch_id,
        "experiment_id":   batch["experiment_id"],
        "experiment_meta": config.get("experiment_meta"),
        "results":         results,
        "summary":         summary,
    }


@app.post("/batch/{batch_id}/cancel")
def cancel_batch_endpoint(batch_id: str):
    batch = get_batch(batch_id)
    if not batch:
        raise HTTPException(status_code=404, detail=f"Batch '{batch_id}' not found.")
    if batch["status"] in ("completed", "cancelled"):
        raise HTTPException(status_code=400, detail=f"Batch is already {batch['status']}.")
    cancel_batch(batch_id)
    return {"batch_id": batch_id, "status": "cancelled"}


@app.get("/batches", response_model=list[BatchSummaryResponse])
def list_batches_endpoint():
    rows = list_batches()
    return [
        BatchSummaryResponse(**{k: r[k] for k in BatchSummaryResponse.model_fields})
        for r in rows
    ]


# ── Human Review ──────────────────────────────────────────────────────────────

class GenerateReviewsRequest(BaseModel):
    batch_id:               str
    review_round:           int   = 1
    blinded:                bool  = True
    max_source_chars:       Optional[int] = None
    severity_threshold:     Optional[int] = None
    confidence_threshold:   Optional[str] = None
    failure_count_threshold: Optional[int] = None
    random_pct:             Optional[float] = None
    manual_run_keys:        Optional[list[str]] = None


@app.post("/human-reviews/generate", status_code=201)
def generate_human_reviews(request: GenerateReviewsRequest):
    batch = get_batch(request.batch_id)
    if not batch:
        raise HTTPException(status_code=404, detail=f"Batch '{request.batch_id}' not found.")

    if not any([
        request.severity_threshold is not None,
        request.confidence_threshold is not None,
        request.failure_count_threshold is not None,
        request.random_pct is not None,
        request.manual_run_keys,
    ]):
        raise HTTPException(
            status_code=400,
            detail="At least one selection rule is required (severity_threshold, "
                   "confidence_threshold, failure_count_threshold, random_pct, or manual_run_keys).",
        )

    candidates = select_runs_for_review(
        request.batch_id,
        severity_threshold=request.severity_threshold,
        confidence_threshold=request.confidence_threshold,
        failure_count_threshold=request.failure_count_threshold,
        random_pct=request.random_pct,
        manual_run_keys=request.manual_run_keys,
        review_round=request.review_round,
    )

    if not candidates:
        return {"created": 0, "skipped_duplicates": 0, "review_ids": [], "message": "No eligible runs found."}

    experiment_id = batch["experiment_id"]
    config = json.loads(batch["config"]) if isinstance(batch["config"], str) else batch["config"]
    task_instructions = config.get("task_instructions") or "Produce a faithful transformation of the source content."

    created_ids = []
    skipped = 0

    for candidate in candidates:
        row    = candidate["row"]
        result = candidate["result"]

        try:
            snapshot = build_packet_snapshot(
                run_key=candidate["run_key"],
                result=result,
                row=row,
                blinded=request.blinded,
                max_source_chars=request.max_source_chars,
            )
        except Exception as exc:
            skipped += 1
            continue

        review_id = create_human_review(
            experiment_id=experiment_id,
            batch_id=request.batch_id,
            run_key=candidate["run_key"],
            dataset_item_id=row["item_id"],
            prompt_variant=row["prompt_variant"],
            transformation_task=task_instructions,
            subject_model=row["model"],
            judge_model=result.get("judge_subject_model_config", {}).get("model_id", JUDGE_MODEL),
            taxonomy_version=snapshot.get("taxonomy_version", "v1"),
            rubric_version=snapshot.get("rubric_version", "v1"),
            review_round=request.review_round,
            selection_reasons=candidate["selection_reasons"],
            blinded=request.blinded,
            packet_snapshot=snapshot,
        )

        if review_id is None:
            skipped += 1
        else:
            created_ids.append(review_id)

    return {
        "created":            len(created_ids),
        "skipped_duplicates": skipped,
        "review_ids":         created_ids,
    }


@app.get("/human-reviews")
def list_human_reviews_endpoint(
    batch_id:      Optional[str] = Query(None),
    experiment_id: Optional[str] = Query(None),
    status:        Optional[str] = Query(None),
    review_round:  Optional[int] = Query(None),
):
    reviews = list_human_reviews(
        batch_id=batch_id,
        experiment_id=experiment_id,
        status=status,
        review_round=review_round,
    )
    # Never include subject_model in list view
    for r in reviews:
        r.pop("subject_model", None)
    return reviews


@app.get("/human-reviews/counts")
def human_review_counts(batch_id: str = Query(...)):
    return count_reviews(batch_id)


@app.get("/human-reviews/summary")
def human_review_summary(
    batch_id:      Optional[str] = Query(None),
    experiment_id: Optional[str] = Query(None),
    review_round:  Optional[int] = Query(None),
):
    return compute_agreement_summary(
        batch_id=batch_id,
        experiment_id=experiment_id,
        review_round=review_round,
    )


@app.get("/human-reviews/export")
def export_human_reviews(
    format:        str            = Query(...),
    review_id:     Optional[str]  = Query(None),
    batch_id:      Optional[str]  = Query(None),
    experiment_id: Optional[str]  = Query(None),
    review_round:  Optional[int]  = Query(None),
):
    if format == "html":
        if not review_id:
            raise HTTPException(status_code=400, detail="format=html requires review_id.")
        try:
            html = export_html(review_id)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        return Response(
            content=html,
            media_type="text/html",
            headers={"Content-Disposition": f'attachment; filename="{review_id}.html"'},
        )

    if format in ("json", "csv_template", "csv_context"):
        if review_id:
            ids = [review_id]
        elif batch_id or experiment_id:
            reviews = list_human_reviews(
                batch_id=batch_id, experiment_id=experiment_id, review_round=review_round
            )
            ids = [r["review_id"] for r in reviews]
        else:
            raise HTTPException(status_code=400, detail="Provide review_id, batch_id, or experiment_id.")

        if not ids:
            raise HTTPException(status_code=404, detail="No reviews found for the given filters.")

        if format == "json":
            packets = export_json_packets(ids)
            return Response(
                content=json.dumps(packets, indent=2, ensure_ascii=False),
                media_type="application/json",
                headers={"Content-Disposition": 'attachment; filename="review_packets.json"'},
            )
        if format == "csv_template":
            csv_content = export_csv_template(ids)
            return Response(
                content=csv_content,
                media_type="text/csv",
                headers={"Content-Disposition": 'attachment; filename="review_template.csv"'},
            )
        if format == "csv_context":
            csv_content = export_csv_context(ids)
            return Response(
                content=csv_content,
                media_type="text/csv",
                headers={"Content-Disposition": 'attachment; filename="review_context.csv"'},
            )

    raise HTTPException(status_code=400, detail=f"Unknown format '{format}'. Use html, json, csv_template, or csv_context.")


@app.post("/human-reviews/import")
async def import_human_review_responses(
    file:      UploadFile = File(...),
    overwrite: bool       = Query(False),
):
    content_bytes = await file.read()
    content = content_bytes.decode("utf-8", errors="replace")
    filename = file.filename or ""

    if filename.lower().endswith(".json"):
        result = import_from_json(content, overwrite=overwrite)
    else:
        result = import_from_csv(content, overwrite=overwrite)

    return result


@app.get("/human-reviews/patterns")
def human_review_patterns(
    batch_id:      Optional[str] = Query(None),
    experiment_id: Optional[str] = Query(None),
    review_round:  Optional[int] = Query(None),
):
    return compute_emerging_patterns(
        batch_id=batch_id,
        experiment_id=experiment_id,
        review_round=review_round,
    )


@app.get("/human-reviews/{review_id}/responses")
def get_review_responses(review_id: str):
    review = get_human_review(review_id)
    if not review:
        raise HTTPException(status_code=404, detail=f"Review '{review_id}' not found.")
    return _get_responses_for_review(review_id)


@app.get("/human-reviews/{review_id}")
def get_human_review_endpoint(review_id: str):
    review = get_human_review(review_id)
    if not review:
        raise HTTPException(status_code=404, detail=f"Review '{review_id}' not found.")
    review.pop("subject_model", None)
    return review


@app.post("/human-reviews/{review_id}/archive")
def archive_human_review(review_id: str):
    review = get_human_review(review_id)
    if not review:
        raise HTTPException(status_code=404, detail=f"Review '{review_id}' not found.")
    mark_review_archived(review_id)
    return {"review_id": review_id, "status": "archived"}
