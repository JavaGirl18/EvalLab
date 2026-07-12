import asyncio
import dataclasses
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

EXPERIMENTS_DIR = Path(__file__).parent / "data" / "experiments"


def _update_experiment_status(
    experiment_id: str,
    phase: Optional[str] = None,
    started_at: Optional[str] = None,
    completed_at: Optional[str] = None,
) -> None:
    """Patch the experiment file's status block in place. Silently skips if file not found."""
    path = EXPERIMENTS_DIR / f"{experiment_id}.json"
    if not path.exists():
        return
    try:
        with open(path) as f:
            exp = json.load(f)
        status = exp.get("status") or {}
        if phase:
            status["phase"] = phase
        if started_at and not status.get("started_at"):
            status["started_at"] = started_at
        if completed_at:
            status["completed_at"] = completed_at
        exp["status"] = status
        with open(path, "w") as f:
            json.dump(exp, f, indent=2, ensure_ascii=False)
            f.write("\n")
    except Exception:
        pass

from config import RESEARCH_SUBJECT_MODELS
from batch_db import (
    create_batch,
    get_batch,
    get_batch_items,
    get_completed_results,
    make_run_key,
    mark_batch_status,
    mark_item_completed,
    mark_item_failed,
    mark_item_running,
    mark_item_skipped,
    reset_stale_running_items,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _valid_model_ids() -> set[str]:
    return {m["model_id"] for m in RESEARCH_SUBJECT_MODELS}


# ── Validation ────────────────────────────────────────────────────────────────

def validate_request(req: dict) -> tuple[list[str], list[str]]:
    """Returns (warnings, errors). Errors block execution; warnings are informational."""
    from research_runner import load_dataset

    warnings: list[str] = []
    errors:   list[str] = []

    if not req.get("experiment_id", "").strip():
        errors.append("experiment_id is required.")
    if not req.get("item_ids"):
        errors.append("item_ids must be a non-empty list.")
    if not req.get("prompt_variants"):
        errors.append("prompt_variants must be a non-empty list.")
    if not req.get("models"):
        errors.append("models must be a non-empty list.")

    valid = _valid_model_ids()
    for m in req.get("models", []):
        if m not in valid:
            errors.append(f"Model '{m}' is not configured. Available: {sorted(valid)}")

    # Stop here if structure is broken — dataset checks would throw misleading errors
    if errors:
        return warnings, errors

    dataset = {item.id: item for item in load_dataset()}
    for item_id in req.get("item_ids", []):
        if item_id not in dataset:
            errors.append(f"Dataset item '{item_id}' not found.")
        elif not dataset[item_id].source_text.strip():
            errors.append(
                f"Dataset item '{item_id}' has no source_text — run ingest_articles.py first."
            )

    # ── Non-fatal warnings ────────────────────────────────────────────────────
    if (
        "faithful_transformation" in req.get("prompt_variants", [])
        and not req.get("task_instructions")
    ):
        warnings.append(
            "faithful_transformation is included without task_instructions. "
            "The generic stub prompt will be used — valid but less targeted."
        )

    temp = req.get("temperature", 0.0)
    if temp > 0 and "faithful_transformation" in req.get("prompt_variants", []):
        warnings.append(
            f"temperature={temp} with faithful_transformation may reduce reproducibility. "
            "Consider temperature=0.0 for deterministic comparisons."
        )

    expected = (
        len(req.get("item_ids", []))
        * len(req.get("prompt_variants", []))
        * len(req.get("models", []))
    )
    max_runs = req.get("max_runs")
    if max_runs and expected > max_runs:
        warnings.append(
            f"{expected} runs expected but max_runs={max_runs}. "
            "Runs beyond the limit are created as 'skipped'."
        )

    return warnings, errors


# ── Batch creation ────────────────────────────────────────────────────────────

def _enumerate_run_items(req: dict) -> list[dict]:
    items = []
    for item_id in req["item_ids"]:
        for variant in req["prompt_variants"]:
            for model in req["models"]:
                items.append({"item_id": item_id, "prompt_variant": variant, "model": model})
    return items


def create_batch_from_request(req: dict, warnings: list[str]) -> str:
    batch_id  = str(uuid.uuid4())
    all_items = _enumerate_run_items(req)
    max_runs  = req.get("max_runs")

    if max_runs and len(all_items) > max_runs:
        active_items = all_items[:max_runs]
        skip_items   = all_items[max_runs:]
    else:
        active_items = all_items
        skip_items   = []

    create_batch(
        batch_id=batch_id,
        experiment_id=req["experiment_id"],
        config=req,
        warnings=warnings,
        run_items=all_items,
    )

    for ri in skip_items:
        rk = make_run_key(req["experiment_id"], ri["item_id"], ri["prompt_variant"], ri["model"])
        mark_item_skipped(rk)

    return batch_id


# ── Async executor ────────────────────────────────────────────────────────────

async def run_batch(batch_id: str) -> None:
    """Background async executor. Called as a FastAPI BackgroundTask."""
    now = _now()
    mark_batch_status(batch_id, "running", started_at=now)
    reset_stale_running_items(batch_id)

    batch = get_batch(batch_id)
    if not batch:
        return

    config          = json.loads(batch["config"]) if isinstance(batch["config"], str) else batch["config"]
    _update_experiment_status(batch["experiment_id"], phase="in_progress", started_at=now)
    max_concurrency = config.get("max_concurrency", 3)
    retry_limit     = config.get("retry_limit", 3)

    all_items = get_batch_items(batch_id)
    to_run = [
        it for it in all_items
        if it["status"] not in ("completed", "skipped", "cancelled")
        and not (it["status"] == "failed" and it["attempt"] >= retry_limit)
    ]

    semaphore = asyncio.Semaphore(max_concurrency)
    await asyncio.gather(
        *[_run_unit(semaphore, batch_id, item, config, retry_limit) for item in to_run],
        return_exceptions=True,
    )

    final_items    = get_batch_items(batch_id)
    has_incomplete = any(it["status"] in ("pending", "running") for it in final_items)
    final_status   = "failed" if has_incomplete else "completed"
    finished_at    = _now()
    mark_batch_status(batch_id, final_status, completed_at=finished_at)
    _update_experiment_status(
        batch["experiment_id"],
        phase="completed" if final_status == "completed" else "paused",
        completed_at=finished_at if final_status == "completed" else None,
    )


async def _run_unit(
    semaphore: asyncio.Semaphore,
    batch_id: str,
    item_row: dict,
    config: dict,
    retry_limit: int,
) -> None:
    from research_runner import get_dataset_item, get_prompt_variant, _call_model_with_retry
    from research_judge import score_research_response

    async with semaphore:
        run_key        = item_row["run_key"]
        item_id        = item_row["item_id"]
        prompt_variant = item_row["prompt_variant"]
        model_id       = item_row["model"]
        temperature    = config.get("temperature", 0.0)
        task_instructions = config.get("task_instructions")

        model_cfg = next(
            (m for m in RESEARCH_SUBJECT_MODELS if m["model_id"] == model_id),
            {"model_id": model_id, "display_name": model_id, "provider": "openai", "version_note": ""},
        )

        # Resolve dataset item and variant once — these failures are not retried
        try:
            dataset_item = await asyncio.to_thread(get_dataset_item, item_id)
            variant      = await asyncio.to_thread(
                get_prompt_variant, dataset_item, prompt_variant, task_instructions
            )
        except ValueError as e:
            mark_item_failed(run_key, str(e))
            return

        attempt = item_row["attempt"]
        while attempt < retry_limit:
            attempt += 1
            mark_item_running(run_key, attempt)
            try:
                response = await asyncio.to_thread(
                    _call_model_with_retry, model_cfg, dataset_item, variant, temperature, 1
                )
                scored = await asyncio.to_thread(
                    score_research_response, dataset_item, response
                )
                mark_item_completed(run_key, json.dumps(dataclasses.asdict(scored)))
                return
            except Exception as e:
                if attempt < retry_limit:
                    await asyncio.sleep(2 ** attempt)
                else:
                    mark_item_failed(run_key, str(e))


# ── Summary computation ───────────────────────────────────────────────────────

def compute_batch_summary(batch_id: str) -> dict:
    results   = get_completed_results(batch_id)
    all_items = get_batch_items(batch_id)

    count_map: dict[str, int] = {}
    for it in all_items:
        count_map[it["status"]] = count_map.get(it["status"], 0) + 1

    ii_by_model:      dict[str, list[float]] = {}
    cf_by_model:      dict[str, list[float]] = {}
    failure_counts:   dict[str, int]         = {}

    for r in results:
        model = r.get("model", "unknown")
        ii_by_model.setdefault(model, []).append(
            r.get("overall_information_integrity_score", 0)
        )
        cf_by_model.setdefault(model, []).append(
            r.get("overall_cultural_fidelity_score", 0)
        )
        for cat in r.get("most_significant_failures", []):
            failure_counts[cat] = failure_counts.get(cat, 0) + 1

    return {
        "expected":  len(all_items),
        "completed": count_map.get("completed", 0),
        "failed":    count_map.get("failed", 0),
        "skipped":   count_map.get("skipped", 0),
        "pending":   count_map.get("pending", 0) + count_map.get("running", 0),
        "avg_ii_by_model": {
            m: round(sum(v) / len(v), 2) for m, v in ii_by_model.items()
        },
        "avg_cf_by_model": {
            m: round(sum(v) / len(v), 2) for m, v in cf_by_model.items()
        },
        "top_failure_categories": sorted(
            failure_counts.items(), key=lambda x: -x[1]
        )[:5],
    }
