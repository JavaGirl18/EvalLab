import csv
import io
import json
from datetime import datetime, timezone
from typing import Optional

from human_review_db import (
    get_human_review,
    mark_review_completed,
    upsert_review_response,
)

VALID_AGREEMENT = {"yes", "partially", "no", "unable_to_determine"}
VALID_DISAGREEMENT_TYPES = {
    "failure_not_present",
    "failure_category_incorrect",
    "severity_too_high",
    "severity_too_low",
    "important_failure_missing",
    "judge_explanation_inaccurate",
    "source_context_misunderstood",
    "other",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_disagreement_types(raw: str) -> tuple:
    """
    Accepts semicolon-separated string or JSON array string.
    Returns (valid_types, warnings).
    """
    if not raw or not raw.strip():
        return [], []

    raw = raw.strip()
    # Try JSON array first
    if raw.startswith("["):
        try:
            items = json.loads(raw)
        except Exception:
            items = [s.strip() for s in raw.split(";") if s.strip()]
    else:
        items = [s.strip() for s in raw.split(";") if s.strip()]

    valid   = [i for i in items if i in VALID_DISAGREEMENT_TYPES]
    unknown = [i for i in items if i not in VALID_DISAGREEMENT_TYPES]
    warnings = [f"Unknown disagreement type '{i}' dropped" for i in unknown]
    return valid, warnings


def import_from_csv(
    content: str,
    overwrite: bool = False,
    import_source: str = "csv",
) -> dict:
    """Parse CSV content and import responses. Returns result summary."""
    # Strip UTF-8 BOM if present (common from Google Forms exports)
    if content.startswith("﻿"):
        content = content[1:]

    reader = csv.DictReader(io.StringIO(content))
    return _process_rows(list(reader), overwrite=overwrite, import_source=import_source)


def import_from_json(
    content: str,
    overwrite: bool = False,
    import_source: str = "json",
) -> dict:
    """Parse JSON array of response objects and import. Returns result summary."""
    try:
        rows = json.loads(content)
    except json.JSONDecodeError as exc:
        return {
            "imported": 0, "updated": 0,
            "errors": [{"row": 0, "review_id": "", "reason": f"Invalid JSON: {exc}"}],
            "unknown_ids": [], "duplicate_skipped": 0,
        }
    if not isinstance(rows, list):
        rows = [rows]
    return _process_rows(rows, overwrite=overwrite, import_source=import_source)


def _process_rows(rows: list, overwrite: bool, import_source: str) -> dict:
    imported         = 0
    updated          = 0
    errors           = []
    unknown_ids      = []
    duplicate_skipped = 0
    imported_at      = _now()

    for i, row in enumerate(rows, start=1):
        review_id  = str(row.get("review_id") or "").strip()
        reviewer_id = str(row.get("reviewer_id") or "").strip()

        if not review_id:
            errors.append({"row": i, "review_id": "", "reason": "Missing review_id"})
            continue
        if not reviewer_id:
            errors.append({"row": i, "review_id": review_id, "reason": "Missing reviewer_id"})
            continue

        # Validate review exists
        review = get_human_review(review_id)
        if not review:
            unknown_ids.append(review_id)
            errors.append({"row": i, "review_id": review_id, "reason": "Unknown review_id"})
            continue

        agreement = str(row.get("agreement_with_judge") or "").strip().lower()
        if not agreement:
            errors.append({"row": i, "review_id": review_id, "reason": "Missing agreement_with_judge"})
            continue
        if agreement not in VALID_AGREEMENT:
            errors.append({
                "row": i, "review_id": review_id,
                "reason": f"Invalid agreement value '{agreement}'. Must be one of: {sorted(VALID_AGREEMENT)}",
            })
            continue

        # Parse disagreement types
        raw_dtypes = str(row.get("disagreement_types") or "")
        dtypes, dtype_warnings = _parse_disagreement_types(raw_dtypes)

        reviewed_at_raw = str(row.get("reviewed_at") or "").strip()
        reviewed_at = reviewed_at_raw if reviewed_at_raw else imported_at

        try:
            _, was_updated = upsert_review_response(
                review_id=review_id,
                reviewer_id=reviewer_id,
                agreement_with_judge=agreement,
                disagreement_types=dtypes,
                comments=str(row.get("comments") or "").strip(),
                missed_failures=str(row.get("missed_failures") or "").strip(),
                incorrectly_flagged=str(row.get("incorrectly_flagged") or "").strip(),
                preserved_meaning=str(row.get("preserved_meaning") or "").strip(),
                cultural_context_preserved=str(row.get("cultural_context_preserved") or "").strip(),
                additional_comments=str(row.get("additional_comments") or "").strip(),
                reviewed_at=reviewed_at,
                imported_at=imported_at,
                import_source=import_source,
                raw_import=dict(row),
                overwrite=overwrite,
            )
        except ValueError as exc:
            duplicate_skipped += 1
            errors.append({"row": i, "review_id": review_id, "reason": str(exc)})
            continue

        mark_review_completed(review_id)

        if was_updated:
            updated += 1
        else:
            imported += 1

        if dtype_warnings:
            for w in dtype_warnings:
                errors.append({"row": i, "review_id": review_id, "reason": f"Warning: {w}"})

    return {
        "imported":          imported,
        "updated":           updated,
        "errors":            errors,
        "unknown_ids":       list(set(unknown_ids)),
        "duplicate_skipped": duplicate_skipped,
    }
