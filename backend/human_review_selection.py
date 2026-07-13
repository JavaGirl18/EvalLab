import json
import random
from typing import Optional

from batch_db import get_batch_items
from human_review_db import run_key_has_review

CONFIDENCE_ORDER = {"Low": 0, "Medium": 1, "High": 2}


def _parse_result(row: dict) -> Optional[dict]:
    rj = row.get("result_json")
    if not rj:
        return None
    try:
        return json.loads(rj) if isinstance(rj, str) else rj
    except Exception:
        return None


def select_runs_for_review(
    batch_id: str,
    *,
    severity_threshold: Optional[int] = None,
    confidence_threshold: Optional[str] = None,
    failure_count_threshold: Optional[int] = None,
    random_pct: Optional[float] = None,
    manual_run_keys: Optional[list] = None,
    representative_sample: bool = False,
    review_round: int = 1,
) -> list:
    """
    Returns list of {run_key, selection_reasons, row, result} for eligible runs.
    Already-reviewed run_keys (same round) are excluded.
    At least one rule must match for a run to be returned.
    """
    all_items = get_batch_items(batch_id)
    completed = [it for it in all_items if it["status"] == "completed"]

    candidates: dict = {}
    for row in completed:
        result = _parse_result(row)
        if result and not run_key_has_review(row["run_key"], review_round):
            candidates[row["run_key"]] = {"row": row, "result": result, "reasons": set()}

    if not candidates:
        return []

    # ── Manual ───────────────────────────────────────────────────────────────
    if manual_run_keys:
        for rk in manual_run_keys:
            if rk in candidates:
                candidates[rk]["reasons"].add("manual")

    # ── Severity threshold ────────────────────────────────────────────────────
    if severity_threshold is not None:
        for rk, v in candidates.items():
            dims = v["result"].get("dimension_scores", {})
            if any(
                (d.get("severity", 0) if isinstance(d, dict) else d.severity) >= severity_threshold
                for d in dims.values()
            ):
                v["reasons"].add("severity_threshold")

    # ── Confidence threshold ──────────────────────────────────────────────────
    if confidence_threshold is not None:
        threshold_val = CONFIDENCE_ORDER.get(confidence_threshold, 0)
        for rk, v in candidates.items():
            dims = v["result"].get("dimension_scores", {})
            for d in dims.values():
                dim_dict = d if isinstance(d, dict) else {"severity": d.severity, "confidence": d.confidence}
                if (
                    dim_dict.get("severity", 0) > 0
                    and CONFIDENCE_ORDER.get(dim_dict.get("confidence", "High"), 2) <= threshold_val
                ):
                    v["reasons"].add("confidence_threshold")
                    break

    # ── Failure count threshold ───────────────────────────────────────────────
    if failure_count_threshold is not None:
        for rk, v in candidates.items():
            failures = v["result"].get("most_significant_failures", [])
            if len(failures) >= failure_count_threshold:
                v["reasons"].add("failure_count_threshold")

    # ── Random sample (applied to the full pool independently) ───────────────
    if random_pct is not None and 0 < random_pct <= 1.0:
        all_keys = list(candidates.keys())
        sample_n = max(1, round(len(all_keys) * random_pct))
        for rk in random.sample(all_keys, min(sample_n, len(all_keys))):
            candidates[rk]["reasons"].add("random_sample")

    # ── Representative sample ─────────────────────────────────────────────────
    # Guarantees at least one review per document, model, and prompt variant.
    # Applied last so it only adds coverage for dimensions not already selected.
    if representative_sample:
        already_selected = {rk for rk, v in candidates.items() if v["reasons"]}

        def _best_in_group(keys: list[str]) -> str:
            return max(keys, key=lambda rk: max(
                (d.get("severity", 0) if isinstance(d, dict) else 0)
                for d in candidates[rk]["result"].get("dimension_scores", {}).values()
            ) if candidates[rk]["result"].get("dimension_scores") else 0)

        for dim_key in ("item_id", "prompt_variant", "model"):
            groups: dict[str, list[str]] = {}
            for rk, v in candidates.items():
                val = v["row"].get(dim_key, "")
                groups.setdefault(val, []).append(rk)
            for val, group_keys in groups.items():
                if not any(rk in already_selected for rk in group_keys):
                    chosen = _best_in_group(group_keys)
                    candidates[chosen]["reasons"].add("representative_sample")
                    already_selected.add(chosen)

    return [
        {
            "run_key":           rk,
            "selection_reasons": sorted(v["reasons"]),
            "row":               v["row"],
            "result":            v["result"],
        }
        for rk, v in candidates.items()
        if v["reasons"]
    ]
