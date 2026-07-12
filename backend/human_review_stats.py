from typing import Optional

from human_review_db import (
    get_completed_reviews_with_responses,
    list_human_reviews,
)

AGREEMENT_VALUES = ["yes", "partially", "no", "unable_to_determine"]


def _blank_agreement() -> dict:
    return {"yes": 0, "partially": 0, "no": 0, "unable_to_determine": 0, "total": 0}


def _add_agreement(bucket: dict, agreement: str) -> None:
    key = agreement if agreement in AGREEMENT_VALUES else "unable_to_determine"
    bucket[key] = bucket.get(key, 0) + 1
    bucket["total"] = bucket.get("total", 0) + 1


def _pct(n: int, total: int) -> float:
    if not total:
        return 0.0
    return round(n / total, 4)


def compute_agreement_summary(
    batch_id: Optional[str] = None,
    experiment_id: Optional[str] = None,
    review_round: Optional[int] = None,
) -> dict:
    """
    Compute agreement statistics across all completed reviews with responses.
    Subject-model breakdown is computed but NOT returned in the top-level dict —
    call compute_model_breakdown() separately for internal use.
    """
    all_reviews = list_human_reviews(
        batch_id=batch_id,
        experiment_id=experiment_id,
        review_round=review_round,
    )
    completed_reviews = get_completed_reviews_with_responses(
        batch_id=batch_id,
        experiment_id=experiment_id,
        review_round=review_round,
    )

    total_assigned  = len(all_reviews)
    total_completed = len(completed_reviews)

    overall        = _blank_agreement()
    by_category: dict    = {}
    by_severity: dict    = {}
    by_variant: dict     = {}
    by_document: dict    = {}
    by_bench_cat: dict   = {}
    by_cult_sig: dict    = {}
    by_reviewer: dict    = {}
    disagreement_counts: dict = {}

    for review in completed_reviews:
        # Snapshot data for this review
        snap     = review.get("packet_snapshot") or {}
        variant  = review.get("prompt_variant", "unknown")
        item_id  = review.get("dataset_item_id", "unknown")
        bench    = snap.get("benchmark_category", "unknown") or "unknown"
        cult     = snap.get("cultural_significance", "unknown") or "unknown"
        findings = snap.get("judge_findings", [])
        flagged_categories = [f["category_id"] for f in findings if f.get("severity", 0) > 0]
        severity_level     = max((f["severity"] for f in findings), default=0)

        # Each response from each reviewer
        for resp in review.get("responses", []):
            agreement   = resp.get("agreement_with_judge", "unable_to_determine")
            reviewer_id = resp.get("reviewer_id", "unknown")
            dtypes      = resp.get("disagreement_types", [])

            # ── Overall ───────────────────────────────────────────────────────
            _add_agreement(overall, agreement)

            # ── By reviewer ───────────────────────────────────────────────────
            if reviewer_id not in by_reviewer:
                by_reviewer[reviewer_id] = _blank_agreement()
            _add_agreement(by_reviewer[reviewer_id], agreement)

            # ── By variant ────────────────────────────────────────────────────
            if variant not in by_variant:
                by_variant[variant] = _blank_agreement()
            _add_agreement(by_variant[variant], agreement)

            # ── By document ───────────────────────────────────────────────────
            if item_id not in by_document:
                by_document[item_id] = _blank_agreement()
            _add_agreement(by_document[item_id], agreement)

            # ── By benchmark category ─────────────────────────────────────────
            if bench not in by_bench_cat:
                by_bench_cat[bench] = _blank_agreement()
            _add_agreement(by_bench_cat[bench], agreement)

            # ── By cultural significance ──────────────────────────────────────
            if cult not in by_cult_sig:
                by_cult_sig[cult] = _blank_agreement()
            _add_agreement(by_cult_sig[cult], agreement)

            # ── By category (each flagged category inherits reviewer agreement) ─
            for cat_id in flagged_categories:
                if cat_id not in by_category:
                    by_category[cat_id] = _blank_agreement()
                _add_agreement(by_category[cat_id], agreement)

            # ── By severity level ─────────────────────────────────────────────
            sev_key = str(severity_level)
            if sev_key not in by_severity:
                by_severity[sev_key] = _blank_agreement()
            _add_agreement(by_severity[sev_key], agreement)

            # ── Disagreement types ────────────────────────────────────────────
            for dt in (dtypes if isinstance(dtypes, list) else []):
                disagreement_counts[dt] = disagreement_counts.get(dt, 0) + 1

    total_responses = overall.get("total", 0)
    top_disagreement = sorted(disagreement_counts.items(), key=lambda x: -x[1])

    return {
        "total_assigned":  total_assigned,
        "total_completed": total_completed,
        "total_responses": total_responses,
        "completion_rate": _pct(total_completed, total_assigned),
        "agreement": {
            "yes":                 overall.get("yes", 0),
            "yes_pct":             _pct(overall.get("yes", 0), total_responses),
            "partially":           overall.get("partially", 0),
            "partially_pct":       _pct(overall.get("partially", 0), total_responses),
            "no":                  overall.get("no", 0),
            "no_pct":              _pct(overall.get("no", 0), total_responses),
            "unable_to_determine": overall.get("unable_to_determine", 0),
            "unable_pct":          _pct(overall.get("unable_to_determine", 0), total_responses),
        },
        "by_category":             by_category,
        "by_severity_level":       by_severity,
        "by_prompt_variant":       by_variant,
        "by_document":             by_document,
        "by_benchmark_category":   by_bench_cat,
        "by_cultural_significance": by_cult_sig,
        "by_reviewer":             by_reviewer,
        "top_disagreement_types":  top_disagreement[:10],
    }


def compute_emerging_patterns(
    batch_id: Optional[str] = None,
    experiment_id: Optional[str] = None,
    review_round: Optional[int] = None,
) -> dict:
    """
    Surface actionable patterns across completed reviews.
    Used by the dashboard Patterns section — never includes subject model identity.
    """
    completed_reviews = get_completed_reviews_with_responses(
        batch_id=batch_id,
        experiment_id=experiment_id,
        review_round=review_round,
    )

    # Accumulate per-category and per-article disagreement data
    cat_agreement:  dict = {}   # {cat_id: {"yes":n, "partially":n, "no":n, "total":n}}
    art_agreement:  dict = {}   # {item_id: {"title":str, "yes":n, "partial_or_no":n, "total":n}}
    reviewer_stats: dict = {}   # {reviewer_id: {"agree":n, "total":n}}
    sev_agreement:  dict = {}   # {sev_label: {"yes":n, "partial_or_no":n, "total":n}}
    dt_counts:      dict = {}   # {disagreement_type: count}

    for review in completed_reviews:
        snap     = review.get("packet_snapshot") or {}
        item_id  = review.get("dataset_item_id", "unknown")
        title    = snap.get("source_title", item_id) or item_id
        findings = snap.get("judge_findings", [])

        flagged = {f["category_id"]: f for f in findings if f.get("severity", 0) > 0}
        max_sev  = max((f["severity"] for f in findings), default=0)
        sev_label = {0: "clean", 1: "minor", 2: "moderate", 3: "severe"}.get(max_sev, "unknown")

        if item_id not in art_agreement:
            art_agreement[item_id] = {"title": title, "yes": 0, "partial_or_no": 0, "total": 0}

        for resp in review.get("responses", []):
            agreement   = resp.get("agreement_with_judge", "unable_to_determine")
            reviewer_id = resp.get("reviewer_id", "unknown")
            dtypes      = resp.get("disagreement_types") or []
            is_agree    = agreement == "yes"
            is_partial  = agreement in ("partially", "no")

            # Article-level
            art_agreement[item_id]["total"] += 1
            if is_agree:
                art_agreement[item_id]["yes"] += 1
            elif is_partial:
                art_agreement[item_id]["partial_or_no"] += 1

            # Severity-level
            if sev_label not in sev_agreement:
                sev_agreement[sev_label] = {"yes": 0, "partial_or_no": 0, "total": 0}
            sev_agreement[sev_label]["total"] += 1
            if is_agree:
                sev_agreement[sev_label]["yes"] += 1
            elif is_partial:
                sev_agreement[sev_label]["partial_or_no"] += 1

            # Per-category
            for cat_id in flagged:
                if cat_id not in cat_agreement:
                    cat_agreement[cat_id] = {"yes": 0, "partially": 0, "no": 0, "total": 0}
                cat_agreement[cat_id]["total"] += 1
                if agreement in cat_agreement[cat_id]:
                    cat_agreement[cat_id][agreement] += 1

            # Reviewer consistency
            if reviewer_id not in reviewer_stats:
                reviewer_stats[reviewer_id] = {"agree": 0, "total": 0}
            reviewer_stats[reviewer_id]["total"] += 1
            if is_agree:
                reviewer_stats[reviewer_id]["agree"] += 1

            # Disagreement types
            for dt in (dtypes if isinstance(dtypes, list) else []):
                dt_counts[dt] = dt_counts.get(dt, 0) + 1

    # High-disagreement categories (agree rate < 60% with ≥2 responses)
    high_disagreement_categories = []
    for cat_id, counts in cat_agreement.items():
        total = counts["total"]
        if total >= 2:
            agree_pct = _pct(counts.get("yes", 0), total)
            if agree_pct < 0.60:
                high_disagreement_categories.append({
                    "category_id":  cat_id,
                    "agree_pct":    agree_pct,
                    "yes":          counts.get("yes", 0),
                    "partially":    counts.get("partially", 0),
                    "no":           counts.get("no", 0),
                    "total":        total,
                })
    high_disagreement_categories.sort(key=lambda x: x["agree_pct"])

    # Disputed articles (≥2 responses, partial_or_no > yes)
    disputed_articles = []
    for item_id, counts in art_agreement.items():
        total = counts["total"]
        if total >= 2 and counts["partial_or_no"] > counts["yes"]:
            disputed_articles.append({
                "item_id":       item_id,
                "title":         counts["title"],
                "partial_or_no": counts["partial_or_no"],
                "yes":           counts["yes"],
                "total":         total,
                "dispute_rate":  _pct(counts["partial_or_no"], total),
            })
    disputed_articles.sort(key=lambda x: -x["dispute_rate"])

    # Reviewer consistency (reviewers with ≥3 reviews)
    reviewer_consistency = []
    for reviewer_id, stats in reviewer_stats.items():
        if stats["total"] >= 3:
            reviewer_consistency.append({
                "reviewer_id":  reviewer_id,
                "agree":        stats["agree"],
                "total":        stats["total"],
                "agree_pct":    _pct(stats["agree"], stats["total"]),
            })
    reviewer_consistency.sort(key=lambda x: -x["total"])

    # Severity calibration
    severity_calibration = [
        {"severity": k, **v, "agree_pct": _pct(v["yes"], v["total"]) if v["total"] else 0}
        for k, v in sorted(
            sev_agreement.items(),
            key=lambda x: {"clean": 0, "minor": 1, "moderate": 2, "severe": 3}.get(x[0], 9)
        )
    ]

    return {
        "high_disagreement_categories":  high_disagreement_categories[:8],
        "disputed_articles":             disputed_articles[:5],
        "reviewer_consistency":          reviewer_consistency[:6],
        "severity_calibration":          severity_calibration,
        "top_disagreement_types":        sorted(dt_counts.items(), key=lambda x: -x[1])[:8],
        "total_completed":               len(completed_reviews),
    }


def compute_model_breakdown(
    batch_id: Optional[str] = None,
    experiment_id: Optional[str] = None,
    review_round: Optional[int] = None,
) -> dict:
    """Internal-only: agreement stats broken down by subject model."""
    completed_reviews = get_completed_reviews_with_responses(
        batch_id=batch_id,
        experiment_id=experiment_id,
        review_round=review_round,
    )
    by_model: dict = {}
    for review in completed_reviews:
        model = review.get("subject_model", "unknown")
        if model not in by_model:
            by_model[model] = _blank_agreement()
        for resp in review.get("responses", []):
            _add_agreement(by_model[model], resp.get("agreement_with_judge", "unable_to_determine"))
    return by_model
