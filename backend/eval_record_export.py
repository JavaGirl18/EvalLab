"""
Evaluation Record — canonical research artifact for one completed evaluation.

Assembly pulls from four live sources:
  batch_run_items  → result_json (model identity, all 14 scores, output)
  batch config     → experiment_meta (frozen at run time)
  benchmark_corpus → dataset item (source text, characteristics)
  human_reviews    → reviewer responses (optional)

On first HTML export the assembled dict is frozen in evaluation_records.record_snapshot.
Subsequent exports render from the frozen snapshot so the record is immutable.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from batch_db import get_batch_item_by_run_key, get_batch
from human_review_db import list_human_reviews, get_responses_for_review
from eval_record_db import get_by_evaluation_id

EVALLAB_VERSION = "1.0.0"
RECORD_FORMAT_VERSION = "1.1"

DATA_DIR        = Path(__file__).parent / "data"
EXPERIMENTS_DIR = DATA_DIR / "experiments"
DATASET_PATH    = DATA_DIR / "benchmark" / "benchmark_corpus.json"

PLAIN_LABELS: dict[str, str] = {
    "FH":  "Fabricated / Hallucinated Facts",
    "CXH": "Context Hallucination",
    "CTH": "Cultural Translation Hallucination",
    "AH":  "Attribution Hallucination",
    "FS":  "False Specificity",
    "UC":  "Unsupported Claims",
    "CL":  "Cultural Loss",
    "EX":  "Exoticization",
    "OC":  "Othering / Community Framing",
    "AL":  "Agency / Leadership Erasure",
    "CM":  "Community Voice Marginalization",
    "FB":  "Framing Bias",
    "PB":  "Political / Ideological Bias",
    "ST":  "Stereotyping",
}

SEVERITY_LABELS  = {0: "Clean", 1: "Minor", 2: "Moderate", 3: "Severe"}
SEVERITY_COLORS  = {0: "#6b7280", 1: "#d97706", 2: "#ea580c", 3: "#dc2626"}
SEVERITY_BG      = {0: "#f3f4f6", 1: "#fffbeb", 2: "#fff7ed", 3: "#fff5f5"}
SEVERITY_BORDER  = {0: "#e5e7eb", 1: "#fde68a", 2: "#fdba74", 3: "#fca5a5"}


def _na(reason: str = "External Submission") -> str:
    return f'<span class="field-na">Not applicable ({_esc(reason)})</span>'


def _notprovided() -> str:
    return '<span class="field-unknown">Not provided by submitter</span>'


def _extfield(value: Optional[str], is_ext: bool, na_reason: Optional[str] = None) -> str:
    v = str(value).strip() if value is not None else ""
    if v and v not in ("—", "None", "null", "N/A"):
        return _esc(v)
    if not is_ext:
        return "—"
    return _na(na_reason) if na_reason else _notprovided()

DISAGREE_LABELS: dict[str, str] = {
    "failure_not_present":          "Failure not present",
    "failure_category_incorrect":   "Wrong category",
    "severity_too_high":            "Severity too high",
    "severity_too_low":             "Severity too low",
    "important_failure_missing":    "Important failure missing",
    "judge_explanation_inaccurate": "Inaccurate explanation",
    "source_context_misunderstood": "Source context misunderstood",
    "other":                        "Other",
}


# ── Data loading helpers ──────────────────────────────────────────────────────

def _load_dataset_item(item_id: str) -> Optional[dict]:
    try:
        with open(DATASET_PATH) as f:
            for item in json.load(f):
                if item.get("id") == item_id:
                    return item
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return None


def _load_experiment_config(experiment_id: str) -> Optional[dict]:
    try:
        path = EXPERIMENTS_DIR / f"{experiment_id}.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return None


def _experiment_meta_from_batch(batch_id: str) -> Optional[dict]:
    """Pull experiment_meta frozen inside the batch config at run time."""
    batch = get_batch(batch_id)
    if not batch:
        return None
    try:
        cfg = json.loads(batch["config"]) if isinstance(batch["config"], str) else batch["config"]
        return cfg.get("experiment_meta")
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def _fmt_score(val: Optional[float]) -> str:
    return f"{val:.1f}" if val is not None else "—"


def _checksum(record: dict) -> str:
    """SHA-256 of the canonical JSON representation (sorted keys, no whitespace)."""
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


# ── Assembly ──────────────────────────────────────────────────────────────────

def build_eval_record(run_key: str, evaluation_id: str) -> dict:
    """
    Assemble the full Evaluation Record dict from all four data sources.
    The evaluation_id must be pre-registered via eval_record_db.register_run_key().
    """
    item = get_batch_item_by_run_key(run_key)
    if not item:
        raise ValueError(f"No batch run item found for run_key: {run_key}")
    if not item.get("result_json"):
        raise ValueError(f"Run {run_key} has no result yet (status: {item.get('status', 'unknown')})")

    result       = json.loads(item["result_json"])
    dataset_item = _load_dataset_item(item["item_id"])
    # Prefer the version frozen at run time over the live experiment file
    exp_meta     = _experiment_meta_from_batch(item["batch_id"]) \
                   or _load_experiment_config(item["experiment_id"]) \
                   or {}
    chars        = (dataset_item or {}).get("characteristics") or {}
    ds           = dataset_item or {}
    ds_meta      = ds.get("metadata") or {}

    # Subject model
    smc              = result.get("subject_model_config") or {}
    subject_model_id = smc.get("model_id") or result.get("model", "")
    subject_model_display = smc.get("display_name") or subject_model_id
    subject_provider = smc.get("provider", "")
    subject_version  = smc.get("version_note", "")

    # Judge model
    jmc            = result.get("judge_subject_model_config") or {}
    judge_model_id = jmc.get("model_id", "")
    judge_display  = jmc.get("display_name") or judge_model_id
    judge_provider = jmc.get("provider", "")
    judge_version  = jmc.get("version_note", "")

    # Human validation
    human_validation = None
    reviews  = list_human_reviews(experiment_id=item["experiment_id"])
    matching = [r for r in reviews if r["run_key"] == run_key]
    if matching:
        rev       = matching[0]
        responses = get_responses_for_review(rev["review_id"])
        human_validation = {
            "review_id":        rev["review_id"],
            "review_status":    rev["review_status"],
            "review_round":     rev["review_round"],
            "blinded":          rev["blinded"],
            "selection_reasons": rev["selection_reasons"],
            "created_at":       rev.get("created_at"),
            "exported_at":      rev.get("exported_at"),
            "completed_at":     rev.get("completed_at"),
            "responses": [
                {
                    "reviewer_id":               r["reviewer_id"],
                    "agreement_with_judge":       r["agreement_with_judge"],
                    "disagreement_types":         json.loads(r.get("disagreement_types") or "[]"),
                    "comments":                   r.get("comments") or "",
                    "missed_failures":            r.get("missed_failures") or "",
                    "incorrectly_flagged":        r.get("incorrectly_flagged") or "",
                    "preserved_meaning":          r.get("preserved_meaning") or "",
                    "cultural_context_preserved": r.get("cultural_context_preserved") or "",
                    "additional_comments":        r.get("additional_comments") or "",
                    "reviewed_at":               r.get("reviewed_at"),
                    "imported_at":               r.get("imported_at"),
                    "import_source":             r.get("import_source", ""),
                }
                for r in responses
            ],
        }

    # Evaluation status
    hv_status = "Not performed"
    if human_validation:
        hv_status = human_validation["review_status"].title()
        if human_validation["responses"]:
            hv_status = f"Completed ({len(human_validation['responses'])} response{'s' if len(human_validation['responses']) != 1 else ''})"

    # Overall judge confidence (average across flagged dimensions)
    dim_scores_raw = result.get("dimension_scores") or {}
    flagged = [s for s in dim_scores_raw.values() if s.get("severity", 0) > 0]
    conf_map = {"High": 1.0, "Medium": 0.67, "Low": 0.33}
    if flagged:
        avg_conf = sum(conf_map.get(s.get("confidence", "Medium"), 0.67) for s in flagged) / len(flagged)
        overall_judge_confidence = round(avg_conf * 100)
    else:
        overall_judge_confidence = 100  # no issues found → fully confident

    # Lifecycle timestamps
    reg_row = get_by_evaluation_id(evaluation_id)
    record_registered_at = reg_row["registered_at"] if reg_row else None

    # Expected failure categories — researcher-only, never shown to reviewers
    expected_failure_categories = [
        PLAIN_LABELS.get(c, c) for c in ds.get("expected_failure_categories", [])
    ]

    # Primary findings for executive summary (severity > 0, sorted by severity desc)
    primary_findings = sorted(
        [
            {
                "category_id":  cat_id,
                "plain_label":  PLAIN_LABELS.get(cat_id, cat_id),
                "severity":     s["severity"],
                "severity_label": SEVERITY_LABELS.get(s["severity"], str(s["severity"])),
                "confidence":   s.get("confidence", ""),
            }
            for cat_id, s in dim_scores_raw.items()
            if s.get("severity", 0) > 0
        ],
        key=lambda x: x["severity"],
        reverse=True,
    )

    now = datetime.now(timezone.utc).isoformat()

    record = {
        # ── Section 1: Evaluation Metadata ───────────────────────────────────
        "evaluation_id":         evaluation_id,
        "run_key":               run_key,
        "experiment_id":         item["experiment_id"],
        "batch_id":              item["batch_id"],
        "item_id":               item["item_id"],
        "experiment_name":       exp_meta.get("experiment_name", ""),
        "research_phase":        exp_meta.get("phase", ""),
        "eval_status":           item.get("status", "completed"),
        "human_validation_status": hv_status,
        "evallab_version":       EVALLAB_VERSION,
        "record_format_version": RECORD_FORMAT_VERSION,
        # Lifecycle timestamps
        "run_started_at":        item.get("started_at"),
        "run_completed_at":      item.get("completed_at"),
        "record_registered_at":  record_registered_at,

        # ── Executive Summary fields ──────────────────────────────────────────
        "primary_findings":             primary_findings,
        "overall_judge_confidence":     overall_judge_confidence,

        # ── Section 2: Source Information ────────────────────────────────────
        "source_title":          result.get("source_title") or ds.get("source_title", ""),
        "source_publisher":      ds_meta.get("publisher", ""),
        "source_published_date": ds_meta.get("published_date", ""),
        "source_url":            ds.get("source_url", ""),
        "source_text":           ds.get("source_text", ""),
        "article_type":          result.get("article_type") or ds.get("article_type", ""),
        "source_type":           ds.get("source_type", ""),
        "benchmark_category":    chars.get("benchmark_category", ""),
        "cultural_significance": chars.get("cultural_significance", ""),
        "primary_community":     chars.get("primary_community") or "",
        "geographic_context":    chars.get("geographic_context", ""),
        "domain":                chars.get("domain", ""),
        "transformation_risk":   chars.get("transformation_risk", ""),
        "benchmark_rationale":   ds.get("benchmark_rationale", ""),
        "expected_failure_categories": expected_failure_categories,

        # ── Section 3: Transformation Configuration ──────────────────────────
        "transformation_task":   exp_meta.get("task_instructions", ""),
        "prompt_variant":        item.get("prompt_variant", ""),
        "prompt_variant_name":   result.get("prompt_variant_name", ""),
        "system_prompt":         result.get("system_prompt", ""),
        "eval_prompt":           result.get("eval_prompt", ""),
        "subject_model_id":      subject_model_id,
        "subject_model_display": subject_model_display,
        "subject_provider":      subject_provider,
        "subject_version_note":  subject_version,
        "temperature":           result.get("temperature"),
        "generation_timestamp":  result.get("timestamp", ""),

        # ── Section 4: Generated Output ──────────────────────────────────────
        "response_text": result.get("response_text", ""),

        # ── Section 5: Automated Evaluation ──────────────────────────────────
        "judge_model_id":       judge_model_id,
        "judge_model_display":  judge_display,
        "judge_provider":       judge_provider,
        "judge_version_note":   judge_version,
        "overall_ii_score":     result.get("overall_information_integrity_score"),
        "overall_cf_score":     result.get("overall_cultural_fidelity_score"),
        "executive_summary":    result.get("executive_summary", ""),
        "most_significant_failures": result.get("most_significant_failures", []),
        "suggested_improvements":    result.get("suggested_improvements", ""),
        "dimension_scores":          dim_scores_raw,

        # ── Section 6: Human Validation ──────────────────────────────────────
        "human_validation": human_validation,

        # ── Section 7: Reproducibility Metadata ──────────────────────────────
        "taxonomy_version":    item.get("taxonomy_version") or matching[0]["taxonomy_version"] if matching else "v1.0",
        "rubric_version":      item.get("rubric_version")   or matching[0]["rubric_version"]   if matching else "v1",
        "research_objective":  exp_meta.get("research_objective", ""),
        "research_question":   exp_meta.get("research_question", ""),
        "hypothesis":          exp_meta.get("hypothesis", ""),
        "experiment_models":   exp_meta.get("models", []),
        "experiment_config_snapshot": {
            "experiment_id":       exp_meta.get("experiment_id", ""),
            "experiment_name":     exp_meta.get("experiment_name", ""),
            "phase":               exp_meta.get("phase", ""),
            "prompt_variant":      exp_meta.get("prompt_variant", ""),
            "task_instructions":   exp_meta.get("task_instructions", ""),
            "models":              exp_meta.get("models", []),
            "status":              exp_meta.get("status"),
        },

        # ── Section 8: Export Metadata ────────────────────────────────────────
        "record_exported_at":    now,
        "export_format_version": RECORD_FORMAT_VERSION,
        "generated_by":          f"EvalLab {EVALLAB_VERSION}",
    }

    record["checksum"] = _checksum({k: v for k, v in record.items() if k != "checksum"})
    return record


# ── HTML renderer ─────────────────────────────────────────────────────────────

def _esc(text: str) -> str:
    """Minimal HTML escaping — enough for user-authored text content."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _fmt_date(iso: Optional[str], fmt: str = "%B %-d, %Y") -> str:
    if not iso:
        return ""
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.strftime(fmt)
    except Exception:
        return iso


def render_eval_record_html(record: dict) -> str:
    is_ext        = record.get("record_type") == "external_package"
    ev_id         = _esc(record.get("evaluation_id", "—"))
    exp_name      = _esc(record.get("experiment_name", "—"))
    research_phase = _esc(record.get("research_phase", "—"))
    subject_model = _esc(record.get("subject_model_display") or record.get("subject_model_id", "—"))
    judge_model   = _esc(record.get("judge_model_display") or record.get("judge_model_id", "—"))
    hv_status     = _esc(record.get("human_validation_status", "Not performed"))
    exported_at_raw = record.get("record_exported_at", "")
    exported_fmt    = _fmt_date(exported_at_raw, "%B %-d, %Y · %H:%M UTC") or exported_at_raw

    hv = record.get("human_validation")
    hv_cover_color  = "#16a34a" if hv and hv.get("responses") else "#6b7280"
    hv_cover_icon   = "✓" if hv and hv.get("responses") else "○"
    hv_cover_label  = hv_status

    ii_score = record.get("overall_ii_score")
    cf_score = record.get("overall_cf_score")

    def score_color(s):
        if s is None: return "#6b7280"
        if s >= 8:    return "#16a34a"
        if s >= 6:    return "#d97706"
        return "#dc2626"

    # ── Cover page ────────────────────────────────────────────────────────────
    cover_html = f"""
  <div class="cover">
    <div class="cover-top">
      <div class="cover-label">Evaluation Record</div>
      <div class="cover-badge">RESEARCHER · INTERNAL</div>
    </div>

    <div class="cover-id">{ev_id}</div>
    <div class="cover-title">{exp_name}</div>

    <div class="cover-grid">
      <div class="cover-field">
        <div class="cf-label">{'Package ID' if is_ext else 'Experiment ID'}</div>
        <div class="cf-value mono">{_esc(record.get('package_id') if is_ext else record.get('experiment_id', '—'))}</div>
      </div>
      <div class="cover-field">
        <div class="cf-label">{'Record Type' if is_ext else 'Research Phase'}</div>
        <div class="cf-value">{'External Package' if is_ext else research_phase}</div>
      </div>
      <div class="cover-field">
        <div class="cf-label">{'Submitting Framework' if is_ext else 'Item'}</div>
        <div class="cf-value {'small' if is_ext else 'mono'}">{_esc('Independent External Package') if is_ext else _esc(record.get('item_id', '—'))}</div>
      </div>
      <div class="cover-field">
        <div class="cf-label">{'Submitted By' if is_ext else 'Prompt Variant'}</div>
        <div class="cf-value">{_esc(record.get('package_submitted_by') or '—') if is_ext else _esc(record.get('prompt_variant', '—'))}</div>
      </div>
      <div class="cover-field">
        <div class="cf-label">{'Subject System Under Evaluation' if is_ext else 'Subject Model'}</div>
        <div class="cf-value strong">{subject_model}</div>
      </div>
      <div class="cover-field">
        <div class="cf-label">Judge Model</div>
        <div class="cf-value strong">{judge_model}</div>
      </div>
      <div class="cover-field">
        <div class="cf-label">Information Integrity</div>
        <div class="cf-value strong" style="color:{score_color(ii_score)}">{_fmt_score(ii_score)} / 10</div>
      </div>
      <div class="cover-field">
        <div class="cf-label">Cultural Fidelity</div>
        <div class="cf-value strong" style="color:{score_color(cf_score)}">{_fmt_score(cf_score)} / 10</div>
      </div>
      <div class="cover-field">
        <div class="cf-label">Human Validation</div>
        <div class="cf-value strong" style="color:{hv_cover_color}">{hv_cover_icon} {hv_cover_label}</div>
      </div>
      <div class="cover-field">
        <div class="cf-label">Exported</div>
        <div class="cf-value">{exported_fmt}</div>
      </div>
    </div>

    <div class="cover-footer">
      <span>EvalLab {_esc(record.get('evallab_version', ''))}</span>
      <span class="sep">·</span>
      <span>Record format {_esc(record.get('record_format_version', ''))}</span>
      <span class="sep">·</span>
      <span class="mono" style="font-size:10px;word-break:break-all">SHA-256 {_esc(record.get('checksum', ''))[:16]}…</span>
    </div>
  </div>"""

    # ── Executive Summary (after cover, before numbered sections) ─────────────
    primary_findings = record.get("primary_findings") or []
    findings_summary_html = ""
    for f in primary_findings:
        sev = f["severity"]
        color  = SEVERITY_COLORS.get(sev, "#6b7280")
        label  = f["severity_label"]
        findings_summary_html += f"""
      <div class="exec-finding">
        <span class="exec-sev-dot" style="background:{color}"></span>
        <span class="exec-sev-label" style="color:{color}">{_esc(label)}</span>
        <span class="exec-finding-name">{_esc(f['plain_label'])}</span>
      </div>"""

    hv = record.get("human_validation")
    hv_summary_label = "Not requested"
    hv_summary_color = "#6b7280"
    if hv:
        status = hv.get("review_status", "")
        if hv.get("responses"):
            hv_summary_label = f"Completed · {len(hv['responses'])} response{'s' if len(hv['responses']) != 1 else ''}"
            hv_summary_color = "#16a34a"
        elif status == "exported":
            hv_summary_label = "Exported · Awaiting responses"
            hv_summary_color = "#d97706"
        else:
            hv_summary_label = "Pending"
            hv_summary_color = "#d97706"

    if is_ext:
        eval_philosophy_html = '<div class="eval-philosophy">This Evaluation Record preserves the submitted External Evaluation Package unchanged and adds an independent EvalLab evaluation layer. The original artifacts remain unmodified and continue to serve as the authoritative evidence package.</div>'
    else:
        eval_philosophy_html = ""

    s_exec = f"""
  <section class="exec-summary-section">
    <h2>Evaluation Summary</h2>
    {eval_philosophy_html}
    <div class="exec-grid">

      <div class="exec-block">
        <div class="exec-block-label">Purpose</div>
        <div class="exec-block-value">{_esc(record.get('research_objective') or record.get('transformation_task') or '—')}</div>
      </div>

      <div class="exec-block">
        <div class="exec-block-label">Overall Result</div>
        <div class="exec-scores">
          <div class="exec-score-item">
            <span class="exec-score-num" style="color:{score_color(ii_score)}">{_fmt_score(ii_score)}</span>
            <span class="exec-score-sub">Information Integrity</span>
          </div>
          <div class="exec-score-divider"></div>
          <div class="exec-score-item">
            <span class="exec-score-num" style="color:{score_color(cf_score)}">{_fmt_score(cf_score)}</span>
            <span class="exec-score-sub">Cultural Fidelity</span>
          </div>
          <div class="exec-score-divider"></div>
          <div class="exec-score-item">
            <span class="exec-score-num" style="color:#6b7280">{record.get('overall_judge_confidence','—')}%</span>
            <span class="exec-score-sub">Judge confidence</span>
          </div>
        </div>
      </div>

      <div class="exec-block">
        <div class="exec-block-label">Primary Findings</div>
        {findings_summary_html if findings_summary_html else '<div class="exec-clean">No issues identified</div>'}
      </div>

      <div class="exec-block">
        <div class="exec-block-label">Human Validation</div>
        <div class="exec-block-value" style="color:{hv_summary_color}">{hv_summary_label}</div>
      </div>

    </div>
  </section>"""

    # ── Section 1: Evaluation Metadata + Lifecycle ────────────────────────────
    def _lc_step(done: bool, label: str, date_str: str) -> str:
        icon  = "✓" if done else "○"
        color = "#16a34a" if done else "#94a3b8"
        date_html = f'<span class="lc-date">{_esc(date_str)}</span>' if date_str else ""
        return f'<div class="lc-step"><span class="lc-icon" style="color:{color}">{icon}</span><span class="lc-label" style="color:{"#0f172a" if done else "#94a3b8"}">{label}</span>{date_html}</div>'

    run_started   = _fmt_date(record.get("run_started_at"))
    run_completed = _fmt_date(record.get("run_completed_at"))
    rec_created   = _fmt_date(record.get("record_registered_at"))

    hr_created  = _fmt_date(hv.get("created_at"))   if hv else ""
    hr_exported = _fmt_date(hv.get("exported_at"))  if hv else ""
    hr_complete = _fmt_date(hv.get("completed_at")) if hv else ""
    has_responses = bool(hv and hv.get("responses"))

    if is_ext:
        pkg_received  = _fmt_date(record.get("package_received_at"))
        pkg_approved  = _fmt_date(record.get("package_approved_at"))
        judged_at_fmt = _fmt_date(record.get("judged_at"))
        lifecycle_html = (
            _lc_step(bool(pkg_received),  "Package received",              pkg_received)
            + _lc_step(bool(pkg_approved), "Package approved — EV record created", pkg_approved)
            + _lc_step(bool(judged_at_fmt), "EvalLab Judge evaluation completed", judged_at_fmt)
            + _lc_step(has_responses,       "Human validation complete",   hr_complete)
        )
    else:
        lifecycle_html = (
            _lc_step(bool(run_started),   "Generation started",           run_started)
            + _lc_step(bool(run_completed), "Judge evaluation completed",  run_completed)
            + _lc_step(bool(rec_created),   "Evaluation Record created",   rec_created)
            + _lc_step(bool(hv),            "Human Review packet generated", hr_created)
            + _lc_step(bool(hr_exported),   "Packet exported to reviewer", hr_exported)
            + _lc_step(has_responses,       "Human validation complete",   hr_complete)
        )

    s1 = f"""
  <section>
    <div class="section-num">01</div>
    <h2>Evaluation Metadata</h2>
    <div class="two-col">
      <table class="mt">
        <tr><th>Evaluation ID</th><td class="mono">{_esc(record.get('evaluation_id','—'))}</td></tr>
        <tr><th>Run Key</th><td class="mono small">{_esc(record.get('run_key','—'))}</td></tr>
        <tr><th>{'Package ID' if is_ext else 'Experiment ID'}</th><td class="mono">{_esc(record.get('package_id') if is_ext else record.get('experiment_id','—'))}</td></tr>
        <tr><th>{'Submitted By' if is_ext else 'Batch ID'}</th><td class="{'mono' if not is_ext else ''}">{_esc(record.get('package_submitted_by') or '—') if is_ext else _esc(record.get('batch_id','—'))}</td></tr>
        <tr><th>{'Source Label' if is_ext else 'Experiment Name'}</th><td>{_esc(record.get('package_source_label') if is_ext else record.get('experiment_name','—'))}</td></tr>
        <tr><th>{'Record Type' if is_ext else 'Research Phase'}</th><td>{'External Package' if is_ext else _esc(record.get('research_phase','—'))}</td></tr>
        <tr><th>Export Timestamp</th><td>{exported_fmt}</td></tr>
        <tr><th>EvalLab Version</th><td>{_esc(record.get('evallab_version','—'))}</td></tr>
      </table>
      <div>
        <div class="lc-title">Evaluation Lifecycle</div>
        <div class="lifecycle">{lifecycle_html}</div>
      </div>
    </div>
  </section>"""

    # ── Section 2: Source Information ─────────────────────────────────────────
    if is_ext:
        mapped = record.get("mapped_meta") or {}
        s2 = f"""
  <section>
    <div class="section-num">02</div>
    <h2>Source Information</h2>
    <table class="mt" style="margin-bottom:20px">
      <tr><th>Title</th><td>{_extfield(record.get('source_title'), is_ext)}</td></tr>
      <tr><th>Researcher</th><td>{_extfield(record.get('researcher'), is_ext)}</td></tr>
      <tr><th>Institution</th><td>{_extfield(record.get('institution'), is_ext)}</td></tr>
      <tr><th>Methodology</th><td>{_extfield(record.get('methodology'), is_ext)}</td></tr>
      <tr><th>Date</th><td>{_extfield(record.get('source_published_date'), is_ext)}</td></tr>
      <tr><th>Contact</th><td>{_extfield(record.get('source_url'), is_ext)}</td></tr>
      <tr><th>Source Type</th><td>External Package</td></tr>
    </table>
    <div class="source-block">
      <div class="source-block-label">Source Artifact</div>
      <div class="artifact-ref">
        <span class="artifact-path">{_esc(record.get('source_file','—'))}</span>
        <span class="artifact-note">File reference within the submitted package. Not rendered — open the original artifact for full content.</span>
      </div>
    </div>
  </section>"""
    else:
        s2 = f"""
  <section>
    <div class="section-num">02</div>
    <h2>Source Information</h2>
    <table class="mt" style="margin-bottom:20px">
      <tr><th>Title</th><td>{_esc(record.get('source_title','—'))}</td></tr>
      <tr><th>Publisher</th><td>{_esc(record.get('source_publisher','—') or '—')}</td></tr>
      <tr><th>Publication Date</th><td>{_esc(record.get('source_published_date','—') or '—')}</td></tr>
      <tr><th>Source URL</th><td><span class="mono small">{_esc(record.get('source_url','—') or '—')}</span></td></tr>
      <tr><th>Source Type</th><td>{_esc(record.get('source_type','—') or '—')}</td></tr>
      <tr><th>Article Type</th><td>{_esc(record.get('article_type','—') or '—')}</td></tr>
    </table>
    <h3>Benchmark Characteristics</h3>
    <table class="mt" style="margin-bottom:16px">
      <tr><th>Benchmark Category</th><td>{_esc(record.get('benchmark_category','—') or '—')}</td></tr>
      <tr><th>Cultural Significance</th><td>{_esc(record.get('cultural_significance','—') or '—')}</td></tr>
      <tr><th>Primary Community</th><td>{_esc(record.get('primary_community','—') or '—')}</td></tr>
      <tr><th>Geographic Context</th><td>{_esc(record.get('geographic_context','—') or '—')}</td></tr>
      <tr><th>Domain</th><td>{_esc(record.get('domain','—') or '—')}</td></tr>
      <tr><th>Transformation Risk</th><td>{_esc(record.get('transformation_risk','—') or '—')}</td></tr>
    </table>
    {f'<div class="rationale"><strong>Purpose in benchmark:</strong> {_esc(record["benchmark_rationale"])}</div>' if record.get('benchmark_rationale') else ''}
    {'<div class="expected-failures"><div class="ef-label">Expected failure categories</div><div class="ef-chips">' + "".join(f'<span class="ef-chip">{_esc(c)}</span>' for c in record["expected_failure_categories"]) + "</div></div>" if record.get("expected_failure_categories") else ""}
    <div class="source-block">
      <div class="source-block-label">Original Source Text</div>
      <div class="source-text">{_esc(record.get('source_text',''))}</div>
    </div>
  </section>"""

    # ── Section 3: Transformation / System Configuration ─────────────────────
    if is_ext:
        s3 = f"""
  <section>
    <div class="section-num">03</div>
    <h2>System Configuration</h2>
    <table class="mt" style="margin-bottom:20px">
      <tr><th>Evaluation Task</th><td>{_extfield(record.get('transformation_task'), is_ext)}</td></tr>
      <tr><th>Source System</th><td class="strong">{subject_model}</td></tr>
      <tr><th>System Identifier</th><td class="mono">{_extfield(record.get('subject_label'), is_ext)}</td></tr>
      <tr><th>System Type</th><td>External System (non-LLM)</td></tr>
      <tr><th>Prompt Variant</th><td>{_na('ASR evaluation — no prompt used')}</td></tr>
      <tr><th>Temperature</th><td>{_na('ASR evaluation — not applicable')}</td></tr>
      <tr><th>Run Timestamp</th><td>{_extfield(record.get('generation_timestamp'), is_ext)}</td></tr>
    </table>
    <div class="source-block">
      <div class="source-block-label">Transformation Artifact</div>
      <div class="artifact-ref">
        <span class="artifact-path">{_esc(record.get('transformation_file','—'))}</span>
        <span class="artifact-note">Output file produced by the source system. Not rendered — open the original artifact for full content.</span>
      </div>
    </div>
  </section>"""
    else:
        s3 = f"""
  <section>
    <div class="section-num">03</div>
    <h2>Transformation Configuration</h2>
    <table class="mt" style="margin-bottom:20px">
      <tr><th>Transformation Task</th><td>{_esc(record.get('transformation_task','—') or '—')}</td></tr>
      <tr><th>Prompt Variant</th><td>{_esc(record.get('prompt_variant','—'))}</td></tr>
      <tr><th>Subject Model</th><td class="strong">{subject_model}</td></tr>
      <tr><th>Model ID</th><td class="mono">{_esc(record.get('subject_model_id','—'))}</td></tr>
      <tr><th>Provider</th><td>{_esc(record.get('subject_provider','—') or '—')}</td></tr>
      <tr><th>Version Note</th><td class="small">{_esc(record.get('subject_version_note','—') or '—')}</td></tr>
      <tr><th>Temperature</th><td>{_esc(str(record.get('temperature','—')))}</td></tr>
      <tr><th>Generation Timestamp</th><td>{_esc(record.get('generation_timestamp','—') or '—')}</td></tr>
    </table>
    <h3>System Prompt</h3>
    <div class="prompt-block">{_esc(record.get('system_prompt',''))}</div>
    <h3>Evaluation Prompt</h3>
    <div class="prompt-block">{_esc(record.get('eval_prompt',''))}</div>
  </section>"""

    # ── Section 4: Output ─────────────────────────────────────────────────────
    if is_ext:
        s4 = f"""
  <section>
    <div class="section-num">04</div>
    <h2>Transformation Output</h2>
    <p class="section-note">Reference to the transformation artifact as submitted. The full content is preserved within the package directory.</p>
    <div class="artifact-ref" style="margin-top:8px">
      <span class="artifact-path">{_esc(record.get('transformation_file','—'))}</span>
      <span class="artifact-note">Open the package artifact directly to review the full transformation output.</span>
    </div>
  </section>"""
    else:
        s4 = f"""
  <section>
    <div class="section-num">04</div>
    <h2>Generated Output</h2>
    <p class="section-note">Complete model response, unmodified.</p>
    <div class="response-block">{_esc(record.get('response_text',''))}</div>
  </section>"""

    # ── Section 5: Evidence Relationships ────────────────────────────────────
    def _ev_node(label: str, detail: str = "", cls: str = "") -> str:
        detail_html = f'<span class="ev-node-detail">{_esc(detail)}</span>' if detail else ""
        return f'<div class="ev-node {cls}"><span class="ev-node-label">{_esc(label)}</span>{detail_html}</div>'

    def _ev_arrow() -> str:
        return '<div class="ev-arrow">↓</div>'

    if is_ext:
        # Derive artifact layers from file_manifest folder prefixes
        manifest = record.get("file_manifest") or []
        folders: dict[str, list[str]] = {}
        for f in manifest:
            name = f.get("name", "")
            parts = name.split("/")
            # Find the numbered folder (e.g. "04_ASR_RUN")
            for part in parts[1:]:  # skip the top-level package folder
                if part and part[0].isdigit() and "_" in part:
                    folders.setdefault(part, []).append(name.split("/")[-1])
                    break

        # Build chain from sorted folder names
        chain_nodes = []
        for folder in sorted(folders.keys()):
            num, _, name = folder.partition("_")
            label = name.replace("_", " ").title()
            files = folders[folder]
            sample = files[0] if files else ""
            chain_nodes.append(_ev_node(label, sample))

        if not chain_nodes:
            # Fallback: derive from source/transformation file paths
            src = record.get("source_file", "")
            tfm = record.get("transformation_file", "")
            chain_nodes = [
                _ev_node("Source Artifact", src),
                _ev_node("Transformation Artifact", tfm, "ev-evallab"),
            ]

        chain_nodes.append(_ev_node("EvalLab Judge Assessment",
                                    record.get("judge_model", ""), "ev-evallab"))
        hv_cls = "ev-done" if has_responses else "ev-pending"
        hv_detail = "Not performed" if not hv else ("Complete" if has_responses else "Pending")
        chain_nodes.append(_ev_node("Human Validation", hv_detail, hv_cls))

        chain_html = _ev_arrow().join(chain_nodes)
        s_evidence = f"""
  <section>
    <div class="section-num">05</div>
    <h2>Evidence Relationships</h2>
    <p class="section-note">
      The chain below shows how each preserved artifact relates to the next.
      Each layer is a distinct, unmodified record of a specific stage of the evaluation.
    </p>
    <div class="evidence-chain">{chain_html}</div>
  </section>"""
    else:
        hv_cls = "ev-done" if has_responses else ("ev-pending" if not hv else "ev-exported")
        hv_detail = "Not requested" if not hv else ("Complete" if has_responses else "Exported — awaiting responses")
        std_chain = _ev_arrow().join([
            _ev_node("Original Source Text", _esc(record.get("source_title", ""))[:60]),
            _ev_node("Prompt", record.get("prompt_variant", "") or record.get("prompt_variant_name", "")),
            _ev_node("Model Output", record.get("subject_model_display") or record.get("subject_model_id", ""), "ev-evallab"),
            _ev_node("Judge Evaluation", record.get("judge_model_display") or record.get("judge_model_id", ""), "ev-evallab"),
            _ev_node("Evaluation Record", record.get("evaluation_id", ""), "ev-evallab"),
            _ev_node("Human Validation", hv_detail, hv_cls),
        ])
        s_evidence = f"""
  <section>
    <div class="section-num">05</div>
    <h2>Evidence Relationships</h2>
    <p class="section-note">
      The chain below shows how each preserved artifact relates to the next,
      from the original source through automated and human validation.
    </p>
    <div class="evidence-chain">{std_chain}</div>
  </section>"""

    # ── Section 6: Automated Evaluation ──────────────────────────────────────
    dim_rows = ""
    dim_scores = record.get("dimension_scores") or {}
    for cat_id, score in dim_scores.items():
        sev    = score.get("severity", 0)
        if sev == 0:
            continue
        sev_label  = SEVERITY_LABELS.get(sev, str(sev))
        sev_color  = SEVERITY_COLORS.get(sev, "#6b7280")
        bg         = SEVERITY_BG.get(sev, "#f9fafb")
        border     = SEVERITY_BORDER.get(sev, "#e5e7eb")
        plain      = _esc(PLAIN_LABELS.get(cat_id, cat_id))
        confidence = _esc(score.get("confidence", ""))
        explanation = _esc(score.get("explanation", ""))
        src_ev     = _esc(score.get("source_evidence", ""))
        out_ev     = _esc(score.get("output_evidence", ""))
        evidence_html = ""
        if src_ev:
            evidence_html += f'<div class="evidence"><span class="ev-label">Source evidence:</span> {src_ev}</div>'
        if out_ev:
            evidence_html += f'<div class="evidence"><span class="ev-label">Output evidence:</span> {out_ev}</div>'
        dim_rows += f"""
    <div class="finding" style="background:{bg};border-color:{border}">
      <div class="finding-header">
        <span class="cat-id">{_esc(cat_id)}</span>
        <span class="cat-name">{plain}</span>
        <span class="sev-badge" style="background:{sev_color}">{sev_label}</span>
        <span class="conf-tag">{confidence} confidence</span>
      </div>
      <p class="explanation">{explanation}</p>
      {evidence_html}
    </div>"""

    clean_cats = [cat_id for cat_id, s in dim_scores.items() if s.get("severity", 0) == 0]
    clean_html = ""
    if clean_cats:
        chips = "".join(f'<span class="clean-chip">{_esc(PLAIN_LABELS.get(c, c))}</span>' for c in clean_cats)
        clean_html = f'<div class="clean-cats"><strong>No issues:</strong> {chips}</div>'

    # Expand abbreviations in most_significant_failures
    failures_html = "".join(
        f"<li>{_esc(PLAIN_LABELS.get(f, f))}</li>"
        for f in (record.get("most_significant_failures") or [])
    )
    judge_confidence = record.get("overall_judge_confidence")
    conf_color = "#16a34a" if (judge_confidence or 0) >= 80 else "#d97706" if (judge_confidence or 0) >= 60 else "#dc2626"

    s5 = f"""
  <section>
    <div class="section-num">06</div>
    <h2>Automated Evaluation</h2>
    <div class="score-row">
      <div class="score-card">
        <div class="score-val" style="color:{score_color(ii_score)}">{_fmt_score(ii_score)}</div>
        <div class="score-label">Information Integrity<br><span class="score-sub">/ 10</span></div>
      </div>
      <div class="score-card">
        <div class="score-val" style="color:{score_color(cf_score)}">{_fmt_score(cf_score)}</div>
        <div class="score-label">Cultural Fidelity<br><span class="score-sub">/ 10</span></div>
      </div>
      <div class="score-card">
        <div class="score-val" style="color:{conf_color}">{judge_confidence if judge_confidence is not None else '—'}%</div>
        <div class="score-label">Judge Confidence<br><span class="score-sub">across flagged dims</span></div>
      </div>
      <div class="score-card" style="flex:2">
        <div class="score-label" style="margin-bottom:8px;font-weight:600">Judge Model</div>
        <div style="font-size:13px;font-weight:600">{judge_model}</div>
        <div style="font-size:11px;color:#6b7280;margin-top:2px">{_esc(record.get('judge_model_id',''))}</div>
      </div>
    </div>
    {f'<div class="exec-summary">{_esc(record.get("executive_summary",""))}</div>' if record.get("executive_summary") else ''}
    {f'<h3>Most Significant Failures</h3><ul class="failure-list">{failures_html}</ul>' if failures_html else ''}
    {f'<h3>Suggested Improvements</h3><p class="improvements">{_esc(record.get("suggested_improvements",""))}</p>' if record.get("suggested_improvements") else ''}
    <h3>Full Taxonomy Results</h3>
    {dim_rows if dim_rows else '<p class="all-clean">No issues identified across all 14 taxonomy dimensions.</p>'}
    {clean_html}
  </section>"""

    # ── Section 7: Human Validation ───────────────────────────────────────────
    if hv:
        resp_html = ""
        for resp in hv.get("responses", []):
            agree      = resp.get("agreement_with_judge", "")
            agree_color = {"yes": "#16a34a", "partially": "#d97706", "no": "#dc2626"}.get(agree, "#6b7280")
            dt_html = "".join(
                f'<span class="dt-chip">{_esc(DISAGREE_LABELS.get(dt, dt))}</span>'
                for dt in (resp.get("disagreement_types") or [])
            )
            extra_rows = ""
            for field, label in [
                ("missed_failures",            "Missed failures"),
                ("incorrectly_flagged",         "Incorrectly flagged"),
                ("preserved_meaning",           "Preserved meaning"),
                ("cultural_context_preserved",  "Cultural context"),
                ("additional_comments",         "Additional comments"),
            ]:
                val = resp.get(field, "")
                if val:
                    extra_rows += f"<tr><th>{label}</th><td>{_esc(val)}</td></tr>"
            resp_html += f"""
      <div class="response-card">
        <div class="resp-header">
          <span class="reviewer-id">{_esc(resp.get('reviewer_id',''))}</span>
          <span class="agree-badge" style="background:{agree_color}">{_esc(agree)}</span>
          <span class="resp-date">{_esc(resp.get('reviewed_at','') or '—')}</span>
        </div>
        {f'<div class="dt-chips">{dt_html}</div>' if dt_html else ''}
        {f'<p class="resp-comment">{_esc(resp.get("comments",""))}</p>' if resp.get("comments") else ''}
        {f'<table class="mt small-table">{extra_rows}</table>' if extra_rows else ''}
        <div class="resp-meta">Imported {_esc(resp.get('imported_at','—') or '—')} via {_esc(resp.get('import_source',''))}</div>
      </div>"""

        # Build lifecycle for the review itself
        hv_lc = (
            _lc_step(True,                       "Packet generated",          _fmt_date(hv.get("created_at")))
            + _lc_step(bool(hv.get("exported_at")), "Packet exported to reviewer", _fmt_date(hv.get("exported_at")))
            + _lc_step(has_responses,               "Responses received",          _fmt_date(hv.get("completed_at")))
        )

        s6 = f"""
  <section>
    <div class="section-num">07</div>
    <h2>Human Validation</h2>
    <div class="two-col" style="margin-bottom:24px">
      <table class="mt">
        <tr><th>Review ID</th><td class="mono">{_esc(hv['review_id'])}</td></tr>
        <tr><th>Status</th><td>{_esc(hv['review_status'].title())}</td></tr>
        <tr><th>Round</th><td>{hv['review_round']}</td></tr>
        <tr><th>Blinded</th><td>{'Yes — model identity withheld' if hv['blinded'] else 'No'}</td></tr>
        <tr><th>Selection Reasons</th><td>{_esc(', '.join(hv['selection_reasons'])) or '—'}</td></tr>
      </table>
      <div>
        <div class="lc-title">Review Lifecycle</div>
        <div class="lifecycle">{hv_lc}</div>
        {f'<p class="muted" style="margin-top:12px;font-size:12px">Awaiting reviewer responses.</p>' if not has_responses else ''}
      </div>
    </div>
    <h3>Reviewer Responses ({len(hv.get('responses',[]))})</h3>
    {resp_html if resp_html else '<p class="muted">No responses have been imported yet.</p>'}
  </section>"""
    else:
        s6 = """
  <section>
    <div class="section-num">07</div>
    <h2>Human Validation</h2>
    <p class="muted">No human validation was performed for this evaluation.</p>
  </section>"""

    # ── Section 8: Reproducibility Metadata ──────────────────────────────────
    model_list_html = "".join(
        f"<li>{_esc(m)}</li>" for m in (record.get("experiment_models") or [])
    )
    prompt_variant_val = _na("External Submission") if is_ext else _esc(record.get('prompt_variant','—'))
    temperature_val    = _na("External Submission") if is_ext else _esc(str(record.get('temperature','—')))
    objective_val      = _extfield(record.get('research_objective'), is_ext)
    rq_val             = _na("External Submission") if is_ext else _extfield(record.get('research_question'), False)
    hypothesis_val     = _na("External Submission") if is_ext else _extfield(record.get('hypothesis'), False)

    if is_ext:
        exp_snapshot_html = f'<p class="muted">{_na("External Submission")} — no experiment configuration snapshot for external packages.</p>'
        s7_note = "This section documents the methodology used by the external system and EvalLab judge."
    else:
        model_list_part = f'<ul class="model-list">{model_list_html}</ul>' if model_list_html else ""
        config_json = _esc(json.dumps(record.get("experiment_config_snapshot", {}), indent=2))
        exp_snapshot_html = f'{model_list_part}<pre class="config-dump">{config_json}</pre>'
        s7_note = "This section documents the methodology. It does not claim deterministic reproducibility of LLM outputs, which are non-deterministic by design."

    s7 = f"""
  <section>
    <div class="section-num">08</div>
    <h2>Reproducibility Metadata</h2>
    <p class="section-note">{s7_note}</p>
    <table class="mt" style="margin-bottom:20px">
      <tr><th>Taxonomy Version</th><td>{_esc(record.get('taxonomy_version','—'))}</td></tr>
      <tr><th>Rubric Version</th><td>{_esc(record.get('rubric_version','—'))}</td></tr>
      <tr><th>Prompt Variant</th><td>{prompt_variant_val}</td></tr>
      <tr><th>Temperature</th><td>{temperature_val}</td></tr>
      <tr><th>{'System Note' if is_ext else 'Subject Model Note'}</th><td>{_extfield(record.get('subject_version_note'), is_ext)}</td></tr>
      <tr><th>Judge Model Note</th><td>{_extfield(record.get('judge_version_note'), is_ext)}</td></tr>
    </table>
    <h3>Research Context</h3>
    <table class="mt" style="margin-bottom:20px">
      <tr><th>Objective</th><td>{objective_val}</td></tr>
      <tr><th>Research Question</th><td>{rq_val}</td></tr>
      <tr><th>Hypothesis</th><td>{hypothesis_val}</td></tr>
    </table>
    <h3>{'Evaluation Configuration' if is_ext else 'Experiment Configuration Snapshot'}</h3>
    <p class="section-note">{'Package provenance — received and approved as submitted.' if is_ext else 'Frozen at run time. Reflects the experiment as it existed when this batch was submitted.'}</p>
    {exp_snapshot_html}
  </section>"""

    # ── Section 9: Export Metadata ────────────────────────────────────────────
    s8 = f"""
  <section>
    <div class="section-num">09</div>
    <h2>Export Metadata</h2>
    <table class="mt">
      <tr><th>Export Timestamp</th><td>{exported_fmt}</td></tr>
      <tr><th>Export Format</th><td>HTML (self-contained)</td></tr>
      <tr><th>Record Format Version</th><td>{_esc(record.get('record_format_version','—'))}</td></tr>
      <tr><th>Generated By</th><td>{_esc(record.get('generated_by','—'))}</td></tr>
      <tr><th>SHA-256 Checksum</th><td class="mono small">{_esc(record.get('checksum','—'))}</td></tr>
    </table>
    <p class="section-note" style="margin-top:16px">
      The checksum is computed over the canonical JSON representation of all record fields
      (sorted keys, no whitespace) before HTML rendering. It can be used to verify
      that this record has not been altered since export.
    </p>
  </section>"""

    # ── CSS ───────────────────────────────────────────────────────────────────
    css = """
  :root {
    --bg: #f8f9fa; --surface: #ffffff; --border: #e2e8f0;
    --text: #0f172a; --muted: #64748b; --accent: #1e40af;
    --accent-light: #eff6ff; --accent-border: #bfdbfe;
    --font: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter', sans-serif;
    --mono: 'SFMono-Regular', 'Consolas', 'Monaco', monospace;
  }
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  html { font-size: 14px; }
  body { font-family: var(--font); background: var(--bg); color: var(--text); line-height: 1.65; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
  .page { max-width: 960px; margin: 0 auto; padding: 0 24px 80px; }

  /* Cover */
  .cover { background: var(--accent); color: white; border-radius: 0 0 16px 16px; padding: 48px 48px 40px; margin-bottom: 40px; }
  .cover-top { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 32px; }
  .cover-label { font-size: 12px; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; opacity: 0.7; }
  .cover-badge { background: rgba(255,255,255,0.15); font-size: 10px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; padding: 4px 10px; border-radius: 4px; }
  .cover-id { font-family: var(--mono); font-size: 28px; font-weight: 700; letter-spacing: -0.5px; margin-bottom: 6px; }
  .cover-title { font-size: 18px; font-weight: 500; opacity: 0.85; margin-bottom: 36px; }
  .cover-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px 32px; margin-bottom: 36px; }
  @media (min-width: 640px) { .cover-grid { grid-template-columns: repeat(3, 1fr); } }
  .cf-label { font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; opacity: 0.6; margin-bottom: 3px; }
  .cf-value { font-size: 13px; }
  .cf-value.mono { font-family: var(--mono); font-size: 11px; word-break: break-all; }
  .cf-value.strong { font-weight: 700; font-size: 14px; }
  .cover-footer { border-top: 1px solid rgba(255,255,255,0.2); padding-top: 16px; font-size: 11px; opacity: 0.55; display: flex; gap: 12px; flex-wrap: wrap; }
  .sep { opacity: 0.4; }

  /* Sections */
  section { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 32px; margin-bottom: 20px; position: relative; }
  .section-num { position: absolute; top: 32px; right: 32px; font-family: var(--mono); font-size: 11px; font-weight: 700; color: var(--border); letter-spacing: 0.05em; }
  h2 { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: var(--muted); margin-bottom: 20px; padding-bottom: 10px; border-bottom: 1px solid var(--border); }
  h3 { font-size: 13px; font-weight: 700; color: var(--text); margin: 24px 0 12px; }
  .section-note { font-size: 12px; color: var(--muted); font-style: italic; margin-bottom: 16px; line-height: 1.6; }

  /* Meta tables */
  .mt { width: 100%; border-collapse: collapse; font-size: 13px; }
  .mt th { width: 180px; text-align: left; color: var(--muted); font-weight: 500; padding: 6px 16px 6px 0; vertical-align: top; white-space: nowrap; }
  .mt td { padding: 6px 0; vertical-align: top; }
  .mt tr { border-bottom: 1px solid #f1f5f9; }
  .mt tr:last-child { border-bottom: none; }
  .small-table th { width: 140px; }

  /* Text blocks */
  .source-block { margin-top: 20px; }
  .source-block-label { font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); margin-bottom: 8px; }
  .source-text, .response-block, .prompt-block {
    font-size: 13px; line-height: 1.8; color: #334155; white-space: pre-wrap;
    border: 1px solid var(--border); border-radius: 6px; padding: 20px;
    font-family: Georgia, 'Times New Roman', serif; max-height: 400px;
    overflow-y: auto; background: var(--bg);
  }
  .response-block { background: var(--accent-light); border-color: var(--accent-border); }
  .prompt-block { background: #fafafa; font-family: var(--mono); font-size: 12px; line-height: 1.6; max-height: 300px; }
  .rationale { font-size: 13px; color: #475569; background: #f8fafc; border-left: 3px solid var(--accent-border); padding: 10px 16px; border-radius: 0 6px 6px 0; margin: 16px 0; }

  /* Scores */
  .score-row { display: flex; gap: 12px; margin-bottom: 20px; flex-wrap: wrap; }
  .score-card { flex: 1; min-width: 100px; background: var(--bg); border: 1px solid var(--border); border-radius: 8px; padding: 16px; text-align: center; }
  .score-val { font-size: 36px; font-weight: 800; line-height: 1; margin-bottom: 8px; }
  .score-label { font-size: 11px; color: var(--muted); font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; line-height: 1.4; }
  .score-sub { font-size: 10px; font-weight: 400; }
  .exec-summary { font-size: 13px; color: #334155; line-height: 1.7; padding: 16px; background: #f8fafc; border-left: 3px solid #94a3b8; border-radius: 0 6px 6px 0; margin-bottom: 16px; }
  .failure-list { padding-left: 18px; margin-bottom: 16px; }
  .failure-list li { font-size: 13px; margin-bottom: 6px; color: #334155; }
  .improvements { font-size: 13px; color: #334155; line-height: 1.7; }

  /* Findings */
  .finding { border: 1px solid var(--border); border-radius: 8px; padding: 16px; margin-bottom: 12px; }
  .finding-header { display: flex; align-items: center; gap: 8px; margin-bottom: 10px; flex-wrap: wrap; }
  .cat-id { font-family: var(--mono); font-size: 11px; font-weight: 700; background: #f1f5f9; padding: 2px 7px; border-radius: 4px; color: #475569; }
  .cat-name { font-size: 13px; font-weight: 700; color: var(--text); }
  .sev-badge { color: white; font-size: 10px; font-weight: 700; padding: 3px 10px; border-radius: 20px; white-space: nowrap; }
  .conf-tag { font-size: 11px; color: var(--muted); margin-left: auto; }
  .explanation { font-size: 13px; color: #334155; line-height: 1.7; margin-bottom: 8px; }
  .evidence { font-size: 12px; color: #475569; background: #f8fafc; border-radius: 4px; padding: 8px 12px; margin-top: 6px; line-height: 1.6; }
  .ev-label { font-weight: 600; color: var(--muted); }
  .all-clean { color: #16a34a; font-size: 13px; font-weight: 600; padding: 12px 0; }
  .clean-cats { margin-top: 16px; font-size: 12px; color: var(--muted); }
  .clean-chip { display: inline-block; background: #f0fdf4; border: 1px solid #bbf7d0; color: #166534; font-size: 11px; padding: 2px 8px; border-radius: 20px; margin: 2px; }

  /* Human validation */
  .response-card { border: 1px solid var(--border); border-radius: 8px; padding: 16px; margin-bottom: 12px; }
  .resp-header { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; flex-wrap: wrap; }
  .reviewer-id { font-family: var(--mono); font-size: 13px; font-weight: 700; }
  .agree-badge { color: white; font-size: 11px; font-weight: 600; padding: 3px 12px; border-radius: 20px; }
  .resp-date { font-size: 11px; color: var(--muted); margin-left: auto; }
  .dt-chips { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 10px; }
  .dt-chip { background: #fff7ed; border: 1px solid #fed7aa; color: #9a3412; font-size: 11px; padding: 2px 9px; border-radius: 4px; }
  .resp-comment { font-size: 13px; color: #334155; line-height: 1.7; margin-bottom: 8px; }
  .resp-meta { font-size: 11px; color: #94a3b8; border-top: 1px solid #f1f5f9; padding-top: 8px; margin-top: 8px; }

  /* Executive Summary section */
  .exec-summary-section { background: #f8fafc; border: 1px solid var(--accent-border); border-radius: 10px; padding: 32px; margin-bottom: 20px; }
  .exec-summary-section h2 { border-color: var(--accent-border); color: #1e40af; }
  .exec-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  @media (min-width: 700px) { .exec-grid { grid-template-columns: 1fr 1fr 1fr 1fr; } }
  .exec-block { background: white; border: 1px solid var(--border); border-radius: 8px; padding: 16px; }
  .exec-block-label { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: var(--muted); margin-bottom: 8px; }
  .exec-block-value { font-size: 13px; color: var(--text); line-height: 1.5; }
  .exec-scores { display: flex; align-items: center; gap: 8px; }
  .exec-score-item { flex: 1; text-align: center; }
  .exec-score-num { display: block; font-size: 28px; font-weight: 800; line-height: 1; }
  .exec-score-sub { display: block; font-size: 10px; color: var(--muted); font-weight: 600; text-transform: uppercase; margin-top: 4px; }
  .exec-score-divider { width: 1px; height: 36px; background: var(--border); flex-shrink: 0; }
  .exec-clean { font-size: 12px; color: #16a34a; font-weight: 600; }
  .exec-finding { display: flex; align-items: center; gap: 6px; margin-bottom: 6px; }
  .exec-sev-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
  .exec-sev-label { font-size: 11px; font-weight: 700; text-transform: uppercase; flex-shrink: 0; }
  .exec-finding-name { font-size: 12px; color: var(--text); }

  /* Evaluation lifecycle */
  .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
  @media (max-width: 600px) { .two-col { grid-template-columns: 1fr; } }
  .lc-title { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: var(--muted); margin-bottom: 10px; }
  .lifecycle { display: flex; flex-direction: column; gap: 2px; }
  .lc-step { display: flex; align-items: flex-start; gap: 8px; padding: 5px 0; border-bottom: 1px solid #f1f5f9; }
  .lc-step:last-child { border-bottom: none; }
  .lc-icon { font-size: 13px; line-height: 1.4; flex-shrink: 0; width: 16px; text-align: center; }
  .lc-label { font-size: 12px; flex: 1; }
  .lc-date { font-size: 11px; color: var(--muted); white-space: nowrap; }

  /* Expected failure categories */
  .expected-failures { margin: 16px 0; }
  .ef-label { font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); margin-bottom: 8px; }
  .ef-chips { display: flex; flex-wrap: wrap; gap: 6px; }
  .ef-chip { background: #fefce8; border: 1px solid #fde047; color: #854d0e; font-size: 11px; padding: 3px 10px; border-radius: 20px; font-weight: 500; }

  /* Footer */
  .record-footer { margin-top: 40px; padding: 24px 0 16px; border-top: 2px solid var(--border); text-align: center; }
  .footer-top { display: flex; justify-content: center; align-items: center; gap: 16px; margin-bottom: 8px; }
  .footer-brand { font-size: 15px; font-weight: 800; color: var(--accent); letter-spacing: -0.5px; }
  .footer-id { font-family: var(--mono); font-size: 12px; color: var(--muted); background: #f1f5f9; padding: 2px 10px; border-radius: 4px; }
  .footer-note { font-size: 12px; color: var(--muted); margin-bottom: 4px; }
  .footer-policy { font-size: 11px; color: #94a3b8; font-style: italic; }

  /* Evaluation philosophy statement */
  .eval-philosophy { font-size: 13px; color: #1e40af; line-height: 1.65; padding: 12px 16px; background: rgba(255,255,255,0.6); border-left: 3px solid rgba(255,255,255,0.5); border-radius: 0 6px 6px 0; margin-bottom: 20px; font-style: italic; }

  /* Context-aware field states */
  .field-na { color: #94a3b8; font-style: italic; font-size: 12px; }
  .field-unknown { color: #94a3b8; font-style: italic; font-size: 12px; }

  /* Artifact reference (external packages) */
  .artifact-ref { display: flex; flex-direction: column; gap: 4px; background: #f8fafc; border: 1px solid var(--border); border-radius: 6px; padding: 14px 18px; margin-top: 4px; }
  .artifact-path { font-family: var(--mono); font-size: 12px; color: var(--accent); word-break: break-all; }
  .artifact-note { font-size: 11px; color: var(--muted); font-style: italic; }

  /* Evidence chain */
  .evidence-chain { display: flex; flex-direction: column; align-items: flex-start; max-width: 480px; }
  .ev-node { background: #f8fafc; border: 1px solid var(--border); border-radius: 8px; padding: 10px 20px; min-width: 300px; display: flex; flex-direction: column; gap: 2px; }
  .ev-node.ev-evallab { background: var(--accent-light); border-color: var(--accent-border); }
  .ev-node.ev-done { background: #f0fdf4; border-color: #bbf7d0; }
  .ev-node.ev-pending { background: #f8fafc; border-color: #e2e8f0; opacity: 0.65; }
  .ev-node.ev-exported { background: #fffbeb; border-color: #fde68a; }
  .ev-node-label { font-size: 13px; font-weight: 600; color: var(--text); }
  .ev-node.ev-evallab .ev-node-label { color: var(--accent); }
  .ev-node.ev-done .ev-node-label { color: #166534; }
  .ev-node.ev-pending .ev-node-label { color: #94a3b8; font-style: italic; }
  .ev-node-detail { font-size: 11px; color: var(--muted); font-family: var(--mono); word-break: break-all; }
  .ev-arrow { padding: 3px 0 3px 22px; color: #94a3b8; font-size: 16px; line-height: 1; }

  /* Misc */
  .muted { color: var(--muted); font-size: 13px; font-style: italic; }
  .mono { font-family: var(--mono); }
  .small { font-size: 11px; }
  .strong { font-weight: 700; }
  code { font-family: var(--mono); font-size: 12px; background: #f1f5f9; padding: 1px 5px; border-radius: 3px; }
  .config-dump { font-family: var(--mono); font-size: 11px; line-height: 1.6; background: #f8fafc; border: 1px solid var(--border); border-radius: 6px; padding: 16px; white-space: pre-wrap; word-break: break-word; overflow-x: auto; max-height: 300px; overflow-y: auto; }
  .model-list { padding-left: 18px; margin-bottom: 12px; }
  .model-list li { font-size: 13px; color: #334155; margin-bottom: 3px; }

  @media print {
    body { background: white; }
    .cover { -webkit-print-color-adjust: exact; }
    .source-text, .response-block, .prompt-block, .config-dump { max-height: none; overflow: visible; }
    section { break-inside: avoid; page-break-inside: avoid; }
  }"""

    # ── Footer ────────────────────────────────────────────────────────────────
    footer_html = f"""
  <footer class="record-footer">
    <div class="footer-top">
      <span class="footer-brand">EvalLab</span>
      <span class="footer-id">{ev_id}</span>
    </div>
    <div class="footer-note">Immutable snapshot as of {exported_fmt}</div>
    <div class="footer-policy">This is a research artifact. Handle according to your institution's data policies.</div>
  </footer>"""

    title = _esc(f"Evaluation Record {ev_id} — {record.get('source_title','')[:50]}")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>{css}</style>
</head>
<body>
<div class="page">
{cover_html}
{s_exec}
{s1}
{s2}
{s3}
{s4}
{s_evidence}
{s5}
{s6}
{s7}
{s8}
{footer_html}
</div>
</body>
</html>"""
