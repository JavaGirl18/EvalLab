import csv
import html as html_lib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from batch_db import get_batch
from human_review_db import get_human_review, list_human_reviews, mark_review_exported
from taxonomy import load_taxonomy

PLAIN_LABELS: dict = {
    "FH":  "Added or invented factual information",
    "CXH": "Introduced content unrelated to the source",
    "CTH": "Invented cultural practices or significance",
    "AH":  "Invented or fabricated a speaker, quote, or source",
    "FS":  "Directly contradicted the source",
    "UC":  "Added information not supported by the source",
    "CL":  "Omitted context needed to understand the source",
    "EX":  "Amplified severity, urgency, or emotional impact",
    "OC":  "Converted uncertainty into a definitive claim",
    "AL":  "Removed attribution to a speaker or organization",
    "CM":  "Misrepresented or generalized cultural meaning",
    "FB":  "Changed how people or events are perceived through word choice or framing",
    "PB":  "Distorted or marginalized a political actor or perspective",
    "ST":  "Attributed generalized characteristics to a group",
}

SEVERITY_LABELS = {0: "Not Present", 1: "Minor", 2: "Moderate", 3: "Severe"}

PACKETS_DIR = Path(__file__).parent / "data" / "human_reviews" / "packets"
IMPORTS_DIR = Path(__file__).parent / "data" / "human_reviews" / "imports"


def _ensure_dirs() -> None:
    PACKETS_DIR.mkdir(parents=True, exist_ok=True)
    IMPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _taxonomy_version() -> str:
    try:
        taxonomy = load_taxonomy()
        return f"v{taxonomy.get('version', '1.0')}"
    except Exception:
        return "v1"


def build_packet_snapshot(
    run_key: str,
    result: dict,
    row: dict,
    blinded: bool,
    max_source_chars: Optional[int] = None,
) -> dict:
    """
    Build the immutable reviewer-facing snapshot for one completed run.
    Called at human_review creation time — never again after that.
    """
    from research_runner import get_dataset_item

    item_id = row["item_id"]

    # Snapshot source document at creation time (immutable from here on)
    try:
        dataset_item = get_dataset_item(item_id)
        source_text        = dataset_item.source_text or ""
        publisher          = (dataset_item.metadata or {}).get("publisher", "")
        pub_date           = (dataset_item.metadata or {}).get("published_date", "")
        benchmark_category = getattr(dataset_item, "characteristics", {})
        if isinstance(benchmark_category, dict):
            bench_cat  = benchmark_category.get("benchmark_category", "")
            cult_sig   = benchmark_category.get("cultural_significance", "")
        else:
            bench_cat  = getattr(benchmark_category, "benchmark_category", "")
            cult_sig   = getattr(benchmark_category, "cultural_significance", "")
    except Exception:
        source_text = result.get("source_title", "")
        publisher   = ""
        pub_date    = ""
        bench_cat   = ""
        cult_sig    = ""

    if max_source_chars and len(source_text) > max_source_chars:
        source_text = (
            source_text[:max_source_chars]
            + f"\n\n[Source text truncated at {max_source_chars:,} characters. "
            f"Full text available in the original dataset.]"
        )

    # Task instructions from batch config
    try:
        batch  = get_batch(row["batch_id"])
        config = json.loads(batch["config"]) if isinstance(batch["config"], str) else batch["config"]
        task_instructions = (
            config.get("task_instructions")
            or "Perform a faithful transformation of the source content."
        )
    except Exception:
        task_instructions = "Perform a faithful transformation of the source content."

    # Build judge findings — only non-zero severity, plain English
    judge_findings = []
    for cat_id, dim in result.get("dimension_scores", {}).items():
        d = dim if isinstance(dim, dict) else {
            "severity": dim.severity, "confidence": dim.confidence,
            "explanation": dim.explanation,
        }
        severity = d.get("severity", 0)
        if severity > 0:
            judge_findings.append({
                "category_id":   cat_id,
                "plain_label":   PLAIN_LABELS.get(cat_id, cat_id),
                "severity":      severity,
                "severity_label": SEVERITY_LABELS.get(severity, str(severity)),
                "confidence":    d.get("confidence", "Low"),
                "explanation":   d.get("explanation", ""),
            })
    judge_findings.sort(key=lambda x: -x["severity"])

    response_label = "Response A" if blinded else result.get("model", "Response A")

    return {
        "source_title":           result.get("source_title", ""),
        "source_publisher":        publisher,
        "source_published_date":   pub_date,
        "source_text":             source_text,
        "benchmark_category":      bench_cat,
        "cultural_significance":   cult_sig,
        "transformation_task":     task_instructions,
        "response_label":          response_label,
        "response_text":           result.get("response_text", ""),
        "judge_findings":          judge_findings,
        "judge_summary":           result.get("executive_summary", ""),
        "judge_ii_score":          result.get("overall_information_integrity_score", 0),
        "judge_cf_score":          result.get("overall_cultural_fidelity_score", 0),
        "taxonomy_version":        _taxonomy_version(),
        "rubric_version":          "v1",
        "snapshot_created_at":     datetime.now(timezone.utc).isoformat(),
    }


def render_html_packet(review_id: str, snapshot: dict) -> str:
    e  = lambda s: html_lib.escape(str(s or ""), quote=True)
    js = lambda s: json.dumps(str(s or ""))  # safe JS string literal

    findings_html = ""
    if snapshot.get("judge_findings"):
        for f in snapshot["judge_findings"]:
            sev    = f["severity"]
            color  = {1: "#d97706", 2: "#ea580c", 3: "#dc2626"}.get(sev, "#6b7280")
            bg     = {1: "#fffbeb", 2: "#fff7ed", 3: "#fef2f2"}.get(sev, "#f9fafb")
            border = {1: "#fcd34d", 2: "#fdba74", 3: "#fca5a5"}.get(sev, "#e5e7eb")
            findings_html += f"""
        <div style="margin-bottom:14px;padding:14px 16px;background:{bg};border:1px solid {border};border-left:4px solid {color};border-radius:0 6px 6px 0;">
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;flex-wrap:wrap;">
            <span style="background:{color};color:white;padding:2px 9px;border-radius:4px;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.04em;">{e(f["severity_label"])}</span>
            <span style="font-weight:600;color:#1f2937;font-size:14px;font-family:system-ui,sans-serif;">{e(f["plain_label"])}</span>
            <span style="margin-left:auto;font-size:11px;color:#6b7280;font-family:system-ui,sans-serif;">Judge confidence: <strong>{e(f["confidence"])}</strong></span>
          </div>
          <p style="margin:0;font-size:13px;color:#374151;font-style:italic;line-height:1.6;">&ldquo;{e(f["explanation"])}&rdquo;</p>
        </div>"""
    else:
        findings_html = '<p style="color:#059669;font-size:13px;font-family:system-ui,sans-serif;">&#10003; No issues identified by the judge.</p>'

    ii = snapshot.get("judge_ii_score", 0)
    cf = snapshot.get("judge_cf_score", 0)
    ii_color = "#059669" if ii >= 8 else ("#d97706" if ii >= 6 else "#dc2626")
    cf_color = "#059669" if cf >= 8 else ("#d97706" if cf >= 6 else "#dc2626")
    snap_date = (snapshot.get("snapshot_created_at") or "")[:10]
    filename  = f"{review_id}_response.csv"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>EvalLab Review &mdash; {e(review_id)}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:Georgia,"Times New Roman",serif;max-width:820px;margin:0 auto;padding:40px 28px;color:#1f2937;background:#fff;line-height:1.7;font-size:15px}}
.sys{{font-family:system-ui,-apple-system,sans-serif}}
h2{{font-family:system-ui,sans-serif;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:#9ca3af;margin-bottom:12px}}
.card{{margin-bottom:24px;padding:22px 24px;border:1px solid #e5e7eb;border-radius:10px}}
.card-violet{{background:#f5f3ff;border-color:#ddd6fe}}
.card-yellow{{background:#fffbeb;border-color:#fde68a}}
.card-form{{background:#f8fafc;border-color:#cbd5e1;border-width:2px}}
.review-id{{font-family:monospace;font-size:22px;font-weight:700;color:#4f46e5;letter-spacing:.02em}}
.source-box{{background:#f9fafb;max-height:380px;overflow-y:auto;padding:16px;border:1px solid #e5e7eb;border-radius:6px;font-size:13px;color:#374151;white-space:pre-wrap;font-family:Georgia,serif;line-height:1.75}}
.response-box{{background:#f0fdf4;padding:16px;border-radius:6px;border:1px solid #bbf7d0;font-size:13px;color:#374151;white-space:pre-wrap;font-family:Georgia,serif;line-height:1.75}}
.scores{{display:flex;gap:14px;margin-bottom:18px}}
.score-badge{{padding:10px 18px;border-radius:8px;background:#f9fafb;border:1px solid #e5e7eb;text-align:center;min-width:90px}}
.score-label{{font-family:system-ui,sans-serif;font-size:10px;text-transform:uppercase;letter-spacing:.06em;color:#9ca3af;display:block;margin-bottom:4px}}
.score-value{{font-family:system-ui,sans-serif;font-size:24px;font-weight:700;display:block;line-height:1}}
.score-denom{{font-family:system-ui,sans-serif;font-size:10px;color:#9ca3af}}
.fq{{margin-bottom:22px}}
.flabel{{display:block;font-family:system-ui,sans-serif;font-size:13px;font-weight:600;color:#1f2937;margin-bottom:10px}}
.req{{color:#dc2626}}
.radio-opt,.check-opt{{display:flex;align-items:flex-start;gap:10px;margin-bottom:8px;padding:10px 12px;border-radius:7px;border:1.5px solid #e5e7eb;background:white;cursor:pointer;transition:border-color .15s,background .15s;font-family:system-ui,sans-serif;font-size:13px;color:#374151}}
.radio-opt:hover,.check-opt:hover{{border-color:#a5b4fc;background:#f5f3ff}}
.radio-opt input,.check-opt input{{margin-top:2px;accent-color:#4f46e5;width:15px;height:15px;flex-shrink:0;cursor:pointer}}
.radio-opt.selected{{border-color:#4f46e5;background:#eef2ff}}
input[type=text]{{width:100%;font-family:system-ui,sans-serif;font-size:13px;border:1.5px solid #d1d5db;border-radius:7px;padding:9px 12px;color:#1f2937;background:white;outline:none;transition:border-color .15s}}
input[type=text]:focus{{border-color:#4f46e5;box-shadow:0 0 0 3px rgba(99,102,241,.1)}}
textarea{{width:100%;font-family:Georgia,serif;font-size:13px;line-height:1.7;border:1.5px solid #d1d5db;border-radius:7px;padding:10px 12px;color:#1f2937;background:white;outline:none;resize:vertical;min-height:100px;transition:border-color .15s}}
textarea:focus{{border-color:#4f46e5;box-shadow:0 0 0 3px rgba(99,102,241,.1)}}
.dl-btn{{display:block;width:100%;padding:14px;background:#4f46e5;color:white;font-family:system-ui,sans-serif;font-size:15px;font-weight:700;text-align:center;border:none;border-radius:9px;cursor:pointer;letter-spacing:.01em;transition:background .15s,transform .1s;margin-top:8px}}
.dl-btn:hover{{background:#4338ca;transform:translateY(-1px)}}
.dl-btn:active{{transform:translateY(0)}}
.dl-btn:disabled{{background:#c7d2fe;cursor:not-allowed;transform:none}}
.success-banner{{display:none;background:#f0fdf4;border:1.5px solid #86efac;border-radius:9px;padding:14px 18px;font-family:system-ui,sans-serif;font-size:13px;color:#166534;margin-top:12px;text-align:center}}
.err{{color:#dc2626;font-family:system-ui,sans-serif;font-size:12px;margin-top:6px;display:none}}
.divider{{border:none;border-top:1px solid #e5e7eb;margin:28px 0}}
#disagreement-section{{display:none}}
@media print{{body{{padding:20px}}.source-box{{max-height:none;overflow:visible}}.dl-btn,.success-banner{{display:none}}}}
</style>
</head>
<body>

<!-- Header -->
<div class="card card-violet">
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;flex-wrap:wrap;gap:8px;">
    <span class="sys" style="font-size:12px;font-weight:600;color:#4f46e5;text-transform:uppercase;letter-spacing:.07em;">EvalLab &middot; Human Evaluation Review</span>
    <span class="sys" style="font-size:11px;color:#9ca3af;background:#ede9fe;padding:3px 9px;border-radius:5px;font-weight:500;">CONFIDENTIAL</span>
  </div>
  <div class="review-id">{e(review_id)}</div>
  <p class="sys" style="font-size:12px;color:#6b7280;margin-top:8px;line-height:1.5;">
    Generated {e(snap_date)} &nbsp;&middot;&nbsp; Taxonomy {e(snapshot.get("taxonomy_version","v1"))} &nbsp;&middot;&nbsp; Rubric {e(snapshot.get("rubric_version","v1"))}
  </p>
</div>

<!-- Instructions -->
<div class="card card-yellow">
  <h2>Instructions</h2>
  <p class="sys" style="font-size:13px;color:#374151;line-height:1.65;">
    You are reviewing a single AI-generated text transformation.
    Your task is to assess whether the AI judge&rsquo;s evaluation is accurate &mdash; not to rank models or compare outputs.
    Read each section below carefully, then complete the form at the bottom and click <strong>Download My Response</strong>.
  </p>
</div>

<!-- ① Source -->
<div class="card">
  <h2>&#9312; Source Document</h2>
  <strong style="font-size:16px;color:#111827;display:block;margin-bottom:6px;">{e(snapshot.get("source_title",""))}</strong>
  <span class="sys" style="font-size:12px;color:#6b7280;display:block;margin-bottom:14px;">
    {e(snapshot.get("source_publisher",""))}{("&nbsp;&middot;&nbsp;" + e(snapshot.get("source_published_date",""))) if snapshot.get("source_published_date") else ""}
  </span>
  <div class="source-box">{e(snapshot.get("source_text",""))}</div>
</div>

<!-- ② Task -->
<div class="card">
  <h2>&#9313; Transformation Task</h2>
  <p style="font-size:14px;color:#374151;font-style:italic;margin-bottom:10px;">The AI was asked to:</p>
  <blockquote style="margin:0;padding:12px 16px;border-left:4px solid #6366f1;background:#f5f3ff;border-radius:0 7px 7px 0;font-size:13px;color:#374151;line-height:1.65;">
    {e(snapshot.get("transformation_task",""))}
  </blockquote>
</div>

<!-- ③ Response -->
<div class="card">
  <h2>&#9314; {e(snapshot.get("response_label","Response A"))}</h2>
  <p class="sys" style="font-size:11px;color:#9ca3af;margin-bottom:12px;">AI-generated output</p>
  <div class="response-box">{e(snapshot.get("response_text",""))}</div>
</div>

<!-- ④ Judge evaluation -->
<div class="card">
  <h2>&#9315; AI Judge Evaluation</h2>
  <div class="scores">
    <div class="score-badge">
      <span class="score-label">Info Integrity</span>
      <span class="score-value" style="color:{ii_color};">{ii}</span>
      <span class="score-denom">/10</span>
    </div>
    <div class="score-badge">
      <span class="score-label">Cultural Fidelity</span>
      <span class="score-value" style="color:{cf_color};">{cf}</span>
      <span class="score-denom">/10</span>
    </div>
  </div>
  <p style="font-size:13px;color:#374151;font-style:italic;margin-bottom:20px;line-height:1.65;">{e(snapshot.get("judge_summary",""))}</p>
  <h2 style="margin-bottom:14px;">Issues Identified</h2>
  {findings_html}
</div>

<!-- ⑤ Interactive form -->
<div class="card card-form">
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:22px;flex-wrap:wrap;gap:8px;">
    <h2 style="margin-bottom:0;">&#9316; Your Review</h2>
    <span class="review-id" style="font-size:16px;">{e(review_id)}</span>
  </div>

  <!-- Reviewer ID -->
  <div class="fq">
    <label class="flabel" for="reviewer-id">Your name or reviewer ID <span class="req">*</span></label>
    <input type="text" id="reviewer-id" placeholder="e.g. Sarah M. or reviewer_02" autocomplete="name" />
    <p class="err" id="err-reviewer">Please enter your name or reviewer ID.</p>
  </div>

  <!-- Agreement -->
  <div class="fq">
    <label class="flabel">Do you agree with the AI judge&rsquo;s evaluation? <span class="req">*</span></label>
    <label class="radio-opt" id="opt-yes">
      <input type="radio" name="agreement" value="yes" onchange="onAgreementChange(this)" />
      <div><strong>Yes</strong> &mdash; I agree with the issues flagged, their severity levels, and the explanations</div>
    </label>
    <label class="radio-opt" id="opt-partially">
      <input type="radio" name="agreement" value="partially" onchange="onAgreementChange(this)" />
      <div><strong>Partially</strong> &mdash; I agree in part but have concerns about the severity, category, or explanation</div>
    </label>
    <label class="radio-opt" id="opt-no">
      <input type="radio" name="agreement" value="no" onchange="onAgreementChange(this)" />
      <div><strong>No</strong> &mdash; I disagree with the overall evaluation</div>
    </label>
    <label class="radio-opt" id="opt-unable">
      <input type="radio" name="agreement" value="unable_to_determine" onchange="onAgreementChange(this)" />
      <div><strong>Unable to determine</strong> &mdash; I cannot assess this evaluation with confidence</div>
    </label>
    <p class="err" id="err-agreement">Please select an agreement level.</p>
  </div>

  <!-- Disagreement types (shown only for partially / no) -->
  <div id="disagreement-section" class="fq">
    <label class="flabel">What type of disagreement? <span class="sys" style="font-weight:400;color:#9ca3af;">(select all that apply)</span></label>
    <label class="check-opt"><input type="checkbox" name="dtype" value="failure_not_present" /> Failure not present in the response</label>
    <label class="check-opt"><input type="checkbox" name="dtype" value="failure_category_incorrect" /> Failure category is incorrect</label>
    <label class="check-opt"><input type="checkbox" name="dtype" value="severity_too_high" /> Severity rated too high</label>
    <label class="check-opt"><input type="checkbox" name="dtype" value="severity_too_low" /> Severity rated too low</label>
    <label class="check-opt"><input type="checkbox" name="dtype" value="important_failure_missing" /> An important failure was missed</label>
    <label class="check-opt"><input type="checkbox" name="dtype" value="judge_explanation_inaccurate" /> Judge explanation is inaccurate</label>
    <label class="check-opt"><input type="checkbox" name="dtype" value="source_context_misunderstood" /> Source context was misunderstood by the judge</label>
    <label class="check-opt"><input type="checkbox" name="dtype" value="other" /> Other</label>
  </div>

  <!-- Comments -->
  <div class="fq">
    <label class="flabel" for="comments">Comments <span class="sys" style="font-weight:400;color:#9ca3af;">(optional)</span></label>
    <textarea id="comments" placeholder="Any additional observations, context, or explanations&hellip;"></textarea>
  </div>

  <!-- Download button -->
  <button class="dl-btn" id="dl-btn" onclick="downloadResponse()">
    &#8681;&nbsp; Download My Response
  </button>
  <div class="success-banner" id="success-banner">
    &#10003;&nbsp; Response saved as <strong id="success-filename"></strong> &mdash; email this file to your research contact.
  </div>
</div>

<hr class="divider">
<p class="sys" style="text-align:center;font-size:11px;color:#d1d5db;line-height:1.8;">
  EvalLab Human Review &nbsp;&middot;&nbsp; {e(review_id)} &nbsp;&middot;&nbsp;
  Generated {e(snap_date)} &nbsp;&middot;&nbsp; Taxonomy {e(snapshot.get("taxonomy_version","v1"))}
</p>

<script>
(function() {{
  var REVIEW_ID = {js(review_id)};
  var FILENAME  = {js(filename)};

  function onAgreementChange(radio) {{
    document.querySelectorAll('.radio-opt').forEach(function(el) {{
      el.classList.remove('selected');
    }});
    if (radio.checked) {{
      radio.closest('.radio-opt').classList.add('selected');
    }}
    var needsDetail = (radio.value === 'partially' || radio.value === 'no');
    document.getElementById('disagreement-section').style.display = needsDetail ? 'block' : 'none';
    document.getElementById('err-agreement').style.display = 'none';
  }}

  function csvCell(val) {{
    var s = String(val == null ? '' : val);
    if (s.indexOf(',') >= 0 || s.indexOf('"') >= 0 || s.indexOf('\\n') >= 0) {{
      return '"' + s.replace(/"/g, '""') + '"';
    }}
    return s;
  }}

  function downloadResponse() {{
    var reviewerId = document.getElementById('reviewer-id').value.trim();
    var agreementEl = document.querySelector('input[name="agreement"]:checked');
    var valid = true;

    if (!reviewerId) {{
      document.getElementById('err-reviewer').style.display = 'block';
      document.getElementById('reviewer-id').focus();
      valid = false;
    }} else {{
      document.getElementById('err-reviewer').style.display = 'none';
    }}

    if (!agreementEl) {{
      document.getElementById('err-agreement').style.display = 'block';
      valid = false;
    }} else {{
      document.getElementById('err-agreement').style.display = 'none';
    }}

    if (!valid) return;

    var agreement = agreementEl.value;
    var dtypes = [];
    if (agreement === 'partially' || agreement === 'no') {{
      document.querySelectorAll('input[name="dtype"]:checked').forEach(function(cb) {{
        dtypes.push(cb.value);
      }});
    }}
    var comments   = document.getElementById('comments').value.trim();
    var reviewedAt = new Date().toISOString();

    var headers = ['review_id','reviewer_id','agreement_with_judge','disagreement_types','comments','reviewed_at'];
    var row     = [REVIEW_ID, reviewerId, agreement, dtypes.join(';'), comments, reviewedAt];
    var csv     = headers.map(csvCell).join(',') + '\\n' + row.map(csvCell).join(',') + '\\n';

    var blob = new Blob([csv], {{type: 'text/csv;charset=utf-8;'}});
    var url  = URL.createObjectURL(blob);
    var a    = document.createElement('a');
    a.href     = url;
    a.download = FILENAME;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    document.getElementById('success-filename').textContent = FILENAME;
    document.getElementById('success-banner').style.display = 'block';
    document.getElementById('dl-btn').textContent = '\\u2713 Downloaded — click again to re-download';
  }}

  window.onAgreementChange   = onAgreementChange;
  window.downloadResponse    = downloadResponse;
}})();
</script>
</body>
</html>"""


def export_html(review_id: str) -> str:
    """Render and return the HTML packet for a review. Marks the review as exported."""
    review = get_human_review(review_id)
    if not review:
        raise ValueError(f"Review '{review_id}' not found.")
    snapshot = review["packet_snapshot"]
    snapshot["review_id"] = review_id
    html = render_html_packet(review_id, snapshot)
    mark_review_exported(review_id)
    return html


def export_json_packets(review_ids: list) -> list:
    """Return list of packet dicts (no subject_model) for the given review_ids."""
    packets = []
    for rid in review_ids:
        review = get_human_review(rid)
        if not review:
            continue
        snapshot = dict(review["packet_snapshot"])
        snapshot["review_id"] = rid
        packets.append({
            "review_id":        rid,
            "experiment_id":    review["experiment_id"],
            "batch_id":         review["batch_id"],
            "dataset_item_id":  review["dataset_item_id"],
            "prompt_variant":   review["prompt_variant"],
            "review_round":     review["review_round"],
            "review_status":    review["review_status"],
            "created_at":       review["created_at"],
            "selection_reasons": review["selection_reasons"],
            "packet":           snapshot,
        })
        mark_review_exported(rid)
    return packets


def export_csv_template(review_ids: list) -> str:
    """Return CSV string: one empty row per review_id for reviewers to fill in."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "review_id", "reviewer_id", "agreement_with_judge",
        "disagreement_types", "comments", "reviewed_at",
    ])
    for rid in review_ids:
        writer.writerow([rid, "", "", "", "", ""])
    return output.getvalue()


def export_csv_context(review_ids: list) -> str:
    """Return CSV string: reference sheet with source title, key findings, scores (not for import)."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "review_id", "source_title", "publisher",
        "response_label", "judge_ii", "judge_cf",
        "key_findings", "judge_summary_preview",
    ])
    for rid in review_ids:
        review = get_human_review(rid)
        if not review:
            continue
        snap = review["packet_snapshot"]
        findings_summary = "; ".join(
            f"{f['severity_label']}: {f['plain_label']}"
            for f in snap.get("judge_findings", [])
        ) or "No issues identified"
        summary_preview = (snap.get("judge_summary") or "")[:120]
        if len(snap.get("judge_summary") or "") > 120:
            summary_preview += "…"
        writer.writerow([
            rid,
            snap.get("source_title", ""),
            snap.get("source_publisher", ""),
            snap.get("response_label", "Response A"),
            snap.get("judge_ii_score", ""),
            snap.get("judge_cf_score", ""),
            findings_summary,
            summary_preview,
        ])
    return output.getvalue()
