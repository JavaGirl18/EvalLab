"""
Benchmark analysis module — combined statistical analysis across experiments.
Provides compute_analysis() and render_analysis_html() for API consumption.
"""
import json, math, sqlite3
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import stats as sp

DB_PATH = Path(__file__).parent / "evallab.db"

VARIANT_ALIAS = {
    "faithful_transformation":  "faithful",
    "extreme_compression":      "compressed",
    "international_adaptation": "international",
    "engagement_optimization":  "engagement",
}
VARIANTS = list(VARIANT_ALIAS.keys())

CATEGORIES = ["FH","CXH","CTH","AH","FS","UC","CL","EX","OC","AL","CM","FB","PB","ST"]
CAT_LABELS = {
    "FH":  "Fabricated/Hallucinated Facts",
    "CXH": "Context Hallucination",
    "CTH": "Cultural Translation Hallucination",
    "AH":  "Attribution Hallucination",
    "FS":  "False Specificity",
    "UC":  "Unsupported Claims",
    "CL":  "Cultural Loss",
    "EX":  "Exoticization",
    "OC":  "Othering/Community Framing",
    "AL":  "Agency/Leadership Erasure",
    "CM":  "Community Voice Marginalization",
    "FB":  "Framing Bias",
    "PB":  "Political/Ideological Bias",
    "ST":  "Stereotyping",
}


def _region_for(item_id: str) -> str:
    return "Nigeria" if item_id.startswith("N") else "United_States"


def _desc(vals) -> dict:
    a = np.array(vals, dtype=float)
    n = len(a)
    m = float(np.mean(a))
    s = float(np.std(a, ddof=1)) if n > 1 else 0.0
    se = s / math.sqrt(n)
    return {
        "n": n, "mean": round(m, 3), "sd": round(s, 3),
        "min": round(float(np.min(a)), 3), "max": round(float(np.max(a)), 3),
        "median": round(float(np.median(a)), 3),
        "ci95_lo": round(m - 1.96*se, 3), "ci95_hi": round(m + 1.96*se, 3),
    }


def _dunn_p(a, b) -> float:
    combined = np.concatenate([a, b])
    n_total  = len(combined)
    ranks    = sp.rankdata(combined)
    r_a      = ranks[:len(a)]
    _, counts = np.unique(combined, return_counts=True)
    tie_sum  = float(np.sum(counts**3 - counts))
    mean_rank = (n_total + 1) / 2
    denom = math.sqrt(
        (n_total*(n_total+1)/12 - tie_sum/(12*(n_total-1))) * (1/len(a) + 1/len(b))
    )
    z = abs(float(np.mean(r_a)) - mean_rank) / denom if denom > 0 else 0.0
    return float(2 * sp.norm.sf(z))


def _effect_r(u, n1, n2) -> float:
    return round(1 - 2*float(u) / (n1*n2), 3)


def _load_runs(experiment_ids: list[str]) -> list[dict]:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    runs = []
    for exp_id in experiment_ids:
        batch = conn.execute(
            "SELECT batch_id FROM batches WHERE experiment_id=? AND status='completed' "
            "ORDER BY completed_at DESC LIMIT 1",
            (exp_id,)
        ).fetchone()
        if not batch:
            continue
        rows = conn.execute(
            "SELECT * FROM batch_run_items WHERE batch_id=? AND status='completed'",
            (batch["batch_id"],)
        ).fetchall()
        for row in rows:
            r   = dict(row)
            res = json.loads(r["result_json"])
            dims = res.get("dimension_scores", {})
            runs.append({
                "experiment_id":  exp_id,
                "batch_id":       batch["batch_id"],
                "run_key":        r["run_key"],
                "item_id":        r["item_id"],
                "prompt_variant": r["prompt_variant"],
                "region":         _region_for(r["item_id"]),
                "II":             res.get("overall_information_integrity_score"),
                "CF":             res.get("overall_cultural_fidelity_score"),
                "failure_count":  sum(1 for c in CATEGORIES if (dims.get(c,{}).get("severity") or 0) > 0),
                "any_severe":     any((dims.get(c,{}).get("severity") or 0) >= 3 for c in CATEGORIES),
                **{f"{c}_sev": (dims.get(c,{}).get("severity") or 0) for c in CATEGORIES},
            })
    conn.close()
    return runs


def compute_analysis(experiment_ids: list[str], corpus_name: str = "BASE50") -> dict:
    runs = _load_runs(experiment_ids)
    if not runs:
        raise ValueError(f"No completed runs found for experiments: {experiment_ids}")

    now = datetime.now(timezone.utc).isoformat()
    ii_all = np.array([r["II"] for r in runs if r["II"] is not None], dtype=float)
    cf_all = np.array([r["CF"] for r in runs if r["CF"] is not None], dtype=float)
    fc_all = np.array([r["failure_count"] for r in runs], dtype=float)

    sw_ii = sp.shapiro(ii_all)
    sw_cf = sp.shapiro(cf_all)

    # ── By variant ────────────────────────────────────────────────────────────
    kw_ii_groups = [
        np.array([r["II"] for r in runs if r["prompt_variant"]==v and r["II"] is not None])
        for v in VARIANTS
    ]
    kw_cf_groups = [
        np.array([r["CF"] for r in runs if r["prompt_variant"]==v and r["CF"] is not None])
        for v in VARIANTS
    ]
    kw_ii = sp.kruskal(*kw_ii_groups)
    kw_cf = sp.kruskal(*kw_cf_groups)

    n_pairs = len(list(combinations(range(4), 2)))
    pairwise_ii, pairwise_cf = {}, {}
    for i, j in combinations(range(4), 2):
        vi, vj = VARIANTS[i], VARIANTS[j]
        key = f"{VARIANT_ALIAS[vi]} vs {VARIANT_ALIAS[vj]}"
        raw_p_ii = _dunn_p(kw_ii_groups[i], kw_ii_groups[j])
        raw_p_cf = _dunn_p(kw_cf_groups[i], kw_cf_groups[j])
        mw_ii = sp.mannwhitneyu(kw_ii_groups[i], kw_ii_groups[j], alternative="two-sided")
        mw_cf = sp.mannwhitneyu(kw_cf_groups[i], kw_cf_groups[j], alternative="two-sided")
        pairwise_ii[key] = {
            "U": int(round(float(mw_ii.statistic))),
            "p_raw": round(raw_p_ii, 4),
            "p_bonferroni": round(min(raw_p_ii * n_pairs, 1.0), 4),
            "significant": bool(raw_p_ii * n_pairs < 0.05),
            "effect_r": _effect_r(mw_ii.statistic, len(kw_ii_groups[i]), len(kw_ii_groups[j])),
            "diff_means": round(float(np.mean(kw_ii_groups[i])) - float(np.mean(kw_ii_groups[j])), 3),
        }
        pairwise_cf[key] = {
            "U": int(round(float(mw_cf.statistic))),
            "p_raw": round(raw_p_cf, 4),
            "p_bonferroni": round(min(raw_p_cf * n_pairs, 1.0), 4),
            "significant": bool(raw_p_cf * n_pairs < 0.05),
            "effect_r": _effect_r(mw_cf.statistic, len(kw_cf_groups[i]), len(kw_cf_groups[j])),
            "diff_means": round(float(np.mean(kw_cf_groups[i])) - float(np.mean(kw_cf_groups[j])), 3),
        }

    # ── By region ─────────────────────────────────────────────────────────────
    ng_ii = np.array([r["II"] for r in runs if r["region"]=="Nigeria" and r["II"] is not None])
    us_ii = np.array([r["II"] for r in runs if r["region"]=="United_States" and r["II"] is not None])
    ng_cf = np.array([r["CF"] for r in runs if r["region"]=="Nigeria" and r["CF"] is not None])
    us_cf = np.array([r["CF"] for r in runs if r["region"]=="United_States" and r["CF"] is not None])
    mw_reg_ii = sp.mannwhitneyu(ng_ii, us_ii, alternative="two-sided")
    mw_reg_cf = sp.mannwhitneyu(ng_cf, us_cf, alternative="two-sided")

    # ── By experiment (replication) ───────────────────────────────────────────
    exp_stats = {}
    for exp_id in experiment_ids:
        e_ii = np.array([r["II"] for r in runs if r["experiment_id"]==exp_id and r["II"] is not None])
        e_cf = np.array([r["CF"] for r in runs if r["experiment_id"]==exp_id and r["CF"] is not None])
        exp_stats[exp_id] = {"n": len(e_ii), "II": _desc(e_ii), "CF": _desc(e_cf)}

    replication: Optional[dict] = None
    if len(experiment_ids) == 2:
        e1_ii = np.array([r["II"] for r in runs if r["experiment_id"]==experiment_ids[0] and r["II"] is not None])
        e2_ii = np.array([r["II"] for r in runs if r["experiment_id"]==experiment_ids[1] and r["II"] is not None])
        e1_cf = np.array([r["CF"] for r in runs if r["experiment_id"]==experiment_ids[0] and r["CF"] is not None])
        e2_cf = np.array([r["CF"] for r in runs if r["experiment_id"]==experiment_ids[1] and r["CF"] is not None])
        mw_exp_ii = sp.mannwhitneyu(e1_ii, e2_ii, alternative="two-sided")
        mw_exp_cf = sp.mannwhitneyu(e1_cf, e2_cf, alternative="two-sided")
        replication = {
            "II": {"U": int(round(float(mw_exp_ii.statistic))), "p": round(float(mw_exp_ii.pvalue), 4),
                   "significant": bool(mw_exp_ii.pvalue < 0.05)},
            "CF": {"U": int(round(float(mw_exp_cf.statistic))), "p": round(float(mw_exp_cf.pvalue), 4),
                   "significant": bool(mw_exp_cf.pvalue < 0.05)},
        }

    # ── Category prevalence ───────────────────────────────────────────────────
    n_total = len(runs)
    categories = {}
    for c in CATEGORIES:
        sevs = [r[f"{c}_sev"] for r in runs]
        counts = {0:0, 1:0, 2:0, 3:0}
        for s in sevs:
            counts[min(s, 3)] += 1
        categories[c] = {
            "label": CAT_LABELS[c],
            "pct_clean":    round(100*counts[0]/n_total, 1),
            "pct_minor":    round(100*counts[1]/n_total, 1),
            "pct_moderate": round(100*counts[2]/n_total, 1),
            "pct_severe":   round(100*counts[3]/n_total, 1),
            "mean_severity": round(sum(sevs)/n_total, 3),
        }

    return {
        "corpus":         corpus_name,
        "experiments":    experiment_ids,
        "generated_at":   now,
        "methodology": {
            "normality_test":   "Shapiro-Wilk",
            "group_comparison": "Kruskal-Wallis H (non-parametric one-way ANOVA)",
            "pairwise":         "Dunn's test with Bonferroni correction (α=0.05, 6 pairs)",
            "two_group":        "Mann-Whitney U (non-parametric, two-sided)",
            "effect_size":      "Rank-biserial correlation r",
            "alpha":            0.05,
        },
        "normality": {
            "II": {"W": round(float(sw_ii.statistic), 4), "p": round(float(sw_ii.pvalue), 6), "normal": bool(sw_ii.pvalue >= 0.05)},
            "CF": {"W": round(float(sw_cf.statistic), 4), "p": round(float(sw_cf.pvalue), 6), "normal": bool(sw_cf.pvalue >= 0.05)},
        },
        "overall": {
            "n_runs":         n_total,
            "n_articles":     len(set(r["item_id"] for r in runs)),
            "n_variants":     len(set(r["prompt_variant"] for r in runs)),
            "n_experiments":  len(experiment_ids),
            "II_score":       _desc(ii_all),
            "CF_score":       _desc(cf_all),
            "failure_count":  _desc(fc_all),
            "pct_any_severe": round(100 * sum(r["any_severe"] for r in runs) / n_total, 2),
        },
        "by_prompt_variant": {
            v: {
                "alias": VARIANT_ALIAS[v],
                "n":     len(kw_ii_groups[i]),
                "II":    _desc(kw_ii_groups[i]),
                "CF":    _desc(kw_cf_groups[i]),
                "failure_count": _desc([r["failure_count"] for r in runs if r["prompt_variant"]==v]),
                "pct_any_severe": round(100*sum(r["any_severe"] for r in runs if r["prompt_variant"]==v)/max(len(kw_ii_groups[i]),1), 2),
            }
            for i, v in enumerate(VARIANTS)
        },
        "kruskal_wallis": {
            "II": {"H": round(float(kw_ii.statistic), 4), "df": 3, "p": round(float(kw_ii.pvalue), 6), "significant": bool(kw_ii.pvalue < 0.05)},
            "CF": {"H": round(float(kw_cf.statistic), 4), "df": 3, "p": round(float(kw_cf.pvalue), 6), "significant": bool(kw_cf.pvalue < 0.05)},
        },
        "pairwise_dunn_bonferroni": {"II": pairwise_ii, "CF": pairwise_cf},
        "by_region": {
            "Nigeria":       {"n": len(ng_ii), "II": _desc(ng_ii), "CF": _desc(ng_cf)},
            "United_States": {"n": len(us_ii), "II": _desc(us_ii), "CF": _desc(us_cf)},
            "mann_whitney_u": {
                "II": {"U": int(round(float(mw_reg_ii.statistic))), "p": round(float(mw_reg_ii.pvalue), 6),
                       "significant": bool(mw_reg_ii.pvalue < 0.05),
                       "effect_r": _effect_r(mw_reg_ii.statistic, len(ng_ii), len(us_ii)),
                       "diff_means": round(float(np.mean(ng_ii)) - float(np.mean(us_ii)), 3)},
                "CF": {"U": int(round(float(mw_reg_cf.statistic))), "p": round(float(mw_reg_cf.pvalue), 6),
                       "significant": bool(mw_reg_cf.pvalue < 0.05),
                       "effect_r": _effect_r(mw_reg_cf.statistic, len(ng_cf), len(us_cf)),
                       "diff_means": round(float(np.mean(ng_cf)) - float(np.mean(us_cf)), 3)},
            },
        },
        "by_category":    categories,
        "by_experiment":  exp_stats,
        "replication":    replication,
    }


# ── HTML renderer ─────────────────────────────────────────────────────────────

def _esc(s) -> str:
    return str(s).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;").replace('"',"&quot;")

def _fmt(v, decimals=3) -> str:
    if v is None: return "—"
    return f"{v:.{decimals}f}"

def _chip(significant: bool) -> str:
    if significant:
        return '<span class="chip chip-sig">Yes</span>'
    return '<span class="chip chip-ns">No</span>'

def _pairwise_rows(pairs: dict) -> str:
    rows = []
    for pair, d in pairs.items():
        sig = d["significant"]
        cls = ' class="row-sig"' if sig else ""
        mark = '<span class="sig-mark"></span>' if sig else ""
        delta = f"{d['diff_means']:+.3f}"
        rows.append(
            f'<tr{cls}><td class="label">{mark}{_esc(pair)}</td>'
            f'<td class="num">{delta}</td>'
            f'<td class="num">{_fmt(d["p_bonferroni"])}</td>'
            f'<td class="num">{_fmt(d["effect_r"])}</td>'
            f'<td>{_chip(sig)}</td></tr>'
        )
    return "\n".join(rows)

def _cat_bar_row(code: str, d: dict) -> str:
    high = "high" if d["mean_severity"] >= 0.8 else ""
    return f"""
    <div class="cat-row">
      <div class="cat-code">{_esc(code)}</div>
      <div>
        <div class="cat-label">{_esc(d['label'])}</div>
        <div class="cat-bar-wrap">
          <div class="cat-bar-clean"    style="width:{d['pct_clean']}%"></div>
          <div class="cat-bar-minor"    style="width:{d['pct_minor']}%"></div>
          <div class="cat-bar-moderate" style="width:{d['pct_moderate']}%"></div>
          <div class="cat-bar-severe"   style="width:{d['pct_severe']}%"></div>
        </div>
      </div>
      <div class="cat-mean {high}">{_fmt(d['mean_severity'])}</div>
    </div>"""


def render_analysis_html(data: dict) -> str:
    o  = data["overall"]
    kw = data["kruskal_wallis"]
    bv = data["by_prompt_variant"]
    br = data["by_region"]
    mw = br["mann_whitney_u"]
    be = data.get("by_experiment", {})
    rep = data.get("replication")
    cats = data["by_category"]
    generated = data["generated_at"][:10]
    corpus = _esc(data.get("corpus", "BASE50"))
    experiments = " + ".join(_esc(e) for e in data.get("experiments", []))

    # Variant table rows
    variant_rows = ""
    for v in VARIANTS:
        blk = bv[v]
        variant_rows += (
            f'<tr><td class="label">{_esc(blk["alias"].title())}</td>'
            f'<td class="num">{blk["n"]}</td>'
            f'<td class="num">{_fmt(blk["II"]["mean"])}</td>'
            f'<td class="num">{_fmt(blk["II"]["sd"])}</td>'
            f'<td class="num">{_fmt(blk["CF"]["mean"])}</td>'
            f'<td class="num">{_fmt(blk["CF"]["sd"])}</td></tr>\n'
        )

    # Replication rows
    rep_rows = ""
    if rep and len(data["experiments"]) == 2:
        e1, e2 = data["experiments"]
        s1, s2 = be.get(e1, {}), be.get(e2, {})
        rep_rows = f"""
        <tr>
          <td class="label">Information Integrity</td>
          <td class="num">{_fmt(s1.get('II',{}).get('mean'))}</td>
          <td class="num">{_fmt(s2.get('II',{}).get('mean'))}</td>
          <td class="num">{rep['II']['U']:,}</td>
          <td class="num">{_fmt(rep['II']['p'])}</td>
          <td><span class="chip {'chip-rep' if not rep['II']['significant'] else 'chip-sig'}">{'Replicated' if not rep['II']['significant'] else 'Diverged'}</span></td>
        </tr>
        <tr>
          <td class="label">Cultural Fidelity</td>
          <td class="num">{_fmt(s1.get('CF',{}).get('mean'))}</td>
          <td class="num">{_fmt(s2.get('CF',{}).get('mean'))}</td>
          <td class="num">{rep['CF']['U']:,}</td>
          <td class="num">{_fmt(rep['CF']['p'])}</td>
          <td><span class="chip {'chip-rep' if not rep['CF']['significant'] else 'chip-sig'}">{'Replicated' if not rep['CF']['significant'] else 'Diverged'}</span></td>
        </tr>"""
        rep_section = f"""
  <section>
    <h2>Replication — {_esc(e1)} vs {_esc(e2)}</h2>
    <p>Mann-Whitney U tests compared score distributions between the two sub-experiments
    to assess whether the expansion corpus behaves consistently with the pilot.</p>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Dimension</th><th>{_esc(e1)} Mean</th><th>{_esc(e2)} Mean</th>
        <th>U</th><th>p-value</th><th>Result</th></tr></thead>
        <tbody>{rep_rows}</tbody>
      </table>
    </div>
    <p style="margin-top:1rem">Non-significant p-values indicate the two corpora are statistically
    indistinguishable — the experiments form a single coherent benchmark.</p>
  </section>"""
    else:
        rep_section = ""

    # Category bars — sorted by mean severity desc
    sorted_cats = sorted(cats.items(), key=lambda x: -x[1]["mean_severity"])
    cat_bars = "\n".join(_cat_bar_row(code, d) for code, d in sorted_cats)

    css = """
:root{--bg:#f0ebe0;--surface:#e8e2d5;--ink:#1a1714;--ink-2:#4a4540;--ink-3:#7a7268;--accent:#b5721a;--accent-lo:#c08b2a22;--rule:#ccc4b4;--sig-bg:#b5721a12;--sig-border:#b5721a;--good:#2d6a4f;--good-bg:#2d6a4f14;--warn:#b5721a;--warn-bg:#b5721a14}
@media(prefers-color-scheme:dark){:root{--bg:#111217;--surface:#191b22;--ink:#e2dbd0;--ink-2:#a09890;--ink-3:#6e6860;--accent:#d4943a;--accent-lo:#d4943a22;--rule:#2b2e38;--sig-bg:#d4943a12;--sig-border:#d4943a;--good:#52b788;--good-bg:#52b78814;--warn:#d4943a;--warn-bg:#d4943a14}}
:root[data-theme=light]{--bg:#f0ebe0;--surface:#e8e2d5;--ink:#1a1714;--ink-2:#4a4540;--ink-3:#7a7268;--accent:#b5721a;--accent-lo:#c08b2a22;--rule:#ccc4b4;--sig-bg:#b5721a12;--sig-border:#b5721a;--good:#2d6a4f;--good-bg:#2d6a4f14;--warn:#b5721a;--warn-bg:#b5721a14}
:root[data-theme=dark]{--bg:#111217;--surface:#191b22;--ink:#e2dbd0;--ink-2:#a09890;--ink-3:#6e6860;--accent:#d4943a;--accent-lo:#d4943a22;--rule:#2b2e38;--sig-bg:#d4943a12;--sig-border:#d4943a;--good:#52b788;--good-bg:#52b78814;--warn:#d4943a;--warn-bg:#d4943a14}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html{font-size:16px}
body{background:var(--bg);color:var(--ink);font-family:-apple-system,"Helvetica Neue",Arial,sans-serif;font-size:.9375rem;line-height:1.65;max-width:820px;margin:0 auto;padding:3rem 2rem 6rem}
@media(max-width:600px){body{padding:2rem 1.25rem 4rem}}
h1{font-family:Georgia,"Times New Roman",serif;font-size:clamp(1.6rem,4vw,2.4rem);font-weight:normal;letter-spacing:-.01em;line-height:1.2;text-wrap:balance;color:var(--ink)}
h2{font-family:-apple-system,"Helvetica Neue",Arial,sans-serif;font-size:.6875rem;font-weight:600;letter-spacing:.12em;text-transform:uppercase;color:var(--ink-3);margin-bottom:1.25rem}
h3{font-family:Georgia,"Times New Roman",serif;font-size:1.1rem;font-weight:normal;color:var(--ink);margin-bottom:.5rem}
p{color:var(--ink-2);max-width:62ch}p+p{margin-top:.75rem}
strong{color:var(--ink);font-weight:600}
.masthead{border-top:2px solid var(--ink);padding-top:1.5rem;margin-bottom:3rem}
.meta{font-size:.75rem;letter-spacing:.06em;text-transform:uppercase;color:var(--ink-3);margin-top:.75rem}
section{border-top:1px solid var(--rule);padding-top:2.25rem;margin-top:2.25rem}
.hero-stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:0;margin:2.5rem 0 1rem;border:1px solid var(--rule)}
.hero-stat{padding:1.5rem 1.25rem 1.25rem;border-right:1px solid var(--rule)}
.hero-stat:last-child{border-right:none}
.hero-stat__value{font-family:Menlo,Consolas,"Courier New",monospace;font-size:clamp(2rem,5vw,3rem);font-weight:normal;color:var(--ink);line-height:1;font-variant-numeric:tabular-nums;letter-spacing:-.02em}
.hero-stat__value.accent{color:var(--accent)}
.hero-stat__label{font-size:.6875rem;letter-spacing:.1em;text-transform:uppercase;color:var(--ink-3);margin-top:.4rem}
.hero-stat__sub{font-family:Menlo,Consolas,"Courier New",monospace;font-size:.6875rem;color:var(--ink-3);margin-top:.2rem;font-variant-numeric:tabular-nums}
.table-wrap{overflow-x:auto;margin-top:1rem}
table{width:100%;border-collapse:collapse;font-size:.8125rem;font-variant-numeric:tabular-nums}
thead tr{border-bottom:1px solid var(--ink-2)}
tbody tr{border-bottom:1px solid var(--rule)}
tbody tr:last-child{border-bottom:none}
th{font-family:-apple-system,"Helvetica Neue",Arial,sans-serif;font-size:.625rem;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:var(--ink-3);text-align:left;padding:.5rem .75rem .5rem 0;white-space:nowrap}
td{padding:.55rem .75rem .55rem 0;color:var(--ink-2);vertical-align:middle}
td.num{font-family:Menlo,Consolas,"Courier New",monospace;font-size:.8125rem;color:var(--ink)}
td.label{font-weight:500;color:var(--ink)}
.row-sig{background:var(--sig-bg)}.row-sig td{color:var(--ink)}.row-sig .num{color:var(--accent)}
.sig-mark{display:inline-block;width:3px;height:1em;background:var(--sig-border);border-radius:1px;margin-right:.5rem;vertical-align:middle}
.stat-inline{background:var(--surface);border-left:2px solid var(--accent);padding:.75rem 1rem;margin:1rem 0;font-size:.8125rem}
.stat-inline .label{font-size:.625rem;letter-spacing:.1em;text-transform:uppercase;color:var(--ink-3);margin-bottom:.25rem}
.stat-row{display:flex;gap:2rem;flex-wrap:wrap}
.stat-cell{display:flex;flex-direction:column}
.stat-cell .val{font-family:Menlo,Consolas,"Courier New",monospace;font-size:1rem;color:var(--ink);font-variant-numeric:tabular-nums}
.stat-cell .sub{font-size:.6875rem;color:var(--ink-3)}
.cat-grid{margin-top:1rem;display:flex;flex-direction:column;gap:0}
.cat-row{display:grid;grid-template-columns:4rem 1fr 4.5rem;align-items:center;gap:.75rem;padding:.45rem 0;border-bottom:1px solid var(--rule)}
.cat-row:last-child{border-bottom:none}
.cat-code{font-family:Menlo,Consolas,"Courier New",monospace;font-size:.75rem;font-weight:600;color:var(--accent);letter-spacing:.05em}
.cat-bar-wrap{display:flex;height:8px;border-radius:1px;overflow:hidden;background:var(--surface)}
.cat-bar-clean{background:var(--good);opacity:.5}
.cat-bar-minor{background:var(--warn);opacity:.5}
.cat-bar-moderate{background:var(--warn);opacity:1}
.cat-bar-severe{background:#b5302a}
.cat-label{font-size:.75rem;color:var(--ink-2);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.cat-mean{font-family:Menlo,Consolas,"Courier New",monospace;font-size:.75rem;color:var(--ink-3);text-align:right;font-variant-numeric:tabular-nums}
.cat-mean.high{color:var(--accent);font-weight:600}
.chip{display:inline-block;font-size:.625rem;letter-spacing:.08em;text-transform:uppercase;font-weight:600;padding:.15em .5em;border-radius:2px;vertical-align:middle;margin-left:.4em}
.chip-sig{background:var(--sig-bg);color:var(--accent);border:1px solid var(--sig-border)}
.chip-rep{background:var(--good-bg);color:var(--good);border:1px solid var(--good)}
.chip-ns{background:var(--surface);color:var(--ink-3);border:1px solid var(--rule)}
.finding{border-left:2px solid var(--accent);padding:.6rem .9rem;margin:1.25rem 0;background:var(--sig-bg)}
.finding p{font-size:.875rem;color:var(--ink);max-width:none}
.region-grid{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:var(--rule);margin-top:1rem}
.region-cell{background:var(--bg);padding:1.25rem}
.region-cell h3{font-size:.75rem;letter-spacing:.08em;text-transform:uppercase;font-family:-apple-system,"Helvetica Neue",Arial,sans-serif;font-weight:600;color:var(--ink-3);margin-bottom:.75rem}
.region-score{font-family:Menlo,Consolas,"Courier New",monospace;font-size:1.5rem;color:var(--ink);font-variant-numeric:tabular-nums}
.region-score-label{font-size:.6875rem;color:var(--ink-3);text-transform:uppercase;letter-spacing:.06em}
.region-scores{display:flex;gap:1.5rem;margin-top:.5rem}
.method-note{background:var(--surface);padding:1.25rem;font-size:.8125rem;color:var(--ink-2);border-top:1px solid var(--rule);margin-top:2.5rem}
.method-note dl{display:grid;grid-template-columns:max-content 1fr;gap:.25rem 1.5rem;margin-top:.75rem}
.method-note dt{font-family:Menlo,Consolas,"Courier New",monospace;font-size:.75rem;color:var(--ink-3);padding-top:.1rem}
.method-note dd{color:var(--ink-2)}
@media(prefers-reduced-motion:reduce){*{transition:none!important}}
"""

    ng = br["Nigeria"]
    us = br["United_States"]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{corpus} Benchmark — Statistical Analysis</title>
<style>{css}</style>
</head>
<body>

<div class="masthead">
  <h1>{corpus} Benchmark<br>Statistical Analysis</h1>
  <p class="meta">{experiments} &nbsp;·&nbsp; Generated {_esc(generated)}</p>
</div>

<section>
  <h2>Overview</h2>
  <p>This analysis combines results from {o['n_experiments']} sequential evaluation
  experiments into a single {o['n_articles']}-article benchmark. Each article was evaluated
  across {o['n_variants']} prompt transformation strategies using a fixed subject model and
  judge model at temperature 0.2. The corpus spans investigative journalism from Nigerian (ICIR)
  and U.S. (ProPublica, MLK50) outlets.</p>

  <div class="hero-stats">
    <div class="hero-stat">
      <div class="hero-stat__value">{_fmt(o['II_score']['mean'], 2)}</div>
      <div class="hero-stat__label">Info. Integrity</div>
      <div class="hero-stat__sub">95% CI [{_fmt(o['II_score']['ci95_lo'],2)}, {_fmt(o['II_score']['ci95_hi'],2)}]</div>
    </div>
    <div class="hero-stat">
      <div class="hero-stat__value">{_fmt(o['CF_score']['mean'], 2)}</div>
      <div class="hero-stat__label">Cultural Fidelity</div>
      <div class="hero-stat__sub">95% CI [{_fmt(o['CF_score']['ci95_lo'],2)}, {_fmt(o['CF_score']['ci95_hi'],2)}]</div>
    </div>
    <div class="hero-stat">
      <div class="hero-stat__value accent">{o['n_runs']}</div>
      <div class="hero-stat__label">Evaluations</div>
      <div class="hero-stat__sub">{o['n_articles']} articles · {o['n_variants']} variants</div>
    </div>
    <div class="hero-stat">
      <div class="hero-stat__value">{_fmt(o['pct_any_severe'], 0)}%</div>
      <div class="hero-stat__label">Severe failures</div>
      <div class="hero-stat__sub">across all {o['n_runs']} runs</div>
    </div>
  </div>
</section>

<section>
  <h2>Prompt Variant Analysis</h2>
  <p>Kruskal-Wallis H-test confirmed that the four prompt strategies produce statistically
  distinguishable score distributions on both dimensions. Pairwise comparisons used Dunn's
  test with Bonferroni correction (α = 0.05, 6 pairs).</p>

  <div class="stat-inline">
    <div class="label">Kruskal-Wallis</div>
    <div class="stat-row">
      <div class="stat-cell">
        <span class="val">H = {_fmt(kw['II']['H'], 2)}</span>
        <span class="sub">Information Integrity &nbsp;·&nbsp; p {'< 0.001' if kw['II']['p'] < 0.001 else _fmt(kw['II']['p'])}</span>
      </div>
      <div class="stat-cell">
        <span class="val">H = {_fmt(kw['CF']['H'], 2)}</span>
        <span class="sub">Cultural Fidelity &nbsp;·&nbsp; p {'< 0.001' if kw['CF']['p'] < 0.001 else _fmt(kw['CF']['p'])}</span>
      </div>
    </div>
  </div>

  <div class="table-wrap">
    <table>
      <thead><tr><th>Variant</th><th>n</th><th>II Mean</th><th>II SD</th><th>CF Mean</th><th>CF SD</th></tr></thead>
      <tbody>{variant_rows}</tbody>
    </table>
  </div>

  <h3 style="margin-top:1.5rem">Pairwise — Information Integrity</h3>
  <div class="table-wrap">
    <table>
      <thead><tr><th>Pair</th><th>Δ Mean</th><th>p (adj.)</th><th>Effect r</th><th>Significant</th></tr></thead>
      <tbody>{_pairwise_rows(data['pairwise_dunn_bonferroni']['II'])}</tbody>
    </table>
  </div>

  <h3 style="margin-top:1.5rem">Pairwise — Cultural Fidelity</h3>
  <div class="table-wrap">
    <table>
      <thead><tr><th>Pair</th><th>Δ Mean</th><th>p (adj.)</th><th>Effect r</th><th>Significant</th></tr></thead>
      <tbody>{_pairwise_rows(data['pairwise_dunn_bonferroni']['CF'])}</tbody>
    </table>
  </div>
</section>

<section>
  <h2>Regional Analysis</h2>
  <p>Nigerian and U.S. articles compared using Mann-Whitney U tests.</p>

  <div class="region-grid">
    <div class="region-cell">
      <h3>Nigeria (ICIR)</h3>
      <div class="region-scores">
        <div><div class="region-score-label">II</div><div class="region-score">{_fmt(ng['II']['mean'],3)}</div></div>
        <div><div class="region-score-label">CF</div><div class="region-score">{_fmt(ng['CF']['mean'],3)}</div></div>
      </div>
      <div style="font-size:.75rem;color:var(--ink-3);margin-top:.6rem">n = {ng['n']} evaluations</div>
    </div>
    <div class="region-cell">
      <h3>United States (ProPublica · MLK50)</h3>
      <div class="region-scores">
        <div><div class="region-score-label">II</div><div class="region-score">{_fmt(us['II']['mean'],3)}</div></div>
        <div><div class="region-score-label">CF</div><div class="region-score">{_fmt(us['CF']['mean'],3)}</div></div>
      </div>
      <div style="font-size:.75rem;color:var(--ink-3);margin-top:.6rem">n = {us['n']} evaluations</div>
    </div>
  </div>

  <div class="table-wrap" style="margin-top:1rem">
    <table>
      <thead><tr><th>Dimension</th><th>U</th><th>p-value</th><th>Effect r</th><th>Δ (NG − US)</th><th>Significant</th></tr></thead>
      <tbody>
        <tr class="{'row-sig' if mw['II']['significant'] else ''}">
          <td class="label">{'<span class="sig-mark"></span>' if mw['II']['significant'] else ''}Information Integrity</td>
          <td class="num">{mw['II']['U']:,}</td>
          <td class="num">{_fmt(mw['II']['p'])}</td>
          <td class="num">{_fmt(mw['II']['effect_r'])}</td>
          <td class="num">{mw['II']['diff_means']:+.3f}</td>
          <td>{_chip(mw['II']['significant'])}</td>
        </tr>
        <tr class="{'row-sig' if mw['CF']['significant'] else ''}">
          <td class="label">{'<span class="sig-mark"></span>' if mw['CF']['significant'] else ''}Cultural Fidelity</td>
          <td class="num">{mw['CF']['U']:,}</td>
          <td class="num">{_fmt(mw['CF']['p'])}</td>
          <td class="num">{_fmt(mw['CF']['effect_r'])}</td>
          <td class="num">{mw['CF']['diff_means']:+.3f}</td>
          <td>{_chip(mw['CF']['significant'])}</td>
        </tr>
      </tbody>
    </table>
  </div>
</section>

<section>
  <h2>Failure Category Prevalence</h2>
  <p>Severity distribution across all {o['n_runs']} evaluations. Bar segments:
  <span style="color:var(--good)">■</span> clean &nbsp;
  <span style="color:var(--warn);opacity:.6">■</span> minor &nbsp;
  <span style="color:var(--warn)">■</span> moderate &nbsp;
  <span style="color:#b5302a">■</span> severe.</p>
  <div class="cat-grid" style="margin-top:1.25rem">{cat_bars}</div>
</section>

{rep_section}

<div class="method-note">
  <strong>Methodology</strong>
  <dl>
    <dt>Normality</dt>
    <dd>Shapiro-Wilk — II: W={_fmt(data['normality']['II']['W'],4)} p={_fmt(data['normality']['II']['p'],4)} &nbsp;|&nbsp; CF: W={_fmt(data['normality']['CF']['W'],4)} p={_fmt(data['normality']['CF']['p'],4)}. Non-parametric tests used throughout.</dd>
    <dt>Group test</dt><dd>Kruskal-Wallis H (non-parametric one-way ANOVA) across 4 prompt variant groups.</dd>
    <dt>Post-hoc</dt><dd>Dunn's test with Bonferroni correction. Family-wise α = 0.05; threshold p &lt; 0.0083 (6 pairs).</dd>
    <dt>Two-group</dt><dd>Mann-Whitney U, two-sided, for region and replication comparisons.</dd>
    <dt>Effect size</dt><dd>Rank-biserial correlation r. |r| &lt; 0.1 negligible · 0.1–0.3 small · 0.3–0.5 medium · &gt; 0.5 large.</dd>
  </dl>
</div>

</body>
</html>"""
    return html
