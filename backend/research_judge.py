import json
import time
import anthropic
from config import ANTHROPIC_API_KEY, JUDGE_MODEL, RESEARCH_JUDGE_CONFIG
from research_runner import DatasetItem, ResearchModelResponse
from taxonomy import load_taxonomy, get_categories, compute_aggregate_scores
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


@dataclass
class DimensionScore:
    severity: int       # 0 = Not Present, 1 = Minor, 2 = Moderate, 3 = Severe
    confidence: str     # Low | Medium | High
    explanation: str
    source_evidence: str
    output_evidence: str


@dataclass
class ResearchScoredResponse:
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
    dimension_scores: dict[str, DimensionScore]
    overall_information_integrity_score: float
    overall_cultural_fidelity_score: float
    executive_summary: str
    most_significant_failures: list[str] = field(default_factory=list)
    suggested_improvements: str = ""


def build_judge_prompt(item: DatasetItem, eval_prompt: str, response_text: str, taxonomy: dict) -> str:
    categories = get_categories(taxonomy)

    rubric_parts = []
    for cat in categories:
        lines = [
            f"### {cat['id']}: {cat['name']}",
            f"Meta-category: {cat['meta_category'].replace('_', ' ').title()}",
            f"Definition: {str(cat['definition']).strip()}",
        ]

        if cat.get("apply_when"):
            lines.append("Apply when:")
            lines.extend(f"  - {rule}" for rule in cat["apply_when"])

        if cat.get("do_not_apply_when"):
            lines.append("Do not apply when:")
            lines.extend(f"  - {rule}" for rule in cat["do_not_apply_when"])

        if cat.get("severity_guidance"):
            lines.append("Severity guidance:")
            lines.extend(f"  - {g}" for g in cat["severity_guidance"])

        if cat.get("example"):
            ex = cat["example"]
            lines.append(f'Example — Source: "{ex.get("source", "")}"')
            output_val = ex.get("output") or ex.get("output_a", "")
            lines.append(f'Example — Output: "{output_val}"')
            lines.append(f'Example — Reason: {str(ex.get("reason", "")).strip()}')

        rubric_parts.append("\n".join(lines))

    rubric = "\n\n".join(rubric_parts)

    # Build the expected JSON schema dynamically from taxonomy category IDs.
    # When categories are added or renamed in taxonomy.yaml, this schema updates automatically.
    schema_lines = [
        f'    "{cat["id"]}": {{'
        f'"severity": <int 0-3>, '
        f'"confidence": "<Low|Medium|High>", '
        f'"explanation": "<string>", '
        f'"source_evidence": "<exact quote or N/A>", '
        f'"output_evidence": "<exact quote or N/A>"}}'
        for cat in categories
    ]
    schema_body = ",\n".join(schema_lines)

    return f"""You are an expert evaluator specializing in information integrity and cultural fidelity assessment.

Evaluate how faithfully the model output preserves the information and cultural content of the source article.
Score each failure category independently. A single response may contain multiple failure types.

---
SOURCE ARTICLE TITLE: {item.source_title}

SOURCE ARTICLE:
{item.source_text}

---
EVALUATION PROMPT GIVEN TO THE MODEL:
{eval_prompt}

---
MODEL OUTPUT:
{response_text}

---
EVALUATION TAXONOMY

For each category below:
1. Determine whether the failure is present.
2. Assign a severity score: 0 = Not Present, 1 = Minor, 2 = Moderate, 3 = Severe
3. Provide a brief explanation.
4. Quote the relevant evidence from the source article and from the model output (or write N/A).
5. State your confidence: Low, Medium, or High.

{rubric}

---
Return ONLY a valid JSON object — no markdown fences, no extra text.

{{
  "dimension_scores": {{
{schema_body}
  }},
  "executive_summary": "<2-4 sentences summarizing the overall evaluation>",
  "most_significant_failures": ["<category_id>", ...],
  "suggested_improvements": "<actionable suggestions for improving the model response>"
}}"""


def _call_judge(prompt: str) -> dict:
    api_response = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=3000,
        temperature=0.1,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = api_response.content[0].text
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def _call_judge_with_retry(prompt: str, max_retries: int = 3) -> dict:
    last_exc: Exception = RuntimeError("No attempts made")
    for attempt in range(max_retries):
        try:
            return _call_judge(prompt)
        except anthropic.RateLimitError as e:
            last_exc = e
            time.sleep((2 ** attempt) * 2)
        except (anthropic.APIError, anthropic.APIConnectionError, anthropic.APITimeoutError) as e:
            last_exc = e
            time.sleep(2 ** attempt)
        except json.JSONDecodeError as e:
            last_exc = e
            time.sleep(1)
    raise last_exc


def score_research_response(item: DatasetItem, response: ResearchModelResponse) -> ResearchScoredResponse:
    taxonomy = load_taxonomy()
    prompt = build_judge_prompt(item, response.eval_prompt, response.response_text, taxonomy)
    raw = _call_judge_with_retry(prompt)

    # Parse dimension scores. Gracefully default any missing category to severity 0.
    categories = get_categories(taxonomy)
    dimension_scores: dict[str, DimensionScore] = {}
    for cat in categories:
        cid = cat["id"]
        entry = raw.get("dimension_scores", {}).get(cid, {})
        dimension_scores[cid] = DimensionScore(
            severity=int(entry.get("severity", 0)),
            confidence=entry.get("confidence", "Low"),
            explanation=entry.get("explanation", "Not evaluated."),
            source_evidence=entry.get("source_evidence", "N/A"),
            output_evidence=entry.get("output_evidence", "N/A"),
        )

    ii_score, cf_score = compute_aggregate_scores(
        {cid: {"severity": ds.severity} for cid, ds in dimension_scores.items()},
        taxonomy,
    )

    return ResearchScoredResponse(
        model=response.model,
        subject_model_config=response.subject_model_config,
        item_id=response.item_id,
        article_type=response.article_type,
        source_title=response.source_title,
        system_prompt=response.system_prompt,
        eval_prompt=response.eval_prompt,
        prompt_variant_name=response.prompt_variant_name,
        response_text=response.response_text,
        temperature=response.temperature,
        timestamp=response.timestamp,
        judge_subject_model_config=RESEARCH_JUDGE_CONFIG,
        dimension_scores=dimension_scores,
        overall_information_integrity_score=ii_score,
        overall_cultural_fidelity_score=cf_score,
        executive_summary=raw.get("executive_summary", ""),
        most_significant_failures=raw.get("most_significant_failures", []),
        suggested_improvements=raw.get("suggested_improvements", ""),
    )


def score_all_research(
    item: DatasetItem, responses: list[ResearchModelResponse]
) -> list[ResearchScoredResponse]:
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(score_research_response, item, r): r for r in responses}
        return [future.result() for future in as_completed(futures)]
