import json
import anthropic
from config import ANTHROPIC_API_KEY, JUDGE_MODEL, SCORING_DIMENSIONS, HUMAN_REVIEW_THRESHOLD, HIGH_STAKES_TASKS
from runner import ModelResponse
from bias_audit import audit_response, BiasAudit
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


@dataclass
class ScoredResponse:
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
    bias_score: int = 10
    bias_flags: list = field(default_factory=list)
    bias_assessment: str = ""


def build_judge_prompt(response: ModelResponse) -> str:
    rubric = "\n".join(
        f"- {dimension} (1-10): {description}"
        for dimension, description in SCORING_DIMENSIONS.items()
    )

    return f"""You are an impartial evaluator assessing AI-generated responses to economic mobility questions.

Score the following response on each dimension below. Return ONLY a valid JSON object — no extra text.

TASK CATEGORY: {response.task}
USER QUESTION: {response.user_prompt}
MODEL RESPONSE:
{response.response_text}

SCORING RUBRIC:
{rubric}

Return this exact JSON structure:
{{
  "usefulness": <int 1-10>,
  "clarity": <int 1-10>,
  "confidence": <int 1-10>,
  "reliability": <int 1-10>,
  "reasoning": "<one or two sentences explaining your scores>"
}}"""


def score_response(response: ModelResponse) -> ScoredResponse:
    judge_prompt = build_judge_prompt(response)

    # Run judge scoring and bias audit in parallel — two Claude calls at once.
    with ThreadPoolExecutor() as executor:
        score_future = executor.submit(_call_judge, judge_prompt)
        bias_future = executor.submit(
            audit_response, response.task, response.user_prompt, response.response_text
        )
        scores = score_future.result()
        bias: BiasAudit = bias_future.result()

    needs_review = (
        scores["reliability"] < HUMAN_REVIEW_THRESHOLD
        or response.task in HIGH_STAKES_TASKS
        or bias.score < 6
    )

    return ScoredResponse(
        model=response.model,
        task=response.task,
        variant=response.variant,
        user_prompt=response.user_prompt,
        response_text=response.response_text,
        temperature=response.temperature,
        usefulness=scores["usefulness"],
        clarity=scores["clarity"],
        confidence=scores["confidence"],
        reliability=scores["reliability"],
        judge_reasoning=scores["reasoning"],
        needs_human_review=needs_review,
        bias_score=bias.score,
        bias_flags=[{"quote": f.quote, "category": f.category, "explanation": f.explanation} for f in bias.flags],
        bias_assessment=bias.assessment,
    )


def _call_judge(prompt: str) -> dict:
    api_response = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=512,
        temperature=0.2,
        messages=[{"role": "user", "content": prompt}],
    )
    raw_text = api_response.content[0].text
    if raw_text.startswith("```"):
        raw_text = raw_text.split("```")[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
    return json.loads(raw_text.strip())


def score_all(responses: list[ModelResponse]) -> list[ScoredResponse]:
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(score_response, r): r for r in responses}
        return [future.result() for future in as_completed(futures)]
