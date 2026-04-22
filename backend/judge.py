import json
import anthropic
from config import ANTHROPIC_API_KEY, JUDGE_MODEL, SCORING_DIMENSIONS, HUMAN_REVIEW_THRESHOLD, HIGH_STAKES_TASKS
from runner import ModelResponse
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

# Initialize the Anthropic client once at the module level.
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


@dataclass
class ScoredResponse:
    # We carry over everything from ModelResponse so the final output is self-contained.
    model: str
    task: str
    variant: str
    user_prompt: str
    response_text: str
    temperature: float
    # Scores from Claude — each is an integer 1-10
    usefulness: int
    clarity: int
    confidence: int
    reliability: int
    # A plain-English summary of why Claude gave these scores
    judge_reasoning: str
    # True if this response should be flagged for a human to review
    needs_human_review: bool


def build_judge_prompt(response: ModelResponse) -> str:
    """
    Builds the prompt we send to Claude for scoring.
    We construct this as an f-string — an f-string starts with f" and lets
    you embed any Python variable directly inside {} brackets.
    """

    # First we build the rubric text by looping over SCORING_DIMENSIONS.
    # .items() on a dict returns each key-value pair as a tuple (key, value).
    # We use an f-string inside the loop to format each line, then
    # "\n".join(...) stitches them together with a newline between each.
    rubric = "\n".join(
        f"- {dimension} (1-10): {description}"
        for dimension, description in SCORING_DIMENSIONS.items()
    )

    # The full prompt tells Claude exactly what to evaluate and what format to return.
    # We ask for JSON so we can reliably parse the scores in code.
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
    """
    Sends a single ModelResponse to Claude for scoring.
    Returns a ScoredResponse with all original fields plus scores.
    """

    judge_prompt = build_judge_prompt(response)

    # Call Claude. We use a low temperature here (0.2) because we want
    # consistent, deterministic scoring — not creative variation.
    api_response = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=512,
        temperature=0.2,
        messages=[
            {"role": "user", "content": judge_prompt}
        ],
    )

    raw_text = api_response.content[0].text

    # Strip markdown code fences if Claude wraps the JSON in ```json ... ```
    if raw_text.startswith("```"):
        raw_text = raw_text.split("```")[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
    raw_text = raw_text.strip()

    scores = json.loads(raw_text)

    # Determine whether this response needs human review.
    # Two conditions trigger a flag:
    #   1. reliability score is below the threshold set in config
    #   2. the task is in our high-stakes list (e.g. tax)
    needs_review = (
        scores["reliability"] < HUMAN_REVIEW_THRESHOLD
        or response.task in HIGH_STAKES_TASKS
    )

    # Build and return the ScoredResponse, combining the original
    # ModelResponse fields with the new scores from Claude.
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
    )


def score_all(responses: list[ModelResponse]) -> list[ScoredResponse]:
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(score_response, r): r for r in responses}
        return [future.result() for future in as_completed(futures)]
