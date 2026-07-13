import dataclasses
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import openai
from openai import OpenAI
import anthropic as anthropic_sdk
from config import OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, RESEARCH_SUBJECT_MODELS, CANONICAL_SYSTEM_PROMPT
from dataclasses import dataclass, field as dc_field
from concurrent.futures import ThreadPoolExecutor, as_completed

DATA_DIR    = Path(__file__).parent / "data"
DATASET_PATH = DATA_DIR / "benchmark" / "benchmark_corpus.json"

openai_client = OpenAI(api_key=OPENAI_API_KEY)
anthropic_client = anthropic_sdk.Anthropic(api_key=ANTHROPIC_API_KEY)

# Gemini client — only initialised if the key is present
_gemini_client = None
if GEMINI_API_KEY:
    try:
        from google import genai as _genai
        _gemini_client = _genai.Client(api_key=GEMINI_API_KEY)
    except ImportError:
        pass


@dataclass
class DatasetItem:
    id: str
    article_type: str
    source_type: str
    high_context: bool
    expected_failure_categories: list[str]
    source_title: str
    source_url: str
    source_text: str
    system_prompt: str
    metadata: dict
    prompt_variants: list[dict]   # [{"name": str, "prompt": str}]
    benchmark_rationale: str
    human_notes: str
    human_override: bool
    characteristics: dict = dc_field(default_factory=dict)


@dataclass
class ResearchModelResponse:
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
    timestamp: str                # ISO 8601 UTC


_ITEM_FIELDS = {f.name for f in dataclasses.fields(DatasetItem)}


def _normalize_item(raw: dict) -> dict:
    """Fill in defaults for fields the benchmark corpus schema omits, drop unknown keys."""
    item = raw.copy()
    item.setdefault("article_type", "news")
    item.setdefault("high_context", False)
    item.setdefault("system_prompt", CANONICAL_SYSTEM_PROMPT)
    return {k: v for k, v in item.items() if k in _ITEM_FIELDS}


def load_dataset(path: Optional[Path] = None) -> list[DatasetItem]:
    corpus = path or DATASET_PATH
    with open(corpus) as f:
        return [DatasetItem(**_normalize_item(item)) for item in json.load(f)]


def get_dataset_item(item_id: str, path: Optional[Path] = None) -> DatasetItem:
    for item in load_dataset(path):
        if item.id == item_id:
            return item
    raise ValueError(f"Dataset item '{item_id}' not found.")


def get_prompt_variant(
    item: DatasetItem,
    variant_name: Optional[str] = None,
    task_instructions: Optional[str] = None,
) -> dict:
    if not item.prompt_variants:
        raise ValueError(f"Dataset item '{item.id}' has no prompt variants defined.")
    if variant_name is None:
        return item.prompt_variants[0]
    for v in item.prompt_variants:
        if v["name"] == variant_name:
            if variant_name == "faithful_transformation" and task_instructions:
                return {
                    "name": v["name"],
                    "prompt": (
                        f"Perform the following task exactly as instructed while preserving "
                        f"factual accuracy, attribution, uncertainty, and cultural context.\n\n"
                        f"Task: {task_instructions}"
                    ),
                }
            return v
    # Fall back to globally-defined standard variants before failing
    from config import STANDARD_PROMPT_VARIANTS
    for v in STANDARD_PROMPT_VARIANTS:
        if v["name"] == variant_name:
            return v
    available = [v["name"] for v in item.prompt_variants]
    raise ValueError(
        f"Prompt variant '{variant_name}' not found in item '{item.id}'. "
        f"Available: {available}"
    )


def _call_openai(model_cfg: dict, item: DatasetItem, variant: dict, temperature: float) -> str:
    user_content = f"ARTICLE:\n{item.source_text}\n\n---\n\n{variant['prompt']}"
    resp = openai_client.chat.completions.create(
        model=model_cfg["model_id"],
        messages=[
            {"role": "system", "content": item.system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=temperature,
    )
    return resp.choices[0].message.content


def _call_anthropic(model_cfg: dict, item: DatasetItem, variant: dict, temperature: float) -> str:
    user_content = f"ARTICLE:\n{item.source_text}\n\n---\n\n{variant['prompt']}"
    resp = anthropic_client.messages.create(
        model=model_cfg["model_id"],
        max_tokens=4096,
        system=item.system_prompt,
        messages=[{"role": "user", "content": user_content}],
        temperature=temperature,
    )
    return resp.content[0].text


def _call_gemini(model_cfg: dict, item: DatasetItem, variant: dict, temperature: float) -> str:
    if _gemini_client is None:
        raise RuntimeError("Gemini API key not configured. Add GEMINI_API_KEY to .env.")
    from google.genai import types as genai_types
    user_content = f"{item.system_prompt}\n\nARTICLE:\n{item.source_text}\n\n---\n\n{variant['prompt']}"
    resp = _gemini_client.models.generate_content(
        model=model_cfg["model_id"],
        contents=user_content,
        config=genai_types.GenerateContentConfig(temperature=temperature),
    )
    return resp.text


def _call_model(
    model_cfg: dict, item: DatasetItem, variant: dict, temperature: float
) -> ResearchModelResponse:
    provider = model_cfg.get("provider", "openai")
    if provider == "google":
        response_text = _call_gemini(model_cfg, item, variant, temperature)
    elif provider == "anthropic":
        response_text = _call_anthropic(model_cfg, item, variant, temperature)
    else:
        response_text = _call_openai(model_cfg, item, variant, temperature)

    return ResearchModelResponse(
        model=model_cfg["model_id"],
        subject_model_config=model_cfg,
        item_id=item.id,
        article_type=item.article_type,
        source_title=item.source_title,
        system_prompt=item.system_prompt,
        eval_prompt=variant["prompt"],
        prompt_variant_name=variant["name"],
        response_text=response_text,
        temperature=temperature,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def _call_model_with_retry(
    model_cfg: dict,
    item: DatasetItem,
    variant: dict,
    temperature: float,
    max_retries: int = 3,
) -> ResearchModelResponse:
    last_exc: Exception = RuntimeError("No attempts made")
    for attempt in range(max_retries):
        try:
            return _call_model(model_cfg, item, variant, temperature)
        except openai.RateLimitError as e:
            last_exc = e
            time.sleep((2 ** attempt) * 2)
        except (openai.APIError, openai.APIConnectionError, openai.APITimeoutError) as e:
            last_exc = e
            time.sleep(2 ** attempt)
        except Exception as e:
            # Covers Gemini errors and any other provider
            last_exc = e
            time.sleep(2 ** attempt)
    raise last_exc


def run_research_eval(
    item_id: Optional[str] = None,
    temperature: float = 0.7,
    prompt_variant_name: Optional[str] = None,
    item: Optional[DatasetItem] = None,
    task_instructions: Optional[str] = None,
    models: Optional[list[str]] = None,
) -> list[ResearchModelResponse]:
    if item is None:
        if item_id is None:
            raise ValueError("Either item_id or item must be provided.")
        item = get_dataset_item(item_id)
    variant = get_prompt_variant(item, prompt_variant_name, task_instructions)

    model_configs = RESEARCH_SUBJECT_MODELS
    if models:
        model_configs = [m for m in RESEARCH_SUBJECT_MODELS if m["model_id"] in models]

    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(_call_model, model_cfg, item, variant, temperature): model_cfg["model_id"]
            for model_cfg in model_configs
        }
        return [future.result() for future in as_completed(futures)]
