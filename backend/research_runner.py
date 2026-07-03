import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from openai import OpenAI
from config import OPENAI_API_KEY, RESEARCH_SUBJECT_MODELS
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

DATASET_PATH = Path(__file__).parent / "data" / "pilot_dataset.json"

client = OpenAI(api_key=OPENAI_API_KEY)


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


def load_dataset() -> list[DatasetItem]:
    with open(DATASET_PATH) as f:
        return [DatasetItem(**item) for item in json.load(f)]


def get_dataset_item(item_id: str) -> DatasetItem:
    for item in load_dataset():
        if item.id == item_id:
            return item
    raise ValueError(f"Dataset item '{item_id}' not found.")


def get_prompt_variant(item: DatasetItem, variant_name: Optional[str] = None) -> dict:
    if not item.prompt_variants:
        raise ValueError(f"Dataset item '{item.id}' has no prompt variants defined.")
    if variant_name is None:
        return item.prompt_variants[0]
    for v in item.prompt_variants:
        if v["name"] == variant_name:
            return v
    available = [v["name"] for v in item.prompt_variants]
    raise ValueError(
        f"Prompt variant '{variant_name}' not found in item '{item.id}'. "
        f"Available: {available}"
    )


def _call_model(
    model_cfg: dict, item: DatasetItem, variant: dict, temperature: float
) -> ResearchModelResponse:
    user_content = f"ARTICLE:\n{item.source_text}\n\n---\n\n{variant['prompt']}"

    api_response = client.chat.completions.create(
        model=model_cfg["model_id"],
        messages=[
            {"role": "system", "content": item.system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=temperature,
    )

    return ResearchModelResponse(
        model=model_cfg["model_id"],
        subject_model_config=model_cfg,
        item_id=item.id,
        article_type=item.article_type,
        source_title=item.source_title,
        system_prompt=item.system_prompt,
        eval_prompt=variant["prompt"],
        prompt_variant_name=variant["name"],
        response_text=api_response.choices[0].message.content,
        temperature=temperature,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def run_research_eval(
    item_id: str,
    temperature: float = 0.7,
    prompt_variant_name: Optional[str] = None,
) -> list[ResearchModelResponse]:
    item = get_dataset_item(item_id)
    variant = get_prompt_variant(item, prompt_variant_name)

    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(_call_model, model_cfg, item, variant, temperature): model_cfg["model_id"]
            for model_cfg in RESEARCH_SUBJECT_MODELS
        }
        return [future.result() for future in as_completed(futures)]
