from openai import OpenAI
from config import OPENAI_API_KEY, SUBJECT_MODELS
from dataclasses import dataclass



from concurrent.futures import ThreadPoolExecutor, as_completed

import tasks.resume as resume
import tasks.tax as tax
import tasks.career as career
import tasks.budgeting as budgeting

TASK_MODULES = {
    "resume": resume,
    "tax": tax,
    "career": career,
    "budgeting": budgeting,
}

client = OpenAI(api_key=OPENAI_API_KEY)


@dataclass
class ModelResponse:
    model: str
    task: str
    variant: str
    user_prompt: str
    response_text: str
    temperature: float


def _call_model(model: str, prompt: dict, task: str, user_prompt: str, temperature: float) -> ModelResponse:
    api_response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt["system"]},
            {"role": "user",   "content": prompt["user"]},
        ],
        temperature=temperature,
    )
    return ModelResponse(
        model=model,
        task=task,
        variant=prompt["variant"],
        user_prompt=user_prompt,
        response_text=api_response.choices[0].message.content,
        temperature=temperature,
    )


def run_eval(task: str, user_prompt: str, temperature: float = 0.7) -> list[ModelResponse]:
    task_module = TASK_MODULES[task]
    prompt_variants = task_module.get_prompts(user_prompt)

    # Build the full list of (model, prompt) combinations to call.
    calls = [
        (model, prompt)
        for model in SUBJECT_MODELS
        for prompt in prompt_variants
    ]

    # Fire all calls in parallel using a thread pool.
    # as_completed() yields each future as it finishes, regardless of order.
    results = []
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(_call_model, model, prompt, task, user_prompt, temperature): (model, prompt)
            for model, prompt in calls
        }
        for future in as_completed(futures):
            results.append(future.result())

    return results
