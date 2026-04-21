from openai import OpenAI
from config import OPENAI_API_KEY, SUBJECT_MODELS
from dataclasses import dataclass

# We import each task module so we can call get_prompts() on any of them.
# Instead of importing them one by one, we store them in a dict keyed by
# their slug — the same slugs defined in config.TASKS.
import tasks.resume as resume
import tasks.tax as tax
import tasks.career as career
import tasks.budgeting as budgeting

# This dict maps each task slug to its module.
# It lets us look up the right module at runtime using a string key,
# e.g. TASK_MODULES["resume"] gives us the resume module.
TASK_MODULES = {
    "resume": resume,
    "tax": tax,
    "career": career,
    "budgeting": budgeting,
}

# Initialize the OpenAI client once using our API key from config.
# We create it here at the module level so it's reused across all calls
# rather than re-created on every request.
client = OpenAI(api_key=OPENAI_API_KEY)


# A dataclass is a clean way to group related values into a named object.
# Think of it like a labeled container. The @dataclass decorator (the line
# above the class) automatically generates boilerplate like __init__ for you.
# Each field is defined as: field_name: type
@dataclass
class ModelResponse:
    model: str          # e.g. "gpt-4o"
    task: str           # e.g. "resume"
    variant: str        # e.g. "zero_shot" or "role_prompted"
    user_prompt: str    # the original question from the user
    response_text: str  # the model's full response
    temperature: float  # the temperature used for this call


def run_eval(task: str, user_prompt: str, temperature: float = 0.7) -> list[ModelResponse]:
    """
    Runs a single user prompt across all subject models and both prompt variants.
    Returns a list of ModelResponse objects — one per (model, variant) combination.

    For example, with 3 models and 2 variants, this returns 6 ModelResponse objects.

    temperature: controls randomness of responses. 0.0 = deterministic and precise,
    1.0 = more varied and creative. Defaults to 0.7. Gets logged in every result
    so you can compare runs at different temperatures.
    """

    # Look up the task module using the slug (e.g. "resume" -> resume module).
    task_module = TASK_MODULES[task]

    # Call get_prompts() on the task module to get our two variants.
    # prompt_variants is now a list of 2 dicts.
    prompt_variants = task_module.get_prompts(user_prompt)

    # This is where we'll collect all the responses before returning them.
    results = []

    # A for loop repeats the indented block once for each item in a list.
    # Here we loop over every model we want to test.
    for model in SUBJECT_MODELS:

        # For each model, we also loop over every prompt variant.
        # This nested loop means: for every model, run every variant.
        for prompt in prompt_variants:

            # Call the OpenAI API.
            # chat.completions.create() sends a conversation to the model
            # and returns its response. We pass:
            #   - model: which model to use
            #   - messages: the conversation as a list of role/content pairs
            #   - temperature: passed in from the caller, not hardcoded
            api_response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": prompt["system"]},
                    {"role": "user",   "content": prompt["user"]},
                ],
                temperature=temperature,
            )

            # The response text is nested inside the API response object.
            # .choices[0] gets the first (and only) completion.
            # .message.content gets the actual text string.
            response_text = api_response.choices[0].message.content

            # Create a ModelResponse dataclass instance to store this result.
            result = ModelResponse(
                model=model,
                task=task,
                variant=prompt["variant"],
                user_prompt=user_prompt,
                response_text=response_text,
                temperature=temperature,
            )

            # .append() adds an item to the end of a list.
            results.append(result)

    return results
