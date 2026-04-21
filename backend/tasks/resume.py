# TODO (v2): accept an uploaded resume file (PDF/DOCX), extract text,
# and inject it into the prompt so the model can give personalized feedback.

def get_prompts(user_prompt: str) -> list[dict]:
    """
    Takes the user's raw question and returns two prompt variants.
    Each variant is a dict with:
      - "variant": a label so we know which strategy was used
      - "system": the system message (sets the model's role/context)
      - "user": the actual question being asked
    """

    # VARIANT 1: Zero-shot
    # No role, no context. Just the raw question.
    # The system message is minimal — we're not coaching the model at all.
    zero_shot = {
        "variant": "zero_shot",
        "system": "You are a helpful assistant.",
        "user": user_prompt,
    }

    # VARIANT 2: Role-prompted
    # We give the model a specific identity and audience before asking.
    # The triple-quoted string lets us write a long system prompt across multiple lines.
    role_prompted = {
        "variant": "role_prompted",
        "system": """You are an experienced career coach who specializes in helping
first-generation professionals and career changers from low-income backgrounds
build competitive resumes. You understand common gaps, non-traditional experience,
and how to frame transferable skills. Keep advice practical, specific, and actionable.""",
        "user": user_prompt,
    }

    # We return both variants in a list so the runner can loop over them.
    # A list in Python is written as [item1, item2, ...].
    return [zero_shot, role_prompted]
