def get_prompts(user_prompt: str) -> list[dict]:
    """
    Career transition prompts. Focused on helping people pivot industries
    or move from hourly/trade work into new roles.
    """

    zero_shot = {
        "variant": "zero_shot",
        "system": "You are a helpful assistant.",
        "user": user_prompt,
    }

    role_prompted = {
        "variant": "role_prompted",
        "system": """You are a career transition specialist who works with adults
making mid-career changes, often from blue-collar or service industry backgrounds
into tech, healthcare, or skilled trades. You focus on realistic pathways,
free or low-cost certifications, and how to leverage existing experience.
Avoid generic advice — be specific about industries, timelines, and costs.""",
        "user": user_prompt,
    }

    return [zero_shot, role_prompted]
