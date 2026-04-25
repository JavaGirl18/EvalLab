def get_prompts(user_prompt: str) -> list[dict]:
    """
    Tax help prompts. This is a high-stakes domain — bad tax advice
    can cause real financial harm, so the role-prompted version
    explicitly emphasizes accuracy and recommending professional help.
    """

    zero_shot = {
        "variant": "zero_shot",
        "system": "You are a helpful assistant. Keep your response under 150 words.",
        "user": user_prompt,
    }

    role_prompted = {
        "variant": "role_prompted",
        "system": """You are a knowledgeable tax assistant helping working-class
and low-to-middle income individuals in the United States navigate their taxes.
Your users may be filing for the first time, working gig economy jobs, or dealing
with life changes like a new dependent or job loss. Prioritize accuracy above all else.
When something is complex or situation-specific, clearly say so and recommend
consulting a free tax service like VITA (Volunteer Income Tax Assistance). Keep your response under 150 words.""",
        "user": user_prompt,
    }

    return [zero_shot, role_prompted]
