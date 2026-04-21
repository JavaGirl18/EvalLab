def get_prompts(user_prompt: str) -> list[dict]:
    """
    Budgeting prompts. Focused on practical money management for people
    living paycheck-to-paycheck or building financial stability for the first time.
    """

    zero_shot = {
        "variant": "zero_shot",
        "system": "You are a helpful assistant.",
        "user": user_prompt,
    }

    role_prompted = {
        "variant": "role_prompted",
        "system": """You are a personal finance coach specializing in helping
individuals and families on tight budgets build financial stability.
Your users often live paycheck to paycheck, carry credit card debt,
or are saving for the first time. Give concrete, realistic advice grounded
in their actual constraints. Avoid advice that assumes disposable income
or financial safety nets. Focus on small, actionable steps.""",
        "user": user_prompt,
    }

    return [zero_shot, role_prompted]
