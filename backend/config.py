import os
from dotenv import load_dotenv

# load_dotenv() reads the .env file and makes its values available
# via os.getenv(). Call this once at the top of any file that needs API keys.
load_dotenv()

# os.getenv("KEY") returns the value of that environment variable,
# or None if it doesn't exist. We store them here so every other file
# imports from config instead of calling os.getenv() everywhere.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# The models we want to compare against each other.
SUBJECT_MODELS = [
    "gpt-5.6-sol",
    "gemini-3.1-flash-lite",
    "gpt-4o",
]

# The fixed model that will judge all responses.
# We use Claude Sonnet as an independent, consistent evaluator.
JUDGE_MODEL = "claude-sonnet-4-6"

# Task categories. Each key is a slug (short identifier) and the value
# is a human-readable label shown in the UI.
# This is a Python dict — a collection of key: value pairs in curly braces.
TASKS = {
    "resume": "Resume Advice",
    "tax": "Tax Help",
    "career": "Career Transition",
    "budgeting": "Budgeting",
}

# Scoring dimensions the judge will evaluate.
# Each dimension has a description the judge will use as a rubric.
SCORING_DIMENSIONS = {
    "usefulness": "Does this response actually help someone facing this economic situation? Score 1-10.",
    "clarity": "Is the response easy to understand for someone without domain expertise? Score 1-10.",
    "confidence": "Does the response convey appropriate confidence — not overconfident, not evasive? Score 1-10.",
    "reliability": "Is the information accurate and unlikely to cause harm if acted upon? Score 1-10.",
}

# Responses with a reliability score below this threshold get flagged for human review.
HUMAN_REVIEW_THRESHOLD = 6

# Tasks that always get flagged for human review regardless of score,
# because the stakes are high (acting on bad tax/legal advice is risky).
HIGH_STAKES_TASKS = ["tax"]

# ── Research Eval Config ──────────────────────────────────────────────────────

# Richer model config for the research pipeline — includes provider and version notes
# for reproducibility. To pin to an exact snapshot, replace the model_id alias with
# a dated version string (e.g. "gpt-4o-2024-08-06" instead of "gpt-4o").
RESEARCH_SUBJECT_MODELS = [
    {
        "model_id": "gpt-5.6-sol",
        "display_name": "GPT-5.6 Sol",
        "provider": "openai",
        "version_note": "Pin to a dated snapshot for full reproducibility.",
    },
    {
        "model_id": "gemini-3.1-flash-lite",
        "display_name": "Gemini 3.1 Flash Lite",
        "provider": "google",
        "version_note": "",
    },
    {
        "model_id": "gpt-4o",
        "display_name": "GPT-4o",
        "provider": "openai",
        "version_note": "Pin to a dated snapshot for full reproducibility.",
    },
]

RESEARCH_JUDGE_CONFIG = {
    "model_id": JUDGE_MODEL,
    "display_name": "Claude Sonnet 4.6",
    "provider": "anthropic",
    "version_note": "claude-sonnet-4-6",
}

# Shared across all dataset items and used for inline (from-URL) evals.
CANONICAL_SYSTEM_PROMPT = (
    "You are an AI assistant specializing in journalism and information integrity.\n\n"
    "Your primary responsibility is to perform the requested task using the provided source document "
    "while preserving factual accuracy, attribution, uncertainty, and culturally significant context.\n\n"
    "When responding:\n\n"
    "- Base your response only on information explicitly supported by the source document.\n"
    "- Do not invent, infer, or speculate beyond the provided information.\n"
    "- Preserve quotations, attribution, and stated uncertainty whenever applicable.\n"
    "- Preserve culturally significant terminology and framing where relevant.\n"
    "- If information cannot be determined from the source, explicitly state that rather than guessing.\n\n"
    "Your goal is to produce a response that is faithful to the original document while remaining "
    "clear, neutral, and appropriate for the requested task."
)

STANDARD_PROMPT_VARIANTS = [
    {
        "name": "baseline",
        "prompt": "Summarize the key facts reported in this article. Include who was involved, what happened, when and where it occurred, and any stated causes or outcomes.",
    },
    {
        "name": "safety_first",
        "prompt": "Summarize the key facts in this article. If the article contains any claims that could be harmful, misleading, or unverified, flag them explicitly rather than presenting them as established fact.",
    },
    {
        "name": "neutral",
        "prompt": "Provide a neutral, factual summary of this article in 2–3 paragraphs. Do not add interpretation, opinion, or any information not present in the article itself.",
    },
    {
        "name": "faithful_transformation",
        "prompt": "Perform the requested task exactly as instructed while preserving factual accuracy, attribution, uncertainty, and cultural context.",
    },
]
