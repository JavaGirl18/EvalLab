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

# The models we want to compare against each other.
# This is a Python list — an ordered collection of items in square brackets.
SUBJECT_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-3.5-turbo",
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
