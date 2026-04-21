from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from runner import run_eval
from judge import score_all
from output import save_to_csv

# FastAPI() creates the application instance.
# Think of it as the container that holds all your routes.
app = FastAPI(title="EvalLab API")

# CORS (Cross-Origin Resource Sharing) is a browser security rule that blocks
# requests from one origin (e.g. localhost:4200 — Angular) to another
# (e.g. localhost:8000 — FastAPI) unless the server explicitly allows it.
# We add this middleware to allow our Angular frontend to talk to this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# BaseModel from Pydantic defines the shape of data coming INTO an endpoint.
# When Angular sends a POST request, FastAPI automatically validates that the
# request body matches this shape — wrong types or missing fields return an error.
class EvalRequest(BaseModel):
    task: str                               # e.g. "resume"
    user_prompt: str                        # the question typed in the UI
    temperature: float = Field(
        default=0.7, ge=0.0, le=1.0        # ge = greater than or equal, le = less than or equal
    )                                       # so temperature must be between 0.0 and 1.0


# BaseModel also defines the shape of data going OUT of an endpoint.
class ScoreResult(BaseModel):
    model: str
    task: str
    variant: str
    user_prompt: str
    response_text: str
    temperature: float
    usefulness: int
    clarity: int
    confidence: int
    reliability: int
    judge_reasoning: str
    needs_human_review: bool


class EvalResponse(BaseModel):
    results: list[ScoreResult]
    csv_path: str


# @app.post("/eval") is a decorator that registers this function as the handler
# for POST requests to the /eval URL. When Angular calls POST /eval with a
# JSON body, FastAPI runs this function and returns its result as JSON.
@app.post("/eval", response_model=EvalResponse)
def run_evaluation(request: EvalRequest):
    """
    Main endpoint. Accepts a task, prompt, and temperature.
    Runs the prompt across all models, scores with Claude, saves CSV.
    Returns all scored results to the frontend.
    """

    # Validate that the task slug is one we support.
    # If not, raise an HTTPException — FastAPI turns this into a 400 error
    # with the detail message, which the frontend can display to the user.
    valid_tasks = ["resume", "tax", "career", "budgeting"]
    if request.task not in valid_tasks:
        raise HTTPException(status_code=400, detail=f"Invalid task. Choose from: {valid_tasks}")

    # Run the eval — sends prompts to OpenAI, gets responses back.
    responses = run_eval(
        task=request.task,
        user_prompt=request.user_prompt,
        temperature=request.temperature,
    )

    # Score all responses with Claude.
    scored = score_all(responses)

    # Save to CSV and get back the file path.
    csv_path = save_to_csv(scored)

    # Build the response. We convert each ScoredResponse dataclass to a dict
    # with vars() so Pydantic can validate and serialize it.
    return EvalResponse(
        results=[ScoreResult(**vars(r)) for r in scored],
        csv_path=csv_path,
    )


# A simple health check endpoint — useful for confirming the server is running.
# GET /health should return {"status": "ok"}.
@app.get("/health")
def health_check():
    return {"status": "ok"}
