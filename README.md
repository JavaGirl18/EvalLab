# EvalLab

EvalLab is an AI evaluation tool built around a core question: **which AI models are most trustworthy for people navigating economic hardship?**

Millions of people turn to AI for help with financial decisions — filing taxes, writing resumes, switching careers, managing debt — often without access to professional advice. EvalLab was built to measure how well today's AI models actually serve those users. It compares multiple OpenAI models side by side on real economic mobility tasks, uses Claude as an independent judge to score each response, and audits every response for bias — flagging language that assumes resources, moralizes about poverty, or excludes users by using jargon they may not know.

The goal is to give researchers, advocates, and developers a clear, data-driven picture of where AI helps and where it falls short for vulnerable populations.

---

## What It Does

- Runs your prompt across multiple OpenAI models simultaneously (GPT-4o, GPT-4o-mini, GPT-3.5 Turbo)
- Tests both zero-shot and role-prompted variants of each prompt
- Scores every response using Claude as an independent judge across four dimensions: **usefulness**, **clarity**, **confidence**, and **reliability**
- Audits each response for bias — flagging assumptions, framing issues, demographic bias, and accessibility problems with exact quotes
- Lets you thumbs up preferred responses and tracks preferences over time in a chart
- Saves all results to a timestamped CSV for further analysis
- Flags high-stakes responses (tax advice, low reliability, high bias) for human review

---

## Try It Live

You can use EvalLab without any setup at:

**[https://eval-lab-six.vercel.app](https://eval-lab-six.vercel.app)**

No API keys or installation required.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Angular 21 + TypeScript + Tailwind CSS v4 |
| Backend | Python + FastAPI |
| Subject models | OpenAI API (GPT-4o, GPT-4o-mini, GPT-3.5 Turbo) |
| Judge & bias auditor | Anthropic API (Claude Sonnet) |
| Database | SQLite (preference tracking) |
| Deployment | Vercel (frontend) + Railway (backend) |

---

## Running Locally

### Prerequisites
- Python 3.11+
- Node.js 18+
- An OpenAI API key (platform.openai.com)
- An Anthropic API key (console.anthropic.com)

### 1. Clone the repo

```bash
git clone https://github.com/JavaGirl18/EvalLab.git
cd EvalLab
```

### 2. Set up environment variables

Create a `.env` file in the `backend/` folder:

```
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
```

### 3. Install backend dependencies

```bash
cd backend
pip3 install -r ../requirements.txt
```

### 4. Start the backend

```bash
python3 -m uvicorn api:app --reload
```

The API will be running at `http://localhost:8000`.

### 5. Install frontend dependencies

Open a second terminal:

```bash
cd frontend
npm install
```

### 6. Start the frontend

```bash
npm start
```

Open your browser at `http://localhost:4200`.

---

## How to Use

### Running an Evaluation

1. **Select a task category** from the dropdown — Resume Advice, Tax Help, Career Transition, or Budgeting
2. **Type your prompt** in the text area (e.g. *"How do I write a resume if I've only had retail jobs?"*)
3. **Adjust the temperature** slider — lower values (0.0–0.3) produce more precise responses, higher values (0.7–1.0) produce more varied ones
4. Click **Run Eval**

EvalLab will send your prompt to all three models across both zero-shot and role-prompted variants (6 total calls), then score every response with Claude. Results appear in 15–30 seconds.

### Reading Results

Results are displayed side by side per model, grouped by prompt strategy:

- **Zero-Shot** — raw response with no role or context given to the model
- **Role-Prompted** — response where the model was given a specific expert persona

Each card shows:
- The model's response (truncated, with a "Show more" toggle)
- Claude's scores for usefulness, clarity, confidence, and reliability (color-coded green/yellow/red)
- A **Bias Audit** section (click to expand) — shows a bias score and any flagged quotes with category labels (assumption, framing, demographic, accessibility)
- A **Judge's Verdict** — Claude's plain-English reasoning for the scores
- A ⚑ **Review** badge if the response was flagged for human review

### Preferences & Insights

- Click **"Do you prefer this response? 👍"** on any card to record a preference
- Open the **Insights** panel in the left sidebar to see:
  - A radar chart showing average scores per model across all dimensions
  - A bar chart showing total thumbs up counts per model
- Preferences are saved and persist across sessions

### Prompt History

Previous prompts appear in the left sidebar under **History**. Click any entry to reload that task, prompt, and temperature setting.

### Exporting Results

Every eval run automatically saves a timestamped CSV to `backend/results/`. Each row contains the model, task, variant, temperature, all scores, bias data, and the full response text.

---

## Project Structure

```
EvalLab/
├── backend/
│   ├── api.py          # FastAPI server
│   ├── runner.py       # OpenAI calls (parallel)
│   ├── judge.py        # Claude scoring
│   ├── bias_audit.py   # Claude bias auditor
│   ├── output.py       # CSV writer
│   ├── db.py           # SQLite preference tracking
│   ├── config.py       # Models, scoring rubric, thresholds
│   └── tasks/          # Prompt templates per domain
├── frontend/
│   └── src/app/
│       ├── run-panel/        # Prompt input + controls
│       ├── results-table/    # Side-by-side scored results
│       ├── insights-panel/   # Charts and preference data
│       └── services/         # HTTP client
└── requirements.txt
```

---

## Scoring Rubric

Claude scores each response 1–10 on:

| Dimension | What it measures |
|---|---|
| **Usefulness** | Does this actually help someone facing this economic situation? |
| **Clarity** | Is it easy to understand for someone without domain expertise? |
| **Confidence** | Is it appropriately confident — not overconfident, not evasive? |
| **Reliability** | Is the information accurate and unlikely to cause harm if acted upon? |

Responses are flagged for human review if reliability < 6, bias score < 6, or the task is tax-related.
