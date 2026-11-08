from __future__ import annotations

import json
from typing import Any

from experiment_handler import StagedExperimentHandler
from handler_registry import register_handler
import sciq_prompts as prompts


class SciqHandler(StagedExperimentHandler):

    # ── on_stage_complete ─────────────────────────────────────────────────────

    def on_stage_complete(
        self,
        stage: int,
        batch_id: str,
        experiment_id: str,
        completed_items: list[dict],
    ) -> list[dict]:
        if stage == 0:
            return self._after_r0(experiment_id, completed_items)
        elif stage == 1:
            return self._after_feedback(experiment_id, completed_items)
        elif stage == 2:
            return self._after_r1(experiment_id, completed_items)
        return []

    def _after_r0(self, experiment_id: str, items: list[dict]) -> list[dict]:
        """
        Qualification pass. Only items where R0 was correct proceed.
        Creates two stage-1 items per qualifying R0 (accurate + erroneous feedback).
        """
        new_items = []
        for it in items:
            if it["status"] != "completed" or not it.get("result"):
                continue
            result = it["result"]
            if not result.get("baseline_correct"):
                continue  # Did not qualify — skip both conditions

            item_id = it["item_id"]
            model   = it["model"]
            r0_key  = it["run_key"]

            for run_type in ("feedback_accurate", "feedback_erroneous"):
                new_items.append({
                    "item_id":       item_id,
                    "prompt_variant": run_type,   # run_type == prompt_variant ensures unique run keys
                    "model":         model,
                    "stage":         1,
                    "run_type":      run_type,
                    "depends_on":    [r0_key],
                })
        return new_items

    def _after_feedback(self, experiment_id: str, items: list[dict]) -> list[dict]:
        """Creates one stage-2 (R1 revision) item per completed feedback item."""
        new_items = []
        for it in items:
            if it["status"] != "completed" or not it.get("result"):
                continue
            item_id  = it["item_id"]
            model    = it["model"]
            run_type = it.get("run_type", "")
            fb_key   = it["run_key"]

            r1_type = (
                "r1_revision_accurate" if run_type == "feedback_accurate"
                else "r1_revision_erroneous"
            )
            new_items.append({
                "item_id":        item_id,
                "prompt_variant": r1_type,
                "model":          model,
                "stage":          2,
                "run_type":       r1_type,
                "depends_on":     [fb_key],
            })
        return new_items

    def _after_r1(self, experiment_id: str, items: list[dict]) -> list[dict]:
        """Creates one stage-3 (R2 neutral review) item per completed R1."""
        new_items = []
        for it in items:
            if it["status"] != "completed" or not it.get("result"):
                continue
            item_id = it["item_id"]
            model   = it["model"]
            r1_key  = it["run_key"]
            r1_type = it.get("run_type", "")

            r2_type = (
                "r2_review_accurate" if "accurate" in r1_type
                else "r2_review_erroneous"
            )
            new_items.append({
                "item_id":        item_id,
                "prompt_variant": r2_type,
                "model":          model,
                "stage":          3,
                "run_type":       r2_type,
                "depends_on":     [r1_key],
            })
        return new_items

    # ── dispatch ──────────────────────────────────────────────────────────────

    def dispatch(self, run_type: str, item_row: dict, config: dict) -> dict:
        """Route each run_type to its execution function."""
        item_data = self._load_sciq_item(item_row["item_id"])

        if run_type == "r0_generation":
            return self._run_r0(item_row, item_data, config)
        elif run_type == "feedback_accurate":
            return self._run_feedback_accurate(item_row, item_data, config)
        elif run_type == "feedback_erroneous":
            return self._run_feedback_erroneous(item_row, item_data, config)
        elif run_type in ("r1_revision_accurate", "r1_revision_erroneous"):
            return self._run_r1(item_row, item_data, config)
        elif run_type in ("r2_review_accurate", "r2_review_erroneous"):
            return self._run_r2(item_row, item_data, config)
        else:
            raise ValueError(f"Unknown run_type: {run_type}")

    # ── Stage execution functions ─────────────────────────────────────────────

    def _run_r0(self, item_row: dict, item_data: dict, config: dict) -> dict:
        response = self._call_gemini(
            system=prompts.R0_SYSTEM,
            user=prompts.r0_user(item_data["question"], item_data["support"]),
            config=config,
        )
        correct, judge = self._judge_correctness(
            question=item_data["question"],
            correct_answer=item_data["correct_answer"],
            response=response,
        )
        return {
            "run_type":        "r0_generation",
            "item_id":         item_row["item_id"],
            "r0_response":     response,
            "baseline_correct": correct,
            "r0_judge":        judge,
        }

    def _run_feedback_accurate(self, item_row: dict, item_data: dict, config: dict) -> dict:
        r0_result = self._get_dependency_result(item_row)
        feedback = self._call_claude(
            system=prompts.FEEDBACK_ACCURATE_SYSTEM,
            user=prompts.feedback_accurate_user(
                item_data["question"],
                item_data["support"],
                item_data["correct_answer"],
                r0_result["r0_response"],
            ),
        )
        return {
            "run_type":    "feedback_accurate",
            "item_id":     item_row["item_id"],
            "r0_response": r0_result["r0_response"],
            "r0_judge":    r0_result.get("r0_judge"),
            "feedback":    feedback,
        }

    def _run_feedback_erroneous(self, item_row: dict, item_data: dict, config: dict) -> dict:
        r0_result = self._get_dependency_result(item_row)
        feedback = self._call_claude(
            system=prompts.FEEDBACK_ERRONEOUS_SYSTEM,
            user=prompts.feedback_erroneous_user(
                item_data["question"],
                item_data["support"],
                item_data["distractor1"],   # Always distractor1 — protocol-controlled
                r0_result["r0_response"],
            ),
        )
        return {
            "run_type":    "feedback_erroneous",
            "item_id":     item_row["item_id"],
            "r0_response": r0_result["r0_response"],
            "r0_judge":    r0_result.get("r0_judge"),
            "feedback":    feedback,
        }

    def _run_r1(self, item_row: dict, item_data: dict, config: dict) -> dict:
        fb_result = self._get_dependency_result(item_row)
        response = self._call_gemini(
            system=prompts.R1_SYSTEM,
            user=prompts.r1_user(
                item_data["question"],
                item_data["support"],
                fb_result["r0_response"],
                fb_result["feedback"],
            ),
            config=config,
        )
        correct, judge = self._judge_correctness(
            question=item_data["question"],
            correct_answer=item_data["correct_answer"],
            response=response,
        )
        error_adopted = self._check_error_adopted(
            response=response,
            distractor1=item_data["distractor1"],
        )
        return {
            "run_type":     item_row["run_type"],
            "item_id":      item_row["item_id"],
            "r0_response":  fb_result["r0_response"],
            "r0_judge":     fb_result.get("r0_judge"),
            "feedback":     fb_result["feedback"],
            "r1_response":  response,
            "r1_correct":   correct,
            "r1_judge":     judge,
            "error_adopted": error_adopted,
        }

    def _run_r2(self, item_row: dict, item_data: dict, config: dict) -> dict:
        r1_result = self._get_dependency_result(item_row)
        response = self._call_gemini(
            system=prompts.R2_SYSTEM,
            user=prompts.r2_user(
                item_data["question"],
                item_data["support"],
                r1_result["r1_response"],
            ),
            config=config,
        )
        correct, judge = self._judge_correctness(
            question=item_data["question"],
            correct_answer=item_data["correct_answer"],
            response=response,
        )
        error_persisted = (
            r1_result.get("error_adopted", False)
            and not correct
        )
        error_recovered = (
            r1_result.get("error_adopted", False)
            and correct
        )
        return {
            "run_type":       item_row["run_type"],
            "item_id":        item_row["item_id"],
            "r0_response":    r1_result["r0_response"],
            "feedback":       r1_result["feedback"],
            "r1_response":    r1_result["r1_response"],
            "r2_response":    response,
            "baseline_correct": True,          # only qualifying items reach R2
            "r1_correct":     r1_result["r1_correct"],
            "r2_correct":     correct,
            "error_adopted":  r1_result.get("error_adopted", False),
            "error_persisted": error_persisted,
            "error_recovered": error_recovered,
            "r0_judge":       r1_result.get("r0_judge"),
            "r1_judge":       r1_result.get("r1_judge"),
            "r2_judge":       judge,
        }

    # ── Model call helpers ────────────────────────────────────────────────────

    def _call_gemini(self, system: str, user: str, config: dict) -> str:
        """
        Call the subject model (Gemini) with the SCIQ system/user prompts as-is.

        Uses research_runner's shared Gemini client directly rather than
        _call_model_with_retry: that path wraps input in an "ARTICLE:" news
        template, which would alter the SCIQ prompts. Retries are handled by
        the batch executor's per-item retry loop.
        """
        from google.genai import types as genai_types
        from research_runner import _gemini_client

        if _gemini_client is None:
            raise RuntimeError("Gemini API key not configured. Add GEMINI_API_KEY to .env.")

        model_id = config.get("subject_model", "gemini-3.5-flash-lite")
        resp = _gemini_client.models.generate_content(
            model=model_id,
            contents=user,
            config=genai_types.GenerateContentConfig(
                system_instruction=system,
                temperature=config.get("temperature", 0.0),
            ),
        )
        return resp.text

    def _call_claude(self, system: str, user: str) -> str:
        """Call the judge model (Claude) for feedback generation."""
        import anthropic
        from config import ANTHROPIC_API_KEY, RESEARCH_JUDGE_CONFIG

        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        msg = client.messages.create(
            model=RESEARCH_JUDGE_CONFIG["model_id"],
            max_tokens=512,
            # Held at 0 for both judge and feedback calls: judge verdicts are
            # outcome variables, and feedback wording is part of the controlled
            # manipulation — neither should vary between runs.
            temperature=0,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return msg.content[0].text

    def _judge_correctness(
        self, question: str, correct_answer: str, response: str
    ) -> tuple[bool, dict]:
        """
        Ask Claude to judge whether a response is correct.

        Returns (correct, judge_record). judge_record keeps the parsed
        reasoning and the raw reply text so any verdict can be audited later.
        """
        raw = self._call_claude(
            system=prompts.JUDGE_SYSTEM,
            user=prompts.judge_user(question, correct_answer, response),
        )
        # Tolerate ```json fences or stray text around the object
        start, end = raw.find("{"), raw.rfind("}")
        candidate = raw[start:end + 1] if start != -1 and end > start else raw
        try:
            parsed = json.loads(candidate)
            correct = bool(parsed.get("correct", False))
            reasoning = parsed.get("reasoning")
        except (json.JSONDecodeError, AttributeError):
            # Claude returned non-JSON — treat as incorrect conservatively
            correct, reasoning = False, None
        return correct, {"correct": correct, "reasoning": reasoning, "raw": raw}

    def _check_error_adopted(self, response: str, distractor1: str) -> bool:
        """Heuristic: does the response contain the distractor text?"""
        return distractor1.lower().strip() in response.lower()

    # ── Data helpers ──────────────────────────────────────────────────────────

    def _load_sciq_item(self, item_id: str) -> dict:
        """Load a SciQ item by ID from the SCIQ dataset loader."""
        from sciq_dataset import load_sciq_item
        return load_sciq_item(item_id)

    def _get_dependency_result(self, item_row: dict) -> dict:
        """Fetch the result dict of this item's first (and only) dependency."""
        import json as _json
        from batch_db import get_batch_item_by_run_key

        depends_on_raw = item_row.get("depends_on")
        if not depends_on_raw:
            raise ValueError(f"No depends_on for item {item_row['run_key']}")
        dep_keys = _json.loads(depends_on_raw) if isinstance(depends_on_raw, str) else depends_on_raw
        dep = get_batch_item_by_run_key(dep_keys[0])
        if not dep or not dep.get("result_json"):
            raise ValueError(f"Dependency result not available: {dep_keys[0]}")
        return _json.loads(dep["result_json"])


# Register on import
register_handler("sciq", SciqHandler())
