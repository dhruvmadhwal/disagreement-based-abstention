"""Correctness judging utilities for Bamboogle baselines."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from evaluation.judge import build_vertex_client
from openai import OpenAI
from prompts.correctness.bamboogle.correctness_prompt import (
    create_correctness_system_prompt,
    create_correctness_user_prompt,
)

DEFAULT_JUDGE_MODEL = os.environ.get("CORRECTNESS_MODEL_NAME", "google/gemini-2.5-flash")
DEFAULT_JUDGE_TEMPERATURE = float(os.environ.get("CORRECTNESS_TEMPERATURE", "0.1"))


@dataclass
class JudgeResult:
    correct: int
    reasoning: str
    raw_response: str
    error: bool = False  # True if judge failed due to API error


class CorrectnessJudge:
    """Thin wrapper around the existing correctness prompt for inline scoring."""

    def __init__(
        self,
        client: Optional[OpenAI] = None,
        model: str = DEFAULT_JUDGE_MODEL,
        temperature: float = DEFAULT_JUDGE_TEMPERATURE,
    ):
        self.client = client or build_vertex_client()
        self.model = model
        self.temperature = temperature
        self.system_prompt = create_correctness_system_prompt()

    @staticmethod
    def _parse_response(text: str) -> Dict[str, Any]:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        # Fallback: try to pull a numeric correctness value from the text.
        lowered = cleaned.lower()
        if "correct" in lowered and ":" in cleaned:
            try:
                token = cleaned.split(":", 1)[1].strip().split()[0]
                return {"correct": int(token), "reasoning": cleaned}
            except Exception:
                return {"correct": 0, "reasoning": cleaned}
        return {"correct": 0, "reasoning": cleaned}

    def score(self, question: str, gold: str, prediction: str) -> JudgeResult:
        """Return (correct, reasoning, raw_response) for one prediction."""
        if not prediction:
            return JudgeResult(correct=-1, reasoning="Missing prediction text", raw_response="")

        body = create_correctness_user_prompt(question.strip(), gold.strip(), prediction.strip())
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": body},
                ],
            )
            raw = response.choices[0].message.content or ""
        except Exception as exc:  # pragma: no cover - network/Vertex errors
            return JudgeResult(correct=0, reasoning=f"API error: {exc}", raw_response=str(exc), error=True)

        parsed = self._parse_response(raw)
        correct_raw = parsed.get("correct", 0)
        try:
            correct_val = int(correct_raw)
        except (TypeError, ValueError):
            correct_val = 0
        if correct_val not in (-1, 0, 1):
            correct_val = 0
        reasoning = parsed.get("reasoning", raw)
        return JudgeResult(correct=correct_val, reasoning=str(reasoning), raw_response=raw.strip())


__all__ = ["CorrectnessJudge", "JudgeResult", "DEFAULT_JUDGE_MODEL"]
