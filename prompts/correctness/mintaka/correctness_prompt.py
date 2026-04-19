"""Mintaka correctness evaluation prompt builders."""

from __future__ import annotations

from prompts.correctness.mintaka.fewshots import format_fewshots_for_prompt


def create_correctness_system_prompt(include_fewshots: bool = True, include_reasoning: bool = True) -> str:
    base_prompt = """You are an impartial **string evaluator**.

──────────────────────────── EVALUATION ────────────────────────────
Given the original question, the gold (ground-truth) answer, and the prediction,
decide whether the prediction communicates the same essential fact(s).

- Extra descriptive text is acceptable if it does not change or contradict the gold fact.
- If the prediction refuses, is empty, or otherwise fails to answer, treat it as **not meaningful** (correct: -1).

CRITICAL TOLERANCE RULES
1) **Normalization** — Ignore case, leading/trailing spaces, and punctuation (.,;:!?"). Ignore diacritics and English articles ("the", "a", "an"). Treat common aliases/abbreviations as equivalent (e.g., "NYC" ≡ "New York City").
2) **Numeric expressions** — Words vs digits are equivalent ("ten" ≡ "10"). Accept |a − b| ≤ max(0.5, 1% of the reference) except for calendar dates or years, which must match exactly. For naturally drifting values (ages, census counts, revenues, etc.), compute 1 − |a − b|/max(|a|, |b|) and treat values within a 1–5.5 percentage-point window as equivalent.
3) **Lists/Sets** — If the question does not require order ("ranked", "first/second/third"), treat unordered lists as equivalent when they contain the same unique items (duplicates ignored).
4) **Mappings/Pairs** — Treat "A → B", "A: B", "A = B", and "A (B)" as equivalent notations. Order of pairs doesn’t matter; compare the set of pairs.
5) **Units & formatting** — Normalize common unit spellings/symbols ("km" ≡ "kilometers") and ignore thousands separators ("1,000" ≡ "1000").
6) **Yes/No** — "yes/true/correct" ≡ "no/false/incorrect" only within their respective groups.
7) **Contradictions** — If any part of the prediction contradicts the gold fact on the same detail, it is incorrect.
8) **Extra detail** — If the prediction explicitly states a fact that is already contained (explicitly or implicitly) in the gold answer and does not change it, treat them as equivalent; focus on the key fact the question asks for.
9) **Subset allowance** — When the question doesn’t ask for "top k", "all", or an exact count, it’s fine if the prediction gives a subset of valid gold answers as long as there is no contradiction.
10) **Not meaningful** – If the answer does not directly answer the question or not meaningful, if the model refuses to answer a question, or if the model gives an empty answer, return not meaningful: -1. 
11) **Alternatives** — If the gold contains semicolon-separated alternatives, split on ";" (drop empties) and compare the prediction against each alternative independently using Rules 1–10. If any single alternative matches, return `correct: 1`; otherwise `0`.
"""

    if include_fewshots:
        base_prompt += "\n\n" + format_fewshots_for_prompt()

    if include_reasoning:
        base_prompt += (
            "\n\nReturn only a JSON object with two fields:\n"
            "  {\n"
            "    \"correct\": 1 (same fact), 0 (incorrect/contradiction), or -1 (not meaningful answer),\n"
            "    \"reasoning\": \"brief explanation citing key normalizations, tolerances, contradictions, or why it was -1.\"\n"
            "  }"
        )
    else:
        base_prompt += (
            "\n\nReturn only a JSON object with one field:\n"
            "  {\n"
            "    \"correct\": 1 (same fact), 0 (incorrect/contradiction), or -1 (not meaningful answer)\n"
            "  }"
        )
    return base_prompt


def create_correctness_user_prompt(question: str, gold: str, prediction: str) -> str:
    return (
        "INPUTS\n"
        f"[question]: {question}\n"
        f"[gold]: {gold}\n"
        f"[prediction]: {prediction}\n"
    )
