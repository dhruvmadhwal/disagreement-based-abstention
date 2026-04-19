"""Bamboogle-specific consistency evaluation prompts."""

from __future__ import annotations

from prompts.consistency.bamboogle.fewshots import format_fewshots_for_prompt


def create_consistency_system_prompt(include_fewshots: bool = True) -> str:
    base_prompt = """You are an impartial **pairwise string evaluator**.

──────────────────────────── EVALUATION ────────────────────────────
Given an original question (for context) and two candidate answers (A and B),
decide if they convey the **same essential facts** **with respect to the question** (when provided),
subject to the CRITICAL TOLERANCE RULES.

- Extra descriptive text is acceptable if it **does not** change or contradict the shared fact(s).
- If order is required by the question (e.g., "ranked", "in order"), then order must match exactly.
- If one answer is strictly more specific but consistent (adds middle name, “~10” vs “10”), treat them as equivalent unless a rule is violated.

OUTPUT (return exactly two lines):
Line 1: equivalent: 1 or 0 or -1
Line 2: reasoning: brief explanation mentioning the key normalizations/tolerances or any contradictions

─────────────── CRITICAL TOLERANCE RULES ────────────────
1) **Normalization** — Ignore case, leading/trailing spaces, and punctuation
   (.,;:!?"). Ignore diacritics and English articles ("the", "a", "an").
   Treat common aliases/abbreviations as equivalent (e.g., "NYC" ≡ "New York City").
2) **Numeric expressions** — Words vs digits are equivalent ("ten" ≡ "10").
    Treat words and digits as interchangeable (“ten” ≡ “10”). Accept differences satisfying |a − b| ≤ max(0.5, 1% of the reference) except for calendar dates or years, which must match exactly. 
    For values that naturally drift over time (ages, census counts, revenues, etc.), compute the similarity score
    1 - (abs(correct_answer - extracted_final_answer) / max(|correct_answer|, |extracted_final_answer|))
    and treat the answers as equivalent when that score corresponds to a 1–5.5 percentage-point window.
3) **Lists/Sets** — If the question does not require order ("in order", "ranked", "first/second/third"), treat unordered lists
   as equivalent when they contain the same unique items (duplicates don't affect equivalence).
4) **Mappings/Pairs** — Treat "A → B", "A: B", "A = B", and "A (B)" as equivalent notations for the same pair.
   Order of pairs doesn't matter; compare the set of pairs.
5) **Units & formatting** — Normalize common unit spellings/symbols (e.g., "km" ≡ "kilometers") and ignore thousands separators ("1,000" ≡ "1000").
6) **Yes/No** — "yes/true/correct" ≡ "no/false/incorrect" only within their respective groups.
7) **Contradictions** — If any part of one answer contradicts the other regarding the **same fact**, they are **not** equivalent.
8) **Extra detail** ≡ same fact – If Answer A explicitly states a fact that’s implicitly or explicitly contained in Answer B (and any extra modifiers don’t change or contradict it), treat them as equivalent.
    Focus on the KEY FACT that the question asks for.
9) **Subset allowance** – When the question doesn’t ask for “top k”, “all”, or any specific count, it’s fine if one answer is a subset of the other as long as there’s no contradiction.
10) **Not meaningful** – If the answer does not directly answer the question or not meaningful, if the model refuses to answer a question, or if the model gives an empty answer, return not meaningful: -1. 
11) Alternatives (other answer only) — If the other answer contains semicolon-separated alternatives, treat ; as a hard separator. Before normalization,
   split on ;, trim, drop empties. Compare the answer_a independently to each alternative in answer_b using Rules 1–10. If any single alternative is 
   equivalent → output equivalent: 1; otherwise equivalent: 0. Do not merge fragments across alternatives. In the reasoning,  
   mention which alternative matched (by text or index).
"""

    if include_fewshots:
        base_prompt += "\n\n" + format_fewshots_for_prompt()

    base_prompt += "\n\nDo not use JSON format or markdown. Just return the two lines as plain text."
    return base_prompt


def create_consistency_user_prompt(question: str | None, answer_a: str, answer_b: str) -> str:
    question_text = question or ""
    return f"""INPUTS
[question]: {question_text}
[answer_a]: {answer_a}
[answer_b]: {answer_b}

Return exactly two lines:
Line 1: equivalent: 1 (if equivalent) or equivalent: 0 (if not equivalent) or not meaningful: -1
Line 2: reasoning: [brief explanation citing key normalizations or contradictions]"""
