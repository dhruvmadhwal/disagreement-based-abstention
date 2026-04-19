"""Utility helpers shared by Crag baseline runners."""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from generate.crag.pipeline import (
    REGIME_FILENAMES,
    parse_open_ended_response,
    strip_think_tags,
)

from .config import (
    CONSISTENCY_RESULTS_ROOT,
    CORRECTNESS_RESULTS_ROOT,
    GENERATION_RESULTS_ROOT,
)


IDK_TOKENS = {
    "i don't know",
    "i do not know",
    "i dont know",
    "idk",
}


def normalize_text(text: str) -> str:
    """Lowercase and trim common trailing punctuation for comparison."""
    if text is None:
        return ""
    cleaned = str(text).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.strip(" \n\t\r.,;:")
    return cleaned


def is_idk(text: Optional[str]) -> bool:
    """Return True if the text is an IDK-style response."""
    if text is None:
        return False
    lowered = normalize_text(text).lower()
    return any(
        lowered == token or lowered.startswith(f"{token}.") or lowered.startswith(f"{token},")
        for token in IDK_TOKENS
    )


def extract_final_answer(text: str) -> str:
    """Extract the shortest plausible final answer from a model reply."""
    if not text:
        return ""
    cleaned = strip_think_tags(text)
    # Prefer explicit "Final answer:" cues.
    match = re.search(r"final\s*answer\s*[:：]\s*(.+)", cleaned, flags=re.IGNORECASE)
    if match:
        candidate = match.group(1).strip()
        # Stop at the next line break to avoid grabbing explanations.
        candidate = candidate.splitlines()[0].strip()
        return candidate
    # Look for a leading "answer:" label; if empty, keep scanning subsequent lines.
    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        label_match = re.match(r"(?i)^(?:final\s*answer|answer|ans)\s*[:：]\s*(.+)$", line)
        if label_match:
            candidate = label_match.group(1).strip()
            if candidate:
                return candidate
            # If the label is empty (e.g., 'answer:' on its own), keep scanning for the next content line.
            continue
        # Fall back to the first non-empty, non-label line.
        return line
    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if line:
            return line
    return cleaned.strip()


def load_generation_lookup(model_slug: str, regimes: Iterable[str]) -> Dict[str, Dict[str, Dict[str, object]]]:
    """Load generation outputs by regime and id for a given model slug."""
    model_dir = GENERATION_RESULTS_ROOT / model_slug
    lookups: Dict[str, Dict[str, Dict[str, object]]] = {}
    for regime in regimes:
        filename = REGIME_FILENAMES.get(regime)
        if not filename:
            continue
        path = model_dir / filename
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        regime_lookup: Dict[str, Dict[str, object]] = {}
        for item in payload:
            example_id = str(item.get("id"))
            if not example_id:
                continue
            regime_lookup[example_id] = item
        lookups[regime] = regime_lookup
    return lookups


def load_correctness_lookup(model_slug: str) -> Dict[str, Dict[str, object]]:
    """Load correctness judgments (open-ended) keyed by id."""
    path = CORRECTNESS_RESULTS_ROOT / model_slug / "crag_open_ended_correctness.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    return {str(k): v for k, v in payload.items()}


def load_consistency_lookup(model_slug: str) -> Dict[str, Dict[str, Dict[str, object]]]:
    """Load consistency results for pairwise comparisons keyed by regime and id.

    Returns a dict like:
        {
            "assistive": {"example_id": {"equivalent": 1, "reasoning": "...", ...}},
            "incremental": {...},
            "model_generated": {...},
        }
    """
    regimes = ("assistive", "incremental", "model_generated")
    lookups: Dict[str, Dict[str, Dict[str, object]]] = {}
    model_dir = CONSISTENCY_RESULTS_ROOT / model_slug
    for regime in regimes:
        filename = f"crag_{regime}_vs_open_ended_consistency.json"
        path = model_dir / filename
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        comparisons = payload.get("comparisons", [])
        regime_lookup: Dict[str, Dict[str, object]] = {}
        for item in comparisons:
            example_id = str(item.get("id", ""))
            if example_id:
                regime_lookup[example_id] = item
        lookups[regime] = regime_lookup
    return lookups


@dataclass
class DemoExample:
    question: str
    answer: str  # either gold answer or "I don't know"
    id: Optional[str] = None


def _is_correct_answer(prediction: str, gold: str, correctness_lookup: Dict[str, Dict[str, object]], example_id: str) -> bool:
    verdict = correctness_lookup.get(example_id, {}).get("correct")
    if verdict is not None:
        try:
            return int(verdict) == 1
        except (TypeError, ValueError):
            pass
    if not prediction:
        return False
    return normalize_text(prediction).lower() == normalize_text(gold).lower()


def build_ic_idk_demos(
    examples: Sequence[dict],
    open_ended_lookup: Dict[str, Dict[str, object]],
    correctness_lookup: Dict[str, Dict[str, object]],
    *,
    k: int,
    d: int,
    seed: int = 13,
) -> List[DemoExample]:
    """Construct a demo bank (K total, D IDK) from open-ended runs or pre-labeled prompts."""
    rng = random.Random(seed)
    idk_pool: List[DemoExample] = []
    correct_pool: List[DemoExample] = []

    for example in examples:
        ex_id = str(example.get("id"))
        if not ex_id:
            continue
        question = str(example.get("question", "")).strip()
        gold_answer = str(example.get("answer", "")).strip()
        open_entry = open_ended_lookup.get(ex_id) or {}
        explicit_answer = example.get("ic_idk_answer") or example.get("answer") or ""
        explicit_label = example.get("ic_idk_is_correct")
        prediction = (
            open_entry.get("answer")
            or open_entry.get("final_answer")
            or open_entry.get("prediction")
            or ""
        )
        is_correct: Optional[bool] = None
        answer_text = ""

        if explicit_label is not None and explicit_answer:
            is_correct = bool(explicit_label)
            answer_text = parse_open_ended_response(str(explicit_answer))
        elif open_entry or correctness_lookup.get(ex_id) is not None:
            is_correct = _is_correct_answer(prediction, gold_answer, correctness_lookup, ex_id)
            answer_text = parse_open_ended_response(str(prediction) or gold_answer)
        elif explicit_answer:
            answer_text = parse_open_ended_response(str(explicit_answer))
            is_correct = not is_idk(answer_text)

        if is_correct is None:
            continue

        if is_correct:
            final_answer = answer_text or gold_answer
            if not final_answer:
                continue
            if is_idk(final_answer):
                final_answer = gold_answer
            correct_pool.append(DemoExample(question=question, answer=final_answer, id=ex_id))
        else:
            idk_pool.append(DemoExample(question=question, answer="I don't know", id=ex_id))

    rng.shuffle(idk_pool)
    rng.shuffle(correct_pool)

    selected_idk = idk_pool[: max(0, min(d, len(idk_pool)))]
    remaining_k = max(0, k - len(selected_idk))
    selected_correct = correct_pool[:remaining_k]

    if len(selected_idk) < d:
        print(f"[IC-IDK] Warning: requested {d} incorrect demos but only found {len(selected_idk)}.")
    if len(selected_correct) < remaining_k:
        print(
            f"[IC-IDK] Warning: requested {remaining_k} correct demos but only found {len(selected_correct)}."
        )

    demos = selected_idk + selected_correct
    rng.shuffle(demos)
    return demos


def compute_metrics(records: Sequence[dict]) -> Dict[str, float]:
    """Compute coverage + paper-style detection metrics.

    Paper framing: we want to detect incorrect claims. Treat a rejection
    (accepted == False) of an incorrect answer as a true positive (TP).
    Accepting an incorrect answer is a false negative (FN).
    Rejecting a correct answer is a false positive (FP).
    Accepting a correct answer is a true negative (TN).
    """
    total = len(records)
    if total == 0:
        return {
            "total": 0,
            "coverage": 0.0,
            "overall_accuracy": 0.0,
            "accuracy_at_coverage": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

    # Confusion matrix under detection framing (incorrect = positive)
    tp = fp = tn = fn = 0
    for r in records:
        accepted = bool(r.get("accepted"))
        try:
            correct = int(r.get("correct"))
        except Exception:
            correct = 0
        # Treat anything other than an explicit 1 (including -1) as incorrect.
        is_correct = correct == 1
        if not accepted and not is_correct:
            tp += 1  # rejected an incorrect claim
        elif not accepted and is_correct:
            fp += 1  # rejected a correct claim
        elif accepted and not is_correct:
            fn += 1  # accepted an incorrect claim
        elif accepted and is_correct:
            tn += 1  # accepted a correct claim

    coverage = (tn + fn) / total  # answered / total
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    # Overall accuracy: fraction of all examples where the claim was correct (accepted correct).
    overall_acc = tn / total
    # Accuracy at coverage: accuracy conditional on answering (only accepted cases).
    acc_at_cov = tn / (tn + fn) if (tn + fn) else 0.0

    return {
        "total": total,
        "coverage": round(coverage, 4),
        "overall_accuracy": round(overall_acc, 4),
        "accuracy_at_coverage": round(acc_at_cov, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


__all__ = [
    "DemoExample",
    "build_ic_idk_demos",
    "compute_metrics",
    "extract_final_answer",
    "is_idk",
    "load_consistency_lookup",
    "load_correctness_lookup",
    "load_generation_lookup",
    "normalize_text",
]
