"""Unified correctness evaluation for the four paper datasets.

Single source of truth replacing the four near-identical
`<dataset>_correctness.py` libraries. Dispatches per-dataset prompts and
paths via `evaluation.specs.EvaluationSpec`.

Public API:
  - `run_correctness_eval(spec, answers_path, output_path, client, *, resume=True)`
    score one answers file against the gold DSL.
  - `run_correctness_batch(dataset, model_slug, regimes, dsl_path=None, *, resume=False)`
    score every available regime for one model.
  - `default_output_path(spec, model_slug, regime)` filename helper.

Error policy: API errors are recorded as `{"correct": 0, "reasoning": "API error: ..."}`
so the run is restartable without losing partial progress (CRAG-style policy
applied uniformly across all four datasets per the C10 reconciliation).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from evaluation.judge import build_vertex_client, parse_judge_json  # noqa: E402
from evaluation.specs import EvaluationSpec, get_spec  # noqa: E402

MODEL_NAME = os.environ.get("CORRECTNESS_MODEL_NAME", "google/gemini-2.5-flash")
TEMPERATURE = float(os.environ.get("CORRECTNESS_TEMPERATURE", "0.1"))
SAVE_EVERY = int(os.environ.get("CORRECTNESS_SAVE_EVERY", "10"))


def _load_json(path: Path) -> Any:
    with path.open("r") as f:
        return json.load(f)


def _save_results(path: Path, results: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(results, f, indent=2)


def _iter_common_ids(answers: Dict[str, Any], dsl: Dict[str, Any]) -> Iterable[str]:
    return sorted(set(answers.keys()) & set(dsl.keys()))


def _extract_prediction(answer_obj: Dict[str, Any]) -> str:
    for key in ("final_answer", "answer", "prediction"):
        value = answer_obj.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def run_correctness_eval(
    spec: EvaluationSpec,
    answers_path: Path,
    output_path: Path,
    client: OpenAI,
    *,
    resume: bool = True,
) -> Tuple[int, int]:
    """Score one answers file against the gold DSL with the Gemini judge."""
    print(f"Loading answers from {answers_path}")
    answers_data = _load_json(answers_path)
    answers_lookup = {str(item["id"]): item for item in answers_data}

    print(f"Loading DSL gold answers from {spec.dsl_path}")
    dsl_data = _load_json(spec.dsl_path)
    dsl_lookup = {
        str(item["id"]): {
            "question": item.get("question") or item.get("top_level_question", ""),
            "gold": item.get("answer")
            or (item.get("original_answers")[-1] if item.get("original_answers") else ""),
        }
        for item in dsl_data
        if item.get("id") is not None
    }

    print(f"Loaded {len(answers_lookup)} model answers")
    print(f"Loaded {len(dsl_lookup)} DSL entries")

    if not answers_lookup:
        raise RuntimeError("Answers file is empty.")
    if not dsl_lookup:
        raise RuntimeError("DSL file is empty.")

    results: Dict[str, Any] = {}
    if output_path.exists():
        if resume:
            print(f"Resuming from {output_path}")
            results = _load_json(output_path)
        else:
            output_path.unlink()
    evaluated_before = len(results)

    ids_to_score = list(_iter_common_ids(answers_lookup, dsl_lookup))
    print(f"Found {len(ids_to_score)} overlapping IDs between answers and DSL")

    system_prompt = spec.correctness_prompt_module.create_correctness_system_prompt()
    user_prompt_fn = spec.correctness_prompt_module.create_correctness_user_prompt

    evaluated_now = 0
    for idx, example_id in enumerate(ids_to_score, 1):
        if example_id in results:
            continue

        answer_obj = answers_lookup[example_id]
        dsl_obj = dsl_lookup[example_id]
        question = (dsl_obj.get("question") or "").strip()
        gold = (dsl_obj.get("gold") or "").strip()
        prediction = _extract_prediction(answer_obj)

        if not gold:
            print(f"[skip] {example_id}: missing gold string")
            continue

        if not prediction:
            print(f"[not meaningful] {example_id}: missing prediction text")
            results[example_id] = {"correct": -1, "reasoning": "Missing prediction text"}
            evaluated_now += 1
            if (evaluated_before + evaluated_now) % SAVE_EVERY == 0:
                print(f"💾 Checkpoint after {evaluated_before + evaluated_now} evaluations")
                _save_results(output_path, results)
            continue

        print(f"[{idx}/{len(ids_to_score)}] Evaluating {example_id}")

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                temperature=TEMPERATURE,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt_fn(question, gold, prediction)},
                ],
            )
            parsed = parse_judge_json(response.choices[0].message.content)
        except Exception as exc:
            print(f"  ERROR evaluating {example_id}: {exc}")
            results[example_id] = {"correct": 0, "reasoning": f"API error: {exc}"}
            evaluated_now += 1
            if (evaluated_before + evaluated_now) % SAVE_EVERY == 0:
                _save_results(output_path, results)
            continue

        correct_raw = parsed.get("correct", 0)
        try:
            correct = int(correct_raw)
        except (TypeError, ValueError):
            correct = 0
        if correct not in (-1, 0, 1):
            correct = 0

        results[example_id] = {"correct": correct, "reasoning": parsed.get("reasoning", "")}
        evaluated_now += 1

        if (evaluated_before + evaluated_now) % SAVE_EVERY == 0:
            print(f"💾 Checkpoint after {evaluated_before + evaluated_now} evaluations")
            _save_results(output_path, results)

    _save_results(output_path, results)

    total_scored = len(results)
    total_correct = sum(
        1 for r in results.values() if isinstance(r, dict) and r.get("correct") == 1
    )

    print("\n" + "═" * 56)
    print(f"{spec.name.title()} correctness evaluation complete")
    print("═" * 56)
    print(f"Examples scored : {total_scored}")
    print(f"Correct answers : {total_correct}")
    if total_scored:
        print(f"Accuracy        : {total_correct / total_scored:.2%}")
    else:
        print("Accuracy        : n/a")
    print(f"Results written : {output_path}")

    return total_scored, total_correct


def _resolve_answers_path(spec: EvaluationSpec, model_dir: Path, regime: str) -> Path:
    return model_dir / spec.regime_filename_map.get(regime, f"{spec.name}_{regime}.json")


def _existing_regimes(
    spec: EvaluationSpec, model_dir: Path, regimes: Iterable[str]
) -> List[str]:
    return [r for r in regimes if _resolve_answers_path(spec, model_dir, r).exists()]


def default_output_path(spec: EvaluationSpec, model_slug: str, regime: str) -> Path:
    safe_regime = regime.replace("-", "_")
    return spec.correctness_dir / model_slug / f"{spec.name}_{safe_regime}_correctness.json"


def run_correctness_batch(
    dataset: str,
    model_slug: str,
    regimes: Iterable[str],
    dsl_path: Optional[Path] = None,
    *,
    resume: bool = False,
) -> None:
    """Score every available regime for one model under one dataset.

    `dsl_path` is accepted for CLI compatibility; the canonical path is
    `spec.dsl_path` (`data/processed/<dataset>/<dataset>_dsl.json`).
    """
    spec = get_spec(dataset)
    model_dir = spec.generation_dir / model_slug
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    target_regimes = _existing_regimes(spec, model_dir, regimes)
    if not target_regimes:
        print(f"No regimes found for {model_slug}. Nothing to do.")
        return

    client = build_vertex_client(task_label="correctness evaluation")
    print(f"Running {spec.name.title()} correctness evaluations for {model_slug}")
    for regime in target_regimes:
        answers_path = _resolve_answers_path(spec, model_dir, regime)
        output_path = default_output_path(spec, model_slug, regime)
        rel_out = output_path.relative_to(REPO_ROOT)
        print(f"  • {regime}: {answers_path.name} → {rel_out}")
        run_correctness_eval(spec, answers_path, output_path, client, resume=resume)


REGIME_CHOICES: List[str] = ["assistive", "incremental", "model_generated", "open_ended"]
DEFAULT_TARGET_REGIMES: List[str] = ["assistive", "incremental", "model_generated"]


__all__ = [
    "run_correctness_eval",
    "run_correctness_batch",
    "default_output_path",
    "build_vertex_client",
    "REGIME_CHOICES",
    "DEFAULT_TARGET_REGIMES",
]
