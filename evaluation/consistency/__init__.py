"""Unified DBA-A consistency comparison for the four paper datasets.

Single source of truth replacing the four near-identical
`compare_<dataset>.py` libraries. Dispatches per-dataset prompts and paths
via `evaluation.specs.EvaluationSpec`.

Public API:
  - `compare_answers(client, question, answer_a, answer_b, spec)`: judge one pair.
  - `compare_answer_sets(client, file_a, file_b, output_file, *, limit, resume, spec)`:
    judge every common-id pair across two regime answer files.
  - `run_comparisons(dataset, model_slug, regimes, limit, *, resume, client)`:
    pair every non-Direct regime against `open_ended` for one model.
  - `default_output_path(spec, model_slug, regime_a, regime_b)` filename helper.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from evaluation.judge import build_vertex_client  # noqa: E402
from evaluation.specs import EvaluationSpec, get_spec  # noqa: E402

CONSISTENCY_MODEL_NAME = os.environ.get("CONSISTENCY_MODEL_NAME", "google/gemini-2.5-flash")
CONSISTENCY_TEMPERATURE = float(os.environ.get("CONSISTENCY_TEMPERATURE", "0.1"))
SAVE_EVERY = int(os.environ.get("CONSISTENCY_SAVE_EVERY", "10"))


def compare_answers(
    client: OpenAI,
    question: Optional[str],
    answer_a: str,
    answer_b: str,
    spec: EvaluationSpec,
) -> Dict[str, Any]:
    """Compare two answers using the per-dataset consistency judge."""
    system_prompt = spec.consistency_prompt_module.create_consistency_system_prompt()
    user_prompt = spec.consistency_prompt_module.create_consistency_user_prompt(
        question, answer_a, answer_b
    )

    print("\n=== DEBUG COMPARISON ===")
    print(f"Question: {question}")
    print(f"Answer A: {answer_a}")
    print(f"Answer B: {answer_b}")
    print("========================\n")

    try:
        response = client.chat.completions.create(
            model=CONSISTENCY_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=CONSISTENCY_TEMPERATURE,
        )
        response_text = response.choices[0].message.content.strip()
        print("Model Response:")
        print(f"'{response_text}'")
        print("--- End Response ---\n")

        equivalent_value = 0
        reasoning = "No reasoning provided"
        for raw_line in response_text.splitlines():
            line = raw_line.strip()
            lowered = line.lower()
            if lowered.startswith("equivalent:"):
                token = line.split(":", 1)[1].strip().split()[0]
                try:
                    parsed = int(token)
                except ValueError:
                    parsed = 0
                equivalent_value = parsed if parsed in (-1, 0, 1) else 0
            elif lowered.startswith("reasoning:") or lowered.startswith("explanation:"):
                reasoning = line.split(":", 1)[1].strip() or reasoning

        print(f"Parsed - Equivalent: {equivalent_value}, Reasoning: {reasoning}\n")

        return {
            "equivalent": equivalent_value,
            "reasoning": reasoning,
            "raw_response": response_text,
        }

    except Exception as exc:
        print(f"Error calling model: {exc}")
        time.sleep(1)
        return {
            "equivalent": 0,
            "reasoning": f"API error: {exc}",
        }


def _load_answers_file(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, "r") as f:
        return json.load(f)


def _load_existing_comparisons(output_file: str) -> List[Dict[str, Any]]:
    output_path = Path(output_file)
    if not output_path.exists():
        return []
    try:
        data = json.loads(output_path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    comparisons = data.get("comparisons", [])
    if not isinstance(comparisons, list):
        return []
    return [item for item in comparisons if isinstance(item, dict)]


def _save_comparison_results(
    results: List[Dict[str, Any]],
    output_file: str,
    equivalent_count: int,
    total_count: int,
    *,
    is_final: bool = False,
) -> None:
    rate = equivalent_count / total_count if total_count > 0 else 0
    output_data = {
        "summary": {
            "total_comparisons": total_count,
            "equivalent_answers": equivalent_count,
            "non_equivalent_answers": total_count - equivalent_count,
            "equivalence_rate": round(rate, 4),
        },
        "comparisons": results,
    }
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(output_data, f, indent=2)
    status = "Final" if is_final else "Checkpoint"
    print(f"{status} save: {total_count} comparisons, {equivalent_count} equivalent ({rate:.1%})")


def compare_answer_sets(
    client: OpenAI,
    file_a: str,
    file_b: str,
    output_file: str,
    *,
    spec: EvaluationSpec,
    limit: Optional[int] = None,
    resume: bool = True,
) -> None:
    """Judge every common-id pair across two regime answer files."""
    print(f"Loading answers from {file_a} and {file_b}")
    answers_a = _load_answers_file(file_a)
    answers_b = _load_answers_file(file_b)

    lookup_a = {item["id"]: item for item in answers_a}
    lookup_b = {item["id"]: item for item in answers_b}
    common_ids = sorted(set(lookup_a.keys()) & set(lookup_b.keys()))
    print(f"Found {len(common_ids)} common question IDs")

    if limit:
        common_ids = common_ids[:limit]
        print(f"Limited to first {limit} comparisons")

    if Path(output_file).exists() and not resume:
        Path(output_file).unlink()

    results = _load_existing_comparisons(output_file)
    completed_ids = {str(item.get("id")) for item in results if item.get("id") is not None}
    if results:
        print(f"Resuming from {output_file}: {len(completed_ids)} comparisons already present")

    equivalent_count = sum(1 for item in results if item.get("equivalent") == 1)
    total_count = len(results)

    pending_ids = [qid for qid in common_ids if str(qid) not in completed_ids]
    if not pending_ids:
        _save_comparison_results(results, output_file, equivalent_count, total_count, is_final=True)
        return

    for i, question_id in enumerate(pending_ids, 1):
        print(f"Comparing {i}/{len(pending_ids)}: {question_id}")

        item_a = lookup_a[question_id]
        item_b = lookup_b[question_id]
        final_answer_a = item_a.get("final_answer", item_a.get("answer", ""))
        final_answer_b = item_b.get("final_answer", item_b.get("answer", ""))
        question = item_a.get("question", "")

        comparison_result = compare_answers(client, question, final_answer_a, final_answer_b, spec)
        results.append(
            {
                "id": question_id,
                "question": question,
                "answer_a": final_answer_a,
                "answer_b": final_answer_b,
                "equivalent": comparison_result.get("equivalent", 0),
                "reasoning": comparison_result.get("reasoning", ""),
                "raw_response": comparison_result.get("raw_response", ""),
            }
        )
        if comparison_result.get("equivalent", 0) == 1:
            equivalent_count += 1
        total_count += 1

        if i % SAVE_EVERY == 0:
            _save_comparison_results(results, output_file, equivalent_count, total_count)

    _save_comparison_results(results, output_file, equivalent_count, total_count, is_final=True)


def _resolve_generation_file(spec: EvaluationSpec, model_dir: Path, regime: str) -> Path:
    return model_dir / spec.regime_filename_map.get(regime, f"{spec.name}_{regime}.json")


def _existing_regimes(
    spec: EvaluationSpec, model_dir: Path, regimes: Iterable[str]
) -> List[str]:
    return [
        r
        for r in regimes
        if r != "open_ended" and _resolve_generation_file(spec, model_dir, r).exists()
    ]


def default_output_path(
    spec: EvaluationSpec, model_slug: str, regime_a: str, regime_b: str
) -> Path:
    safe_a = regime_a.replace("-", "_")
    safe_b = regime_b.replace("-", "_")
    return spec.consistency_dir / model_slug / f"{spec.name}_{safe_a}_vs_{safe_b}_consistency.json"


def run_comparisons(
    dataset: str,
    model_slug: str,
    regimes: Iterable[str],
    limit: Optional[int] = None,
    *,
    resume: bool = True,
    client: Optional[OpenAI] = None,
) -> None:
    """Pair every non-Direct regime against `open_ended` for one model."""
    spec = get_spec(dataset)
    model_dir = spec.generation_dir / model_slug
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    open_ended_path = _resolve_generation_file(spec, model_dir, "open_ended")
    if not open_ended_path.exists():
        raise FileNotFoundError(
            f"Open-ended file missing for {model_slug}: {open_ended_path}"
        )

    target_regimes = _existing_regimes(spec, model_dir, regimes)
    if not target_regimes:
        print(f"No regimes to compare for {model_slug}. Nothing to do.")
        return

    if client is None:
        client = build_vertex_client(task_label="consistency evaluation")

    print(f"Running {spec.name.title()} consistency comparisons for {model_slug}")
    for regime in target_regimes:
        regime_path = _resolve_generation_file(spec, model_dir, regime)
        output_path = default_output_path(spec, model_slug, regime, "open_ended")
        print(f"  • {regime}: {regime_path.name} ↔ {open_ended_path.name}")
        compare_answer_sets(
            client,
            str(regime_path),
            str(open_ended_path),
            str(output_path),
            spec=spec,
            limit=limit,
            resume=resume,
        )


REGIME_CHOICES: List[str] = ["assistive", "incremental", "model_generated", "open_ended"]
DEFAULT_TARGET_REGIMES: List[str] = ["assistive", "incremental", "model_generated"]


__all__ = [
    "compare_answers",
    "compare_answer_sets",
    "run_comparisons",
    "default_output_path",
    "build_vertex_client",
    "REGIME_CHOICES",
    "DEFAULT_TARGET_REGIMES",
]
