#!/usr/bin/env python3
"""Analyze decomposition-chain behavior against assistive-vs-open consistency.

This script joins three artifacts per dataset/model:
- decomposition-equivalence per-item scores (`equivalent_final`)
- consistency checks for assistive vs open-ended (`equivalent`)
- correctness labels for both prompts (open-ended and assistive)

It reports the 4-way partition:
  - SS: decomp same, answer same
  - SD: decomp same, answer different
  - DS: decomp different, answer same
  - DD: decomp different, answer different
and further splits DD by open/assistive correctness.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DECOMP_ROOT = Path("decomposition_equivalence") / "results" / "eval"
CONSISTENCY_ROOT = Path("results") / "consistency"
CORRECTNESS_ROOT = Path("results") / "correctness"
DEFAULT_OUTPUT_DIR = Path("analysis") / "derived"

PAIR_BUCKETS = ("SS", "SD", "DS", "DD")
DD_BUCKETS = ("both_wrong", "open_wrong_assistive_correct", "open_correct_assistive_wrong", "both_correct", "missing_label")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze decomposition/consistency mismatches for assistive prompts.")
    parser.add_argument("--decomp-root", type=Path, default=DECOMP_ROOT, help="Decomposition eval root")
    parser.add_argument("--consistency-root", type=Path, default=CONSISTENCY_ROOT, help="Consistency results root")
    parser.add_argument("--correctness-root", type=Path, default=CORRECTNESS_ROOT, help="Correctness results root")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for generated outputs")
    parser.add_argument("--output-prefix", default="assistive_decomp_answer_mismatch", help="Filename prefix for outputs")
    parser.add_argument(
        "--max-dd-examples-per-bucket",
        type=int,
        default=None,
        help="Limit DD examples per mismatch bucket. Defaults to all DD examples.",
    )
    parser.add_argument(
        "--skip-missing-labels",
        action="store_true",
        help="Skip rows in DD when either prompt lacks a correctness label.",
    )
    return parser.parse_args()


def _safe_load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _normalize_model_name(model_name: str) -> str:
    if not model_name:
        return ""
    normalized = model_name.strip().lower().replace("_", "-").replace(".", "-")
    # Collapse repeated separators introduced by manual naming differences.
    while "--" in normalized:
        normalized = normalized.replace("--", "-")
    return normalized


def _canonicalize_consistency_model(model_name: str, decomp_models: Iterable[str]) -> Optional[str]:
    normalized = _normalize_model_name(model_name)
    decomp_set = set(decomp_models)

    # Direct lookup for exact names.
    if normalized in decomp_set:
        return normalized

    if normalized in {"llama-31-8b-instruct", "llama-3-1-8b-instruct"}:
        candidate = "meta-llama-llama-3-1-8b-instruct"
        if candidate in decomp_set:
            return candidate

    if normalized in {"llama-33-70b-instruct", "llama-3-3-70b-instruct"}:
        candidate = "meta-llama-llama-3-3-70b-instruct"
        if candidate in decomp_set:
            return candidate

    if normalized.startswith("mistral-7b-instruct-v0-3") or normalized.startswith("mistralai-mistral-7b-instruct-v0-3"):
        candidate = "mistralai-mistral-7b-instruct-v0-3"
        if candidate in decomp_set:
            return candidate

    if normalized.startswith("qwen-2-5-72b") and candidate_in_set("qwen-qwen2-5-72b-instruct", decomp_set):
        return "qwen-qwen2-5-72b-instruct"

    if normalized.startswith("qwen-3-8") and candidate_in_set("qwen-qwen3-8b", decomp_set):
        return "qwen-qwen3-8b"

    if normalized.startswith("qwen-3-32") and candidate_in_set("qwen-qwen3-32b", decomp_set):
        return "qwen-qwen3-32b"

    if normalized.startswith("gemma-3-4b-it"):
        candidate = "google-gemma-3-4b-it"
        if candidate in decomp_set:
            return candidate

    if normalized.startswith("google-gemma-3-4b-it"):
        if "google-gemma-3-4b-it" in decomp_set:
            return "google-gemma-3-4b-it"

    return None


def candidate_in_set(candidate: str, candidates: Iterable[str]) -> bool:
    return candidate in candidates


def _coerce_binary(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value if value in (0, 1) else None
    if isinstance(value, float):
        if value in (0.0, 1.0):
            return int(value)
        return None
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"0", "false", "no", "not equivalent", "none"}:
            return 0
        if v in {"1", "true", "yes", "equivalent"}:
            return 1
        try:
            fv = float(v)
        except ValueError:
            return None
        if fv in (0.0, 1.0):
            return int(fv)
    return None


def _pair_key(dc: Optional[int], ac: Optional[int]) -> Optional[str]:
    if dc is None or ac is None:
        return None
    if dc == 1 and ac == 1:
        return "SS"
    if dc == 1 and ac == 0:
        return "SD"
    if dc == 0 and ac == 1:
        return "DS"
    if dc == 0 and ac == 0:
        return "DD"
    return None


def _dd_bucket(open_correct: Optional[int], assistive_correct: Optional[int]) -> str:
    if open_correct == 0 and assistive_correct == 0:
        return "both_wrong"
    if open_correct == 0 and assistive_correct == 1:
        return "open_wrong_assistive_correct"
    if open_correct == 1 and assistive_correct == 0:
        return "open_correct_assistive_wrong"
    if open_correct == 1 and assistive_correct == 1:
        return "both_correct"
    return "missing_label"


def _load_correctness_map(path: Path) -> Dict[str, int]:
    payload = _safe_load_json(path)
    out: Dict[str, int] = {}
    if payload is None:
        return out

    if isinstance(payload, dict):
        iterable = []
        for key, value in payload.items():
            if isinstance(value, dict):
                value = {**value, "id": str(value.get("id", key))}
            iterable.append(value)
    elif isinstance(payload, list):
        iterable = payload
    else:
        return out

    for item in iterable:
        if not isinstance(item, dict):
            continue
        example_id = item.get("id")
        if not isinstance(example_id, str):
            example_id = str(example_id) if example_id is not None else None
        if not example_id:
            continue
        correct = item.get("correct")
        if isinstance(correct, bool):
            out[example_id] = int(correct)
        elif isinstance(correct, (int, float)):
            if int(correct) in (0, 1, -1):
                out[example_id] = int(correct)
    return out


def _extract_consistency_records(payload: dict, dataset: str) -> List[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return []

    output: List[Dict[str, Any]] = []

    comparisons = payload.get("comparisons")
    if isinstance(comparisons, list):
        for item in comparisons:
            if not isinstance(item, dict):
                continue
            eq = _coerce_binary(item.get("equivalent"))
            if eq not in (0, 1):
                continue
            output.append(
                {
                    "id": str(item.get("id")) if item.get("id") is not None else None,
                    "equivalent": eq,
                    "answer_a": item.get("answer_a") if item.get("answer_a") is not None else item.get("open_ended_answer"),
                    "answer_b": item.get("answer_b") if item.get("answer_b") is not None else item.get("assistive_answer"),
                    "question": item.get("question") or "",
                    "regime": "assistive",
                    "dataset": dataset,
                }
            )
        return output

    evaluations = payload.get("evaluations")
    if isinstance(evaluations, list):
        for item in evaluations:
            if not isinstance(item, dict):
                continue
            eq = _coerce_binary(item.get("equivalent"))
            if eq not in (0, 1):
                eq = _coerce_binary(item.get("consistency_score"))
            if eq not in (0, 1):
                continue
            output.append(
                {
                    "id": str(item.get("id")) if item.get("id") is not None else None,
                    "equivalent": eq,
                    "answer_a": item.get("answer_open_ended"),
                    "answer_b": item.get("answer_assistive"),
                    "question": item.get("question") or "",
                    "regime": "assistive",
                    "dataset": dataset,
                }
            )
        return output

    return []


def _read_pair_counts(
    summary_rows: List[Dict[str, Any]],
    dd_rows: List[Dict[str, Any]],
    decomp_path: Path,
    consistency_path: Path,
    open_correctness_path: Path,
    assistive_correctness_path: Path,
) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
    decomp_payload = _safe_load_json(decomp_path)
    if not isinstance(decomp_payload, list):
        return {}, {}, {}
    decomp_map = {
        str(item.get("id")): item
        for item in decomp_payload
        if isinstance(item, dict) and "id" in item
    }

    consistency_payload = _safe_load_json(consistency_path)
    comp_records = _extract_consistency_records(consistency_payload, decomp_path.parent.parent.name)
    if not comp_records:
        return {}, {}, {}

    open_correct_map = _load_correctness_map(open_correctness_path)
    assistive_correct_map = _load_correctness_map(assistive_correctness_path)

    pair_counts = {key: 0 for key in PAIR_BUCKETS}
    dd_bucket_counts = {key: 0 for key in DD_BUCKETS}
    pair_examples: Dict[str, List[Dict[str, Any]]] = {key: [] for key in DD_BUCKETS}

    for item in comp_records:
        example_id = item.get("id")
        if example_id is None:
            continue
        decomp_item = decomp_map.get(example_id)
        if not isinstance(decomp_item, dict):
            continue

        dc = _coerce_binary(decomp_item.get("equivalent_final"))
        ac = _coerce_binary(item.get("equivalent"))
        if dc is None or ac is None:
            continue

        bucket = _pair_key(dc, ac)
        if not bucket:
            continue
        pair_counts[bucket] = pair_counts.get(bucket, 0) + 1

        if bucket != "DD":
            continue

        open_correct = open_correct_map.get(example_id)
        assistive_correct = assistive_correct_map.get(example_id)

        dd_key = _dd_bucket(open_correct, assistive_correct)
        dd_bucket_counts[dd_key] = dd_bucket_counts.get(dd_key, 0) + 1
        pair_examples[dd_key].append(
            {
                "dataset": decomp_path.parent.name,
                "decomp_model": decomp_path.stem,
                "consistency_model": consistency_path.parent.name,
                "id": example_id,
                "question": item.get("question", ""),
                "decomp_equivalent_final": dc,
                "consistency_equivalent": ac,
                "answer_assistive": item.get("answer_b"),
                "answer_open_ended": item.get("answer_a"),
                "decomp_question": decomp_item.get("question", ""),
                "gold_dsl": decomp_item.get("gold_dsl", ""),
                "model_dsl": decomp_item.get("model_dsl", ""),
                "open_ended_correct": open_correct,
                "assistive_correct": assistive_correct,
                "dd_bucket": dd_key,
            }
        )
        dd_rows.append(pair_examples[dd_key][-1])

    return pair_counts, dd_bucket_counts, pair_examples


def _find_consistency_file(model_dir: Path, dataset: str) -> Optional[Path]:
    candidate_names = [
        f"{dataset}_assistive_vs_open_ended_consistency.json",
        f"{dataset}_open_ended_vs_assistive.json",
        "consistency_open_ended_vs_assistive.json",
    ]
    for name in candidate_names:
        candidate = model_dir / name
        if candidate.exists():
            return candidate
    return None


def _find_correctness_file(correctness_dataset_dir: Path, model: str, dataset: str, answer_regime: str) -> Optional[Path]:
    if answer_regime not in {"open_ended", "assistive"}:
        raise ValueError(f"Unsupported answer regime: {answer_regime}")

    candidates = [
        correctness_dataset_dir / model / f"{dataset}_{answer_regime}_correctness.json",
        correctness_dataset_dir / model / f"{answer_regime}_correctness.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_dataset_decomp_models(dataset_dir: Path) -> Tuple[set[str], list[Path]]:
    paths: list[Path] = sorted(dataset_dir.glob("*.json"))
    return {p.stem for p in paths}, paths


def run_analysis(
    decomp_root: Path,
    consistency_root: Path,
    correctness_root: Path,
    output_dir: Path,
    output_prefix: str,
    max_dd_examples_per_bucket: Optional[int] = None,
    skip_missing_labels: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    dataset_summary: List[Dict[str, Any]] = []
    dd_examples: List[Dict[str, Any]] = []

    if not decomp_root.exists():
        return dataset_summary, dd_examples

    for dataset_dir in sorted(p for p in decomp_root.iterdir() if p.is_dir()):
        dataset = dataset_dir.name
        decomp_models_set, decomp_files = _load_dataset_decomp_models(dataset_dir)
        if not decomp_models_set:
            continue

        consistency_dataset_dir = consistency_root / dataset
        if not consistency_dataset_dir.exists():
            continue
        correctness_dataset_dir = correctness_root / dataset

        for consistency_model_dir in sorted(p for p in consistency_dataset_dir.iterdir() if p.is_dir()):
            consistency_model = consistency_model_dir.name
            decomp_model = _canonicalize_consistency_model(consistency_model, decomp_models_set)
            if not decomp_model:
                continue

            consistency_path = _find_consistency_file(consistency_model_dir, dataset)
            if not consistency_path:
                continue

            decomp_path = dataset_dir / f"{decomp_model}.json"
            if not decomp_path.exists():
                continue

            open_correctness_path = _find_correctness_file(
                correctness_dataset_dir,
                consistency_model,
                dataset,
                answer_regime="open_ended",
            )
            assistive_correctness_path = _find_correctness_file(
                correctness_dataset_dir,
                consistency_model,
                dataset,
                answer_regime="assistive",
            )
            if open_correctness_path is None or assistive_correctness_path is None:
                continue

            pair_counts: Dict[str, int]
            dd_bucket_counts: Dict[str, int]
            pair_counts, dd_bucket_counts, _ = _read_pair_counts(
                dataset_summary,
                dd_examples,
                decomp_path,
                consistency_path,
                open_correctness_path,
                assistive_correctness_path,
            )

            total = sum(pair_counts.values())
            if total == 0:
                continue

            row = {
                "dataset": dataset,
                "decomp_model": decomp_model,
                "consistency_model": consistency_model,
                "total": total,
                "SS_count": pair_counts["SS"],
                "SD_count": pair_counts["SD"],
                "DS_count": pair_counts["DS"],
                "DD_count": pair_counts["DD"],
                "SS_rate": pair_counts["SS"] / total,
                "SD_rate": pair_counts["SD"] / total,
                "DS_rate": pair_counts["DS"] / total,
                "DD_rate": pair_counts["DD"] / total,
                "DD_both_wrong": dd_bucket_counts["both_wrong"],
                "DD_open_wrong_assistive_correct": dd_bucket_counts["open_wrong_assistive_correct"],
                "DD_open_correct_assistive_wrong": dd_bucket_counts["open_correct_assistive_wrong"],
                "DD_both_correct": dd_bucket_counts["both_correct"],
                "DD_missing_label": dd_bucket_counts["missing_label"],
            }
            dataset_summary.append(row)

    if max_dd_examples_per_bucket is not None:
        trimmed: List[Dict[str, Any]] = []
        kept = defaultdict(int)
        for record in dd_examples:
            key = record["dd_bucket"]
            if key == "missing_label" and skip_missing_labels:
                continue
            if kept[key] >= max_dd_examples_per_bucket:
                continue
            trimmed.append(record)
            kept[key] += 1
        dd_examples = trimmed

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_summary_csv(output_dir / f"{output_prefix}_summary.csv", dataset_summary)
    _write_examples_jsonl(output_dir / f"{output_prefix}_dd_examples.jsonl", dd_examples)

    return dataset_summary, dd_examples


def _write_summary_csv(output_path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    columns = [
        "dataset",
        "decomp_model",
        "consistency_model",
        "total",
        "SS_count",
        "SD_count",
        "DS_count",
        "DD_count",
        "SS_rate",
        "SD_rate",
        "DS_rate",
        "DD_rate",
        "DD_both_wrong",
        "DD_open_wrong_assistive_correct",
        "DD_open_correct_assistive_wrong",
        "DD_both_correct",
        "DD_missing_label",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            serialized = {
                key: (
                    f"{value:.3f}" if key.endswith("_rate") and isinstance(value, (int, float)) else value
                )
                for key, value in row.items()
            }
            writer.writerow(serialized)


def _write_examples_jsonl(output_path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()

    summary_rows, example_rows = run_analysis(
        decomp_root=args.decomp_root,
        consistency_root=args.consistency_root,
        correctness_root=args.correctness_root,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        max_dd_examples_per_bucket=args.max_dd_examples_per_bucket,
        skip_missing_labels=args.skip_missing_labels,
    )
    print(f"Summary rows written: {len(summary_rows)}")
    print(f"DD examples written: {len(example_rows)}")
    print(f"Output directory: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
