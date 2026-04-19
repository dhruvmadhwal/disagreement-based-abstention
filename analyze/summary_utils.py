"""Utilities for computing aggregate QA evaluation statistics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "results"

REGIME_ALIASES = {
    "sequential": "incremental",
    "self_ask": "incremental",
    "selfask": "incremental",
    "model_generated_plan": "model_generated",
    "model_generated_plan_v2": "model_generated",
    "modelgenerated": "model_generated",
    "model_generated_plan_dsl": "model_generated",
    "model_generated_plan_samples": "model_generated",
    "assist": "assistive",
    "openended": "open_ended",
    "open-ended": "open_ended",
    "openended_generation": "open_ended",
}

DEFAULT_REGIME_ORDER = ["open_ended", "assistive", "incremental", "model_generated"]

MODEL_ALIASES = {
    "gemini-2.5-flash": "google-gemini-2-5-flash",
    "gemini-2.5-pro": "google-gemini-2-5-pro",
    "llama-3.1-8b-instruct": "meta-llama-llama-3-1-8b-instruct",
    "llama-3.3-70b-instruct": "meta-llama-llama-3-3-70b-instruct",
    "mistral-7b-instruct-v0.3": "mistralai-mistral-7b-instruct-v0-3",
    "qwen-2.5-72b-instruct": "qwen-qwen2-5-72b-instruct",
    "qwen3-8b": "qwen-qwen3-8b",
    "qwen-3-8b": "qwen-qwen3-8b",
    "qwen3-32b": "qwen-qwen3-32b",
    "qwen-3-32b": "qwen-qwen3-32b",
}

def _median(values: List[float]) -> Optional[float]:
    if not values:
        return None
    sorted_vals = sorted(values)
    mid = len(sorted_vals) // 2
    if len(sorted_vals) % 2 == 1:
        return float(sorted_vals[mid])
    return float((sorted_vals[mid - 1] + sorted_vals[mid]) / 2)


def _load_dsl_lengths(dataset: str) -> Optional[List[int]]:
    processed_dir = REPO_ROOT / "data" / "processed" / dataset
    dsl_path = processed_dir / f"{dataset}_dsl.json"
    if not dsl_path.exists():
        return None
    payload = _safe_load_json(dsl_path)
    if not isinstance(payload, list):
        return None
    lengths: List[int] = []
    for item in payload:
        dsl = item.get("dsl")
        if not isinstance(dsl, str):
            continue
        steps = sum(1 for line in dsl.splitlines() if line.strip())
        if steps:
            lengths.append(steps)
    return lengths or None


def canonicalize_regime(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    slug = name.strip().lower()
    slug = slug.replace("-", "_").replace(" ", "_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    slug = slug.strip("_")
    return REGIME_ALIASES.get(slug, slug)


def canonicalize_model(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    normalized = name.strip()
    lookup = normalized.lower()
    canonical = MODEL_ALIASES.get(lookup)
    return canonical or normalized


def _maybe_int(value) -> Optional[int]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_load_json(path: Path) -> Optional[dict]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _tally_correctness_records(records: Iterable[dict]) -> Optional[Tuple[int, int, int, int]]:
    evaluated = 0
    correct = 0
    incorrect = 0
    not_meaningful = 0
    found = False
    for record in records:
        if not isinstance(record, dict):
            continue
        val = record.get("correct")
        if val is None:
            continue
        found = True
        if isinstance(val, str):
            try:
                val = float(val)
            except ValueError:
                continue
        if val == 1:
            correct += 1
            evaluated += 1
        elif val == 0:
            incorrect += 1
            evaluated += 1
        elif val == -1:
            not_meaningful += 1
            evaluated += 1
        else:
            not_meaningful += 1
            evaluated += 1
    if not found or evaluated == 0:
        return None
    return evaluated, correct, incorrect, not_meaningful


def _extract_correctness_counts(payload: dict) -> Optional[Tuple[int, int, int, int]]:
    if not isinstance(payload, dict):
        return None

    summary = payload.get("summary")
    if isinstance(summary, dict):
        total = _maybe_int(
            summary.get("total_evaluated")
            or summary.get("evaluated")
            or summary.get("total")
            or summary.get("count")
        )
        correct = _maybe_int(summary.get("correct"))
        incorrect = _maybe_int(summary.get("incorrect"))
        not_meaningful = _maybe_int(
            summary.get("not_meaningful")
            or summary.get("not_meaningful_answers")
            or summary.get("unjudged")
            or summary.get("errors")
        )
        parts = [val for val in (correct, incorrect, not_meaningful) if val is not None]
        if total is None and parts:
            total = sum(parts)
        if incorrect is None and total is not None and correct is not None:
            remaining = total - correct - (not_meaningful or 0)
            incorrect = max(0, remaining)
        if total is None:
            total = 0
        return total, correct or 0, incorrect or 0, not_meaningful or 0

    evaluations = payload.get("evaluations")
    if isinstance(evaluations, list):
        counts = _tally_correctness_records(evaluations)
        if counts:
            return counts

    record_values = [
        record
        for key, record in payload.items()
        if key not in {"summary", "evaluations"}
    ]
    return _tally_correctness_records(record_values)


def _iter_dataset_model_dirs(base_dir: Path) -> Iterator[Tuple[str, str, Path]]:
    if not base_dir.exists():
        return iter(())
    for dataset_dir in sorted([d for d in base_dir.iterdir() if d.is_dir()]):
        dataset = dataset_dir.name
        for model_dir in sorted([m for m in dataset_dir.iterdir() if m.is_dir()]):
            model_name = canonicalize_model(model_dir.name) or model_dir.name
            yield dataset, model_name, model_dir


def _regime_sequence(available: Iterable[str]) -> List[str]:
    sequence = list(DEFAULT_REGIME_ORDER)
    extras = [reg for reg in available if reg not in sequence]
    sequence.extend(sorted(extras))
    return sequence


def _extract_regime_from_generation_file(path: Path, dataset: str) -> Optional[str]:
    stem = path.stem
    if stem.endswith("_legacy"):
        return None
    if stem.endswith("_full") or "regimes_full" in stem:
        return None
    prefix = f"{dataset}_"
    if stem.startswith(prefix):
        stem = stem[len(prefix) :]
    elif stem.startswith("regimes_"):
        return None
    return canonicalize_regime(stem)


def _extract_regime_from_correctness_file(path: Path, dataset: str) -> Optional[str]:
    stem = path.stem
    if stem.endswith("_summary"):
        return None
    suffix = "_correctness"
    if stem.endswith(suffix):
        stem = stem[: -len(suffix)]
    prefix = f"{dataset}_"
    if stem.startswith(prefix):
        stem = stem[len(prefix) :]
    return canonicalize_regime(stem)


def _parse_consistency_pair(path: Path, dataset: str, summary: Optional[dict]) -> Tuple[Optional[str], Optional[str]]:
    comparison = (summary or {}).get("comparison")
    if isinstance(comparison, str) and "_vs_" in comparison:
        left, right = comparison.split("_vs_", 1)
        return canonicalize_regime(left), canonicalize_regime(right)

    stem = path.stem
    for suffix in ("_consistency", "_summary"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    for prefix in (f"{dataset}_", "consistency_", "comparison_"):
        if stem.startswith(prefix):
            stem = stem[len(prefix) :]
    if "_vs_" in stem:
        left, right = stem.split("_vs_", 1)
        return canonicalize_regime(left), canonicalize_regime(right)
    regime = canonicalize_regime((summary or {}).get("regime"))
    comparison_regime = canonicalize_regime((summary or {}).get("comparison_regime"))
    return regime, comparison_regime


def load_generation_counts(results_dir: Path = RESULTS_ROOT) -> pd.DataFrame:
    rows: List[dict] = []
    base_dir = results_dir / "generation"
    if not base_dir.exists():
        return pd.DataFrame(columns=["dataset", "model", "regime", "examples", "path"])

    for dataset, model, model_dir in _iter_dataset_model_dirs(base_dir):
        for json_file in sorted(model_dir.glob("*.json")):
            regime = _extract_regime_from_generation_file(json_file, dataset)
            if not regime:
                continue
            payload = _safe_load_json(json_file)
            if not isinstance(payload, list):
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "regime": regime,
                    "examples": len(payload),
                    "path": str(json_file.relative_to(REPO_ROOT)),
                }
            )

    return pd.DataFrame(rows)


def load_baseline_reject_overlap(results_dir: Path = RESULTS_ROOT) -> pd.DataFrame:
    rows: List[dict] = []
    base_dir = results_dir / "baselines"
    if not base_dir.exists():
        return pd.DataFrame(
            columns=[
                "dataset",
                "model",
                "baseline_a",
                "baseline_b",
                "rejects_a",
                "rejects_b",
                "overlap",
                "overlap_pct_a",
                "overlap_pct_b",
                "jaccard",
                "incorrect_overlap",
                "overlap_precision",
                "overlap_recall",
            ]
        )

    for dataset_dir in sorted(d for d in base_dir.iterdir() if d.is_dir()):
        dataset = dataset_dir.name
        for model_dir in sorted(d for d in dataset_dir.iterdir() if d.is_dir()):
            model = canonicalize_model(model_dir.name) or model_dir.name
            rejects_by_baseline: Dict[str, set] = {}
            correctness_by_id: Dict[str, int] = {}
            for json_file in sorted(model_dir.glob("*.json")):
                if "ic_idk_pools" in json_file.parts or "pool" in json_file.name:
                    continue
                payload = _safe_load_json(json_file)
                if not isinstance(payload, dict):
                    continue
                records = payload.get("records")
                if not isinstance(records, list):
                    continue
                stem = json_file.stem
                baseline = stem
                if stem.startswith(f"{dataset}_"):
                    baseline = stem[len(dataset) + 1 :]
                rejected = set()
                for record in records:
                    if not isinstance(record, dict):
                        continue
                    if not record.get("accepted"):
                        ex_id = record.get("id")
                        if isinstance(ex_id, str):
                            rejected.add(ex_id)
                    ex_id = record.get("id")
                    if isinstance(ex_id, str) and ex_id not in correctness_by_id:
                        correct_val = record.get("correct")
                        if correct_val in (0, 1, -1):
                            correctness_by_id[ex_id] = int(correct_val)
                rejects_by_baseline[baseline] = rejected

            if not rejects_by_baseline:
                continue

            total_incorrect = sum(1 for v in correctness_by_id.values() if v == 0)

            baselines = sorted(rejects_by_baseline.keys())
            for baseline_a in baselines:
                for baseline_b in baselines:
                    if baseline_a == baseline_b:
                        continue
                    rej_a = rejects_by_baseline.get(baseline_a, set())
                    rej_b = rejects_by_baseline.get(baseline_b, set())
                    overlap_set = rej_a & rej_b
                    overlap = len(overlap_set)
                    rejects_a = len(rej_a)
                    rejects_b = len(rej_b)
                    union = len(rej_a | rej_b)
                    incorrect_overlap = sum(
                        1 for ex_id in overlap_set if correctness_by_id.get(ex_id) == 0
                    )
                    rows.append(
                        {
                            "dataset": dataset,
                            "model": model,
                            "baseline_a": baseline_a,
                            "baseline_b": baseline_b,
                            "rejects_a": rejects_a,
                            "rejects_b": rejects_b,
                            "overlap": overlap,
                            "overlap_pct_a": (overlap / rejects_a) if rejects_a else None,
                            "overlap_pct_b": (overlap / rejects_b) if rejects_b else None,
                            "jaccard": (overlap / union) if union else None,
                            "incorrect_overlap": incorrect_overlap,
                            "overlap_precision": (incorrect_overlap / overlap) if overlap else None,
                            "overlap_recall": (incorrect_overlap / total_incorrect) if total_incorrect else None,
                        }
                    )

    return pd.DataFrame(rows)


def _sort_by_regime(df: pd.DataFrame, column: str = "regime") -> pd.DataFrame:
    if df.empty or column not in df.columns:
        return df
    order_map = {reg: idx for idx, reg in enumerate(DEFAULT_REGIME_ORDER)}
    df = df.copy()
    df["__regime_order"] = df[column].apply(lambda reg: order_map.get(reg, len(order_map)))
    sort_fields = [field for field in ("dataset", "model") if field in df.columns]
    sort_fields.append("__regime_order")
    if column not in sort_fields:
        sort_fields.append(column)
    df = df.sort_values(sort_fields).drop(columns="__regime_order")
    return df


def load_model_generated_steps(results_dir: Path = RESULTS_ROOT) -> pd.DataFrame:
    rows: List[dict] = []
    base_dir = results_dir / "generation"
    if not base_dir.exists():
        return pd.DataFrame(
            columns=[
                "dataset",
                "model",
                "examples",
                "with_chain",
                "avg_steps",
                "median_steps",
                "avg_dsl_steps",
                "median_dsl_steps",
            ]
        )

    dsl_cache: Dict[str, Dict[str, Optional[float]]] = {}

    def _get_dsl_stats(dataset: str) -> Dict[str, Optional[float]]:
        if dataset in dsl_cache:
            return dsl_cache[dataset]
        lengths = _load_dsl_lengths(dataset)
        if lengths:
            avg_len = sum(lengths) / len(lengths)
            median_len = _median(lengths)
        else:
            avg_len = None
            median_len = None
        dsl_cache[dataset] = {
            "avg_dsl_steps": avg_len,
            "median_dsl_steps": median_len,
        }
        return dsl_cache[dataset]

    for dataset, model, model_dir in _iter_dataset_model_dirs(base_dir):
        for json_file in sorted(model_dir.glob("*.json")):
            regime = _extract_regime_from_generation_file(json_file, dataset)
            if regime != "model_generated":
                continue
            payload = _safe_load_json(json_file)
            if not isinstance(payload, list):
                continue
            chain_lengths: List[int] = []
            for item in payload:
                chain = item.get("qa_chain") if isinstance(item, dict) else None
                if isinstance(chain, list):
                    chain_lengths.append(len(chain))
            if not payload:
                continue
            avg_steps = (sum(chain_lengths) / len(chain_lengths)) if chain_lengths else None
            median_steps = _median(chain_lengths) if chain_lengths else None
            dsl_stats = _get_dsl_stats(dataset)
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "examples": len(payload),
                    "with_chain": len(chain_lengths),
                    "avg_steps": avg_steps,
                    "median_steps": median_steps,
                    "avg_dsl_steps": dsl_stats["avg_dsl_steps"],
                    "median_dsl_steps": dsl_stats["median_dsl_steps"],
                }
            )

    return pd.DataFrame(rows)


def _build_regime_coverage(
    generation_df: pd.DataFrame,
    correctness_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "dataset",
        "model",
        "regime",
        "has_generation",
        "generation_examples",
        "has_correctness",
        "has_consistency",
        "needs_correctness",
        "needs_consistency",
    ]
    if generation_df.empty:
        return pd.DataFrame(columns=columns)

    corr_keys = {
        (row.dataset, row.model, row.regime)
        for row in correctness_df.itertuples(index=False)
    }
    cons_keys = {
        (row.dataset, row.model, row.focused_regime)
        for row in consistency_df.dropna(subset=["focused_regime"]).itertuples(index=False)
    }

    grouped = (
        generation_df.groupby(["dataset", "model", "regime"], dropna=False)
        .agg({"examples": "max"})
        .reset_index()
    )
    entries: List[dict] = []
    model_pairs = grouped[["dataset", "model"]].drop_duplicates().sort_values(["dataset", "model"])
    for dataset, model in model_pairs.itertuples(index=False):
        subset = grouped[(grouped["dataset"] == dataset) & (grouped["model"] == model)]
        available_regimes = subset["regime"].tolist()
        for regime in _regime_sequence(available_regimes):
            match = subset[subset["regime"] == regime]
            has_generation = not match.empty
            examples = int(match.iloc[0]["examples"]) if has_generation else None
            key = (dataset, model, regime)
            has_correctness = key in corr_keys
            has_consistency = key in cons_keys
            needs_consistency = bool(has_generation and not has_consistency and regime != "open_ended")
            entries.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "regime": regime,
                    "has_generation": has_generation,
                    "generation_examples": examples,
                    "has_correctness": has_correctness,
                    "has_consistency": has_consistency,
                    "needs_correctness": bool(has_generation and not has_correctness),
                    "needs_consistency": needs_consistency,
                }
            )

    coverage = pd.DataFrame(entries, columns=columns)
    return _sort_by_regime(coverage)


def load_correctness_dataframe(results_dir: Path = RESULTS_ROOT) -> pd.DataFrame:
    rows: List[dict] = []
    base_dir = results_dir / "correctness"
    if not base_dir.exists():
        return pd.DataFrame(columns=[
            "dataset",
            "model",
            "regime",
            "evaluated",
            "correct",
            "incorrect",
            "not_meaningful",
            "accuracy",
            "path",
        ])

    for dataset, model, model_dir in _iter_dataset_model_dirs(base_dir):
        for json_file in sorted(model_dir.glob("*.json")):
            regime = _extract_regime_from_correctness_file(json_file, dataset)
            if not regime:
                continue
            payload = _safe_load_json(json_file)
            if not isinstance(payload, dict):
                continue
            counts = _extract_correctness_counts(payload)
            if not counts:
                continue
            evaluated, correct, incorrect, not_meaningful = counts
            accuracy = (correct / evaluated) if evaluated else None
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "regime": regime,
                    "evaluated": evaluated,
                    "correct": correct,
                    "incorrect": incorrect,
                    "not_meaningful": not_meaningful,
                    "accuracy": accuracy,
                    "path": str(json_file.relative_to(REPO_ROOT)),
                }
            )

    return pd.DataFrame(rows)


def _extract_consistency_counts(payload: dict) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[float], Optional[int]]:
    summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
    total = (
        summary.get("total_comparisons")
        or summary.get("total_evaluated")
        or summary.get("total")
    )
    equivalent = (
        summary.get("equivalent_answers")
        or summary.get("equivalent")
        or summary.get("consistent")
    )
    non_equivalent = (
        summary.get("non_equivalent_answers")
        or summary.get("non_equivalent")
        or summary.get("inconsistent")
    )
    eq_rate = (
        summary.get("equivalence_rate")
        or summary.get("consistency_rate")
        or summary.get("consistent_fraction")
    )
    not_meaningful = None

    comparisons = payload.get("comparisons")
    if isinstance(comparisons, list) and comparisons:
        eq_counts = {1: 0, 0: 0, -1: 0}
        for item in comparisons:
            eq = item.get("equivalent")
            if eq in eq_counts:
                eq_counts[eq] += 1
        equivalent = eq_counts[1]
        non_equivalent = eq_counts[0]
        not_meaningful = eq_counts[-1]
        total = eq_counts[1] + eq_counts[0]
        eq_rate = (equivalent / total) if total else None
    else:
        if isinstance(eq_rate, (int, float)) and eq_rate > 1.5:
            eq_rate = eq_rate / 100.0
        if total is not None and equivalent is not None and not isinstance(non_equivalent, (int, float)):
            non_equivalent = total - equivalent
    return total, equivalent, non_equivalent, eq_rate, not_meaningful


def load_consistency_dataframe(results_dir: Path = RESULTS_ROOT) -> pd.DataFrame:
    rows: List[dict] = []
    base_dir = results_dir / "consistency"
    if not base_dir.exists():
        return pd.DataFrame(columns=[
            "dataset",
            "model",
            "regime_a",
            "regime_b",
            "focused_regime",
            "comparisons",
            "equivalent",
            "non_equivalent",
            "not_meaningful",
            "equivalence_rate",
            "path",
        ])

    for dataset, model, model_dir in _iter_dataset_model_dirs(base_dir):
        for json_file in sorted(model_dir.glob("*.json")):
            if "_vs_" not in json_file.stem:
                continue
            payload = _safe_load_json(json_file)
            if not isinstance(payload, dict):
                continue
            regime_a, regime_b = _parse_consistency_pair(json_file, dataset, payload.get("summary"))
            total, equivalent, non_equivalent, eq_rate, not_meaningful = _extract_consistency_counts(payload)
            focused_regime = None
            if regime_a == "open_ended" and regime_b:
                focused_regime = regime_b
            elif regime_b == "open_ended" and regime_a:
                focused_regime = regime_a
            else:
                focused_regime = regime_a or regime_b
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "regime_a": regime_a,
                    "regime_b": regime_b,
                    "focused_regime": focused_regime,
                    "comparisons": total,
                    "equivalent": equivalent,
                    "non_equivalent": non_equivalent,
                    "not_meaningful": not_meaningful,
                    "equivalence_rate": eq_rate,
                    "path": str(json_file.relative_to(REPO_ROOT)),
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty and "focused_regime" in df.columns:
        df = df[df["focused_regime"] != "open_ended"].reset_index(drop=True)
    return df


def _load_correctness_map(results_dir: Path, dataset: str, model: str, regime: str) -> Dict[str, Optional[int]]:
    """Return a mapping from example id to correctness label for a given setup."""
    path = (
        Path(results_dir)
        / "correctness"
        / dataset
        / model
        / f"{dataset}_{regime}_correctness.json"
    )
    payload = _safe_load_json(path) or {}
    out: Dict[str, Optional[int]] = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            if not isinstance(value, dict):
                continue
            correct = value.get("correct")
            if correct in (0, 1, -1):
                out[key] = int(correct)
    return out


def build_consistency_correctness_examples(results_dir: Path = RESULTS_ROOT) -> pd.DataFrame:
    """Create per-example rows linking consistency (vs open-ended) with correctness."""
    base_dir = results_dir / "consistency"
    if not base_dir.exists():
        return pd.DataFrame(
            columns=[
                "dataset",
                "model",
                "example_id",
                "num_consistent",
                "available_pairs",
                "open_correct",
                "assistive_correct",
                "incremental_correct",
                "model_generated_correct",
                "structured_mean_correct",
                "all_structured_correct",
            ]
        )

    records: Dict[tuple, dict] = {}
    for dataset, model, model_dir in _iter_dataset_model_dirs(base_dir):
        for json_file in sorted(model_dir.glob("*.json")):
            if "_vs_" not in json_file.stem:
                continue
            payload = _safe_load_json(json_file)
            if not isinstance(payload, dict):
                continue
            comparisons = payload.get("comparisons")
            if not isinstance(comparisons, list):
                continue
            regime_a, regime_b = _parse_consistency_pair(json_file, dataset, payload.get("summary"))
            # We only anchor on open-ended to study 3-pair consistency.
            if regime_a == "open_ended":
                target_regime = regime_b
            elif regime_b == "open_ended":
                target_regime = regime_a
            else:
                continue
            if not target_regime:
                continue
            for comp in comparisons:
                ex_id = comp.get("id")
                if not isinstance(ex_id, str):
                    continue
                key = (dataset, model, ex_id)
                record = records.setdefault(
                    key,
                    {
                        "dataset": dataset,
                        "model": model,
                        "example_id": ex_id,
                        "assistive_equiv": None,
                        "incremental_equiv": None,
                        "model_generated_equiv": None,
                    },
                )
                if comp.get("equivalent") == 1:
                    record[f"{target_regime}_equiv"] = 1
                elif comp.get("equivalent") in (0, -1):
                    record[f"{target_regime}_equiv"] = 0

        if not records:
            continue

        # Load correctness maps once per model/dataset.
        correctness_maps = {
            reg: _load_correctness_map(results_dir, dataset, model, reg)
            for reg in ("open_ended", "assistive", "incremental", "model_generated")
        }

        for key, rec in list(records.items()):
            ds, mdl, ex_id = key
            if ds != dataset or mdl != model:
                continue
            open_val = correctness_maps["open_ended"].get(ex_id)
            rec["open_correct"] = None if open_val is None else int(open_val == 1)
            for reg in ("assistive", "incremental", "model_generated"):
                val = correctness_maps[reg].get(ex_id)
                rec[f"{reg}_correct"] = None if val is None else int(val == 1)

            equivalents = [
                rec.get("assistive_equiv"),
                rec.get("incremental_equiv"),
                rec.get("model_generated_equiv"),
            ]
            rec["available_pairs"] = sum(val in (0, 1) for val in equivalents)
            rec["num_consistent"] = sum(val == 1 for val in equivalents if val in (0, 1))

            struct_vals = [
                rec.get("assistive_correct"),
                rec.get("incremental_correct"),
                rec.get("model_generated_correct"),
            ]
            valid_struct = [val for val in struct_vals if isinstance(val, int)]
            if valid_struct:
                rec["structured_mean_correct"] = sum(valid_struct) / len(valid_struct)
                rec["all_structured_correct"] = int(all(val == 1 for val in valid_struct))
            else:
                rec["structured_mean_correct"] = None
                rec["all_structured_correct"] = None

    return pd.DataFrame(records.values())


def _aggregate_correctness(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=group_cols + ["evaluated", "correct", "incorrect", "not_meaningful", "accuracy"])
    grouped = (
        df.groupby(group_cols, dropna=False)[["evaluated", "correct", "incorrect", "not_meaningful"]]
        .sum()
        .reset_index()
    )
    grouped["accuracy"] = grouped.apply(
        lambda row: (row["correct"] / row["evaluated"]) if row["evaluated"] else None,
        axis=1,
    )
    return grouped


def _aggregate_consistency(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=group_cols + ["comparisons", "equivalent", "non_equivalent", "not_meaningful", "equivalence_rate"])
    grouped = (
        df.groupby(group_cols, dropna=False)[["comparisons", "equivalent", "non_equivalent", "not_meaningful"]]
        .sum()
        .reset_index()
    )
    grouped["equivalence_rate"] = grouped.apply(
        lambda row: (row["equivalent"] / row["comparisons"]) if row["comparisons"] else None,
        axis=1,
    )
    return grouped


def _build_rq1_table(correctness_df: pd.DataFrame) -> pd.DataFrame:
    if correctness_df.empty:
        return pd.DataFrame(columns=["dataset", "model", "regime", "accuracy", "open_ended_accuracy", "delta_vs_open"])
    open_acc = correctness_df[correctness_df["regime"] == "open_ended"][
        ["dataset", "model", "accuracy"]
    ].rename(columns={"accuracy": "open_ended_accuracy"})
    merged = correctness_df.merge(open_acc, on=["dataset", "model"], how="left")
    merged["delta_vs_open"] = merged.apply(
        lambda row: row["accuracy"] - row["open_ended_accuracy"]
        if pd.notna(row["accuracy"]) and pd.notna(row["open_ended_accuracy"])
        else None,
        axis=1,
    )
    focus_regimes = {"assistive", "incremental", "model_generated"}
    return merged[merged["regime"].isin(focus_regimes)].sort_values(["dataset", "model", "regime"])


def _merge_consistency_correctness(
    consistency_df: pd.DataFrame, correctness_df: pd.DataFrame
) -> pd.DataFrame:
    if consistency_df.empty or correctness_df.empty:
        return pd.DataFrame(columns=["dataset", "model", "regime", "accuracy", "equivalence_rate", "comparisons"])
    filtered = consistency_df.dropna(subset=["focused_regime"])
    merged = filtered.merge(
        correctness_df,
        left_on=["dataset", "model", "focused_regime"],
        right_on=["dataset", "model", "regime"],
        suffixes=("_consistency", "_correctness"),
    )
    if "regime" in merged.columns:
        merged = merged.drop(columns=["regime"])
    merged = merged.rename(columns={"focused_regime": "regime"})
    selected = merged[[
        "dataset",
        "model",
        "regime",
        "accuracy",
        "equivalence_rate",
        "comparisons",
    ]].copy()
    return selected.sort_values(["dataset", "model", "regime"])


def _bucketize_consistency_correctness(
    examples: pd.DataFrame, group_cols: List[str]
) -> pd.DataFrame:
    """Aggregate correctness conditioned on how many pairs are consistent."""
    if examples.empty:
        return pd.DataFrame(
            columns=group_cols
            + [
                "consistent_pairs",
                "examples",
                "open_accuracy",
                "assistive_accuracy",
                "incremental_accuracy",
                "model_generated_accuracy",
                "structured_mean_accuracy",
                "all_struct_accuracy",
            ]
        )
    subset = examples[examples["available_pairs"] == 3].copy()
    if subset.empty:
        return pd.DataFrame(
            columns=group_cols
            + [
                "consistent_pairs",
                "examples",
                "open_accuracy",
                "assistive_accuracy",
                "incremental_accuracy",
                "model_generated_accuracy",
                "structured_mean_accuracy",
                "all_struct_accuracy",
            ]
        )

    group_fields = list(group_cols) + ["num_consistent"]

    def _mean_safe(series: pd.Series) -> Optional[float]:
        clean = series.dropna()
        return None if clean.empty else clean.mean()

    grouped = (
        subset.groupby(group_fields, dropna=False)
        .agg(
            examples=("example_id", "count"),
            open_accuracy=("open_correct", _mean_safe),
            assistive_accuracy=("assistive_correct", _mean_safe),
            incremental_accuracy=("incremental_correct", _mean_safe),
            model_generated_accuracy=("model_generated_correct", _mean_safe),
            all_struct_accuracy=("all_structured_correct", _mean_safe),
        )
        .reset_index()
        .rename(columns={"num_consistent": "consistent_pairs"})
    )
    sort_fields = list(group_cols) + ["consistent_pairs"]
    grouped = grouped.sort_values(sort_fields)
    # Compute average across the three structured regimes where available.
    struct_cols = ["assistive_accuracy", "incremental_accuracy", "model_generated_accuracy"]
    grouped["structured_mean_accuracy"] = grouped[struct_cols].apply(
        lambda row: row.dropna().mean() if row.dropna().size else None, axis=1
    )
    return grouped


def compute_analysis_tables(results_dir: Path = RESULTS_ROOT) -> Dict[str, pd.DataFrame]:
    results_dir = Path(results_dir).resolve()
    correctness_df = load_correctness_dataframe(results_dir)
    consistency_df = load_consistency_dataframe(results_dir)
    generation_df = load_generation_counts(results_dir)
    baseline_overlap_df = load_baseline_reject_overlap(results_dir)
    steps_df = load_model_generated_steps(results_dir)
    coverage_df = _build_regime_coverage(generation_df, correctness_df, consistency_df)
    consistency_examples = build_consistency_correctness_examples(results_dir)

    tables: Dict[str, pd.DataFrame] = {
        "correctness_raw": correctness_df,
        "consistency_raw": consistency_df,
        "generation_counts": generation_df,
        "baseline_overlap": baseline_overlap_df,
        "model_generated_steps": steps_df,
        "regime_coverage": coverage_df,
        "correctness_by_dataset": _aggregate_correctness(correctness_df, ["dataset", "regime"]),
        "correctness_by_model": _aggregate_correctness(correctness_df, ["model", "regime"]),
        "consistency_by_dataset": _aggregate_consistency(consistency_df, ["dataset", "focused_regime"]).rename(columns={"focused_regime": "regime"}),
        "consistency_by_model": _aggregate_consistency(consistency_df, ["model", "focused_regime"]).rename(columns={"focused_regime": "regime"}),
        "rq1_deltas": _build_rq1_table(correctness_df),
        "consistency_vs_accuracy": _merge_consistency_correctness(consistency_df, correctness_df),
        "consistency_correctness_examples": consistency_examples,
    }

    if not tables["rq1_deltas"].empty:
        dataset_summary = (
            tables["rq1_deltas"].groupby(["dataset", "regime"], dropna=False)
            .agg(
                accuracy=("accuracy", "mean"),
                open_ended_accuracy=("open_ended_accuracy", "mean"),
                delta_vs_open=("delta_vs_open", "mean"),
                models=("model", "nunique"),
            )
            .reset_index()
        )
        tables["rq1_by_dataset"] = dataset_summary
    else:
        tables["rq1_by_dataset"] = pd.DataFrame(columns=["dataset", "regime", "accuracy", "open_ended_accuracy", "delta_vs_open", "models"])

    if not tables["consistency_vs_accuracy"].empty:
        dataset_group = (
            tables["consistency_vs_accuracy"].groupby(["dataset", "regime"], dropna=False)
            .agg(
                accuracy=("accuracy", "mean"),
                equivalence_rate=("equivalence_rate", "mean"),
                comparisons=("comparisons", "sum"),
                models=("model", "nunique"),
            )
            .reset_index()
        )
        tables["consistency_vs_accuracy_by_dataset"] = dataset_group
    else:
        tables["consistency_vs_accuracy_by_dataset"] = pd.DataFrame(columns=["dataset", "regime", "accuracy", "equivalence_rate", "comparisons", "models"])

    tables["consistency_correctness_buckets_overall"] = _bucketize_consistency_correctness(
        consistency_examples, []
    )
    tables["consistency_correctness_buckets_by_dataset"] = _bucketize_consistency_correctness(
        consistency_examples, ["dataset"]
    )
    tables["consistency_correctness_buckets_by_model"] = _bucketize_consistency_correctness(
        consistency_examples, ["model"]
    )
    tables["consistency_correctness_buckets_by_dataset_model"] = _bucketize_consistency_correctness(
        consistency_examples, ["dataset", "model"]
    )

    return tables
