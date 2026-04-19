#!/usr/bin/env python3
"""Summarize decomposition equivalence metrics across datasets/models.

This script scans `decomposition_equivalence/results/summary/<dataset>/<model>.json`
and aggregates the equivalence metrics comparing model-generated decompositions
against gold (human-annotated) decompositions.

Usage:
    python analysis/decomposition_equivalence_summary.py
    python analysis/decomposition_equivalence_summary.py --output results/decomposition_equivalence_results.md
    python analysis/decomposition_equivalence_summary.py --format json --output analysis/derived/decomposition_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DECOMP_RESULTS = REPO_ROOT / "decomposition_equivalence" / "results"


def _safe_load(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _rnd(x: float, decimals: int = 3) -> float:
    return round(x, decimals)


def iter_summary_files() -> Iterable[Tuple[str, str, Path]]:
    """Yield (dataset, model, path) for summary JSON files."""
    summary_dir = DECOMP_RESULTS / "summary"
    if not summary_dir.exists():
        return
    for dataset_dir in sorted(d for d in summary_dir.iterdir() if d.is_dir()):
        dataset = dataset_dir.name
        for json_file in sorted(dataset_dir.glob("*.json")):
            model = json_file.stem
            yield dataset, model, json_file


def load_decomposition_table() -> List[Dict[str, Any]]:
    """Load all decomposition equivalence summaries into a list of dicts."""
    rows: List[Dict[str, Any]] = []
    for dataset, model, path in iter_summary_files():
        data = _safe_load(path)
        if not isinstance(data, dict):
            continue
        rows.append({
            "dataset": dataset,
            "model": model,
            "path": str(path),
            "total": data.get("total", 0),
            "equivalence_rate": data.get("equivalence_rate", 0),
            "precision": data.get("precision_macro", 0),
            "recall": data.get("recall_macro", 0),
            "f1": data.get("f1_macro", 0),
            "avg_gold_hops": data.get("average_gold_hops", 0),
            "avg_model_hops": data.get("average_model_hops", 0),
            "hop_ratio": data.get("average_hop_ratio", 0),
            "hop_diff": data.get("average_hop_diff", 0),
            "equiv_count": data.get("equivalence_counts", {}).get("equivalent", 0),
            "not_equiv_count": data.get("equivalence_counts", {}).get("not_equivalent", 0),
            "uncertain_count": data.get("equivalence_counts", {}).get("uncertain", 0),
        })
    return rows


def aggregate_by_model(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate metrics across datasets for each model."""
    model_data: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        model_data[r["model"]].append(r)

    results = []
    for model in sorted(model_data.keys()):
        rs = model_data[model]
        n = len(rs)
        results.append({
            "model": model,
            "datasets": n,
            "avg_equivalence_rate": _rnd(sum(r["equivalence_rate"] for r in rs) / n),
            "avg_precision": _rnd(sum(r["precision"] for r in rs) / n),
            "avg_recall": _rnd(sum(r["recall"] for r in rs) / n),
            "avg_f1": _rnd(sum(r["f1"] for r in rs) / n),
            "avg_hop_ratio": _rnd(sum(r["hop_ratio"] for r in rs) / n),
        })
    return results


def aggregate_by_dataset(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate metrics across models for each dataset."""
    dataset_data: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        dataset_data[r["dataset"]].append(r)

    results = []
    for dataset in sorted(dataset_data.keys()):
        rs = dataset_data[dataset]
        n = len(rs)
        results.append({
            "dataset": dataset,
            "models": n,
            "avg_equivalence_rate": _rnd(sum(r["equivalence_rate"] for r in rs) / n),
            "avg_precision": _rnd(sum(r["precision"] for r in rs) / n),
            "avg_recall": _rnd(sum(r["recall"] for r in rs) / n),
            "avg_f1": _rnd(sum(r["f1"] for r in rs) / n),
            "avg_gold_hops": _rnd(sum(r["avg_gold_hops"] for r in rs) / n, 2),
            "avg_model_hops": _rnd(sum(r["avg_model_hops"] for r in rs) / n, 2),
        })
    return results


def format_markdown(rows: List[Dict[str, Any]]) -> str:
    """Format results as markdown tables."""
    lines = [
        "# Decomposition Equivalence Results",
        "",
        "Source: decomposition_equivalence/results/summary/",
        "",
        "This analysis compares model-generated question decompositions against gold (human-annotated) decompositions.",
        "",
        "## Key Metrics",
        "",
        "- **equivalence_rate**: % of questions where model decomposition is semantically equivalent to gold",
        "- **precision**: How many model hops match gold hops (from model's perspective)",
        "- **recall**: How many gold hops are covered by model hops (from gold's perspective)",
        "- **f1**: Harmonic mean of precision and recall",
        "- **hop_ratio**: avg(model_hops / gold_hops) - 1.0 means same number of hops",
        "- **avg_gold_hops / avg_model_hops**: Average number of reasoning steps",
        "",
    ]

    # Group by dataset
    datasets = sorted(set(r["dataset"] for r in rows))

    for dataset in datasets:
        ds_rows = [r for r in rows if r["dataset"] == dataset]
        if not ds_rows:
            continue

        lines.append(f"## {dataset}")
        lines.append("")
        lines.append("| model | total | equiv_rate | precision | recall | f1 | gold_hops | model_hops | hop_ratio |")
        lines.append("|---|---|---|---|---|---|---|---|---|")

        for r in sorted(ds_rows, key=lambda x: x["model"]):
            lines.append(
                f"| {r['model']} | {r['total']} | {r['equivalence_rate']:.3f} | "
                f"{r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | "
                f"{r['avg_gold_hops']:.2f} | {r['avg_model_hops']:.2f} | {r['hop_ratio']:.3f} |"
            )
        lines.append("")

    # Averages by model
    model_agg = aggregate_by_model(rows)
    lines.append("## Averages by Model (across all datasets)")
    lines.append("")
    lines.append("| model | datasets | avg_equiv_rate | avg_precision | avg_recall | avg_f1 | avg_hop_ratio |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in model_agg:
        lines.append(
            f"| {r['model']} | {r['datasets']} | {r['avg_equivalence_rate']:.3f} | "
            f"{r['avg_precision']:.3f} | {r['avg_recall']:.3f} | {r['avg_f1']:.3f} | "
            f"{r['avg_hop_ratio']:.3f} |"
        )
    lines.append("")

    # Averages by dataset
    dataset_agg = aggregate_by_dataset(rows)
    lines.append("## Averages by Dataset (across all models)")
    lines.append("")
    lines.append("| dataset | models | avg_equiv_rate | avg_precision | avg_recall | avg_f1 | avg_gold_hops | avg_model_hops |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in dataset_agg:
        lines.append(
            f"| {r['dataset']} | {r['models']} | {r['avg_equivalence_rate']:.3f} | "
            f"{r['avg_precision']:.3f} | {r['avg_recall']:.3f} | {r['avg_f1']:.3f} | "
            f"{r['avg_gold_hops']:.2f} | {r['avg_model_hops']:.2f} |"
        )
    lines.append("")

    return "\n".join(lines)


def format_json(rows: List[Dict[str, Any]]) -> str:
    """Format results as JSON."""
    return json.dumps({
        "source": "decomposition_equivalence/results/summary/",
        "detailed": rows,
        "by_model": aggregate_by_model(rows),
        "by_dataset": aggregate_by_dataset(rows),
    }, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize decomposition equivalence metrics"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output file path (default: stdout)"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format (default: markdown)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rows = load_decomposition_table()
    if not rows:
        print("No decomposition equivalence results found.", file=sys.stderr)
        sys.exit(1)

    if args.format == "json":
        output = format_json(rows)
    else:
        output = format_markdown(rows)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
        print(f"Written to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
