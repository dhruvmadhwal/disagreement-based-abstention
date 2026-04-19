#!/usr/bin/env python3
"""Summarize baseline metrics across datasets/models.

This script scans `results/baselines/<dataset>/<model>/*.json` (excluding
IC-IDK pool files) and aggregates the summary metrics written by the baseline
pipelines (coverage, overall_accuracy, precision, recall, f1, etc.).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def _safe_load(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _float_close(a: object, b: object, tol: float = 1e-4) -> bool:
    return isinstance(a, (int, float)) and isinstance(b, (int, float)) and abs(float(a) - float(b)) <= tol


def _compute_detection_metrics(records: Sequence[dict]) -> Dict[str, float]:
    """Recompute detection-style metrics from raw records."""
    total = len(records)
    if total == 0:
        return {
            "total": 0,
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
            "coverage": 0.0,
            "overall_accuracy": 0.0,
            "accuracy_at_coverage": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

    tp = fp = tn = fn = 0
    for r in records:
        accepted = bool(r.get("accepted"))
        try:
            correct = int(r.get("correct"))
        except Exception:
            correct = 0
        is_correct = correct == 1
        if not accepted and not is_correct:
            tp += 1
        elif not accepted and is_correct:
            fp += 1
        elif accepted and not is_correct:
            fn += 1
        else:
            tn += 1

    coverage = (tn + fn) / total  # answered / total
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    # Overall accuracy: correct answered / total
    overall_acc = tn / total
    # Accuracy at coverage: correct answered / answered
    acc_at_cov = tn / (tn + fn) if (tn + fn) else 0.0

    def _rnd(x: float) -> float:
        return round(x, 4)

    return {
        "total": total,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "coverage": _rnd(coverage),
        "overall_accuracy": _rnd(overall_acc),
        "accuracy_at_coverage": _rnd(acc_at_cov),
        "precision": _rnd(precision),
        "recall": _rnd(recall),
        "f1": _rnd(f1),
    }


def _validate_accuracy_at_coverage(summary: Dict[str, object], payload: Dict[str, object]) -> Optional[float]:
    """Return validated accuracy@coverage if records match detection framing; otherwise None."""
    records = payload.get("records")
    if not isinstance(records, list):
        return None

    recomputed = _compute_detection_metrics(records)
    keys = ("coverage", "overall_accuracy", "precision", "recall", "f1")
    for key in keys:
        if not _float_close(summary.get(key), recomputed.get(key)):
            return None

    acc_at_cov = summary.get("accuracy_at_coverage")
    if acc_at_cov is None:
        return None
    if not _float_close(acc_at_cov, recomputed.get("accuracy_at_coverage")):
        return None
    return recomputed.get("accuracy_at_coverage")


def _iter_baseline_files(results_dir: Path) -> Iterable[Tuple[str, str, Path]]:
    """Yield (dataset, model, path) for baseline result JSONs."""
    base = results_dir / "baselines"
    if not base.exists():
        return
    for dataset_dir in sorted(d for d in base.iterdir() if d.is_dir()):
        dataset = dataset_dir.name
        for model_dir in sorted(d for d in dataset_dir.iterdir() if d.is_dir()):
            model = model_dir.name
            for json_file in sorted(model_dir.glob("*.json")):
                # Skip IC-IDK pool files.
                if "ic_idk_pools" in json_file.parts or "pool" in json_file.name:
                    continue
                yield dataset, model, json_file


def load_baseline_table(results_dir: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for dataset, model, path in _iter_baseline_files(results_dir):
        payload = _safe_load(path)
        if not isinstance(payload, dict):
            continue
        records = payload.get("records")
        summary = payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}
        if isinstance(records, list) and records:
            # Recompute metrics to keep definitions consistent and always populate acc@cov.
            summary = _compute_detection_metrics(records)
        if not summary:
            continue

        # Infer baseline name from filename: <dataset>_<baseline>.json
        stem = path.stem
        baseline = stem
        if stem.startswith(f"{dataset}_"):
            baseline = stem[len(dataset) + 1 :]
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "baseline": baseline,
                "path": str(path),
                **summary,
            }
        )
    return rows


def _group_stats(rows: List[Dict[str, object]], group_keys: List[str]) -> List[Dict[str, object]]:
    """Aggregate mean metrics by the provided keys."""
    from collections import defaultdict

    buckets: Dict[Tuple, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = tuple(row.get(k) for k in group_keys)
        buckets[key].append(row)

    metrics = ["coverage", "accuracy_at_coverage", "overall_accuracy", "precision", "recall", "f1"]
    aggregates: List[Dict[str, object]] = []
    for key, items in buckets.items():
        agg: Dict[str, object] = {k: v for k, v in zip(group_keys, key)}
        count = len(items)
        agg["models"] = count if "model" in group_keys else len({item.get("model") for item in items})
        for metric in metrics:
            vals = [item.get(metric) for item in items if isinstance(item.get(metric), (int, float))]
            agg[metric] = sum(vals) / len(vals) if vals else None
        aggregates.append(agg)
    aggregates.sort(key=lambda r: (r.get("dataset") or "", r.get("baseline") or "", r.get("model") or ""))
    return aggregates


def _print_table(rows: List[Dict[str, object]], title: str) -> None:
    if not rows:
        print(f"{title}: no data\n")
        return
    headers = [
        "dataset",
        "baseline",
        "model",
        "coverage",
        "accuracy_at_coverage",
        "overall_accuracy",
        "precision",
        "recall",
        "f1",
        "path",
    ]
    print(title)
    print("-" * len(title))
    for row in rows:
        line = []
        for h in headers:
            line.append(f"{h}={row.get(h)}")
        print(", ".join(line))
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate baseline performance metrics.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Root results directory (default: results/).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write a JSON payload with raw rows and aggregates.",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=None,
        help="Optional path to write a Markdown summary table.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("analysis/figures"),
        help="Directory to write summary figures (passed to generate_summary_figures).",
    )
    parser.add_argument(
        "--metrics-json",
        type=Path,
        default=Path("analysis/derived/summary_metrics.json"),
        help="Path for serialized metrics JSON (passed to generate_summary_figures).",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip running the figure generation pipeline.",
    )
    parser.add_argument(
        "--no-general-figures",
        action="store_true",
        help="Skip the general summary figures (still runs Mintaka figures unless disabled).",
    )
    parser.add_argument(
        "--no-mintaka-figures",
        action="store_true",
        help="Skip running the Mintaka-specific baseline figure script.",
    )
    parser.add_argument(
        "--no-baseline-bar-figures",
        action="store_true",
        help="Skip running the per-dataset baseline bar figure script.",
    )
    args = parser.parse_args()

    rows = load_baseline_table(args.results_dir)
    per_model = rows
    by_baseline = _group_stats(rows, ["dataset", "baseline"])
    by_model = _group_stats(rows, ["dataset", "model"])
    by_dataset = _group_stats(rows, ["dataset"])

    _print_table(per_model, "Per-model baselines")
    _print_table(by_baseline, "Averages by dataset & baseline")
    _print_table(by_dataset, "Averages by dataset (all baselines)")

    if args.output:
        payload = {
            "per_model": per_model,
            "by_baseline": by_baseline,
            "by_dataset": by_dataset,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2))
        print(f"Wrote summary JSON to {args.output}")
    if args.markdown:
        lines: List[str] = []
        lines.append("# Baseline Evaluation Results")
        lines.append("")
        lines.append(f"Source: {args.output or 'analysis/baseline_summary.py'}")
        lines.append("")
        # Per-model tables grouped by dataset/model
        grouped: Dict[tuple, List[Dict[str, object]]] = {}
        for row in per_model:
            key = (row.get("dataset"), row.get("model"))
            grouped.setdefault(key, []).append(row)
        for (dataset, model) in sorted(grouped):
            lines.append(f"## {dataset} — {model}")
            lines.append("")
            lines.append("| baseline | total | tp | fp | tn | fn | coverage | acc_at_cov | overall_acc | precision | recall | f1 | path |")
            lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
            for r in sorted(grouped[(dataset, model)], key=lambda x: str(x.get("baseline"))):
                def fmt(val):
                    return f"{val:.3f}" if isinstance(val, (int, float)) else (str(val) if val is not None else "")
                lines.append(
                    f"| {r.get('baseline','')} | {r.get('total','')} | {r.get('tp','')} | {r.get('fp','')} | "
                    f"{r.get('tn','')} | {r.get('fn','')} | {fmt(r.get('coverage'))} | {fmt(r.get('accuracy_at_coverage'))} | "
                    f"{fmt(r.get('overall_accuracy'))} | {fmt(r.get('precision'))} | {fmt(r.get('recall'))} | {fmt(r.get('f1'))} | "
                    f"{r.get('path','')} |"
                )
            lines.append("")
        # Averages by baseline
        if by_baseline:
            lines.append("## Averages by baseline")
            lines.append("")
            lines.append("| dataset | baseline | models | coverage | acc_at_cov | overall_acc | precision | recall | f1 |")
            lines.append("|---|---|---|---|---|---|---|---|---|")
            for r in sorted(by_baseline, key=lambda x: (str(x.get("dataset")), str(x.get("baseline")))):
                def fmt(val):
                    return f"{val:.3f}" if isinstance(val, (int, float)) else (str(val) if val is not None else "")
                lines.append(
                    f"| {r.get('dataset','')} | {r.get('baseline','')} | {r.get('models','')} | {fmt(r.get('coverage'))} | "
                    f"{fmt(r.get('accuracy_at_coverage'))} | {fmt(r.get('overall_accuracy'))} | {fmt(r.get('precision'))} | "
                    f"{fmt(r.get('recall'))} | {fmt(r.get('f1'))} |"
                )
            lines.append("")
        if by_model:
            lines.append("## Averages by model")
            lines.append("")
            lines.append("| dataset | model | baselines | coverage | acc_at_cov | overall_acc | precision | recall | f1 |")
            lines.append("|---|---|---|---|---|---|---|---|---|")
            for r in sorted(by_model, key=lambda x: (str(x.get("dataset")), str(x.get("model")))):
                def fmt(val):
                    return f"{val:.3f}" if isinstance(val, (int, float)) else (str(val) if val is not None else "")
                lines.append(
                    f"| {r.get('dataset','')} | {r.get('model','')} | {r.get('models','')} | {fmt(r.get('coverage'))} | "
                    f"{fmt(r.get('accuracy_at_coverage'))} | {fmt(r.get('overall_accuracy'))} | {fmt(r.get('precision'))} | "
                    f"{fmt(r.get('recall'))} | {fmt(r.get('f1'))} |"
                )
            lines.append("")
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text("\n".join(lines))
        print(f"Wrote Markdown summary to {args.markdown}")
    if not args.no_figures:
        cmd = [
            sys.executable,
            "-m",
            "analysis.generate_summary_figures",
            "--results-dir",
            str(args.results_dir),
            "--figures-dir",
            str(args.figures_dir),
            "--metrics-json",
            str(args.metrics_json),
        ]
        if not args.no_general_figures:
            print("Running figure generation:", " ".join(cmd))
            subprocess.run(cmd, check=True)
        if not args.no_mintaka_figures:
            mintaka_cmd = [
                sys.executable,
                str(Path(__file__).resolve().parents[0] / "plot_mintaka_baseline_figures.py"),
            ]
            print("Running Mintaka baseline figures:", " ".join(mintaka_cmd))
            subprocess.run(mintaka_cmd, check=True)
        if not args.no_baseline_bar_figures:
            bar_cmd = [
                sys.executable,
                str(Path(__file__).resolve().parents[0] / "plot_baseline_bars.py"),
            ]
            print("Running baseline bar figures:", " ".join(bar_cmd))
            subprocess.run(bar_cmd, check=True)


if __name__ == "__main__":
    main()
