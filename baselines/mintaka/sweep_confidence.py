#!/usr/bin/env python3
"""Sweep confidence thresholds post-hoc and report the best F1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from baselines.mintaka.judge import CorrectnessJudge


def load_records(path: Path) -> List[dict]:
    payload = json.loads(path.read_text())
    records = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError(f"Unexpected payload format in {path}")
    return records


def ensure_labels(records: List[dict], judge: Optional[CorrectnessJudge] = None) -> List[Tuple[float, int]]:
    """Return list of (confidence, label) with label in {0,1}.

    If a judge is provided, we re-score every prediction to avoid relying on
    stored labels (which may be missing or all zeros when threshold was unset).
    """
    items: List[Tuple[float, int]] = []
    for rec in records:
        conf = rec.get("confidence")
        if conf is None:
            continue
        if judge is not None:
            question = rec.get("question", "")
            gold = rec.get("gold_answer", "")
            pred = rec.get("prediction", "")
            res = judge.score(question, gold, pred)
            label = 1 if res.correct == 1 else 0
        else:
            label = rec.get("correct")
            if label not in (0, 1):
                # Treat missing as incorrect to keep sweep offline if no judge provided.
                label = 0
        items.append((float(conf), int(label)))
    return items


def compute_metrics(threshold: float, items: List[Tuple[float, int]]) -> Dict[str, float]:
    accepted = [(c, y) for c, y in items if c >= threshold]
    total = len(items)
    if total == 0:
        return {"coverage": 0.0, "acc_at_cov": 0.0, "overall_acc": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    cov = len(accepted) / total
    if not accepted:
        return {"coverage": cov, "acc_at_cov": 0.0, "overall_acc": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    tp = sum(1 for _, y in accepted if y == 1)
    fp = len(accepted) - tp
    fn = sum(1 for _, y in items if y == 1) - tp
    acc_at_cov = tp / len(accepted)
    overall_acc = tp / total
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "coverage": round(cov, 4),
        "acc_at_cov": round(acc_at_cov, 4),
        "overall_acc": round(overall_acc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def sweep(items: List[Tuple[float, int]]) -> Tuple[float, Dict[str, float]]:
    thresholds = sorted({c for c, _ in items})
    if not thresholds:
        return 0.0, compute_metrics(0.0, items)
    # Add a low buffer and a high buffer
    thresholds = [thresholds[0] - 1e-4] + thresholds + [thresholds[-1] + 1e-4]
    best_thr = thresholds[0]
    best_metrics = compute_metrics(best_thr, items)
    for thr in thresholds[1:]:
        metrics = compute_metrics(thr, items)
        if metrics["f1"] > best_metrics["f1"] or (
            metrics["f1"] == best_metrics["f1"] and metrics["coverage"] > best_metrics["coverage"]
        ):
            best_thr = thr
            best_metrics = metrics
    return best_thr, best_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep confidence thresholds for Mintaka confidence baseline.")
    parser.add_argument(
        "--file",
        type=Path,
        default=Path("results/baselines/mintaka/google-gemini-2-5-flash/mintaka_confidence.json"),
        help="Path to confidence baseline JSON with confidence scores.",
    )
    parser.add_argument(
        "--model",
        default="google/gemini-2.5-flash",
        help="Judge model name (passes through to CorrectnessJudge).",
    )
    parser.add_argument("--temperature", type=float, default=0.1, help="Judge temperature.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_records(args.file)
    judge = CorrectnessJudge(model=args.model, temperature=args.temperature)
    items = ensure_labels(records, judge)
    if not items:
        print("No records with confidence scores found.")
        return
    best_thr, metrics = sweep(items)
    print(f"Best threshold: {best_thr:.6f}")
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()
