#!/usr/bin/env python3
"""Plot baseline rejection overlap heatmaps (cross-method) per dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze.summary_utils import load_baseline_reject_overlap


DEFAULT_ORDER = [
    "ays",
    "idk",
    "ic_idk",
    "pairwise_assistive",
    "pairwise_incremental",
    "pairwise_model_generated",
    "self_consistency",
]


def _pivot_matrix(df: pd.DataFrame, baselines: List[str], value_col: str) -> np.ndarray:
    matrix = np.full((len(baselines), len(baselines)), np.nan, dtype=float)
    lookup = df.set_index(["baseline_a", "baseline_b"])[value_col].to_dict()
    for i, a in enumerate(baselines):
        for j, b in enumerate(baselines):
            if a == b:
                matrix[i, j] = 1.0 if value_col == "jaccard" else np.nan
                continue
            val = lookup.get((a, b))
            if isinstance(val, (int, float)):
                matrix[i, j] = float(val)
    return matrix


def _plot_heatmap(ax, data: np.ndarray, labels: List[str], title: str) -> None:
    im = ax.imshow(data, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.set_ylim(len(labels) - 0.5, -0.5)
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = data[i, j]
            if np.isnan(val):
                continue
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="white", fontsize=7)
    return im


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot baseline rejection overlap heatmaps.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Root results directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analyze/figures"),
        help="Directory to save heatmaps.",
    )
    parser.add_argument(
        "--order",
        nargs="*",
        default=DEFAULT_ORDER,
        help="Baseline order for heatmaps.",
    )
    args = parser.parse_args()

    df = load_baseline_reject_overlap(args.results_dir)
    if df.empty:
        raise SystemExit("No baseline overlap data found.")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for dataset in sorted(df["dataset"].unique()):
        subset = df[df["dataset"] == dataset]
        if subset.empty:
            continue
        # Average across models for the heatmap.
        grouped = (
            subset.groupby(["baseline_a", "baseline_b"], dropna=False)
            .agg(jaccard=("jaccard", "mean"), overlap_precision=("overlap_precision", "mean"))
            .reset_index()
        )
        baselines = [b for b in args.order if b in set(subset["baseline_a"]) | set(subset["baseline_b"])]
        if not baselines:
            baselines = sorted(set(subset["baseline_a"]) | set(subset["baseline_b"]))

        jaccard_mat = _pivot_matrix(grouped, baselines, "jaccard")
        precision_mat = _pivot_matrix(grouped, baselines, "overlap_precision")

        fig, axes = plt.subplots(1, 2, figsize=(1.2 * len(baselines) + 3, 5))
        im0 = _plot_heatmap(axes[0], jaccard_mat, baselines, f"{dataset}: reject Jaccard")
        im1 = _plot_heatmap(axes[1], precision_mat, baselines, f"{dataset}: overlap precision (incorrect)")
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        fig.tight_layout()
        out_path = output_dir / f"{dataset}_baseline_reject_overlap.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
