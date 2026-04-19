#!/usr/bin/env python3
"""Create figures for assistive decomp mismatch behavior analysis."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DD_BUCKETS = (
    "DD_both_wrong",
    "DD_open_wrong_assistive_correct",
    "DD_open_correct_assistive_wrong",
    "DD_both_correct",
    "DD_missing_label",
)
LABEL_MAP = {
    "DD_both_wrong": "Both wrong",
    "DD_open_wrong_assistive_correct": "Open wrong, assistive correct",
    "DD_open_correct_assistive_wrong": "Open correct, assistive wrong",
    "DD_both_correct": "Both correct",
    "DD_missing_label": "Missing labels",
}
PALETTE = [
    "#4c78a8",
    "#72b7b2",
    "#f28e2b",
    "#54a24b",
    "#b7b7b7",
]


def _read_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "dataset",
        "consistency_model",
        "decomp_model",
        "total",
        "DD_count",
        "DD_rate",
        *DD_BUCKETS,
    }
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Summary file missing required columns: {sorted(missing)}")

    for column in ["total", "DD_count", *DD_BUCKETS]:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0).astype(int)
    df["DD_rate"] = pd.to_numeric(df["DD_rate"], errors="coerce").fillna(0.0)
    return df


def _aggregate(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    by_model = df.groupby("consistency_model", dropna=False)[["total", "DD_count", *DD_BUCKETS]].sum().reset_index()
    by_dataset = df.groupby("dataset", dropna=False)[["total", "DD_count", *DD_BUCKETS]].sum().reset_index()
    by_dataset_model = (
        df.groupby(["dataset", "consistency_model"], dropna=False)[["total", "DD_count", *DD_BUCKETS]]
        .sum()
        .reset_index()
    )

    for frame in (by_model, by_dataset, by_dataset_model):
        frame["DD_rate"] = frame["DD_count"] / frame["total"].replace(0, np.nan)
        frame["DD_rate"] = frame["DD_rate"].fillna(0.0)
        frame["labeled_dd"] = frame["DD_count"] - frame["DD_missing_label"]
        frame["label_coverage"] = (frame["labeled_dd"] / frame["DD_count"]).fillna(0.0)

        for bucket in DD_BUCKETS:
            denom = frame["labeled_dd"] if bucket != "DD_missing_label" else frame["DD_count"]
            frame[f"share_{bucket}"] = frame.apply(
                lambda row: row[bucket] / row[denom.name] if row[denom.name] > 0 else 0.0,
                axis=1,
            )

    return {
        "by_model": by_model.sort_values("DD_count", ascending=False),
        "by_dataset": by_dataset.sort_values("DD_count", ascending=False),
        "by_dataset_model": by_dataset_model,
    }


def _plot_stack_by_group(
    agg: pd.DataFrame,
    *,
    group_col: str,
    use_labeled_only: bool,
    output: Path,
    title: str,
) -> None:
    use_buckets = DD_BUCKETS if not use_labeled_only else DD_BUCKETS[:-1]
    denom = "labeled_dd" if use_labeled_only else "DD_count"
    y = np.arange(len(agg))
    width = 0.7

    fig, ax = plt.subplots(figsize=(max(10, len(agg) * 0.75), 5))
    bottom = np.zeros(len(agg), dtype=float)

    for bucket, color in zip(use_buckets, PALETTE):
        values = agg[f"share_{bucket}"].to_numpy()
        ax.bar(
            y,
            values,
            width,
            bottom=bottom,
            label=LABEL_MAP[bucket],
            color=color,
        )
        bottom += values

    ax.set_xticks(y)
    ax.set_xticklabels(agg[group_col], rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Composition")
    ax.set_title(title)
    ax.set_xticks(y)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2)
    ax.grid(axis="y", alpha=0.2)

    for idx, row in enumerate(agg.itertuples(index=False)):
        dd_n = int(getattr(row, "DD_count"))
        dd_rate = getattr(row, "DD_rate")
        lbl_n = int(getattr(row, "labeled_dd"))
        ax.text(
            idx,
            1.01,
            f"N={dd_n}\nL={lbl_n}\nDD {dd_rate:.1%}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_dd_rate_heatmap(agg: pd.DataFrame, *, output: Path, title: str, fmt: str = "{:.0%}") -> None:
    heat = agg.pivot(index="dataset", columns="consistency_model", values="DD_rate")
    if heat.empty:
        return
    # Sort for readability and consistent appearance.
    heat = heat.loc[sorted(heat.index), sorted(heat.columns)]
    data = heat.to_numpy()

    fig, ax = plt.subplots(figsize=(max(10, heat.shape[1] * 0.85), max(4, heat.shape[0] * 0.55)))
    im = ax.imshow(data, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(heat.shape[1]))
    ax.set_yticks(np.arange(heat.shape[0]))
    ax.set_xticklabels(heat.columns, rotation=45, ha="right")
    ax.set_yticklabels(heat.index)
    ax.set_title(title)

    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            value = heat.iat[i, j]
            if np.isnan(value):
                continue
            ax.text(
                j,
                i,
                fmt.format(value),
                ha="center",
                va="center",
                color="white" if value > 0.6 else "black",
                fontsize=7,
            )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("DD rate")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_coverage_bar(agg: pd.DataFrame, *, output: Path, title: str, sort_by: str = "label_coverage") -> None:
    ordered = agg.sort_values(sort_by)
    y = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(max(9, len(ordered) * 0.6), 4.5))

    ax.bar(y, ordered["label_coverage"], color="#4c78a8")
    ax.set_xticks(y)
    ax.set_xticklabels(ordered["consistency_model"], rotation=35, ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Labeled DD share")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)

    for idx, row in enumerate(ordered.itertuples(index=False)):
        miss = int(row.DD_missing_label)
        lbl = int(row.labeled_dd)
        ax.text(idx, max(0.01, row.label_coverage + 0.02), f"L={lbl}\nM={miss}", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_summary(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    agg = _aggregate(df)
    agg["by_model"].to_csv(output_dir / "assistive_mismatch_by_model_aggregated.csv", index=False)
    agg["by_dataset"].to_csv(output_dir / "assistive_mismatch_by_dataset_aggregated.csv", index=False)

    # Labeled-only composition shows the corrected behavior in matched cases.
    _plot_stack_by_group(
        agg["by_model"],
        group_col="consistency_model",
        use_labeled_only=True,
        output=output_dir / "dd_bucket_composition_by_model_labeled.png",
        title="DD bucket composition by model (labeled DD only)",
    )

    _plot_stack_by_group(
        agg["by_dataset"],
        group_col="dataset",
        use_labeled_only=True,
        output=output_dir / "dd_bucket_composition_by_dataset_labeled.png",
        title="DD bucket composition by dataset (labeled DD only)",
    )

    # Missing-label awareness in practice is central to this analysis.
    _plot_stack_by_group(
        agg["by_model"],
        group_col="consistency_model",
        use_labeled_only=False,
        output=output_dir / "dd_bucket_composition_by_model_with_missing.png",
        title="DD bucket composition by model (including missing labels)",
    )

    _plot_dd_rate_heatmap(
        agg["by_dataset_model"],
        output=output_dir / "dd_rate_heatmap_dataset_x_model.png",
        title="DD rate by dataset and consistency model",
        fmt="{:.1%}",
    )

    _plot_coverage_bar(
        agg["by_model"],
        output=output_dir / "dd_label_coverage_by_model.png",
        title="DD label coverage by consistency model",
    )

    for file in sorted(output_dir.glob("dd_*.png")):
        print(file)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot assistive mismatch analysis outputs.")
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("analysis/derived/assistive_decomp_answer_mismatch_summary.csv"),
        help="Path to assistive_decomp_answer_mismatch_summary.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/figures/decomp_assistive_mismatch"),
        help="Output directory for plots.",
    )
    args = parser.parse_args()

    df = _read_summary(args.summary)
    _write_summary(df, args.output_dir)


if __name__ == "__main__":
    main()
