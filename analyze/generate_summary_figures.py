#!/usr/bin/env python3
"""Build aggregate QA metrics and helper figures for the paper."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .summary_utils import DEFAULT_REGIME_ORDER, canonicalize_regime, compute_analysis_tables

FOCUS_REGIMES = ["open_ended", "assistive", "incremental", "model_generated"]
REGIME_PLOT_ORDER = ["assistive", "incremental", "model_generated", "open_ended"]
REGIME_DISPLAY_NAMES = {
    "open_ended": "Open-ended",
    "assistive": "Assistive",
    "incremental": "Sequential",
    "model_generated": "Model-generated plan",
}
BAR_COLORS = {
    "accuracy": "#4c78a8",
    "consistency": "#72b7b2",
}


def _make_cmap(name: str) -> matplotlib.colors.Colormap:
    base = plt.cm.get_cmap(name)
    if hasattr(base, "colors"):
        # ListedColormap exposes colors directly.
        new_map = matplotlib.colors.ListedColormap(base.colors)
    else:
        new_map = matplotlib.colors.ListedColormap(base(np.linspace(0, 1, 256)))
    new_map.set_bad("#f2f2f2")
    return new_map


def _prepare_correctness_df(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=["dataset", "model", "regime", "evaluated", "correct"])
    df = raw.copy()
    df["regime"] = df["regime"].apply(canonicalize_regime)
    return df[df["regime"].isin(FOCUS_REGIMES)]


def _prepare_consistency_df(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=["dataset", "model", "regime", "comparisons", "equivalent"])
    df = raw.copy()
    df["regime"] = df["focused_regime"].apply(canonicalize_regime)
    return df[df["regime"].isin(FOCUS_REGIMES)]


def _aggregate_accuracy(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=group_cols + ["correct", "evaluated", "accuracy"])
    grouped = (
        df.groupby(group_cols, dropna=False)[["correct", "evaluated"]]
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
        return pd.DataFrame(columns=group_cols + ["equivalent", "comparisons", "equivalence_rate"])
    grouped = (
        df.groupby(group_cols, dropna=False)[["equivalent", "comparisons"]]
        .sum()
        .reset_index()
    )
    grouped["equivalence_rate"] = grouped.apply(
        lambda row: (row["equivalent"] / row["comparisons"]) if row["comparisons"] else None,
        axis=1,
    )
    return grouped


def _order_by_regime(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "regime" not in df.columns:
        return df
    order = {reg: idx for idx, reg in enumerate(DEFAULT_REGIME_ORDER)}
    df = df.copy()
    df["__order"] = df["regime"].map(lambda reg: order.get(reg, len(order)))
    ordered = df.sort_values("__order").drop(columns="__order")
    return ordered


def _display_regime(regime: Optional[str]) -> str:
    if not regime:
        return "Unknown"
    canonical = canonicalize_regime(regime)
    if not canonical:
        return "Unknown"
    return REGIME_DISPLAY_NAMES.get(canonical, canonical.replace("_", " ").title())


def _prepare_regime_summary(acc: pd.DataFrame, cons: pd.DataFrame) -> pd.DataFrame:
    acc_view = acc[["regime", "accuracy"]] if not acc.empty else pd.DataFrame(columns=["regime", "accuracy"])
    cons_view = cons[["regime", "equivalence_rate"]] if not cons.empty else pd.DataFrame(columns=["regime", "equivalence_rate"])
    merged = acc_view.merge(cons_view, on="regime", how="outer")
    if merged.empty:
        base = pd.DataFrame({"regime": REGIME_PLOT_ORDER})
        base["accuracy"] = None
        base["equivalence_rate"] = None
        base["display"] = base["regime"].apply(_display_regime)
        return base
    merged["regime"] = merged["regime"].apply(canonicalize_regime)
    merged = (
        merged.groupby("regime", dropna=False)[["accuracy", "equivalence_rate"]]
        .max()
        .reset_index()
    )
    merged = merged.set_index("regime").reindex(REGIME_PLOT_ORDER).reset_index()
    merged["display"] = merged["regime"].apply(_display_regime)
    return merged


def _records(df: pd.DataFrame) -> List[Dict[str, Optional[float]]]:
    if df.empty:
        return []
    rows: List[Dict[str, Optional[float]]] = []
    for record in df.to_dict("records"):
        cleaned: Dict[str, Optional[float]] = {}
        for key, value in record.items():
            if isinstance(value, float):
                cleaned[key] = None if math.isnan(value) else round(value, 6)
            elif isinstance(value, (int, str)):
                cleaned[key] = value
            elif pd.isna(value):
                cleaned[key] = None
            else:
                cleaned[key] = value
        rows.append(cleaned)
    return rows


def aggregate_metrics(
    results_dir: Path, tables: Optional[Dict[str, pd.DataFrame]] = None
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, List[dict]]]:
    if tables is None:
        tables = compute_analysis_tables(results_dir)
    correctness = _prepare_correctness_df(tables["correctness_raw"])
    consistency = _prepare_consistency_df(tables["consistency_raw"])

    frames: Dict[str, pd.DataFrame] = {
        "accuracy_by_regime": _order_by_regime(_aggregate_accuracy(correctness, ["regime"])),
        "consistency_by_regime": _order_by_regime(_aggregate_consistency(consistency, ["regime"])),
        "accuracy_by_dataset": _aggregate_accuracy(correctness, ["dataset"]).sort_values("accuracy", ascending=False),
        "consistency_by_dataset": _aggregate_consistency(consistency, ["dataset"]).sort_values("equivalence_rate", ascending=False),
        "accuracy_by_model": _aggregate_accuracy(correctness, ["model"]).sort_values("accuracy", ascending=False),
        "consistency_by_model": _aggregate_consistency(consistency, ["model"]).sort_values("equivalence_rate", ascending=False),
        "accuracy_by_dataset_regime": _order_by_regime(_aggregate_accuracy(correctness, ["dataset", "regime"])),
        "consistency_by_dataset_regime": _order_by_regime(_aggregate_consistency(consistency, ["dataset", "regime"])),
        "accuracy_by_model_regime": _order_by_regime(_aggregate_accuracy(correctness, ["model", "regime"])),
        "consistency_by_model_regime": _order_by_regime(_aggregate_consistency(consistency, ["model", "regime"])),
        "accuracy_by_dataset_model": _aggregate_accuracy(correctness, ["dataset", "model"]),
        "consistency_by_dataset_model": _aggregate_consistency(consistency, ["dataset", "model"]),
        "consistency_correctness_buckets_overall": tables.get("consistency_correctness_buckets_overall", pd.DataFrame()),
        "consistency_correctness_buckets_by_dataset": tables.get("consistency_correctness_buckets_by_dataset", pd.DataFrame()),
        "consistency_correctness_buckets_by_model": tables.get("consistency_correctness_buckets_by_model", pd.DataFrame()),
        "consistency_correctness_buckets_by_dataset_model": tables.get("consistency_correctness_buckets_by_dataset_model", pd.DataFrame()),
    }

    serializable = {name: _records(df) for name, df in frames.items()}
    return frames, serializable


def _plot_regime_figure(frames: Dict[str, pd.DataFrame], output_dir: Path) -> Optional[Path]:
    acc = frames.get("accuracy_by_regime", pd.DataFrame())
    cons = frames.get("consistency_by_regime", pd.DataFrame())
    summary = _prepare_regime_summary(acc, cons)
    if summary[["accuracy", "equivalence_rate"]].dropna(how="all").empty:
        return None

    x_positions = list(range(len(summary)))
    width = 0.4
    fig, ax = plt.subplots(figsize=(10, 5))

    acc_values = summary["accuracy"].fillna(0)
    cons_values = summary["equivalence_rate"].fillna(0)
    acc_bars = ax.bar(
        [pos - width / 2 for pos in x_positions],
        acc_values,
        width,
        label="Accuracy",
        color=BAR_COLORS["accuracy"],
    )
    cons_bars = ax.bar(
        [pos + width / 2 for pos in x_positions],
        cons_values,
        width,
        label="Consistency vs open-ended",
        color=BAR_COLORS["consistency"],
    )

    # Highlight the top regime for each metric.
    valid_acc = summary["accuracy"].dropna()
    if not valid_acc.empty:
        top_acc_idx = valid_acc.idxmax()
        if 0 <= top_acc_idx < len(acc_bars):
            acc_bars[top_acc_idx].set_edgecolor("#222222")
            acc_bars[top_acc_idx].set_linewidth(1.4)
    valid_cons = summary["equivalence_rate"].dropna()
    if not valid_cons.empty:
        top_cons_idx = valid_cons.idxmax()
        if 0 <= top_cons_idx < len(cons_bars):
            cons_bars[top_cons_idx].set_edgecolor("#222222")
            cons_bars[top_cons_idx].set_linewidth(1.4)

    for bar, val in zip(acc_bars, summary["accuracy"]):
        if pd.isna(val):
            ax.text(bar.get_x() + bar.get_width() / 2, 0.02, "N/A", ha="center", va="bottom", fontsize=8, color="#666666")
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.1%}", ha="center", va="bottom", fontsize=9)
    for bar, val in zip(cons_bars, summary["equivalence_rate"]):
        if pd.isna(val):
            bar.set_hatch("//")
            ax.text(bar.get_x() + bar.get_width() / 2, 0.02, "N/A", ha="center", va="bottom", fontsize=8, color="#666666")
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.1%}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(summary["display"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score (0-1)")
    ax.set_xlabel("Approach")
    ax.set_title("Q1 & Q2: Four approaches vs. accuracy and consistency", pad=16, fontsize=12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=2, frameon=False)
    ax.text(
        0.5,
        -0.18,
        "Consistency bars quantify agreement with open-ended answers; open-ended itself has no pairwise consistency.",
        transform=ax.transAxes,
        ha="center",
        fontsize=8,
        color="#555555",
    )

    fig.tight_layout()
    out_path = output_dir / "q1_q2_approach_comparison.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _align_second_frame(base: pd.DataFrame, other: pd.DataFrame, key: str) -> pd.DataFrame:
    if base.empty:
        return pd.DataFrame(columns=other.columns)
    aligned = other.set_index(key).reindex(base[key]).reset_index()
    return aligned


def _plot_dataset_figure(frames: Dict[str, pd.DataFrame], output_dir: Path) -> Optional[Path]:
    acc = frames["accuracy_by_dataset"]
    cons = frames["consistency_by_dataset"]
    if acc.empty:
        return None
    acc_ordered = acc.sort_values("accuracy")
    cons_aligned = _align_second_frame(acc_ordered, cons, "dataset")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    bars_acc = axes[0].barh(acc_ordered["dataset"], acc_ordered["accuracy"], color=BAR_COLORS["accuracy"])
    axes[0].set_xlim(0, 1)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Accuracy")
    axes[0].set_title("Q3: Benchmark performance (ordered by accuracy)")
    for bar in bars_acc:
        axes[0].text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2, f"{bar.get_width():.1%}", va="center")

    if cons.empty:
        axes[1].axis("off")
    else:
        cons_vals = cons_aligned["equivalence_rate"].fillna(0)
        bars_cons = axes[1].barh(acc_ordered["dataset"], cons_vals, color=BAR_COLORS["consistency"])
        axes[1].set_xlim(0, 1)
        axes[1].set_xlabel("Equivalence Rate")
        axes[1].set_title("Q3: Benchmark consistency vs open-ended")
        for bar, val in zip(bars_cons, cons_aligned["equivalence_rate"]):
            label = "N/A" if pd.isna(val) else f"{bar.get_width():.1%}"
            axes[1].text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2, label, va="center")

    fig.tight_layout()
    out_path = output_dir / "q3_benchmark_breakdown.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_model_figure(frames: Dict[str, pd.DataFrame], output_dir: Path) -> Optional[Path]:
    acc = frames["accuracy_by_model"]
    cons = frames["consistency_by_model"]
    if acc.empty and cons.empty:
        return None
    if cons.empty:
        ordered_acc = acc.sort_values("accuracy", ascending=False)
        cons_aligned = _align_second_frame(ordered_acc, cons, "model")
    else:
        ordered_acc = _align_second_frame(cons.sort_values("equivalence_rate", ascending=False), acc, "model")
        cons_aligned = cons.sort_values("equivalence_rate", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    bars_acc = axes[0].barh(ordered_acc["model"], ordered_acc["accuracy"], color=BAR_COLORS["accuracy"])
    axes[0].set_xlim(0, 1)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Accuracy")
    acc_title_suffix = " (ordered by consistency)" if not cons.empty else ""
    axes[0].set_title(f"Q4: Model accuracy{acc_title_suffix}")
    for bar, val in zip(bars_acc, ordered_acc["accuracy"]):
        label = "N/A" if pd.isna(val) else f"{bar.get_width():.1%}"
        axes[0].text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2, label, va="center")

    if cons.empty:
        axes[1].axis("off")
    else:
        cons_vals = cons_aligned["equivalence_rate"].fillna(0)
        bars_cons = axes[1].barh(cons_aligned["model"], cons_vals, color=BAR_COLORS["consistency"])
        axes[1].set_xlim(0, 1)
        axes[1].set_xlabel("Equivalence Rate")
        axes[1].set_title("Q4: Model consistency (highest → lowest)")
        for bar, val in zip(bars_cons, cons_aligned["equivalence_rate"]):
            label = "N/A" if pd.isna(val) else f"{bar.get_width():.1%}"
            axes[1].text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2, label, va="center")

    fig.tight_layout()
    out_path = output_dir / "q4_model_rankings.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_consistency_correctness_overall(frames: Dict[str, pd.DataFrame], output_dir: Path) -> Optional[Path]:
    df = frames.get("consistency_correctness_buckets_overall", pd.DataFrame())
    if df.empty:
        return None
    ordered = df.sort_values("consistent_pairs")
    buckets = ordered["consistent_pairs"].tolist()
    x_positions = list(range(len(buckets)))
    width = 0.18

    fig, ax = plt.subplots(figsize=(11, 5))
    series = [
        ("Open-ended", ordered["open_accuracy"], BAR_COLORS["accuracy"], -1.5),
        ("Assistive", ordered["assistive_accuracy"], "#e15759", -0.5),
        ("Sequential", ordered["incremental_accuracy"], "#59a14f", 0.5),
        ("Model-generated", ordered["model_generated_accuracy"], "#f28e2b", 1.5),
    ]
    for label, vals, color, offset in series:
        positions = [x + offset * width for x in x_positions]
        bars = ax.bar(positions, vals.fillna(0), width, label=label, color=color)
        for bar, acc, count in zip(bars, vals, ordered["examples"]):
            label_txt = "N/A" if pd.isna(acc) else f"{acc:.0%}"
            ax.text(bar.get_x() + bar.get_width() / 2, (acc or 0) + 0.012, label_txt, ha="center", va="bottom", fontsize=8)
            ax.text(bar.get_x() + bar.get_width() / 2, 0.005, f"n={count}", ha="center", va="bottom", fontsize=7, rotation=90)

    ax.plot(
        x_positions,
        ordered["structured_mean_accuracy"].fillna(0),
        color="#2f4b7c",
        linestyle="--",
        marker="o",
        label="Mean of three structured regimes",
    )
    ax.plot(
        x_positions,
        ordered["all_struct_accuracy"].fillna(0),
        color="#b30000",
        linestyle="-.",
        marker="s",
        label="All structured correct",
    )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{b} pairs consistent" for b in buckets])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Correctness")
    ax.set_title("Consistency vs correctness (examples with all three pairs)")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    fig.tight_layout()
    out_path = output_dir / "consistency_correctness_overall.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path

def _prepare_heatmap_matrix(
    df: pd.DataFrame,
    row_key: str,
    col_key: str,
    value_key: str,
    row_order: Optional[List[str]] = None,
    col_order: Optional[List[str]] = None,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    pivot = df.pivot(index=row_key, columns=col_key, values=value_key)
    if row_order:
        ordered_rows = [row for row in row_order if row in pivot.index]
        if ordered_rows:
            pivot = pivot.reindex(ordered_rows)
    if col_order:
        pivot = pivot.reindex(columns=col_order)
    return pivot


def _draw_heatmap(ax, data: pd.DataFrame, title: str, cbar_label: str, cmap_name: str) -> None:
    if data.empty:
        ax.axis("off")
        ax.set_title(f"{title}\n(No data available)", fontsize=11, pad=10)
        return
    cmap = _make_cmap(cmap_name)
    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels(data.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels(data.index)
    ax.set_title(title, fontsize=11, pad=12)
    for i, row_label in enumerate(data.index):
        for j, col_label in enumerate(data.columns):
            val = data.iloc[i, j]
            if pd.isna(val):
                label = "N/A"
                color = "#555555"
            else:
                label = f"{val:.0%}"
                color = "#0d0d0d" if val < 0.65 else "#f7f7f7"
            ax.text(j, i, label, ha="center", va="center", fontsize=8, color=color)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, rotation=90)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _plot_dataset_regime_heatmaps(frames: Dict[str, pd.DataFrame], output_dir: Path) -> Optional[Path]:
    acc = frames.get("accuracy_by_dataset_regime", pd.DataFrame())
    cons = frames.get("consistency_by_dataset_regime", pd.DataFrame())
    if acc.empty:
        return None
    dataset_order = frames.get("accuracy_by_dataset", pd.DataFrame())
    dataset_labels = (
        dataset_order["dataset"].tolist()
        if not dataset_order.empty
        else sorted(acc["dataset"].unique())
    )
    acc_matrix = _prepare_heatmap_matrix(
        acc, "dataset", "regime", "accuracy", dataset_labels, REGIME_PLOT_ORDER
    )
    cons_matrix = _prepare_heatmap_matrix(
        cons, "dataset", "regime", "equivalence_rate", dataset_labels, REGIME_PLOT_ORDER
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    _draw_heatmap(
        axes[0],
        acc_matrix,
        "Accuracy by benchmark and approach",
        "Accuracy",
        "Blues",
    )
    _draw_heatmap(
        axes[1],
        cons_matrix,
        "Consistency vs open-ended by benchmark and approach",
        "Equivalence rate",
        "BuGn",
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    out_path = output_dir / "benchmark_approach_heatmaps.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_dataset_model_heatmaps(frames: Dict[str, pd.DataFrame], output_dir: Path) -> Optional[Path]:
    acc = frames.get("accuracy_by_dataset_model", pd.DataFrame())
    cons = frames.get("consistency_by_dataset_model", pd.DataFrame())
    if acc.empty and cons.empty:
        return None
    dataset_order = frames.get("accuracy_by_dataset", pd.DataFrame())
    dataset_labels = (
        dataset_order["dataset"].tolist()
        if not dataset_order.empty
        else sorted(acc["dataset"].unique() if not acc.empty else cons["dataset"].unique())
    )
    model_order = frames.get("consistency_by_model", pd.DataFrame())
    model_labels = (
        model_order["model"].tolist()
        if not model_order.empty
        else sorted(acc["model"].unique() if not acc.empty else cons["model"].unique())
    )

    acc_matrix = _prepare_heatmap_matrix(
        acc, "dataset", "model", "accuracy", dataset_labels, model_labels
    )
    cons_matrix = _prepare_heatmap_matrix(
        cons, "dataset", "model", "equivalence_rate", dataset_labels, model_labels
    )

    fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharey=True)
    _draw_heatmap(
        axes[0],
        acc_matrix,
        "Accuracy by benchmark × model",
        "Accuracy",
        "Purples",
    )
    _draw_heatmap(
        axes[1],
        cons_matrix,
        "Consistency vs open-ended by benchmark × model",
        "Equivalence rate",
        "Greens",
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    out_path = output_dir / "benchmark_model_heatmaps.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_model_regime_heatmaps(frames: Dict[str, pd.DataFrame], output_dir: Path) -> Optional[Path]:
    acc = frames.get("accuracy_by_model_regime", pd.DataFrame())
    cons = frames.get("consistency_by_model_regime", pd.DataFrame())
    if acc.empty and cons.empty:
        return None

    model_order = frames.get("consistency_by_model", pd.DataFrame())
    model_labels = (
        model_order["model"].tolist()
        if not model_order.empty
        else sorted(acc["model"].unique() if not acc.empty else cons["model"].unique())
    )
    acc_matrix = _prepare_heatmap_matrix(acc, "model", "regime", "accuracy", model_labels, REGIME_PLOT_ORDER)
    cons_matrix = _prepare_heatmap_matrix(cons, "model", "regime", "equivalence_rate", model_labels, REGIME_PLOT_ORDER)

    fig, axes = plt.subplots(1, 2, figsize=(14, max(6, len(model_labels) * 0.45)), sharey=True)
    _draw_heatmap(
        axes[0],
        acc_matrix,
        "Accuracy by model × approach",
        "Accuracy",
        "Blues",
    )
    _draw_heatmap(
        axes[1],
        cons_matrix,
        "Consistency vs open-ended by model × approach",
        "Equivalence rate",
        "GnBu",
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.14)
    out_path = output_dir / "model_regime_heatmaps.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_consistency_correctness_dataset_heatmap(frames: Dict[str, pd.DataFrame], output_dir: Path) -> Optional[Path]:
    df = frames.get("consistency_correctness_buckets_by_dataset", pd.DataFrame())
    if df.empty:
        return None
    bucket_labels = sorted(df["consistent_pairs"].dropna().unique())
    dataset_labels = sorted(df["dataset"].unique())

    matrices = [
        ("Open-ended", "open_accuracy", "PuBu"),
        ("Assistive", "assistive_accuracy", "OrRd"),
        ("Sequential", "incremental_accuracy", "YlGn"),
        ("Model-generated", "model_generated_accuracy", "YlOrBr"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=True)
    axes = axes.flatten()
    for ax, (title, field, cmap) in zip(axes, matrices):
        mat = _prepare_heatmap_matrix(df, "dataset", "consistent_pairs", field, dataset_labels, bucket_labels)
        _draw_heatmap(
            ax,
            mat,
            f"{title} accuracy by benchmark × consistent pairs",
            "Accuracy",
            cmap,
        )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.12)
    out_path = output_dir / "consistency_correctness_by_dataset.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_consistency_correctness_benchmark_panels(frames: Dict[str, pd.DataFrame], output_dir: Path) -> Optional[Path]:
    df = frames.get("consistency_correctness_buckets_by_dataset", pd.DataFrame())
    if df.empty:
        return None
    datasets = sorted(df["dataset"].unique())
    cols = 3
    rows = int(math.ceil(len(datasets) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.4, rows * 3.6), sharey=True)
    axes = axes.flatten()
    for ax in axes[len(datasets) :]:
        ax.axis("off")
    for ax, dataset in zip(axes, datasets):
        subset = df[df["dataset"] == dataset].sort_values("consistent_pairs")
        if subset.empty:
            ax.axis("off")
            continue
        x = subset["consistent_pairs"].tolist()
        width = 0.16
        series = [
            ("Open", subset["open_accuracy"], BAR_COLORS["accuracy"], -1.5),
            ("Assistive", subset["assistive_accuracy"], "#e15759", -0.5),
            ("Seq", subset["incremental_accuracy"], "#59a14f", 0.5),
            ("Model-gen", subset["model_generated_accuracy"], "#f28e2b", 1.5),
        ]
        for label, vals, color, offset in series:
            positions = [xi + offset * width for xi in x]
            ax.bar(positions, vals.fillna(0), width, color=color, label=label)
            for pos, acc, n in zip(positions, vals, subset["examples"]):
                ax.text(pos, (acc or 0) + 0.01, "N/A" if pd.isna(acc) else f"{acc:.0%}", ha="center", va="bottom", fontsize=7)
                ax.text(pos, 0.02, f"n={n}", ha="center", va="bottom", fontsize=7, rotation=90)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(v)}" for v in x])
        ax.set_ylim(0, 1.05)
        ax.set_title(dataset)
    axes[0].set_ylabel("Correctness")
    axes[min(len(datasets) - 1, 1)].legend(loc="upper right", fontsize=8, frameon=False, ncol=2)
    fig.suptitle("Consistency vs correctness by benchmark (pairs consistent)", fontsize=12, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = output_dir / "consistency_correctness_by_benchmark_panels.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_consistency_correctness_model_panels(frames: Dict[str, pd.DataFrame], output_dir: Path) -> Optional[Path]:
    df = frames.get("consistency_correctness_buckets_by_model", pd.DataFrame())
    if df.empty:
        return None
    models = sorted(df["model"].unique())
    cols = 3
    rows = int(math.ceil(len(models) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.6, rows * 3.6), sharey=True)
    axes = axes.flatten()
    for ax in axes[len(models) :]:
        ax.axis("off")
    for ax, model in zip(axes, models):
        subset = df[df["model"] == model].sort_values("consistent_pairs")
        if subset.empty:
            ax.axis("off")
            continue
        x = subset["consistent_pairs"].tolist()
        width = 0.16
        series = [
            ("Open", subset["open_accuracy"], BAR_COLORS["accuracy"], -1.5),
            ("Assistive", subset["assistive_accuracy"], "#e15759", -0.5),
            ("Seq", subset["incremental_accuracy"], "#59a14f", 0.5),
            ("Model-gen", subset["model_generated_accuracy"], "#f28e2b", 1.5),
        ]
        for label, vals, color, offset in series:
            positions = [xi + offset * width for xi in x]
            ax.bar(positions, vals.fillna(0), width, color=color, label=label)
            for pos, acc, n in zip(positions, vals, subset["examples"]):
                ax.text(pos, (acc or 0) + 0.01, "N/A" if pd.isna(acc) else f"{acc:.0%}", ha="center", va="bottom", fontsize=7)
                ax.text(pos, 0.02, f"n={n}", ha="center", va="bottom", fontsize=7, rotation=90)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(v)}" for v in x])
        ax.set_ylim(0, 1.05)
        ax.set_title(model)
    axes[0].set_ylabel("Correctness")
    axes[min(len(models) - 1, 1)].legend(loc="upper right", fontsize=8, frameon=False, ncol=2)
    fig.suptitle("Consistency vs correctness by model (pairs consistent)", fontsize=12, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = output_dir / "consistency_correctness_by_model_panels.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_figures(frames: Dict[str, pd.DataFrame], output_dir: Path) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for builder in (
        _plot_regime_figure,
        _plot_dataset_figure,
        _plot_model_figure,
        _plot_dataset_regime_heatmaps,
        _plot_dataset_model_heatmaps,
        _plot_consistency_correctness_overall,
        _plot_consistency_correctness_dataset_heatmap,
        _plot_consistency_correctness_benchmark_panels,
        _plot_consistency_correctness_model_panels,
        _plot_model_regime_heatmaps,
    ):
        path = builder(frames, output_dir)
        if path:
            paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Create aggregate metrics and paper-ready figures.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"), help="Directory containing experiment outputs.")
    parser.add_argument("--figures-dir", type=Path, default=Path("analysis/figures"), help="Where to store the generated figures.")
    parser.add_argument("--metrics-json", type=Path, default=Path("analysis/derived/summary_metrics.json"), help="Path to the serialized metrics JSON.")
    args = parser.parse_args()

    frames, serializable = aggregate_metrics(args.results_dir)
    metrics_path = args.metrics_json
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")

    figure_paths = build_figures(frames, args.figures_dir)
    print(f"Wrote metrics to {metrics_path}")
    for path in figure_paths:
        print(f"Wrote figure to {path}")


if __name__ == "__main__":
    main()
