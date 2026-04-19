"""CLI for building consistency/correctness summary tables."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

try:
    from .generate_summary_figures import aggregate_metrics, build_figures
    from .summary_utils import compute_analysis_tables
except ImportError:  # Allow direct script execution.
    import sys as _sys
    from pathlib import Path as _Path
    _REPO_ROOT = _Path(__file__).resolve().parents[1]
    if str(_REPO_ROOT) not in _sys.path:
        _sys.path.append(str(_REPO_ROOT))
    from analysis.generate_summary_figures import aggregate_metrics, build_figures  # type: ignore
    from analysis.summary_utils import compute_analysis_tables  # type: ignore


def _format_float(value: Optional[float], decimals: int = 3) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.{decimals}f}"


def _format_value(value) -> str:
    if value is None or pd.isna(value):
        return "—"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _format_bool(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    return "✅" if bool(value) else "—"


def _prepare_table(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    float_cols: Optional[Iterable[str]] = None,
    bool_cols: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=columns or [])
    display = df.copy()
    if columns:
        valid_cols = [col for col in columns if col in display.columns]
        display = display[valid_cols]
    if float_cols:
        for col in float_cols:
            if col in display.columns:
                display[col] = display[col].apply(lambda val: _format_float(val))
    if bool_cols:
        for col in bool_cols:
            if col in display.columns:
                display[col] = display[col].apply(_format_bool)
    for col in display.columns:
        display[col] = display[col].apply(_format_value)
    return display


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data available._"
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        values = [str(row[col]) for col in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _build_raw_consistency_table(consistency_df: pd.DataFrame, coverage_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "dataset",
        "model",
        "regime",
        "comparisons",
        "equivalent",
        "non_equivalent",
        "not_meaningful",
        "equivalence_rate",
    ]
    if coverage_df.empty:
        return pd.DataFrame(columns=columns)
    metrics = consistency_df.copy()
    metrics = metrics.dropna(subset=["focused_regime"])
    if metrics.empty:
        merged = coverage_df[coverage_df["has_generation"]][["dataset", "model", "regime"]].copy()
        return merged.assign(
            comparisons=None,
            equivalent=None,
            non_equivalent=None,
            not_meaningful=None,
            equivalence_rate=None,
        )
    metrics = metrics.rename(columns={"focused_regime": "regime"})
    metrics = metrics[[
        "dataset",
        "model",
        "regime",
        "comparisons",
        "equivalent",
        "non_equivalent",
        "not_meaningful",
        "equivalence_rate",
    ]]
    base = coverage_df[coverage_df["has_generation"]][["dataset", "model", "regime"]]
    merged = base.merge(metrics, on=["dataset", "model", "regime"], how="left")
    merged = merged[merged["regime"] != "open_ended"]
    return merged[columns]


def _build_raw_correctness_table(correctness_df: pd.DataFrame, coverage_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "dataset",
        "model",
        "regime",
        "evaluated",
        "correct",
        "incorrect",
        "not_meaningful",
        "accuracy",
    ]
    if coverage_df.empty:
        return pd.DataFrame(columns=columns)
    metrics = correctness_df[[
        "dataset",
        "model",
        "regime",
        "evaluated",
        "correct",
        "incorrect",
        "not_meaningful",
        "accuracy",
    ]].copy()
    base = coverage_df[coverage_df["has_generation"]][["dataset", "model", "regime"]]
    merged = base.merge(metrics, on=["dataset", "model", "regime"], how="left")
    return merged[columns]


def _build_rq1_summary_text(rq1_df: pd.DataFrame) -> List[str]:
    if rq1_df.empty:
        return ["No paired correctness vs open-ended regimes detected."]
    lines: List[str] = []
    for dataset in sorted(rq1_df["dataset"].unique()):
        section = rq1_df[rq1_df["dataset"] == dataset]
        parts: List[str] = []
        for regime in ["assistive", "incremental", "model_generated"]:
            row = section[section["regime"] == regime]
            if row.empty:
                continue
            value = row.iloc[0]["delta_vs_open"]
            parts.append(f"{regime}: {_format_float(value)} Δ accuracy")
        if parts:
            lines.append(f"- {dataset}: " + "; ".join(parts))
    return lines if lines else ["No comparable regimes available."]


def _compute_correlation(data: pd.DataFrame) -> Dict[str, str]:
    def corr_value(frame: pd.DataFrame) -> Optional[float]:
        subset = frame.dropna(subset=["accuracy", "equivalence_rate"])
        if len(subset) < 2:
            return None
        return subset["accuracy"].corr(subset["equivalence_rate"])

    overall = corr_value(data)
    details = {"overall": _format_float(overall) if overall is not None else "—"}
    for dataset in sorted(data["dataset"].unique()):
        dataset_corr = corr_value(data[data["dataset"] == dataset])
        details[dataset] = _format_float(dataset_corr) if dataset_corr is not None else "—"
    return details


def build_markdown(tables: Dict[str, pd.DataFrame]) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
    lines: List[str] = [
        "# Consistency and Correctness Evaluation Summary",
        "",
        "_Auto-generated by `analysis/run_analysis.py`. Do not edit by hand._",
        f"_Generated on {timestamp}._",
        "",
    ]

    coverage_df = tables.get("regime_coverage", pd.DataFrame())
    raw_consistency = _prepare_table(
        _build_raw_consistency_table(tables["consistency_raw"], coverage_df),
        float_cols=["equivalence_rate"],
    )
    raw_correctness = _prepare_table(
        _build_raw_correctness_table(tables["correctness_raw"], coverage_df),
        float_cols=["accuracy"],
    )

    sections = [
        ("Raw consistency vs open-ended", raw_consistency),
        ("Raw correctness vs DSL gold", raw_correctness),
        ("Consistency aggregated by dataset", _prepare_table(tables["consistency_by_dataset"], float_cols=["equivalence_rate"])),
        ("Consistency aggregated by model", _prepare_table(tables["consistency_by_model"], float_cols=["equivalence_rate"])),
        ("Correctness aggregated by dataset", _prepare_table(tables["correctness_by_dataset"], float_cols=["accuracy"])),
        ("Correctness aggregated by model", _prepare_table(tables["correctness_by_model"], float_cols=["accuracy"])),
        ("Baseline rejection overlap (cross-method)", _prepare_table(
            tables.get("baseline_overlap", pd.DataFrame()),
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
            ],
            float_cols=["overlap_pct_a", "overlap_pct_b", "jaccard", "overlap_precision", "overlap_recall"],
        )),
        ("Model-generated plan steps per setup", _prepare_table(
            tables["model_generated_steps"],
            columns=[
                "dataset",
                "model",
                "examples",
                "with_chain",
                "avg_steps",
                "median_steps",
                "avg_dsl_steps",
                "median_dsl_steps",
            ],
            float_cols=["avg_steps", "median_steps", "avg_dsl_steps", "median_dsl_steps"],
        )),
    ]

    for title, df in sections:
        lines.append(f"## {title}")
        lines.append("")
        lines.append(_dataframe_to_markdown(df))
        lines.append("")

    # Coverage summary omitted for now

    # RQ1
    rq1_model = _prepare_table(
        tables["rq1_deltas"][[
            "dataset",
            "model",
            "regime",
            "evaluated",
            "correct",
            "incorrect",
            "not_meaningful",
            "accuracy",
            "open_ended_accuracy",
            "delta_vs_open",
        ]],
        float_cols=["accuracy", "open_ended_accuracy", "delta_vs_open"],
    )
    rq1_dataset = _prepare_table(
        tables["rq1_by_dataset"],
        float_cols=["accuracy", "open_ended_accuracy", "delta_vs_open"],
    )
    lines.append("## Research Question 1 — Do structured regimes beat open-ended?")
    lines.append("")
    lines.append("### Per model deltas")
    lines.append(_dataframe_to_markdown(rq1_model))
    lines.append("")
    lines.append("### Aggregate deltas by dataset")
    lines.append(_dataframe_to_markdown(rq1_dataset))
    lines.append("")
    lines.append("Key takeaways:")
    lines.extend(_build_rq1_summary_text(tables["rq1_by_dataset"]))
    lines.append("")

    # RQ2
    rq2_model = _prepare_table(
        tables["consistency_vs_accuracy"],
        float_cols=["accuracy", "equivalence_rate"],
    )
    rq2_dataset = _prepare_table(
        tables["consistency_vs_accuracy_by_dataset"],
        float_cols=["accuracy", "equivalence_rate"],
    )
    lines.append("## Research Question 2 — Does consistency signal correctness?")
    lines.append("")
    lines.append("### Per setup consistency vs accuracy")
    lines.append(_dataframe_to_markdown(rq2_model))
    lines.append("")
    lines.append("### Aggregate view by dataset")
    lines.append(_dataframe_to_markdown(rq2_dataset))
    lines.append("")
    corr_details = _compute_correlation(tables["consistency_vs_accuracy"])
    lines.append("Correlation (Pearson r) between equivalence rate and accuracy:")
    for label, value in corr_details.items():
        lines.append(f"- {label}: {value}")

    return "\n".join(lines).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build evaluation summary tables and figures.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing evaluation outputs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/evaluation_summary.md"),
        help="Path to the Markdown file to write.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("analysis/figures"),
        help="Directory to store generated figures.",
    )
    parser.add_argument(
        "--metrics-json",
        type=Path,
        default=Path("analysis/derived/summary_metrics.json"),
        help="Path to the serialized aggregate metrics JSON file.",
    )
    args = parser.parse_args()

    tables = compute_analysis_tables(args.results_dir)
    markdown = build_markdown(tables)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8")
    print(f"Wrote summary to {args.output}")

    frames, serializable = aggregate_metrics(args.results_dir, tables=tables)
    args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_json.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    print(f"Wrote aggregate metrics to {args.metrics_json}")

    figure_paths = build_figures(frames, args.figures_dir)
    if figure_paths:
        for path in figure_paths:
            print(f"Wrote figure to {path}")
    else:
        print("No figures generated (insufficient data).")


if __name__ == "__main__":
    main()
