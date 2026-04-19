#!/usr/bin/env python3
"""Format semantic self-consistency sweep TSV into a LaTeX table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


DEFAULT_MODELS = [
    "qwen-qwen3-8b",
    "qwen-qwen3-32b",
    "meta-llama-llama-3-3-70b-instruct",
    "google-gemini-2-5-pro",
]

MODEL_DISPLAY = {
    "qwen-qwen3-8b": "Qwen3-8B",
    "qwen-qwen3-32b": "Qwen3-32B",
    "meta-llama-llama-3-3-70b-instruct": "Llama 3.3 70B",
    "google-gemini-2-5-pro": "Gemini Pro",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a LaTeX table from semantic self-consistency sweep TSV.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/analysis/bamboogle_self_consistency_semantic_sweep.tsv"),
        help="Input TSV path from scripts/self_consistency_semantic_sweep.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tables/bamboogle_self_consistency_semantic_sweep.tex"),
        help="Output LaTeX table path",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help="Model slugs (row order). Missing rows are rendered as --.",
    )
    parser.add_argument("--caption", default=None, help="Optional caption override.")
    parser.add_argument("--label", default="tab:bamboogle_self_consistency_semantic_sweep", help="LaTeX label.")
    return parser.parse_args()


def fmt_float(val: str, *, ndigits: int = 3) -> str:
    if val is None:
        return "--"
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return "--"
    try:
        f = float(s)
    except ValueError:
        return "--"
    return f"{f:.{ndigits}f}"


def fmt_int(val: str) -> str:
    if val is None:
        return "--"
    s = str(val).strip()
    if not s:
        return "--"
    try:
        return str(int(float(s)))
    except ValueError:
        return "--"


def load_rows(path: Path) -> Dict[str, Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows: Dict[str, Dict[str, str]] = {}
        for row in reader:
            model = (row.get("model") or "").strip()
            if model:
                rows[model] = row
        return rows


def main() -> None:
    args = parse_args()
    rows_by_model = load_rows(args.input)

    caption = args.caption
    if caption is None:
        caption = (
            "Semantic self-consistency threshold sweep on Bamboogle. "
            "$m$ is the size of the largest semantic cluster among 7 open-ended samples (IDK excluded), "
            "with clustering judged by Gemini 2.5 Flash. "
            "AUROC uses $7-m$ as the score (higher disagreement indicates higher error likelihood). "
            "$t^*$ is the threshold on $m$ that maximizes F1; @4 is the fixed 4/7 operating point."
        )

    lines: List[str] = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength\tabcolsep{3pt}")
    lines.append(r"\caption{" + caption + r"}")
    lines.append(r"\label{" + args.label + r"}")
    lines.append(r"\begin{tabular}{lccccccccccc}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Model} & \textbf{AUROC} & \textbf{$t^*$} & "
        r"\textbf{P} & \textbf{R} & \textbf{F1} & \textbf{Cov} & "
        r"\textbf{P@4} & \textbf{R@4} & \textbf{F1@4} & \textbf{Cov@4} \\"
    )
    lines.append(r"\midrule")

    for model in args.models:
        row = rows_by_model.get(model)
        display = MODEL_DISPLAY.get(model, model)
        if row is None:
            lines.append(display + r" & -- & -- & -- & -- & -- & -- & -- & -- & -- & -- \\")
            continue
        lines.append(
            " & ".join(
                [
                    display,
                    fmt_float(row.get("auroc"), ndigits=3),
                    fmt_int(row.get("best_t")),
                    fmt_float(row.get("best_precision"), ndigits=3),
                    fmt_float(row.get("best_recall"), ndigits=3),
                    fmt_float(row.get("best_f1"), ndigits=3),
                    fmt_float(row.get("best_coverage"), ndigits=3),
                    fmt_float(row.get("t4_precision"), ndigits=3),
                    fmt_float(row.get("t4_recall"), ndigits=3),
                    fmt_float(row.get("t4_f1"), ndigits=3),
                    fmt_float(row.get("t4_coverage"), ndigits=3),
                ]
            )
            + r" \\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    lines.append("")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()

