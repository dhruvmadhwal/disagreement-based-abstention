#!/usr/bin/env python3
"""
Generate a presentable table summarizing token overhead for DBA (Assistive).

Reads token summaries produced by `generate_answers/run_*_regimes.py`:
  results/generation/<dataset>/<model>/token_summary.json

For each dataset, we compute:
  - mean open-ended total tokens/question (prompt+completion)
  - mean assistive total tokens/question
  - mean DBA(A) multiplier = (open+assistive)/open
  - min/max multiplier over available models

Outputs:
  - LaTeX table to tables/efficiency_overhead_assistive.tex (default)
  - Markdown table to results/efficiency_overhead_assistive.md (default)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Tuple


DEFAULT_DATASETS = ("bamboogle", "crag", "hotpotqa", "mintaka")

DISPLAY_DATASET: Dict[str, str] = {
    "bamboogle": "Bamboogle",
    "crag": "CRAG",
    "hotpotqa": "HotpotQA",
    "mintaka": "Mintaka",
}


def _load_json(path: Path) -> Optional[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _total_tokens(token_summary: dict, regime: str) -> Optional[float]:
    av = (token_summary.get("averages") or {}).get(regime)
    if not isinstance(av, dict):
        return None
    try:
        return float(av.get("avg_in", 0.0)) + float(av.get("avg_out", 0.0))
    except Exception:
        return None


def _fmt_int(x: float) -> str:
    return f"{int(round(x)):,}"


def _fmt_mult(x: float) -> str:
    return f"{x:.2f}"


def _collect_dataset_rows(dataset: str) -> List[Tuple[str, float, float, float, float, float, float, float]]:
    """Return per-model rows.

    Schema:
        (model, open_prompt, open_completion, assist_prompt, assist_completion,
         dba_prompt_mult, dba_completion_mult, dba_total_mult)
    """
    ds_dir = Path("results/generation") / dataset
    if not ds_dir.exists():
        return []
    rows: List[Tuple[str, float, float, float, float, float, float, float]] = []
    for token_path in ds_dir.glob("*/token_summary.json"):
        model = token_path.parent.name
        summary = _load_json(token_path)
        if not summary:
            continue
        open_stats = (summary.get("averages") or {}).get("open_ended")
        assist_stats = (summary.get("averages") or {}).get("assistive")
        if not isinstance(open_stats, dict) or not isinstance(assist_stats, dict):
            continue
        try:
            open_prompt = float(open_stats.get("avg_in", 0.0))
            open_completion = float(open_stats.get("avg_out", 0.0))
            assist_prompt = float(assist_stats.get("avg_in", 0.0))
            assist_completion = float(assist_stats.get("avg_out", 0.0))
        except Exception:
            continue
        open_total = open_prompt + open_completion
        assist_total = assist_prompt + assist_completion
        if open_prompt <= 0 or open_completion <= 0 or open_total <= 0:
            continue
        dba_prompt_mult = (open_prompt + assist_prompt) / open_prompt
        dba_completion_mult = (open_completion + assist_completion) / open_completion
        dba_total_mult = (open_total + assist_total) / open_total
        rows.append(
            (
                model,
                open_prompt,
                open_completion,
                assist_prompt,
                assist_completion,
                dba_prompt_mult,
                dba_completion_mult,
                dba_total_mult,
            )
        )
    return rows


def _to_markdown(rows: List[dict]) -> str:
    headers = list(rows[0].keys()) if rows else []
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        out.append("| " + " | ".join(str(r[h]) for h in headers) + " |")
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DBA(A) efficiency table (assistive).")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help="Datasets to include (default: bamboogle crag hotpotqa mintaka).",
    )
    parser.add_argument(
        "--out-tex",
        type=Path,
        default=Path("tables/efficiency_overhead_assistive.tex"),
        help="Output path for LaTeX table.",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("results/efficiency_overhead_assistive.md"),
        help="Output path for Markdown table.",
    )
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets if d.strip()]
    if not datasets:
        raise SystemExit("No datasets provided.")

    table_rows: List[dict] = []
    overall_open: List[float] = []
    overall_open_prompt: List[float] = []
    overall_open_completion: List[float] = []
    overall_mult_prompt: List[float] = []
    overall_mult_completion: List[float] = []
    overall_mult_total: List[float] = []

    for ds in datasets:
        per_model = _collect_dataset_rows(ds)
        if not per_model:
            continue
        open_prompt = [r[1] for r in per_model]
        open_completion = [r[2] for r in per_model]
        mult_prompt = [r[5] for r in per_model]
        mult_completion = [r[6] for r in per_model]
        mult_total = [r[7] for r in per_model]

        overall_open_prompt.extend(open_prompt)
        overall_open_completion.extend(open_completion)
        overall_mult_prompt.extend(mult_prompt)
        overall_mult_completion.extend(mult_completion)
        overall_mult_total.extend(mult_total)

        table_rows.append(
            {
                "Dataset": DISPLAY_DATASET.get(ds, ds),
                "Models": len(per_model),
                "Open prompt": _fmt_int(mean(open_prompt)),
                "Open compl.": _fmt_int(mean(open_completion)),
                "DBA(A)/Open prompt×": _fmt_mult(mean(mult_prompt)),
                "DBA(A)/Open compl.×": _fmt_mult(mean(mult_completion)),
                "DBA(A)/Open total×": _fmt_mult(mean(mult_total)),
            }
        )

    if overall_mult_total:
        table_rows.append(
            {
                "Dataset": "Average",
                "Models": len(overall_mult_total),
                "Open prompt": _fmt_int(mean(overall_open_prompt)),
                "Open compl.": _fmt_int(mean(overall_open_completion)),
                "DBA(A)/Open prompt×": _fmt_mult(mean(overall_mult_prompt)),
                "DBA(A)/Open compl.×": _fmt_mult(mean(overall_mult_completion)),
                "DBA(A)/Open total×": _fmt_mult(mean(overall_mult_total)),
            }
        )

    # Markdown
    md = "# DBA(A) token overhead (Assistive)\n\n" + _to_markdown(table_rows) if table_rows else ""
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(md, encoding="utf-8")

    # LaTeX
    # Note: keep this table self-contained and aligned with other booktabs tables in /tables.
    latex_lines: List[str] = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Token overhead for DBA in the Assistive setting. DBA(A) generates both an open-ended and an assistive answer. We report prompt and completion tokens separately, along with multipliers $m_{\\text{prompt}}=(P_{\\text{open}}+P_{\\text{assist}})/P_{\\text{open}}$, $m_{\\text{compl}}=(C_{\\text{open}}+C_{\\text{assist}})/C_{\\text{open}}$, and $m_{\\text{total}}=(T_{\\text{open}}+T_{\\text{assist}})/T_{\\text{open}}$. Tokens are averaged over the available models per dataset.}",
        "\\label{tab:efficiency_overhead_assistive}",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "\\textbf{Dataset} & \\textbf{\\#Models} & \\textbf{Open (prompt)} & \\textbf{Open (compl.)} & \\textbf{DBA(A)/Open (prompt$\\times$)} & \\textbf{DBA(A)/Open (compl.$\\times$)} & \\textbf{DBA(A)/Open (total$\\times$)} \\\\",
        "\\midrule",
    ]
    for r in table_rows:
        if r["Dataset"] == "Average":
            latex_lines.append("\\midrule")
            dataset_label = "\\textit{Average}"
        else:
            dataset_label = str(r["Dataset"])
        latex_lines.append(
            f"{dataset_label} & {r['Models']} & {r['Open prompt']} & {r['Open compl.']} & {r['DBA(A)/Open prompt×']} & {r['DBA(A)/Open compl.×']} & {r['DBA(A)/Open total×']} \\\\"
        )
    latex_lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    args.out_tex.parent.mkdir(parents=True, exist_ok=True)
    args.out_tex.write_text("\n".join(latex_lines), encoding="utf-8")

    print(f"Wrote {args.out_tex}")
    print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()
