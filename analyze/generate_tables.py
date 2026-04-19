#!/usr/bin/env python3
"""Generate LaTeX tables for ACL submission."""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_TABLES_DIR = REPO_ROOT.parent / "Consistency-in-Multi-Hop-QA" / "latex" / "tables"

DATASET_ORDER = ['bamboogle', 'mintaka', 'hotpotqa', 'crag']
DATASET_ORDER_ALL = ['bamboogle', 'mintaka', 'hotpotqa', 'crag', 'frames', 'musique']
DATASET_DISPLAY = {
    'bamboogle': 'Bamboogle',
    'crag': 'CRAG',
    'hotpotqa': 'HotpotQA',
    'mintaka': 'Mintaka',
    'frames': 'FRAMES',
    'musique': 'MuSiQue',
}

BASELINE_ORDER = [
    'ays', 'idk',
]
PROPOSED_ORDER = [
    'pairwise_assistive', 'pairwise_incremental',
    'ensemble_pairwise_both',
    'ensemble_ays_pairwise_assistive', 'ensemble_ays_pairwise_incremental',
    'ensemble_idk_pairwise_assistive', 'ensemble_idk_pairwise_incremental',
    'ensemble_ays_idk_pairwise_assistive', 'ensemble_ays_idk_pairwise_incremental',
    'ensemble_ays_idk_pairwise_both',
]
ALL_BASELINES_ORDER = BASELINE_ORDER + PROPOSED_ORDER
BASELINE_DISPLAY = {
    'ays': 'Are You Sure (AYS)',
    'idk': "I Don't Know (IDK)",
    'pairwise_assistive': 'Disagree (A)',
    'pairwise_incremental': 'Disagree (S)',
    'ensemble_pairwise_both': 'Disagree (A) $\\cup$ Disagree (S)',
    'ensemble_ays_pairwise_assistive': 'AYS $\\cup$ Disagree (A)',
    'ensemble_ays_pairwise_incremental': 'AYS $\\cup$ Disagree (S)',
    'ensemble_idk_pairwise_assistive': 'IDK $\\cup$ Disagree (A)',
    'ensemble_idk_pairwise_incremental': 'IDK $\\cup$ Disagree (S)',
    'ensemble_ays_idk_pairwise_assistive': 'AYS $\\cup$ IDK $\\cup$ Disagree (A)',
    'ensemble_ays_idk_pairwise_incremental': 'AYS $\\cup$ IDK $\\cup$ Disagree (S)',
    'ensemble_ays_idk_pairwise_both': 'AYS $\\cup$ IDK $\\cup$ Disagree (A) $\\cup$ Disagree (S)',
}


def generate_baseline_prf_table(df: pd.DataFrame, output_path: Path) -> Path:
    """
    Generate LaTeX table with P, R, F1, Coverage, Acc@Coverage for each baseline across datasets.
    Aggregates by summing TP, FP, TN, FN across models, then computing metrics.
    """
    # Aggregate across models
    agg = df.groupby(['dataset', 'baseline']).agg({
        'TP': 'sum',
        'FP': 'sum',
        'TN': 'sum',
        'FN': 'sum'
    }).reset_index()

    # Compute all metrics
    agg['P'] = agg['TP'] / (agg['TP'] + agg['FP'])
    agg['R'] = agg['TP'] / (agg['TP'] + agg['FN'])
    agg['F1'] = 2 * agg['P'] * agg['R'] / (agg['P'] + agg['R'])
    agg['Total'] = agg['TP'] + agg['FP'] + agg['TN'] + agg['FN']
    agg['Cov'] = (agg['TN'] + agg['FN']) / agg['Total']  # Coverage = accepted / total
    agg['AccCov'] = agg['TN'] / (agg['TN'] + agg['FN'])  # Acc@Cov = correct among accepted

    # Handle NaN
    agg = agg.fillna(0)

    # Find best F1 per dataset for bolding (among all baselines we show)
    best_f1 = {}
    for dataset in DATASET_ORDER:
        subset = agg[(agg['dataset'] == dataset) & (agg['baseline'].isin(ALL_BASELINES_ORDER))]
        if len(subset) > 0:
            best_f1[dataset] = subset['F1'].max()

    # Build LaTeX
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\setlength\tabcolsep{2.0pt}")
    lines.append(r"\scriptsize")
    lines.append(r"\begin{center}")

    # Header - 5 columns per dataset
    header_cols = "l " + " | ".join(["ccccc"] * len(DATASET_ORDER))
    lines.append(r"\begin{tabular}{" + header_cols + "}")

    # Dataset names row
    dataset_headers = " & ".join([
        f"\\multicolumn{{5}}{{c}}{{{DATASET_DISPLAY.get(d, d)}}}"
        for d in DATASET_ORDER
    ])
    lines.append(f" & {dataset_headers} \\\\ [0.1cm]")

    # Metric headers row
    metric_headers = " & ".join([r"\textbf{P} & \textbf{R} & \textbf{F1} & \textbf{Cov} & \textbf{A@C}"] * len(DATASET_ORDER))
    lines.append(f"\\multicolumn{{1}}{{c}}{{}} & {metric_headers} \\\\")
    lines.append(r"\toprule")

    # Data rows - baselines first, then proposed methods
    for baseline in ALL_BASELINES_ORDER:
        row_data = [BASELINE_DISPLAY.get(baseline, baseline)]

        for dataset in DATASET_ORDER:
            subset = agg[(agg['dataset'] == dataset) & (agg['baseline'] == baseline)]
            if len(subset) > 0:
                p = subset['P'].values[0] * 100
                r = subset['R'].values[0] * 100
                f1 = subset['F1'].values[0] * 100
                cov = subset['Cov'].values[0] * 100
                acc_cov = subset['AccCov'].values[0] * 100

                # Bold if best F1 (for proposed methods only)
                is_best = abs(subset['F1'].values[0] - best_f1.get(dataset, 0)) < 0.001

                if is_best and baseline in PROPOSED_ORDER:
                    row_data.extend([
                        f"$\\mathbf{{{p:.0f}}}$",
                        f"$\\mathbf{{{r:.0f}}}$",
                        f"$\\mathbf{{{f1:.0f}}}$",
                        f"$\\mathbf{{{cov:.0f}}}$",
                        f"$\\mathbf{{{acc_cov:.0f}}}$"
                    ])
                else:
                    row_data.extend([f"${p:.0f}$", f"${r:.0f}$", f"${f1:.0f}$", f"${cov:.0f}$", f"${acc_cov:.0f}$"])
            else:
                row_data.extend(["--", "--", "--", "--", "--"])

        # Add midrule after baselines (idk), before proposed methods
        if baseline == 'idk':
            lines.append(" & ".join(row_data) + r" \\ \midrule")
        else:
            lines.append(" & ".join(row_data) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{center}")
    lines.append(r"\caption{Error detection metrics across datasets. P=Precision, R=Recall, F1=F1 score, Cov=Coverage (\% accepted), A@C=Accuracy among accepted. Our proposed methods (bottom) achieve higher F1 with better accuracy at coverage.}")
    lines.append(r"\label{table:baseline_prf}")
    lines.append(r"\end{table*}")

    # Write
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    print(f"Saved: {output_path}")
    return output_path


def generate_accuracy_table(df: pd.DataFrame, output_path: Path) -> Path:
    """
    Generate LaTeX table with accuracy for each model × dataset × approach.
    Uses all 6 datasets.
    """
    approaches = ['open_ended', 'assistive', 'sequential']
    approach_abbrev = {'open_ended': 'D', 'assistive': 'A', 'sequential': 'S'}

    model_order = [
        'gemma-3-4B-it', 'mistral-7B-Instruct-v0.3', 'qwen-3-8B', 'llama-3.1-8B-Instruct',
        'qwen-3-32B', 'qwen-2.5-72B-Instruct', 'llama-3.3-70B-Instruct',
        'gemini-2.5-flash', 'gemini-2.5-pro', 'gpt-5.1'
    ]
    model_display = {
        'gemini-2.5-flash': 'Gemini Flash',
        'gemini-2.5-pro': 'Gemini Pro',
        'gpt-5.1': 'GPT-5.1',
        'llama-3.3-70B-Instruct': 'Llama 70B',
        'llama-3.1-8B-Instruct': 'Llama 8B',
        'qwen-2.5-72B-Instruct': 'Qwen 72B',
        'qwen-3-32B': 'Qwen 32B',
        'qwen-3-8B': 'Qwen 8B',
        'mistral-7B-Instruct-v0.3': 'Mistral 7B',
        'gemma-3-4B-it': 'Gemma 4B',
    }

    # Abbreviated dataset names for 6-column layout
    dataset_abbrev = {
        'bamboogle': 'Bamb', 'mintaka': 'Mint', 'hotpotqa': 'Hotp',
        'crag': 'CRAG', 'frames': 'FRAM', 'musique': 'MuSi'
    }

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\setlength\tabcolsep{2.5pt}")
    lines.append(r"\scriptsize")
    lines.append(r"\begin{center}")

    # 6 datasets × 3 approaches = 18 columns + model name
    header_cols = "l " + " | ".join(["ccc"] * len(DATASET_ORDER_ALL))
    lines.append(r"\begin{tabular}{" + header_cols + "}")

    # Dataset names
    dataset_headers = " & ".join([
        f"\\multicolumn{{3}}{{c}}{{{dataset_abbrev.get(d, d)}}}"
        for d in DATASET_ORDER_ALL
    ])
    lines.append(f"\\textbf{{Model}} & {dataset_headers} \\\\ [0.1cm]")

    # Approach names (abbreviated)
    approach_headers = " & ".join([
        " & ".join([approach_abbrev[a] for a in approaches])
        for _ in DATASET_ORDER_ALL
    ])
    lines.append(f" & {approach_headers} \\\\")
    lines.append(r"\toprule")

    # Data rows
    for model in model_order:
        row_data = [model_display.get(model, model)]

        for dataset in DATASET_ORDER_ALL:
            for approach in approaches:
                subset = df[(df['dataset'] == dataset) &
                           (df['model'] == model) &
                           (df['approach'] == approach)]
                if len(subset) > 0:
                    acc = subset['accuracy'].values[0]
                    row_data.append(f"{acc:.1f}")
                else:
                    row_data.append("--")

        lines.append(" & ".join(row_data) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{center}")
    lines.append(r"\caption{Accuracy (\%) by model, dataset, and reasoning approach. D=Direct, A=Assistive, S=Sequential.}")
    lines.append(r"\label{table:accuracy_full}")
    lines.append(r"\end{table*}")

    output_path.write_text("\n".join(lines))
    print(f"Saved: {output_path}")
    return output_path


def generate_consistency_table(df: pd.DataFrame, output_path: Path) -> Path:
    """
    Generate LaTeX table with consistency rates for all 6 datasets.
    """
    approaches = ['assistive', 'sequential']
    approach_abbrev = {'assistive': 'A', 'sequential': 'S'}

    model_order = [
        'gemma-3-4B-it', 'mistral-7B-Instruct-v0.3', 'qwen-3-8B', 'llama-3.1-8B-Instruct',
        'qwen-3-32B', 'qwen-2.5-72B-Instruct', 'llama-3.3-70B-Instruct',
        'gemini-2.5-flash', 'gemini-2.5-pro', 'gpt-5.1'
    ]
    model_display = {
        'gemini-2.5-flash': 'Gemini Flash',
        'gemini-2.5-pro': 'Gemini Pro',
        'gpt-5.1': 'GPT-5.1',
        'llama-3.3-70B-Instruct': 'Llama 70B',
        'llama-3.1-8B-Instruct': 'Llama 8B',
        'qwen-2.5-72B-Instruct': 'Qwen 72B',
        'qwen-3-32B': 'Qwen 32B',
        'qwen-3-8B': 'Qwen 8B',
        'mistral-7B-Instruct-v0.3': 'Mistral 7B',
        'gemma-3-4B-it': 'Gemma 4B',
    }

    # Abbreviated dataset names for 6-column layout
    dataset_abbrev = {
        'bamboogle': 'Bamb', 'mintaka': 'Mint', 'hotpotqa': 'Hotp',
        'crag': 'CRAG', 'frames': 'FRAM', 'musique': 'MuSi'
    }

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\setlength\tabcolsep{3.0pt}")
    lines.append(r"\scriptsize")
    lines.append(r"\begin{center}")

    header_cols = "l " + " | ".join(["cc"] * len(DATASET_ORDER_ALL))
    lines.append(r"\begin{tabular}{" + header_cols + "}")

    # Dataset names
    dataset_headers = " & ".join([
        f"\\multicolumn{{2}}{{c}}{{{dataset_abbrev.get(d, d)}}}"
        for d in DATASET_ORDER_ALL
    ])
    lines.append(f"\\textbf{{Model}} & {dataset_headers} \\\\ [0.1cm]")

    # Approach names (abbreviated)
    approach_headers = " & ".join([
        " & ".join([approach_abbrev[a] for a in approaches])
        for _ in DATASET_ORDER_ALL
    ])
    lines.append(f" & {approach_headers} \\\\")
    lines.append(r"\toprule")

    # Data rows
    for model in model_order:
        row_data = [model_display.get(model, model)]

        for dataset in DATASET_ORDER_ALL:
            for approach in approaches:
                subset = df[(df['dataset'] == dataset) &
                           (df['model'] == model) &
                           (df['approach'] == approach)]
                if len(subset) > 0:
                    rate = subset['consistency_rate'].values[0]
                    row_data.append(f"{rate:.1f}")
                else:
                    row_data.append("--")

        lines.append(" & ".join(row_data) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{center}")
    lines.append(r"\caption{Consistency rate (\%) with direct answers by model, dataset, and reasoning approach. A=Assistive, S=Sequential.}")
    lines.append(r"\label{table:consistency_full}")
    lines.append(r"\end{table*}")

    output_path.write_text("\n".join(lines))
    print(f"Saved: {output_path}")
    return output_path


def generate_delta_table(df: pd.DataFrame, output_path: Path) -> Path:
    """
    Generate table showing accuracy delta (Decomposed - Direct) for all 6 datasets.
    """
    model_order = [
        'gemma-3-4B-it', 'mistral-7B-Instruct-v0.3', 'qwen-3-8B', 'llama-3.1-8B-Instruct',
        'qwen-3-32B', 'qwen-2.5-72B-Instruct', 'llama-3.3-70B-Instruct',
        'gemini-2.5-flash', 'gemini-2.5-pro', 'gpt-5.1'
    ]
    model_display = {
        'gemini-2.5-flash': 'Gemini Flash',
        'gemini-2.5-pro': 'Gemini Pro',
        'gpt-5.1': 'GPT-5.1',
        'llama-3.3-70B-Instruct': 'Llama 70B',
        'llama-3.1-8B-Instruct': 'Llama 8B',
        'qwen-2.5-72B-Instruct': 'Qwen 72B',
        'qwen-3-32B': 'Qwen 32B',
        'qwen-3-8B': 'Qwen 8B',
        'mistral-7B-Instruct-v0.3': 'Mistral 7B',
        'gemma-3-4B-it': 'Gemma 4B',
    }

    # Abbreviated dataset names for 6-column layout
    dataset_abbrev = {
        'bamboogle': 'Bamb', 'mintaka': 'Mint', 'hotpotqa': 'Hotp',
        'crag': 'CRAG', 'frames': 'FRAM', 'musique': 'MuSi'
    }

    # Pivot
    pivot = df.pivot_table(index=['dataset', 'model'], columns='approach',
                          values='accuracy', aggfunc='first').reset_index()
    pivot['delta_assistive'] = pivot['assistive'] - pivot['open_ended']
    pivot['delta_sequential'] = pivot['sequential'] - pivot['open_ended']

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\setlength\tabcolsep{2.5pt}")
    lines.append(r"\scriptsize")
    lines.append(r"\begin{center}")

    # 6 datasets × 2 deltas = 12 columns + model
    header_cols = "l " + " | ".join(["cc"] * len(DATASET_ORDER_ALL))
    lines.append(r"\begin{tabular}{" + header_cols + "}")

    # Dataset names
    dataset_headers = " & ".join([
        f"\\multicolumn{{2}}{{c}}{{{dataset_abbrev.get(d, d)}}}"
        for d in DATASET_ORDER_ALL
    ])
    lines.append(f"\\textbf{{Model}} & {dataset_headers} \\\\ [0.1cm]")

    # Delta type names
    delta_headers = " & ".join(["$\\Delta$A & $\\Delta$S"] * len(DATASET_ORDER_ALL))
    lines.append(f" & {delta_headers} \\\\")
    lines.append(r"\toprule")

    # Data rows
    for model in model_order:
        row_data = [model_display.get(model, model)]

        for dataset in DATASET_ORDER_ALL:
            subset = pivot[(pivot['dataset'] == dataset) & (pivot['model'] == model)]
            if len(subset) > 0 and not pd.isna(subset['delta_assistive'].values[0]):
                da = subset['delta_assistive'].values[0]
                ds = subset['delta_sequential'].values[0]

                # Format with + for positive
                da_str = f"+{da:.1f}" if da > 0 else f"{da:.1f}"
                ds_str = f"+{ds:.1f}" if ds > 0 else f"{ds:.1f}"

                row_data.extend([da_str, ds_str])
            else:
                row_data.extend(["--", "--"])

        lines.append(" & ".join(row_data) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{center}")
    lines.append(r"\caption{Accuracy change (\%) from Direct to Assistive ($\Delta$A) and Sequential ($\Delta$S).}")
    lines.append(r"\label{table:delta}")
    lines.append(r"\end{table*}")

    output_path.write_text("\n".join(lines))
    print(f"Saved: {output_path}")
    return output_path


def generate_accuracy_table_all(df: pd.DataFrame, output_path: Path) -> Path:
    """
    Generate LaTeX table with accuracy for all 6 datasets (appendix version).
    """
    approaches = ['open_ended', 'assistive', 'sequential']
    approach_display = {'open_ended': 'Direct', 'assistive': 'Assistive', 'sequential': 'Sequential'}

    model_order = [
        'gemma-3-4B-it', 'mistral-7B-Instruct-v0.3', 'qwen-3-8B', 'llama-3.1-8B-Instruct',
        'qwen-3-32B', 'qwen-2.5-72B-Instruct', 'llama-3.3-70B-Instruct',
        'gemini-2.5-flash', 'gemini-2.5-pro', 'gpt-5.1'
    ]
    model_display = {
        'gemini-2.5-flash': 'Gemini Flash',
        'gemini-2.5-pro': 'Gemini Pro',
        'gpt-5.1': 'GPT-5.1',
        'llama-3.3-70B-Instruct': 'Llama 70B',
        'llama-3.1-8B-Instruct': 'Llama 8B',
        'qwen-2.5-72B-Instruct': 'Qwen 72B',
        'qwen-3-32B': 'Qwen 32B',
        'qwen-3-8B': 'Qwen 8B',
        'mistral-7B-Instruct-v0.3': 'Mistral 7B',
        'gemma-3-4B-it': 'Gemma 4B',
    }

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\setlength\tabcolsep{2.5pt}")
    lines.append(r"\scriptsize")
    lines.append(r"\begin{center}")

    # 6 datasets × 3 approaches = 18 columns + model name
    header_cols = "l " + " | ".join(["ccc"] * len(DATASET_ORDER_ALL))
    lines.append(r"\begin{tabular}{" + header_cols + "}")

    # Dataset names (abbreviated for space)
    dataset_abbrev = {
        'bamboogle': 'Bamb', 'mintaka': 'Mint', 'hotpotqa': 'Hotp',
        'crag': 'CRAG', 'frames': 'FRAM', 'musique': 'MuSi'
    }
    dataset_headers = " & ".join([
        f"\\multicolumn{{3}}{{c}}{{{dataset_abbrev.get(d, d)}}}"
        for d in DATASET_ORDER_ALL
    ])
    lines.append(f"\\textbf{{Model}} & {dataset_headers} \\\\ [0.1cm]")

    # Approach names (abbreviated)
    approach_abbrev = {'open_ended': 'D', 'assistive': 'A', 'sequential': 'S'}
    approach_headers = " & ".join([
        " & ".join([approach_abbrev[a] for a in approaches])
        for _ in DATASET_ORDER_ALL
    ])
    lines.append(f" & {approach_headers} \\\\")
    lines.append(r"\toprule")

    # Data rows
    for model in model_order:
        row_data = [model_display.get(model, model)]

        for dataset in DATASET_ORDER_ALL:
            for approach in approaches:
                subset = df[(df['dataset'] == dataset) &
                           (df['model'] == model) &
                           (df['approach'] == approach)]
                if len(subset) > 0:
                    acc = subset['accuracy'].values[0]
                    row_data.append(f"{acc:.1f}")
                else:
                    row_data.append("--")

        lines.append(" & ".join(row_data) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{center}")
    lines.append(r"\caption{Accuracy (\%) across all 6 datasets. D=Direct, A=Assistive, S=Sequential.}")
    lines.append(r"\label{table:accuracy_all}")
    lines.append(r"\end{table*}")

    output_path.write_text("\n".join(lines))
    print(f"Saved: {output_path}")
    return output_path


def generate_consistency_table_all(df: pd.DataFrame, output_path: Path) -> Path:
    """
    Generate LaTeX table with consistency rates for all 6 datasets (appendix version).
    """
    approaches = ['assistive', 'sequential']
    approach_display = {'assistive': 'A', 'sequential': 'S'}

    model_order = [
        'gemma-3-4B-it', 'mistral-7B-Instruct-v0.3', 'qwen-3-8B', 'llama-3.1-8B-Instruct',
        'qwen-3-32B', 'qwen-2.5-72B-Instruct', 'llama-3.3-70B-Instruct',
        'gemini-2.5-flash', 'gemini-2.5-pro', 'gpt-5.1'
    ]
    model_display = {
        'gemini-2.5-flash': 'Gemini Flash',
        'gemini-2.5-pro': 'Gemini Pro',
        'gpt-5.1': 'GPT-5.1',
        'llama-3.3-70B-Instruct': 'Llama 70B',
        'llama-3.1-8B-Instruct': 'Llama 8B',
        'qwen-2.5-72B-Instruct': 'Qwen 72B',
        'qwen-3-32B': 'Qwen 32B',
        'qwen-3-8B': 'Qwen 8B',
        'mistral-7B-Instruct-v0.3': 'Mistral 7B',
        'gemma-3-4B-it': 'Gemma 4B',
    }

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\setlength\tabcolsep{3.0pt}")
    lines.append(r"\scriptsize")
    lines.append(r"\begin{center}")

    header_cols = "l " + " | ".join(["cc"] * len(DATASET_ORDER_ALL))
    lines.append(r"\begin{tabular}{" + header_cols + "}")

    # Dataset names (abbreviated)
    dataset_abbrev = {
        'bamboogle': 'Bamb', 'mintaka': 'Mint', 'hotpotqa': 'Hotp',
        'crag': 'CRAG', 'frames': 'FRAM', 'musique': 'MuSi'
    }
    dataset_headers = " & ".join([
        f"\\multicolumn{{2}}{{c}}{{{dataset_abbrev.get(d, d)}}}"
        for d in DATASET_ORDER_ALL
    ])
    lines.append(f"\\textbf{{Model}} & {dataset_headers} \\\\ [0.1cm]")

    # Approach names
    approach_headers = " & ".join([
        " & ".join([approach_display[a] for a in approaches])
        for _ in DATASET_ORDER_ALL
    ])
    lines.append(f" & {approach_headers} \\\\")
    lines.append(r"\toprule")

    # Data rows
    for model in model_order:
        row_data = [model_display.get(model, model)]

        for dataset in DATASET_ORDER_ALL:
            for approach in approaches:
                subset = df[(df['dataset'] == dataset) &
                           (df['model'] == model) &
                           (df['approach'] == approach)]
                if len(subset) > 0:
                    rate = subset['consistency_rate'].values[0]
                    row_data.append(f"{rate:.1f}")
                else:
                    row_data.append("--")

        lines.append(" & ".join(row_data) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{center}")
    lines.append(r"\caption{Consistency rate (\%) with direct answers across all 6 datasets. A=Assistive, S=Sequential.}")
    lines.append(r"\label{table:consistency_all}")
    lines.append(r"\end{table*}")

    output_path.write_text("\n".join(lines))
    print(f"Saved: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=PAPER_TABLES_DIR)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")

    derived_dir = REPO_ROOT / "analysis" / "derived"

    # Load baselines (4 datasets with data)
    baselines_df = pd.read_csv(derived_dir / "baselines.tsv", sep='\t')

    # Load ALL data (6 datasets) for accuracy, consistency, delta tables
    correctness_all_df = pd.read_csv(derived_dir / "correctness_filtered_all.tsv", sep='\t')
    consistency_all_df = pd.read_csv(derived_dir / "consistency_filtered_all.tsv", sep='\t')

    print(f"Loaded: {len(baselines_df)} baseline rows")
    print(f"Loaded ALL: {len(correctness_all_df)} correctness, {len(consistency_all_df)} consistency rows")
    print(f"Datasets: {sorted(correctness_all_df['dataset'].unique())}")

    # Generate tables
    print("\nGenerating tables...")

    # Baseline table (4 datasets with baseline data)
    generate_baseline_prf_table(baselines_df, output_dir / "baseline_prf.tex")

    # All other tables use all 6 datasets
    generate_accuracy_table(correctness_all_df, output_dir / "accuracy_full.tex")
    generate_consistency_table(consistency_all_df, output_dir / "consistency_full.tex")
    generate_delta_table(correctness_all_df, output_dir / "delta.tex")

    print("\nDone!")


if __name__ == "__main__":
    main()
