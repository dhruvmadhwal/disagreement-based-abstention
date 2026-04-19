#!/usr/bin/env python3
"""Generate baseline_main_prf.tex and baseline_other_prf.tex tables."""

import pandas as pd
import numpy as np

# Model name mapping - handles all variants in TSV
MODEL_NAME_MAP = {
    # GPT-5.1
    'gpt-5-1': 'GPT-5.1',
    'gpt-5.1': 'GPT-5.1',
    # Llama 3.3 70B
    'meta-llama-llama-3-3-70b-instruct': 'Llama 3.3 70B',
    'llama-3.3-70b-instruct': 'Llama 3.3 70B',
    'llama-3.3-70B-Instruct': 'Llama 3.3 70B',
    # Qwen3-8B
    'qwen-qwen3-8b': 'Qwen3-8B',
    'qwen3-8b': 'Qwen3-8B',
    'qwen-3-8B': 'Qwen3-8B',
    # Mistral 7B
    'mistralai-mistral-7b-instruct-v0-3': 'Mistral 7B',
    'mistral-7B-Instruct-v0.3': 'Mistral 7B',
    # Llama 3.1 8B
    'meta-llama-llama-3-1-8b-instruct': 'Llama 3.1 8B',
    'llama-3.1-8B-Instruct': 'Llama 3.1 8B',
    # Qwen3-32B
    'qwen-qwen3-32b': 'Qwen3-32B',
    'qwen-3-32B': 'Qwen3-32B',
    # Qwen 2.5 72B
    'qwen-qwen2-5-72b-instruct': 'Qwen 2.5 72B',
    'qwen-2.5-72B-Instruct': 'Qwen 2.5 72B',
    # Gemini Flash
    'google-gemini-2-5-flash': 'Gemini Flash',
    'gemini-2.5-flash': 'Gemini Flash',
    # Gemini Pro
    'google-gemini-2-5-pro': 'Gemini Pro',
    'gemini-2.5-pro': 'Gemini Pro',
}

# Method name mapping
METHOD_MAP = {
    'ays': 'AYS',
    'idk': 'IC-IDK',
    'ic_idk': 'IC-IDK',
    'IC-IDK': 'IC-IDK',
    'pairwise_assistive': 'Disagree(A)',
    'pairwise_incremental': 'Disagree(I)',
    'ensemble_assistive': 'Ensemble(A)',
    'ensemble_incremental': 'Ensemble(I)',
}

# Datasets in order
DATASETS = ['bamboogle', 'crag', 'frames', 'hotpotqa', 'mintaka', 'musique']
DATASET_DISPLAY = {
    'bamboogle': 'Bamboogle',
    'crag': 'CRAG',
    'frames': 'FRAMES',
    'hotpotqa': 'HotpotQA',
    'mintaka': 'Mintaka',
    'musique': 'MuSiQue',
}

# Models for each table
MAIN_MODELS = ['GPT-5.1', 'Llama 3.3 70B', 'Qwen3-8B']
OTHER_MODELS = ['Mistral 7B', 'Llama 3.1 8B', 'Qwen3-32B', 'Qwen 2.5 72B', 'Gemini Flash', 'Gemini Pro']

# Methods order - single methods first, then ensemble
SINGLE_METHODS = ['AYS', 'IC-IDK', 'Disagree(A)', 'Disagree(I)']
ENSEMBLE_METHODS = ['Ensemble(A)', 'Ensemble(I)']


def load_data(tsv_path):
    """Load and normalize the TSV data."""
    df = pd.read_csv(tsv_path, sep='\t')

    # Normalize model names
    df['model_display'] = df['model'].map(MODEL_NAME_MAP)

    # Normalize method names
    df['method_display'] = df['baseline'].map(METHOD_MAP)

    # Normalize dataset names
    df['dataset_lower'] = df['dataset'].str.lower()

    return df


def fmt_val(val, is_best_single, is_best_overall):
    """Format a value with underline/bold as needed."""
    if pd.isna(val) or val == '--':
        return '--'

    try:
        num = float(val)
        formatted = f'{num:.2f}'
    except (ValueError, TypeError):
        return '--'

    if is_best_overall:
        return f'\\textbf{{{formatted}}}'
    elif is_best_single:
        return f'\\underline{{{formatted}}}'
    return formatted


def get_best_values(df, model, dataset, metric, single_methods, all_methods):
    """Get best single and best overall values for a metric."""
    subset = df[(df['model_display'] == model) & (df['dataset_lower'] == dataset)]

    # Best among single methods
    single_df = subset[subset['method_display'].isin(single_methods)]
    if len(single_df) > 0 and metric in single_df.columns:
        single_vals = pd.to_numeric(single_df[metric], errors='coerce')
        best_single = single_vals.max() if not single_vals.isna().all() else None
    else:
        best_single = None

    # Best overall (including ensemble)
    all_df = subset[subset['method_display'].isin(all_methods)]
    if len(all_df) > 0 and metric in all_df.columns:
        all_vals = pd.to_numeric(all_df[metric], errors='coerce')
        best_overall = all_vals.max() if not all_vals.isna().all() else None
    else:
        best_overall = None

    return best_single, best_overall


def get_value(df, model, dataset, method, metric):
    """Get a specific value from the dataframe."""
    row = df[(df['model_display'] == model) &
             (df['dataset_lower'] == dataset) &
             (df['method_display'] == method)]

    if len(row) == 0 or metric not in row.columns:
        return None

    val = row[metric].iloc[0]
    if pd.isna(val):
        return None
    return val


def generate_table(df, models, include_ensemble, caption, label):
    """Generate a LaTeX table."""

    if include_ensemble:
        all_methods = SINGLE_METHODS + ENSEMBLE_METHODS
    else:
        all_methods = SINGLE_METHODS

    metrics = ['Precision', 'Recall', 'F1', 'AUROC']
    metric_headers = ['P', 'R', 'F1', 'AUC']

    lines = []
    lines.append(r'\begin{table*}[t]')
    lines.append(r'\setlength\tabcolsep{2pt}')
    lines.append(r'\scriptsize')
    lines.append(r'\centering')

    # Column spec: model, method, then 4 cols per dataset
    col_spec = 'll | ' + ' | '.join(['cccc'] * len(DATASETS))
    lines.append(r'\begin{tabular}{' + col_spec + ' }')
    lines.append(r'\toprule')

    # Dataset header row
    dataset_headers = ' & '.join([f'\\multicolumn{{4}}{{c}}{{\\textbf{{{DATASET_DISPLAY[d]}}}}}' for d in DATASETS])
    lines.append(f' &  & {dataset_headers} \\\\')

    # Metric header row
    metric_line = r'\textbf{Model} & \textbf{Method}'
    for _ in DATASETS:
        metric_line += ' & ' + ' & '.join(metric_headers)
    metric_line += r' \\'
    lines.append(metric_line)
    lines.append(r'\midrule')

    # Generate rows for each model
    for mi, model in enumerate(models):
        if mi > 0:
            lines.append(r'\midrule')

        # For each method
        for method_idx, method in enumerate(all_methods):
            # Model name only on first row
            model_cell = model if method_idx == 0 else ''

            row_vals = [model_cell, method]

            for dataset in DATASETS:
                for metric in metrics:
                    val = get_value(df, model, dataset, method, metric)

                    # Get best values for formatting
                    best_single, best_overall = get_best_values(
                        df, model, dataset, metric, SINGLE_METHODS, all_methods
                    )

                    if val is not None:
                        is_best_single = (best_single is not None and
                                         abs(val - best_single) < 0.001 and
                                         method in SINGLE_METHODS)
                        is_best_overall = (best_overall is not None and
                                          abs(val - best_overall) < 0.001)
                        formatted = fmt_val(val, is_best_single, is_best_overall)
                    else:
                        formatted = '--'

                    row_vals.append(formatted)

            lines.append(' & '.join(row_vals) + r' \\')

            # Add cmidrule after IC-IDK (before Disagree methods)
            if method == 'IC-IDK':
                lines.append(r'\cmidrule{2-26}')

            # Add cmidrule before Ensemble methods (if included)
            if include_ensemble and method == 'Disagree(I)':
                lines.append(r'\cmidrule{2-26}')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(f'\\caption{{{caption}}}')
    lines.append(f'\\label{{{label}}}')
    lines.append(r'\end{table*}')
    lines.append('')

    return '\n'.join(lines)


def main():
    # Load data
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[1]
    df = load_data(str(repo_root / 'analysis' / 'derived' / 'baselines_all.tsv'))

    # Debug: show what we have
    print("Models found:", df['model_display'].dropna().unique())
    print("Methods found:", df['method_display'].dropna().unique())
    print("Datasets found:", df['dataset_lower'].unique())

    # Check for main models data
    for model in MAIN_MODELS:
        model_data = df[df['model_display'] == model]
        print(f"\n{model}: {len(model_data)} rows")
        if len(model_data) > 0:
            print(f"  Datasets: {model_data['dataset_lower'].unique()}")
            print(f"  Methods: {model_data['method_display'].unique()}")

    # Generate main table (GPT-5.1, Llama 3.3 70B, Qwen3-8B) - with ensemble
    main_caption = (
        "Complete error detection results (Precision, Recall, F1, AUROC) for the three main models "
        "across all datasets. \\underline{Underline} = best among single methods; \\textbf{bold} = best overall."
    )
    main_table = generate_table(df, MAIN_MODELS, include_ensemble=True,
                                caption=main_caption, label='tab:baseline_main_prf')

    with open('/home/antonxue/david/Consistency-in-Multi-Hop-QA/latex/tables/baseline_main_prf.tex', 'w') as f:
        f.write(main_table)
    print("\nWrote baseline_main_prf.tex")

    # Generate other table (remaining models) - with ensemble for all models
    other_caption = (
        "Complete error detection results (Precision, Recall, F1, AUROC) for all remaining models. "
        "\\underline{Underline} = best among single methods; \\textbf{bold} = best overall."
    )
    other_table = generate_table(df, OTHER_MODELS, include_ensemble=True,
                                 caption=other_caption, label='tab:baseline_other_prf')

    with open('/home/antonxue/david/Consistency-in-Multi-Hop-QA/latex/tables/baseline_other_prf.tex', 'w') as f:
        f.write(other_table)
    print("Wrote baseline_other_prf.tex")


if __name__ == '__main__':
    main()
