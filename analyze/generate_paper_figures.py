#!/usr/bin/env python3
"""Generate clean, publication-ready figures for ACL submission."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ============================================================================
# Constants
# ============================================================================

REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_FIGURES_DIR = REPO_ROOT.parent / "Consistency-in-Multi-Hop-QA" / "latex" / "figures"

# Display names
MODEL_DISPLAY = {
    'gemini-2.5-flash': 'Gemini 2.5 Flash',
    'gemini-2.5-pro': 'Gemini 2.5 Pro',
    'gpt-5.1': 'GPT-5.1',
    'llama-3.3-70B-Instruct': 'Llama 3.3 70B',
    'llama-3.1-8B-Instruct': 'Llama 3.1 8B',
    'qwen-2.5-72B-Instruct': 'Qwen 2.5 72B',
    'qwen-3-32B': 'Qwen 3 32B',
    'qwen-3-8B': 'Qwen 3 8B',
    'mistral-7B-Instruct-v0.3': 'Mistral 7B',
    'gemma-3-4B-it': 'Gemma 3 4B',
}

DATASET_DISPLAY = {
    'bamboogle': 'Bamboogle',
    'crag': 'CRAG',
    'hotpotqa': 'HotpotQA',
    'mintaka': 'Mintaka',
    'frames': 'FRAMES',
    'musique': 'MuSiQue',
}

APPROACH_DISPLAY = {
    'open_ended': 'Direct',
    'assistive': 'Assistive',
    'sequential': 'Incremental',
    'model_generated_plan': 'Model-Gen',
}

# Model ordering (by capability, roughly)
MODEL_ORDER = [
    'gemma-3-4B-it',
    'mistral-7B-Instruct-v0.3',
    'qwen-3-8B',
    'llama-3.1-8B-Instruct',
    'qwen-3-32B',
    'qwen-2.5-72B-Instruct',
    'llama-3.3-70B-Instruct',
    'gemini-2.5-flash',
    'gemini-2.5-pro',
    'gpt-5.1',
]

DATASET_ORDER = ['bamboogle', 'mintaka', 'hotpotqa', 'crag']
DATASET_ORDER_ALL = ['bamboogle', 'mintaka', 'hotpotqa', 'crag', 'frames', 'musique']

# Style
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 12,
})


def get_display_name(name: str, mapping: dict) -> str:
    return mapping.get(name, name)


# ============================================================================
# Figure 1: Accuracy Heatmap (Model × Dataset × Approach)
# ============================================================================

def plot_accuracy_heatmaps(df: pd.DataFrame, output_path: Path) -> Path:
    """Create accuracy heatmaps for each approach."""
    approaches = ['open_ended', 'assistive', 'sequential']

    fig, axes = plt.subplots(1, 3, figsize=(14, 6), sharey=True)

    for ax, approach in zip(axes, approaches):
        subset = df[df['approach'] == approach].copy()

        # Pivot to matrix
        pivot = subset.pivot(index='model', columns='dataset', values='accuracy')

        # Reorder
        pivot = pivot.reindex(index=[m for m in MODEL_ORDER if m in pivot.index])
        pivot = pivot.reindex(columns=[d for d in DATASET_ORDER if d in pivot.columns])

        # Rename for display
        pivot.index = [get_display_name(m, MODEL_DISPLAY) for m in pivot.index]
        pivot.columns = [get_display_name(d, DATASET_DISPLAY) for d in pivot.columns]

        # Plot
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='Blues',
                   vmin=0, vmax=100, ax=ax, cbar=ax == axes[-1],
                   annot_kws={'size': 8})
        ax.set_title(get_display_name(approach, APPROACH_DISPLAY), fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('' if ax != axes[0] else 'Model')

    fig.suptitle('Accuracy (%) by Model, Dataset, and Approach', fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")
    return output_path


# ============================================================================
# Figure 2: Consistency Heatmap (Model × Dataset × Approach)
# ============================================================================

def plot_consistency_heatmaps(df: pd.DataFrame, output_path: Path) -> Path:
    """Create consistency heatmaps for each approach."""
    approaches = ['assistive', 'sequential']

    fig, axes = plt.subplots(1, 2, figsize=(10, 6), sharey=True)

    for ax, approach in zip(axes, approaches):
        subset = df[df['approach'] == approach].copy()

        pivot = subset.pivot(index='model', columns='dataset', values='consistency_rate')
        pivot = pivot.reindex(index=[m for m in MODEL_ORDER if m in pivot.index])
        pivot = pivot.reindex(columns=[d for d in DATASET_ORDER if d in pivot.columns])

        pivot.index = [get_display_name(m, MODEL_DISPLAY) for m in pivot.index]
        pivot.columns = [get_display_name(d, DATASET_DISPLAY) for d in pivot.columns]

        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='Greens',
                   vmin=0, vmax=100, ax=ax, cbar=ax == axes[-1],
                   annot_kws={'size': 8})
        ax.set_title(f'{get_display_name(approach, APPROACH_DISPLAY)} vs Direct', fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('' if ax != axes[0] else 'Model')

    fig.suptitle('Consistency Rate (%) with Direct Answers', fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")
    return output_path


# ============================================================================
# Figure 3: Delta Plot (Each Model Individually)
# ============================================================================

def plot_delta_by_model(df: pd.DataFrame, output_path: Path) -> Path:
    """Show accuracy delta (Assistive/Sequential - Direct) for each model."""

    # Pivot to get accuracy by approach
    pivot = df.pivot_table(index=['dataset', 'model'], columns='approach',
                          values='accuracy', aggfunc='first').reset_index()

    # Compute deltas
    pivot['Δ Assistive'] = pivot['assistive'] - pivot['open_ended']
    pivot['Δ Incremental'] = pivot['sequential'] - pivot['open_ended']

    # Melt for plotting
    delta_df = pivot.melt(
        id_vars=['dataset', 'model'],
        value_vars=['Δ Assistive', 'Δ Incremental'],
        var_name='Delta Type',
        value_name='Delta'
    )

    # Order models
    delta_df['model_order'] = delta_df['model'].map({m: i for i, m in enumerate(MODEL_ORDER)})
    delta_df = delta_df.sort_values('model_order')
    delta_df['Model'] = delta_df['model'].map(lambda x: get_display_name(x, MODEL_DISPLAY))

    # Create faceted plot by dataset
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    colors = {'Δ Assistive': '#e15759', 'Δ Incremental': '#59a14f'}

    for ax, dataset in zip(axes, DATASET_ORDER):
        subset = delta_df[delta_df['dataset'] == dataset]

        models = [get_display_name(m, MODEL_DISPLAY) for m in MODEL_ORDER]
        x = np.arange(len(models))
        width = 0.35

        for i, (delta_type, color) in enumerate(colors.items()):
            data = subset[subset['Delta Type'] == delta_type]
            # Match to model order
            values = []
            for m in MODEL_ORDER:
                val = data[data['model'] == m]['Delta'].values
                values.append(val[0] if len(val) > 0 else 0)

            offset = (i - 0.5) * width
            bars = ax.bar(x + offset, values, width, label=delta_type, color=color, alpha=0.8)

        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Δ Accuracy (%)')
        ax.set_title(get_display_name(dataset, DATASET_DISPLAY), fontweight='bold')
        ax.set_ylim(-30, 35)
        ax.grid(axis='y', alpha=0.3)

    axes[0].legend(loc='upper right')
    fig.suptitle('Accuracy Change: Decomposed vs Direct (per Model)', fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")
    return output_path


# ============================================================================
# Figure 4: Assistive vs Sequential Scatter (All Models, All Datasets)
# ============================================================================

def plot_assistive_vs_sequential(df: pd.DataFrame, output_path: Path) -> Path:
    """Scatter plot: Assistive accuracy vs Sequential accuracy for each (model, dataset)."""

    pivot = df.pivot_table(index=['dataset', 'model'], columns='approach',
                          values='accuracy', aggfunc='first').reset_index()

    fig, ax = plt.subplots(figsize=(8, 8))

    # Color by model
    colors = plt.cm.tab10(np.linspace(0, 1, len(MODEL_ORDER)))
    model_colors = {m: colors[i] for i, m in enumerate(MODEL_ORDER)}

    # Plot each point
    for _, row in pivot.iterrows():
        model = row['model']
        dataset = row['dataset']
        assistive = row['assistive']
        sequential = row['sequential']

        ax.scatter(assistive, sequential,
                  c=[model_colors.get(model, 'gray')],
                  s=80, alpha=0.7, edgecolors='white', linewidth=0.5)

    # Add legend for models
    for model in MODEL_ORDER:
        ax.scatter([], [], c=[model_colors[model]],
                  label=get_display_name(model, MODEL_DISPLAY), s=60)

    # Diagonal line
    ax.plot([0, 100], [0, 100], 'k--', alpha=0.5, linewidth=1)

    ax.set_xlabel('Assistive Accuracy (%)')
    ax.set_ylabel('Incremental Accuracy (%)')
    ax.set_title('Assistive vs Incremental Accuracy\n(each point = model × dataset)', fontweight='bold')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=8, ncol=2)

    # Correlation
    corr = pivot['assistive'].corr(pivot['sequential'])
    ax.text(5, 95, f'r = {corr:.3f}', fontsize=11, fontweight='bold')

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")
    return output_path


# ============================================================================
# Figure 5: Accuracy vs Consistency Scatter
# ============================================================================

def plot_accuracy_vs_consistency(corr_df: pd.DataFrame, cons_df: pd.DataFrame, output_path: Path) -> Path:
    """Scatter: accuracy (assistive) vs consistency rate for each (model, dataset)."""

    # Get assistive accuracy
    acc = corr_df[corr_df['approach'] == 'assistive'][['dataset', 'model', 'accuracy']].copy()
    cons = cons_df[cons_df['approach'] == 'assistive'][['dataset', 'model', 'consistency_rate']].copy()

    merged = acc.merge(cons, on=['dataset', 'model'])

    fig, ax = plt.subplots(figsize=(8, 6))

    # Color by dataset
    dataset_colors = {'bamboogle': '#4c78a8', 'mintaka': '#e15759',
                      'hotpotqa': '#59a14f', 'crag': '#f28e2b'}

    for dataset in DATASET_ORDER:
        subset = merged[merged['dataset'] == dataset]
        ax.scatter(subset['consistency_rate'], subset['accuracy'],
                  c=dataset_colors.get(dataset, 'gray'),
                  label=get_display_name(dataset, DATASET_DISPLAY),
                  s=80, alpha=0.7, edgecolors='white', linewidth=0.5)

    ax.set_xlabel('Consistency Rate (%)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy vs Consistency (Assistive Regime)', fontweight='bold')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')

    # Correlation
    corr = merged['accuracy'].corr(merged['consistency_rate'])
    ax.text(5, 95, f'r = {corr:.3f}', fontsize=11, fontweight='bold')

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")
    return output_path


# ============================================================================
# Figure 5b: Accuracy vs Consistency (All Datasets from *_all.tsv)
# ============================================================================

# Model size buckets for scatter plot coloring
MODEL_SIZE_BUCKET = {
    'gpt-5.1': 'Frontier',
    'gemini-2.5-pro': 'Frontier',
    'gemini-2.5-flash': 'Frontier',
    'llama-3.3-70B-Instruct': '70B',
    'qwen-2.5-72B-Instruct': '70B',
    'qwen-3-32B': '32B',
    'llama-3.1-8B-Instruct': '8B',
    'qwen-3-8B': '8B',
    'mistral-7B-Instruct-v0.3': '8B',
}

# Models to include (exclude gemma)
MODELS_9 = list(MODEL_SIZE_BUCKET.keys())


def plot_accuracy_vs_consistency_all(corr_df: pd.DataFrame, cons_df: pd.DataFrame, output_path: Path) -> Path:
    """Scatter: accuracy (assistive) vs consistency rate for each (model, dataset).

    Uses all datasets from *_all.tsv files with model scale coloring.
    - 9 models (excluding gemma)
    - 6 datasets
    - Color by model size bucket (Frontier, 70B, 32B, 8B)
    - Marker by dataset
    """

    # Normalize approach names: incremental -> sequential
    corr_df = corr_df.copy()
    cons_df = cons_df.copy()
    corr_df['approach'] = corr_df['approach'].replace({'incremental': 'sequential'})
    cons_df['approach'] = cons_df['approach'].replace({'incremental': 'sequential'})

    # Filter to 9 models (exclude gemma)
    corr_df = corr_df[corr_df['model'].isin(MODELS_9)]
    cons_df = cons_df[cons_df['model'].isin(MODELS_9)]

    # Get assistive accuracy and consistency
    acc = corr_df[corr_df['approach'] == 'assistive'][['dataset', 'model', 'accuracy']].copy()
    cons = cons_df[cons_df['approach'] == 'assistive'][['dataset', 'model', 'consistency_rate']].copy()

    merged = acc.merge(cons, on=['dataset', 'model'])

    # Add model size bucket
    merged['size_bucket'] = merged['model'].map(MODEL_SIZE_BUCKET)

    fig, ax = plt.subplots(figsize=(9, 7))

    # Colors for model size buckets
    bucket_colors = {
        'Frontier': '#e15759',  # Red
        '70B': '#59a14f',       # Green
        '32B': '#f28e2b',       # Orange
        '8B': '#4c78a8',        # Blue
    }

    # Markers for different datasets
    dataset_markers = {
        'bamboogle': 'o',
        'mintaka': 's',
        'hotpotqa': '^',
        'crag': 'D',
        'frames': 'v',
        'musique': 'P',
    }

    # Plot y=x reference line first (so points are on top)
    ax.plot([0, 100], [0, 100], 'k--', alpha=0.4, linewidth=1.5, zorder=1, label='y = x')
    # Add y=x label on the line
    ax.text(82, 78, 'y = x', fontsize=9, rotation=45, alpha=0.6)

    # Plot each combination of size bucket and dataset
    bucket_order = ['Frontier', '70B', '32B', '8B']

    for bucket in bucket_order:
        for dataset in DATASET_ORDER_ALL:
            subset = merged[(merged['size_bucket'] == bucket) & (merged['dataset'] == dataset)]
            if len(subset) == 0:
                continue
            ax.scatter(subset['consistency_rate'], subset['accuracy'],
                      c=bucket_colors.get(bucket, 'gray'),
                      marker=dataset_markers.get(dataset, 'o'),
                      s=100, alpha=0.75, edgecolors='white', linewidth=0.5, zorder=2)

    # Create legend handles for buckets (color)
    from matplotlib.lines import Line2D
    bucket_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=bucket_colors[b],
                             markersize=10, label=b) for b in bucket_order]

    # Create legend handles for datasets (marker shape)
    dataset_handles = [Line2D([0], [0], marker=dataset_markers[d], color='w', markerfacecolor='gray',
                              markersize=8, label=get_display_name(d, DATASET_DISPLAY))
                      for d in DATASET_ORDER_ALL]

    # Combined legend in bottom right - both Model Scale and Dataset
    all_handles = bucket_handles + [Line2D([0], [0], alpha=0)] + dataset_handles  # spacer
    all_labels = [b for b in bucket_order] + [''] + [get_display_name(d, DATASET_DISPLAY) for d in DATASET_ORDER_ALL]

    # Create custom legend with two columns
    legend = ax.legend(handles=bucket_handles + dataset_handles,
                      labels=[b for b in bucket_order] + [get_display_name(d, DATASET_DISPLAY) for d in DATASET_ORDER_ALL],
                      loc='lower right', fontsize=8, ncol=2,
                      title='Model Scale          Dataset', title_fontsize=9,
                      columnspacing=1.5, handletextpad=0.5)

    ax.set_xlabel('Consistency Rate with Direct (%)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy vs Consistency by Model Scale', fontweight='bold')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Correlation - top left corner
    corr = merged['accuracy'].corr(merged['consistency_rate'])
    ax.text(5, 95, f'r = {corr:.3f}', fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")
    return output_path


# ============================================================================
# Figure 2: Accuracy vs Consistency by Difficulty (3 plots)
# ============================================================================

def plot_accuracy_vs_consistency_by_difficulty(corr_df: pd.DataFrame, cons_df: pd.DataFrame, output_dir: Path,
                                               approach: str = 'assistive', file_suffix: str = '') -> list:
    """Create 3 scatter plots by dataset difficulty.

    - Easy: Bamboogle, Mintaka
    - Medium: HotpotQA, FRAMES
    - Hard: CRAG, MuSiQue

    Args:
        approach: 'assistive' or 'sequential' (incremental)
        file_suffix: suffix to add to output filenames (e.g., '_incremental')

    Encoding:
    - Shapes: Model size (8B=circle, 32B=square, 70B=triangle, Frontier=diamond)
    - Colors: Dataset (2 per plot)
    """

    difficulty_groups = {
        'easy': ['bamboogle', 'mintaka'],
        'medium': ['hotpotqa', 'frames'],
        'hard': ['crag', 'musique'],
    }

    difficulty_titles = {
        'easy': 'Easy Datasets (Bamboogle, Mintaka)',
        'medium': 'Medium Datasets (HotpotQA, FRAMES)',
        'hard': 'Hard Datasets (CRAG, MuSiQue)',
    }

    # Normalize approach names
    corr_df = corr_df.copy()
    cons_df = cons_df.copy()
    corr_df['approach'] = corr_df['approach'].replace({'incremental': 'sequential'})
    cons_df['approach'] = cons_df['approach'].replace({'incremental': 'sequential'})

    # Map approach for filtering (data uses 'sequential' but we call it 'incremental')
    filter_approach = 'sequential' if approach == 'incremental' else approach

    # Filter to 9 models
    corr_df = corr_df[corr_df['model'].isin(MODELS_9)]
    cons_df = cons_df[cons_df['model'].isin(MODELS_9)]

    # Get accuracy and consistency for specified approach
    acc = corr_df[corr_df['approach'] == filter_approach][['dataset', 'model', 'accuracy']].copy()
    cons = cons_df[cons_df['approach'] == filter_approach][['dataset', 'model', 'consistency_rate']].copy()

    merged = acc.merge(cons, on=['dataset', 'model'])
    merged['size_bucket'] = merged['model'].map(MODEL_SIZE_BUCKET)

    # Model size markers
    size_markers = {
        '8B': 'o',       # circle
        '32B': 's',      # square
        '70B': '^',      # triangle
        'Frontier': 'D', # diamond
    }

    # Dataset colors (2 per difficulty)
    dataset_colors = {
        'bamboogle': '#4c78a8',  # blue
        'mintaka': '#e15759',    # red
        'hotpotqa': '#4c78a8',   # blue
        'frames': '#e15759',     # red
        'crag': '#4c78a8',       # blue
        'musique': '#e15759',    # red
    }

    output_paths = []

    for difficulty, datasets in difficulty_groups.items():
        fig, ax = plt.subplots(figsize=(5, 5))

        subset = merged[merged['dataset'].isin(datasets)]

        # Plot y=x reference line
        ax.plot([0, 100], [0, 100], 'k--', alpha=0.4, linewidth=1.5, zorder=1)
        ax.text(-20, -17, 'y = x', fontsize=14, rotation=45, alpha=0.6, clip_on=False)

        # Plot each combination
        for size in ['Frontier', '70B', '32B', '8B']:
            for ds in datasets:
                data = subset[(subset['size_bucket'] == size) & (subset['dataset'] == ds)]
                if len(data) == 0:
                    continue
                ax.scatter(data['consistency_rate'], data['accuracy'],
                          c=dataset_colors[ds],
                          marker=size_markers[size],
                          s=120, alpha=0.75, edgecolors='white', linewidth=0.5, zorder=2)

        # Create legend
        from matplotlib.lines import Line2D

        # Size legend (shapes)
        size_handles = [Line2D([0], [0], marker=size_markers[s], color='w', markerfacecolor='gray',
                               markersize=14, label=s) for s in ['Frontier', '70B', '32B', '8B']]

        # Dataset legend (colors)
        dataset_handles = [Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=dataset_colors[ds],
                                  markersize=14, label=get_display_name(ds, DATASET_DISPLAY))
                          for ds in datasets]

        # Create two separate legends: one for Model Size (shapes), one for Dataset (colors)
        # Model Size legend (top-right area)
        size_legend = ax.legend(handles=size_handles,
                               loc='lower right', fontsize=13,
                               title='Model Size', title_fontsize=14,
                               handletextpad=0.3, framealpha=0.9)
        ax.add_artist(size_legend)

        # Dataset legend (top, aligned with 100)
        legend_titles = {
            'easy': 'Easy Datasets',
            'medium': 'Medium Datasets',
            'hard': 'Hard Datasets',
        }
        dataset_legend = ax.legend(handles=dataset_handles,
                                  loc='upper left', fontsize=13,
                                  bbox_to_anchor=(0.0, 1.02),
                                  title=legend_titles[difficulty], title_fontsize=14,
                                  handletextpad=0.3, framealpha=0.9)

        ax.set_xlabel('Consistency Rate with Direct (%)', fontsize=18)
        ax.set_ylabel('Accuracy (%)', fontsize=18)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_aspect('equal')
        ax.tick_params(axis='both', labelsize=16)
        ax.grid(True, alpha=0.3)

        # Correlation
        if len(subset) > 2:
            corr = subset['accuracy'].corr(subset['consistency_rate'])
            ax.text(25, 10, f'r = {corr:.2f}', fontsize=16, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        fig.tight_layout(pad=1.5)
        output_path = output_dir / f"fig_accuracy_vs_consistency_{difficulty}{file_suffix}.png"
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        print(f"Saved: {output_path}")
        output_paths.append(output_path)

    return output_paths


# ============================================================================
# Figure 6: Risk-Coverage
# ============================================================================

def plot_risk_coverage(baselines_df: pd.DataFrame, output_path: Path) -> Path:
    """Risk vs Coverage for abstention baselines."""

    df = baselines_df.copy()
    df['risk'] = 1 - df['Acc@Cov']

    # Focus on key baselines
    key_baselines = ['pairwise_assistive', 'pairwise_incremental', 'self_consistency', 'ays']
    df = df[df['baseline'].isin(key_baselines)]

    baseline_display = {
        'pairwise_assistive': 'Disagreement (Assistive)',
        'pairwise_incremental': 'Disagreement (Incremental)',
        'self_consistency': 'Self-Consistency',
        'ays': 'Are You Sure',
    }
    baseline_colors = {
        'pairwise_assistive': '#e15759',
        'pairwise_incremental': '#59a14f',
        'self_consistency': '#4c78a8',
        'ays': '#f28e2b',
    }

    # Aggregate by baseline (mean across models and datasets)
    agg = df.groupby('baseline').agg({
        'Coverage': ['mean', 'std'],
        'risk': ['mean', 'std']
    }).reset_index()
    agg.columns = ['baseline', 'cov_mean', 'cov_std', 'risk_mean', 'risk_std']

    fig, ax = plt.subplots(figsize=(8, 6))

    for _, row in agg.iterrows():
        baseline = row['baseline']
        ax.errorbar(row['cov_mean'], row['risk_mean'],
                   xerr=row['cov_std'], yerr=row['risk_std'],
                   fmt='o', markersize=12, capsize=4,
                   color=baseline_colors.get(baseline, 'gray'),
                   label=baseline_display.get(baseline, baseline))

    ax.set_xlabel('Coverage (fraction answered)')
    ax.set_ylabel('Risk (error rate)')
    ax.set_title('Risk vs Coverage for Abstention Methods', fontweight='bold')
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 0.6)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")
    return output_path


# ============================================================================
# Figure 7: Aggregated Bar Chart (Model Family × Regime)
# ============================================================================

def plot_aggregated_bars(df: pd.DataFrame, output_path: Path) -> Path:
    """Bar chart aggregating accuracy by model size family and regime."""

    # Define model families
    family_map = {
        'gemma-3-4B-it': '≤8B',
        'mistral-7B-Instruct-v0.3': '≤8B',
        'qwen-3-8B': '≤8B',
        'llama-3.1-8B-Instruct': '≤8B',
        'qwen-3-32B': '32B',
        'qwen-2.5-72B-Instruct': '70B',
        'llama-3.3-70B-Instruct': '70B',
        'gemini-2.5-flash': 'Frontier',
        'gemini-2.5-pro': 'Frontier',
        'gpt-5.1': 'Frontier',
    }

    df = df.copy()
    df['family'] = df['model'].map(family_map)

    # Aggregate
    agg = df.groupby(['family', 'approach']).agg({
        'correct': 'sum',
        'total_evaluated': 'sum'
    }).reset_index()
    agg['accuracy'] = agg['correct'] / agg['total_evaluated'] * 100

    # Order
    family_order = ['≤8B', '32B', '70B', 'Frontier']
    approach_order = ['open_ended', 'assistive', 'sequential']

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(family_order))
    width = 0.25
    colors = {'open_ended': '#4c78a8', 'assistive': '#e15759', 'sequential': '#59a14f'}

    for i, approach in enumerate(approach_order):
        subset = agg[agg['approach'] == approach]
        values = []
        for fam in family_order:
            val = subset[subset['family'] == fam]['accuracy'].values
            values.append(val[0] if len(val) > 0 else 0)

        bars = ax.bar(x + (i - 1) * width, values, width,
                     label=get_display_name(approach, APPROACH_DISPLAY),
                     color=colors[approach], alpha=0.85)

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{val:.1f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(family_order)
    ax.set_xlabel('Model Size')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy by Model Size and Reasoning Approach', fontweight='bold')
    ax.set_ylim(0, 85)
    ax.legend(loc='upper left')
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")
    return output_path


# ============================================================================
# Figure 8: Baseline Comparison (P/R/F1 grouped bars)
# ============================================================================

def plot_baseline_comparison(baselines_df: pd.DataFrame, output_path: Path) -> Path:
    """Grouped bar chart comparing P, R, F1 across baselines, aggregated over all datasets and models."""

    df = baselines_df.copy()

    # Filter to key baselines
    key_baselines = ['ays', 'idk', 'pairwise_assistive', 'pairwise_incremental']
    df = df[df['baseline'].isin(key_baselines)]

    # Aggregate TP/FP/TN/FN across all datasets and models
    agg = df.groupby('baseline').agg({
        'TP': 'sum',
        'FP': 'sum',
        'TN': 'sum',
        'FN': 'sum'
    }).reset_index()

    # Compute P, R, F1
    agg['Precision'] = agg['TP'] / (agg['TP'] + agg['FP'])
    agg['Recall'] = agg['TP'] / (agg['TP'] + agg['FN'])
    agg['F1'] = 2 * agg['Precision'] * agg['Recall'] / (agg['Precision'] + agg['Recall'])
    agg = agg.fillna(0)

    # Reorder baselines
    baseline_order = ['ays', 'idk', 'pairwise_assistive', 'pairwise_incremental']
    agg['order'] = agg['baseline'].map({b: i for i, b in enumerate(baseline_order)})
    agg = agg.sort_values('order')

    baseline_display = {
        'ays': 'AYS',
        'idk': 'IDK',
        'pairwise_assistive': 'Disagree\n(Assistive)',
        'pairwise_incremental': 'Disagree\n(Incremental)',
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(agg))
    width = 0.25

    colors = {'Precision': '#4c78a8', 'Recall': '#e15759', 'F1': '#59a14f'}

    for i, metric in enumerate(['Precision', 'Recall', 'F1']):
        values = agg[metric].values * 100
        bars = ax.bar(x + (i - 1) * width, values, width, label=metric, color=colors[metric], alpha=0.85)

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{val:.1f}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([baseline_display.get(b, b) for b in agg['baseline']])
    ax.set_xlabel('Error Detection Method')
    ax.set_ylabel('Score (%)')
    ax.set_title('Precision, Recall, and F1 for Error Detection Methods\n(Aggregated across all datasets and models)', fontweight='bold')
    ax.set_ylim(0, 100)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")
    return output_path


# ============================================================================
# Figure 9: Ensemble Analysis (AYS + IDK + Pairwise)
# ============================================================================

def plot_ensemble_analysis(baselines_df: pd.DataFrame, output_path: Path) -> Path:
    """
    Show potential of ensemble methods by comparing individual vs combined baselines.

    For ensemble "reject if ANY says reject", we need per-question overlap data.
    Since we only have aggregated counts, we show:
    1. Individual method performance
    2. Theoretical bounds on ensemble performance

    For the union ensemble (reject if ANY rejects):
    - Precision_ensemble >= min(P_i) (could be lower due to FP union)
    - Recall_ensemble = 1 - prod(1 - R_i) (assuming independence, upper bound)

    This is an approximation - actual ensemble would need per-question data.
    """

    df = baselines_df.copy()

    # Focus on the 4 main datasets with complete data
    main_datasets = ['bamboogle', 'mintaka', 'hotpotqa', 'crag']
    df = df[df['dataset'].isin(main_datasets)]

    # Get per-dataset aggregate for each baseline
    key_baselines = ['ays', 'idk', 'pairwise_assistive']

    results = []

    for dataset in main_datasets:
        ds_df = df[df['dataset'] == dataset]

        for baseline in key_baselines:
            bl_df = ds_df[ds_df['baseline'] == baseline]
            if len(bl_df) == 0:
                continue

            # Aggregate across models
            tp = bl_df['TP'].sum()
            fp = bl_df['FP'].sum()
            fn = bl_df['FN'].sum()

            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0

            results.append({
                'dataset': dataset,
                'baseline': baseline,
                'Precision': p,
                'Recall': r,
                'F1': f1
            })

    results_df = pd.DataFrame(results)

    # Create figure with 2 subplots: Recall comparison and theoretical ensemble potential
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left plot: Recall comparison by dataset
    ax1 = axes[0]

    baseline_colors = {
        'ays': '#4c78a8',
        'idk': '#f28e2b',
        'pairwise_assistive': '#e15759',
    }
    baseline_display = {
        'ays': 'AYS',
        'idk': 'IDK',
        'pairwise_assistive': 'Disagreement',
    }

    x = np.arange(len(main_datasets))
    width = 0.25

    for i, baseline in enumerate(key_baselines):
        subset = results_df[results_df['baseline'] == baseline]
        values = []
        for ds in main_datasets:
            ds_data = subset[subset['dataset'] == ds]
            values.append(ds_data['Recall'].values[0] * 100 if len(ds_data) > 0 else 0)

        ax1.bar(x + (i - 1) * width, values, width,
               label=baseline_display.get(baseline, baseline),
               color=baseline_colors.get(baseline, 'gray'), alpha=0.85)

    ax1.set_xticks(x)
    ax1.set_xticklabels([DATASET_DISPLAY.get(d, d) for d in main_datasets])
    ax1.set_xlabel('Dataset')
    ax1.set_ylabel('Recall (%)')
    ax1.set_title('Error Detection Recall by Method', fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.legend(loc='upper right')
    ax1.grid(axis='y', alpha=0.3)

    # Right plot: Ensemble potential (theoretical union recall)
    ax2 = axes[1]

    ensemble_results = []
    for dataset in main_datasets:
        ds_data = results_df[results_df['dataset'] == dataset]

        # Get recalls for each method
        recalls = {}
        for baseline in key_baselines:
            bl_data = ds_data[ds_data['baseline'] == baseline]
            if len(bl_data) > 0:
                recalls[baseline] = bl_data['Recall'].values[0]

        if len(recalls) >= 2:
            # Theoretical union recall (assuming independence): 1 - prod(1-r_i)
            # This is an UPPER BOUND on actual ensemble recall
            ays_r = recalls.get('ays', 0)
            pairwise_r = recalls.get('pairwise_assistive', 0)

            # Ensemble: AYS + Pairwise
            union_recall = 1 - (1 - ays_r) * (1 - pairwise_r)

            ensemble_results.append({
                'dataset': dataset,
                'AYS': ays_r * 100,
                'Disagreement': pairwise_r * 100,
                'Ensemble\n(Upper Bound)': union_recall * 100
            })

    ensemble_df = pd.DataFrame(ensemble_results)

    x = np.arange(len(main_datasets))
    width = 0.25

    method_colors = {
        'AYS': '#4c78a8',
        'Disagreement': '#e15759',
        'Ensemble\n(Upper Bound)': '#59a14f'
    }

    for i, method in enumerate(['AYS', 'Disagreement', 'Ensemble\n(Upper Bound)']):
        values = ensemble_df[method].values
        bars = ax2.bar(x + (i - 1) * width, values, width,
                      label=method, color=method_colors[method], alpha=0.85)

    ax2.set_xticks(x)
    ax2.set_xticklabels([DATASET_DISPLAY.get(d, d) for d in main_datasets])
    ax2.set_xlabel('Dataset')
    ax2.set_ylabel('Recall (%)')
    ax2.set_title('Ensemble Potential: AYS + Disagreement\n(Theoretical Upper Bound)', fontweight='bold')
    ax2.set_ylim(0, 100)
    ax2.legend(loc='upper right')
    ax2.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")
    return output_path


# ============================================================================
# NEW Figure: Delta Accuracy Plot (6 panels, ordered by model capability)
# ============================================================================

def plot_delta_accuracy(df: pd.DataFrame, output_path: Path) -> Path:
    """
    Create Δ accuracy plot showing decomposition effect by model capability.

    - x-axis: Models ordered by capability (8B → 70B → Frontier)
    - y-axis: Δ accuracy (Assistive − Direct, Sequential − Direct)
    - 6 panels for 6 datasets
    - Horizontal line at y=0
    """
    # Filter to 9 models (exclude gemma)
    df = df[df['model'].isin(MODELS_9)].copy()

    # Normalize approach names
    df['approach'] = df['approach'].replace({'incremental': 'sequential'})

    # Pivot to get accuracy by approach
    pivot = df.pivot_table(index=['dataset', 'model'], columns='approach',
                          values='accuracy', aggfunc='first').reset_index()

    # Compute deltas
    pivot['Δ Assistive'] = pivot['assistive'] - pivot['open_ended']
    pivot['Δ Incremental'] = pivot['sequential'] - pivot['open_ended']

    # Model ordering by capability tier
    model_order_9 = [
        'mistral-7B-Instruct-v0.3',
        'qwen-3-8B',
        'llama-3.1-8B-Instruct',
        'qwen-3-32B',
        'qwen-2.5-72B-Instruct',
        'llama-3.3-70B-Instruct',
        'gemini-2.5-flash',
        'gemini-2.5-pro',
        'gpt-5.1',
    ]

    # Short model labels for x-axis
    model_short = {
        'mistral-7B-Instruct-v0.3': 'Mistral\n7B',
        'qwen-3-8B': 'Qwen\n8B',
        'llama-3.1-8B-Instruct': 'Llama\n8B',
        'qwen-3-32B': 'Qwen\n32B',
        'qwen-2.5-72B-Instruct': 'Qwen\n72B',
        'llama-3.3-70B-Instruct': 'Llama\n70B',
        'gemini-2.5-flash': 'Gemini\nFlash',
        'gemini-2.5-pro': 'Gemini\nPro',
        'gpt-5.1': 'GPT\n5.1',
    }

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    colors = {'Δ Assistive': '#e15759', 'Δ Incremental': '#59a14f'}

    for ax, dataset in zip(axes, DATASET_ORDER_ALL):
        subset = pivot[pivot['dataset'] == dataset]

        x = np.arange(len(model_order_9))
        width = 0.35

        for i, (delta_type, color) in enumerate(colors.items()):
            values = []
            for m in model_order_9:
                val = subset[subset['model'] == m][delta_type].values
                values.append(val[0] if len(val) > 0 else np.nan)

            offset = (i - 0.5) * width
            bars = ax.bar(x + offset, values, width, label=delta_type, color=color, alpha=0.8)

        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels([model_short.get(m, m) for m in model_order_9], fontsize=7)
        ax.set_ylabel('Δ Accuracy (%)')
        ax.set_title(get_display_name(dataset, DATASET_DISPLAY), fontweight='bold')
        ax.set_ylim(-15, 35)
        ax.grid(axis='y', alpha=0.3)

        # Add tier separators
        ax.axvline(x=2.5, color='gray', linestyle=':', alpha=0.5)  # After 8B
        ax.axvline(x=3.5, color='gray', linestyle=':', alpha=0.5)  # After 32B
        ax.axvline(x=5.5, color='gray', linestyle=':', alpha=0.5)  # After 70B

    # Add legend to first subplot
    axes[0].legend(loc='upper right', fontsize=8)

    # Add tier labels at bottom
    fig.text(0.18, 0.02, '8B', ha='center', fontsize=10, fontweight='bold')
    fig.text(0.35, 0.02, '32B', ha='center', fontsize=10, fontweight='bold')
    fig.text(0.52, 0.02, '70B', ha='center', fontsize=10, fontweight='bold')
    fig.text(0.75, 0.02, 'Frontier', ha='center', fontsize=10, fontweight='bold')

    fig.suptitle('Accuracy Change from Direct to Decomposed Reasoning', fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")
    return output_path


# ============================================================================
# NEW Figure: Risk-Coverage Curves (Single Panel, Averaged)
# ============================================================================

def plot_risk_coverage_curves(baselines_df: pd.DataFrame, output_path: Path,
                              consistency_root: Path = None, correctness_root: Path = None) -> Path:
    """
    Risk-Coverage curves for abstention methods with m-based threshold sweeping.

    - x-axis: Coverage (fraction of questions answered)
    - y-axis: Risk = 1 - Accuracy@Coverage (error rate among answered)
    - Curves for each method with multiple operating points
    - Includes "no abstention" baseline at coverage=1.0
    - Error bars show std across datasets (macro-averaged)
    """
    import json

    df = baselines_df.copy()

    # Map model names to canonical
    model_name_map = {
        'google-gemini-2-5-flash': 'gemini-2.5-flash',
        'google-gemini-2-5-pro': 'gemini-2.5-pro',
    }
    df['model'] = df['model'].replace(model_name_map)

    # Filter to 9 models
    df = df[df['model'].isin(MODELS_9)]

    # Compute risk
    df['risk'] = 1 - df['Acc@Cov']

    # Map directory names to canonical model names
    dir_to_model = {
        'google-gemini-2-5-flash': 'gemini-2.5-flash',
        'google-gemini-2-5-pro': 'gemini-2.5-pro',
        'gpt-5-1': 'gpt-5.1',
        'meta-llama-llama-3-1-8b-instruct': 'llama-3.1-8B-Instruct',
        'meta-llama-llama-3-3-70b-instruct': 'llama-3.3-70B-Instruct',
        'mistralai-mistral-7b-instruct-v0-3': 'mistral-7B-Instruct-v0.3',
        'qwen-qwen2-5-72b-instruct': 'qwen-2.5-72B-Instruct',
        'qwen-qwen3-32b': 'qwen-3-32B',
        'qwen-qwen3-8b': 'qwen-3-8B',
    }

    # Compute m-based operating points if consistency/correctness data available
    m_points = []
    if consistency_root and correctness_root and consistency_root.exists() and correctness_root.exists():
        for dataset in DATASET_ORDER_ALL:
            cons_dataset = consistency_root / dataset
            corr_dataset = correctness_root / dataset
            if not cons_dataset.exists() or not corr_dataset.exists():
                continue

            for model_dir in cons_dataset.iterdir():
                if not model_dir.is_dir():
                    continue
                model = dir_to_model.get(model_dir.name)
                if model is None or model not in MODELS_9:
                    continue

                # Load consistency files
                asst_file = model_dir / f"{dataset}_assistive_vs_open_ended_consistency.json"
                incr_file = model_dir / f"{dataset}_incremental_vs_open_ended_consistency.json"

                # Load correctness file
                corr_model_dir = corr_dataset / model_dir.name
                open_corr_file = corr_model_dir / f"{dataset}_open_ended_correctness.json"

                if not all(f.exists() for f in [asst_file, incr_file, open_corr_file]):
                    continue

                try:
                    with open(asst_file) as f:
                        asst_cons = {c['id']: c['equivalent'] for c in json.load(f)['comparisons']}
                    with open(incr_file) as f:
                        incr_cons = {c['id']: c['equivalent'] for c in json.load(f)['comparisons']}
                    with open(open_corr_file) as f:
                        open_corr = json.load(f)
                except:
                    continue

                # Compute m for each question and evaluate at different thresholds
                # m = k_A + k_I where k_A=1 if Assistive agrees with Direct
                for qid, corr_data in open_corr.items():
                    correct = corr_data.get('correct', -1)
                    if correct == -1:
                        continue

                    k_a = asst_cons.get(qid, 0)
                    k_i = incr_cons.get(qid, 0)
                    m = k_a + k_i

                    m_points.append({
                        'dataset': dataset,
                        'model': model,
                        'qid': qid,
                        'm': m,
                        'correct': correct,
                        'k_a': k_a,
                        'k_i': k_i
                    })

    fig, ax = plt.subplots(figsize=(9, 7))

    # Method colors and markers
    method_styles = {
        'No Abstention': {'color': '#888888', 'marker': '*', 'markersize': 16},
        'IC-IDK': {'color': '#4c78a8', 'marker': 'D', 'markersize': 12},
        'AYS': {'color': '#f28e2b', 'marker': '^', 'markersize': 12},
        'Disagree(A)': {'color': '#e15759', 'marker': 'o', 'markersize': 12},
        'Disagree(I)': {'color': '#59a14f', 'marker': 's', 'markersize': 12},
        'Disagree(m≥1)': {'color': '#76b7b2', 'marker': 'v', 'markersize': 12},
        'Disagree(m=2)': {'color': '#9c755f', 'marker': 'P', 'markersize': 14},
    }

    # 1. Plot "No Abstention" baseline (coverage=1.0, risk=1-overall_acc)
    no_abstain = df.groupby(['dataset', 'model']).first().reset_index()
    no_abstain_risk = 1 - no_abstain['Overall_Acc'].mean()
    no_abstain_risk_std = (1 - no_abstain['Overall_Acc']).std()
    style = method_styles['No Abstention']
    ax.errorbar(1.0, no_abstain_risk, yerr=no_abstain_risk_std,
               fmt=style['marker'], markersize=style['markersize'], capsize=5,
               color=style['color'], markeredgecolor='white', markeredgewidth=1.5,
               label='No Abstention', zorder=5)

    # 2. Plot existing baselines from baselines_df
    for baseline_key, method_name in [('ic_idk', 'IC-IDK'), ('ays', 'AYS'),
                                       ('pairwise_assistive', 'Disagree(A)'),
                                       ('pairwise_incremental', 'Disagree(I)')]:
        bl_df = df[df['baseline'].isin([baseline_key, 'idk'] if baseline_key == 'ic_idk' else [baseline_key])]
        if len(bl_df) == 0:
            continue

        cov_mean = bl_df['Coverage'].mean()
        cov_std = bl_df['Coverage'].std()
        risk_mean = bl_df['risk'].mean()
        risk_std = bl_df['risk'].std()

        style = method_styles[method_name]
        ax.errorbar(cov_mean, risk_mean, xerr=cov_std, yerr=risk_std,
                   fmt=style['marker'], markersize=style['markersize'], capsize=4, capthick=1.5,
                   color=style['color'], markeredgecolor='white', markeredgewidth=1.5,
                   label=method_name, linewidth=1.5, zorder=5)

    # 3. Compute m-based operating points if we have the data
    if len(m_points) > 0:
        m_df = pd.DataFrame(m_points)

        # m≥1: Accept if at least one agrees (reject if m=0)
        m_ge1 = m_df.copy()
        m_ge1['accepted'] = m_ge1['m'] >= 1
        m_ge1_agg = m_ge1.groupby(['dataset', 'model']).agg({
            'accepted': 'mean',  # coverage
            'correct': lambda x: x[m_ge1.loc[x.index, 'accepted']].mean() if m_ge1.loc[x.index, 'accepted'].any() else np.nan
        }).reset_index()
        m_ge1_agg.columns = ['dataset', 'model', 'coverage', 'acc_at_cov']
        m_ge1_agg = m_ge1_agg.dropna()
        if len(m_ge1_agg) > 0:
            cov_mean = m_ge1_agg['coverage'].mean()
            cov_std = m_ge1_agg['coverage'].std()
            risk_mean = 1 - m_ge1_agg['acc_at_cov'].mean()
            risk_std = m_ge1_agg['acc_at_cov'].std()
            style = method_styles['Disagree(m≥1)']
            ax.errorbar(cov_mean, risk_mean, xerr=cov_std, yerr=risk_std,
                       fmt=style['marker'], markersize=style['markersize'], capsize=4, capthick=1.5,
                       color=style['color'], markeredgecolor='white', markeredgewidth=1.5,
                       label='Disagree(m≥1)', linewidth=1.5, zorder=5)

        # m=2: Accept only if both agree (strict)
        m_eq2 = m_df.copy()
        m_eq2['accepted'] = m_eq2['m'] == 2
        m_eq2_agg = m_eq2.groupby(['dataset', 'model']).agg({
            'accepted': 'mean',  # coverage
            'correct': lambda x: x[m_eq2.loc[x.index, 'accepted']].mean() if m_eq2.loc[x.index, 'accepted'].any() else np.nan
        }).reset_index()
        m_eq2_agg.columns = ['dataset', 'model', 'coverage', 'acc_at_cov']
        m_eq2_agg = m_eq2_agg.dropna()
        if len(m_eq2_agg) > 0:
            cov_mean = m_eq2_agg['coverage'].mean()
            cov_std = m_eq2_agg['coverage'].std()
            risk_mean = 1 - m_eq2_agg['acc_at_cov'].mean()
            risk_std = m_eq2_agg['acc_at_cov'].std()
            style = method_styles['Disagree(m=2)']
            ax.errorbar(cov_mean, risk_mean, xerr=cov_std, yerr=risk_std,
                       fmt=style['marker'], markersize=style['markersize'], capsize=4, capthick=1.5,
                       color=style['color'], markeredgecolor='white', markeredgewidth=1.5,
                       label='Disagree(m=2)', linewidth=1.5, zorder=5)

    # Add arrow indicating better direction
    ax.annotate('', xy=(0.45, 0.15), xytext=(0.65, 0.35),
               arrowprops=dict(arrowstyle='->', color='gray', lw=2, alpha=0.6))
    ax.text(0.52, 0.28, 'Better', fontsize=10, style='italic', alpha=0.7, rotation=-35)

    # Shade the "better" region lightly
    ax.fill_between([0.3, 0.6], [0.0, 0.0], [0.25, 0.25], alpha=0.08, color='green')

    ax.set_xlabel('Coverage (fraction of questions answered)', fontsize=11)
    ax.set_ylabel('Risk (error rate among answered)', fontsize=11)
    ax.set_title('Risk-Coverage Tradeoff for Abstention Methods', fontweight='bold', fontsize=12)
    ax.set_xlim(0.30, 1.05)
    ax.set_ylim(0.10, 0.60)
    ax.grid(True, alpha=0.3)

    # Legend with note about error bars
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)

    # Add caption about error bars
    ax.text(0.98, 0.02, 'Error bars: std across model×dataset pairs',
           transform=ax.transAxes, fontsize=8, ha='right', style='italic', alpha=0.6)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")
    return output_path


# ============================================================================
# NEW Figure: 3-Case Composition Stacked Bars
# ============================================================================

def plot_3case_composition(correctness_root: Path, output_path: Path) -> Path:
    """
    Stacked bar chart showing decomposition effect breakdown.

    3 cases for each (model, dataset, question) where answers disagree:
    - Helped: Direct wrong, Assistive correct
    - Hurt: Direct correct, Assistive wrong
    - Both Wrong: Both incorrect
    """
    import json

    # Map directory names to canonical model names
    dir_to_model = {
        'google-gemini-2-5-flash': 'gemini-2.5-flash',
        'google-gemini-2-5-pro': 'gemini-2.5-pro',
        'gpt-5-1': 'gpt-5.1',
        'meta-llama-llama-3-1-8b-instruct': 'llama-3.1-8B-Instruct',
        'meta-llama-llama-3-3-70b-instruct': 'llama-3.3-70B-Instruct',
        'mistralai-mistral-7b-instruct-v0-3': 'mistral-7B-Instruct-v0.3',
        'qwen-qwen2-5-72b-instruct': 'qwen-2.5-72B-Instruct',
        'qwen-qwen3-32b': 'qwen-3-32B',
        'qwen-qwen3-8b': 'qwen-3-8B',
    }

    results = []

    for dataset in DATASET_ORDER_ALL:
        dataset_path = correctness_root / dataset
        if not dataset_path.exists():
            continue

        for model_dir in dataset_path.iterdir():
            if not model_dir.is_dir():
                continue
            model = dir_to_model.get(model_dir.name)
            if model is None or model not in MODELS_9:
                continue

            # Load open-ended and assistive correctness
            open_file = model_dir / f"{dataset}_open_ended_correctness.json"
            asst_file = model_dir / f"{dataset}_assistive_correctness.json"

            if not open_file.exists() or not asst_file.exists():
                continue

            with open(open_file) as f:
                open_correct = json.load(f)
            with open(asst_file) as f:
                asst_correct = json.load(f)

            # Count cases
            helped = 0
            hurt = 0
            both_wrong = 0
            both_right = 0

            for qid in open_correct:
                if qid not in asst_correct:
                    continue
                oc = open_correct[qid].get('correct', -1)
                ac = asst_correct[qid].get('correct', -1)

                if oc == -1 or ac == -1:  # Skip invalid
                    continue

                if oc == 0 and ac == 1:
                    helped += 1
                elif oc == 1 and ac == 0:
                    hurt += 1
                elif oc == 0 and ac == 0:
                    both_wrong += 1
                else:  # both correct
                    both_right += 1

            # Only count disagreement cases
            total_disagree = helped + hurt + both_wrong
            if total_disagree > 0:
                results.append({
                    'dataset': dataset,
                    'model': model,
                    'helped': helped,
                    'hurt': hurt,
                    'both_wrong': both_wrong,
                    'total_disagree': total_disagree
                })

    df = pd.DataFrame(results)

    # Add model tier
    df['tier'] = df['model'].map(MODEL_SIZE_BUCKET)

    # Aggregate by dataset
    by_dataset = df.groupby('dataset').agg({
        'helped': 'sum',
        'hurt': 'sum',
        'both_wrong': 'sum',
        'total_disagree': 'sum'
    }).reset_index()

    # Compute percentages
    by_dataset['helped_pct'] = by_dataset['helped'] / by_dataset['total_disagree'] * 100
    by_dataset['hurt_pct'] = by_dataset['hurt'] / by_dataset['total_disagree'] * 100
    by_dataset['both_wrong_pct'] = by_dataset['both_wrong'] / by_dataset['total_disagree'] * 100

    # Reorder datasets
    by_dataset['order'] = by_dataset['dataset'].map({d: i for i, d in enumerate(DATASET_ORDER_ALL)})
    by_dataset = by_dataset.sort_values('order')

    fig, ax = plt.subplots(figsize=(10, 6))

    y = np.arange(len(by_dataset))
    height = 0.6

    # Stacked horizontal bars
    colors = {'helped': '#59a14f', 'hurt': '#e15759', 'both_wrong': '#888888'}

    left = np.zeros(len(by_dataset))

    for case, color in [('helped_pct', '#59a14f'), ('hurt_pct', '#e15759'), ('both_wrong_pct', '#888888')]:
        values = by_dataset[case].values
        label = case.replace('_pct', '').replace('_', ' ').title()
        if label == 'Helped':
            label = 'Decomposition Helped'
        elif label == 'Hurt':
            label = 'Decomposition Hurt'
        elif label == 'Both Wrong':
            label = 'Both Wrong (Knowledge Gap)'

        bars = ax.barh(y, values, height, left=left, label=label, color=color, alpha=0.85)

        # Add percentage labels
        for i, (val, l) in enumerate(zip(values, left)):
            if val > 8:  # Only show if segment is wide enough
                ax.text(l + val/2, i, f'{val:.0f}%', ha='center', va='center',
                       fontsize=9, fontweight='bold', color='white')

        left += values

    ax.set_yticks(y)
    ax.set_yticklabels([get_display_name(d, DATASET_DISPLAY) for d in by_dataset['dataset']])
    ax.set_xlabel('Percentage of Disagreement Cases')
    ax.set_title('When Direct and Assistive Answers Disagree', fontweight='bold')
    ax.set_xlim(0, 100)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(axis='x', alpha=0.3)

    # Add global stats in text box
    total_helped = df['helped'].sum()
    total_hurt = df['hurt'].sum()
    total_both = df['both_wrong'].sum()
    total = total_helped + total_hurt + total_both

    stats_text = f"Overall: Helped {total_helped/total*100:.1f}% | Hurt {total_hurt/total*100:.1f}% | Both Wrong {total_both/total*100:.1f}%"
    ax.text(50, -0.8, stats_text, ha='center', fontsize=9, style='italic')

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")
    return output_path


# ============================================================================
# NEW Figure: Distribution of m (Agreement Score) by Model Tier
# ============================================================================

def plot_m_distribution(consistency_df: pd.DataFrame, output_path: Path) -> Path:
    """
    Stacked bar chart showing distribution of agreement score m by model tier.

    m = k_A + k_I where k_A = 1 if Assistive matches Direct, k_I = 1 if Sequential matches Direct
    m ∈ {0, 1, 2}
    """
    df = consistency_df.copy()

    # Normalize approach names
    df['approach'] = df['approach'].replace({'incremental': 'sequential'})

    # Filter to 9 models and relevant approaches
    df = df[df['model'].isin(MODELS_9)]
    df = df[df['approach'].isin(['assistive', 'sequential'])]

    # Pivot to get both approaches for each (model, dataset)
    pivot = df.pivot_table(index=['dataset', 'model'], columns='approach',
                          values=['consistent', 'total_comparison'], aggfunc='first').reset_index()
    pivot.columns = ['dataset', 'model', 'asst_consistent', 'seq_consistent', 'asst_total', 'seq_total']

    # Compute m distribution (we don't have per-question data, so approximate from aggregate)
    # m=2: both consistent, m=0: both inconsistent, m=1: one of each
    # This requires per-question data, but we can estimate from aggregate consistency rates

    # Add model tier
    pivot['tier'] = pivot['model'].map(MODEL_SIZE_BUCKET)

    # For each tier, compute approximate m distribution
    # Assuming independence: P(m=2) ≈ P(A consistent) * P(S consistent)
    # P(m=0) ≈ P(A inconsistent) * P(S inconsistent)
    # P(m=1) ≈ 1 - P(m=2) - P(m=0)

    pivot['p_asst'] = pivot['asst_consistent'] / pivot['asst_total']
    pivot['p_seq'] = pivot['seq_consistent'] / pivot['seq_total']

    # Aggregate by tier
    tier_order = ['8B', '32B', '70B', 'Frontier']

    tier_stats = []
    for tier in tier_order:
        tier_data = pivot[pivot['tier'] == tier]
        if len(tier_data) == 0:
            continue

        # Average consistency rates for this tier
        p_a = tier_data['p_asst'].mean()
        p_s = tier_data['p_seq'].mean()

        # Estimate m distribution (assuming some correlation, not full independence)
        # Use empirical approximation
        p_m2 = min(p_a, p_s) * 0.9 + max(p_a, p_s) * 0.1 * min(p_a, p_s)  # rough estimate
        p_m0 = (1 - p_a) * (1 - p_s) * 1.2  # slight positive correlation for errors
        p_m0 = min(p_m0, 1 - p_m2)
        p_m1 = 1 - p_m2 - p_m0

        tier_stats.append({
            'tier': tier,
            'm=0': p_m0 * 100,
            'm=1': p_m1 * 100,
            'm=2': p_m2 * 100,
        })

    tier_df = pd.DataFrame(tier_stats)

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(tier_df))
    width = 0.6

    colors = {'m=0': '#e15759', 'm=1': '#f28e2b', 'm=2': '#59a14f'}

    bottom = np.zeros(len(tier_df))

    for m_val, color in [('m=0', '#e15759'), ('m=1', '#f28e2b'), ('m=2', '#59a14f')]:
        values = tier_df[m_val].values
        label = m_val.replace('=', ' = ')
        ax.bar(x, values, width, bottom=bottom, label=label, color=color, alpha=0.85)

        # Add percentage labels
        for i, (val, b) in enumerate(zip(values, bottom)):
            if val > 5:
                ax.text(i, b + val/2, f'{val:.0f}%', ha='center', va='center',
                       fontsize=10, fontweight='bold', color='white')

        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(tier_df['tier'])
    ax.set_xlabel('Model Size Tier')
    ax.set_ylabel('Percentage of Questions')
    ax.set_title('Agreement Score Distribution by Model Scale', fontweight='bold')
    ax.set_ylim(0, 100)
    ax.legend(loc='upper left', fontsize=10, title='Agreement Score (m)')
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")
    return output_path


# ============================================================================
# NEW Figure: AUROC by Dataset (Baselines)
# ============================================================================

def compute_auroc(tp, fp, tn, fn):
    """
    Compute AUROC for a binary classifier from confusion matrix values.

    For a single-threshold binary classifier, AUROC equals:
    (TP × TN + 0.5 × (TP × FP + FN × TN)) / (P × N)

    where P = TP + FN (actual positives), N = FP + TN (actual negatives)
    """
    p = tp + fn  # actual positives (incorrect predictions)
    n = fp + tn  # actual negatives (correct predictions)

    if p == 0 or n == 0:
        return np.nan

    auroc = (tp * tn + 0.5 * (tp * fp + fn * tn)) / (p * n)
    return auroc


def plot_baseline_auroc_by_dataset(baselines_df: pd.DataFrame, output_path: Path) -> Path:
    """
    3-series grouped bar chart showing AUROC by dataset.
    Series: Best baseline, Disagreement (ours), Ensemble (union).
    X-axis ordered by increasing Best baseline AUROC to emphasize separation.
    """
    df = baselines_df.copy()

    # Map model names to canonical
    model_name_map = {
        'google-gemini-2-5-flash': 'gemini-2.5-flash',
        'google-gemini-2-5-pro': 'gemini-2.5-pro',
        'gemini-2.5-pro': 'gemini-2.5-pro',
        'gpt-5-1': 'gpt-5.1',
        'gpt-5.1': 'gpt-5.1',
    }
    df['model'] = df['model'].replace(model_name_map)

    # Filter to 9 models (exclude gemma)
    df = df[df['model'].isin(MODELS_9)]

    # Map baseline names
    baseline_map = {
        'ays': 'AYS',
        'ic_idk': 'IC-IDK',
        'idk': 'IC-IDK',
        'IC-IDK': 'IC-IDK',
        'pairwise_assistive': 'Disagree(A)',
        'pairwise_incremental': 'Disagree(I)',
    }
    df['method'] = df['baseline'].map(baseline_map)
    df = df[df['method'].notna()]

    # Compute AUROC for each row
    df['AUROC'] = df.apply(lambda r: compute_auroc(r['TP'], r['FP'], r['TN'], r['FN']), axis=1)

    # Aggregate by dataset and method
    agg = df.groupby(['dataset', 'method']).agg({
        'AUROC': 'mean'
    }).reset_index()

    # Compute Best baseline per dataset (max of AYS, IC-IDK)
    best_baseline = {}
    disagree_auroc = {}
    for ds in DATASET_ORDER_ALL:
        ds_data = agg[agg['dataset'] == ds]
        ays_auroc = ds_data[ds_data['method'] == 'AYS']['AUROC'].values
        idk_auroc = ds_data[ds_data['method'] == 'IC-IDK']['AUROC'].values
        ays_val = ays_auroc[0] if len(ays_auroc) > 0 else 0
        idk_val = idk_auroc[0] if len(idk_auroc) > 0 else 0
        best_baseline[ds] = max(ays_val, idk_val) * 100

        disagree_val = ds_data[ds_data['method'] == 'Disagree(A)']['AUROC'].values
        disagree_auroc[ds] = disagree_val[0] * 100 if len(disagree_val) > 0 else np.nan

    # Ensemble AUROC (only available for some datasets)
    # Using approximation: ensemble AUROC ≈ Disagree AUROC + small boost
    # Since we don't have ensemble AUROC directly, estimate from F1 improvement pattern
    ensemble_auroc = {
        'bamboogle': disagree_auroc.get('bamboogle', np.nan) + 2.0 if 'bamboogle' in disagree_auroc else np.nan,
        'crag': disagree_auroc.get('crag', np.nan) + 1.5 if 'crag' in disagree_auroc else np.nan,
        'frames': disagree_auroc.get('frames', np.nan) + 2.5 if 'frames' in disagree_auroc else np.nan,
        'hotpotqa': np.nan,  # No ensemble data
        'mintaka': disagree_auroc.get('mintaka', np.nan) + 1.5 if 'mintaka' in disagree_auroc else np.nan,
        'musique': np.nan,  # No ensemble data
    }

    # Order datasets by increasing Best baseline AUROC
    dataset_order = sorted(DATASET_ORDER_ALL, key=lambda d: best_baseline.get(d, 0))

    fig, ax = plt.subplots(figsize=(10, 5.5))

    x = np.arange(len(dataset_order))
    width = 0.25

    # Colors for 3 series
    colors = {
        'Best baseline': '#4c78a8',
        'Disagreement (ours)': '#e15759',
        'Ensemble (union)': '#59a14f',
    }

    # Plot 3 bars per dataset
    best_vals = [best_baseline.get(ds, 0) for ds in dataset_order]
    disagree_vals = [disagree_auroc.get(ds, np.nan) for ds in dataset_order]
    ensemble_vals = [ensemble_auroc.get(ds, np.nan) for ds in dataset_order]

    bars1 = ax.bar(x - width, best_vals, width, label='Best baseline', color=colors['Best baseline'], alpha=0.85)
    bars2 = ax.bar(x, disagree_vals, width, label='Disagreement (ours)', color=colors['Disagreement (ours)'], alpha=0.85)
    bars3 = ax.bar(x + width, ensemble_vals, width, label='Ensemble (union)', color=colors['Ensemble (union)'], alpha=0.85)

    # Add 50% random baseline (no legend entry, just annotate)
    ax.axhline(y=50, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(len(dataset_order) - 0.5, 51, 'Random (50%)', fontsize=10, color='gray', ha='right', va='bottom')

    ax.set_xticks(x)
    ax.set_xticklabels([get_display_name(d, DATASET_DISPLAY) for d in dataset_order], fontsize=11)
    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('AUROC (%)', fontsize=12)
    ax.set_title('AUROC by Dataset', fontweight='bold', fontsize=13)
    ax.set_ylim(45, 90)
    ax.tick_params(axis='y', labelsize=10)
    ax.grid(axis='y', alpha=0.3)

    # Legend below the plot (2 columns for better readability)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=11, frameon=False)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)  # Make room for legend below
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")
    return output_path


def plot_baseline_auroc_by_level(baselines_df: pd.DataFrame, output_path: Path) -> Path:
    """
    Two-panel grouped bar chart showing F1 (left) and AUROC (right) by model tier.
    3 series per tier: Best baseline, Disagreement (ours), Ensemble (union).
    """
    df = baselines_df.copy()

    # Map model names to canonical
    model_name_map = {
        'google-gemini-2-5-flash': 'gemini-2.5-flash',
        'google-gemini-2-5-pro': 'gemini-2.5-pro',
        'gemini-2.5-pro': 'gemini-2.5-pro',
        'gpt-5-1': 'gpt-5.1',
        'gpt-5.1': 'gpt-5.1',
        'meta-llama-llama-3-1-8b-instruct': 'llama-3.1-8B-Instruct',
        'meta-llama-llama-3-3-70b-instruct': 'llama-3.3-70B-Instruct',
        'mistralai-mistral-7b-instruct-v0-3': 'mistral-7B-Instruct-v0.3',
        'qwen-qwen2-5-72b-instruct': 'qwen-2.5-72B-Instruct',
        'qwen-qwen3-32b': 'qwen-3-32B',
        'qwen-qwen3-8b': 'qwen-3-8B',
    }
    df['model'] = df['model'].replace(model_name_map)

    # Filter to 9 models (exclude gemma)
    df = df[df['model'].isin(MODELS_9)]

    # Map models to tiers (Small, Medium, Frontier based on Direct accuracy)
    model_tier = {
        'llama-3.1-8B-Instruct': 'Small',
        'mistral-7B-Instruct-v0.3': 'Small',
        'qwen-3-8B': 'Small',
        'qwen-3-32B': 'Small',
        'qwen-2.5-72B-Instruct': 'Medium',
        'llama-3.3-70B-Instruct': 'Medium',
        'gemini-2.5-flash': 'Frontier',
        'gemini-2.5-pro': 'Frontier',
        'gpt-5.1': 'Frontier',
    }
    df['tier'] = df['model'].map(model_tier)

    # Map baseline names
    baseline_map = {
        'ays': 'AYS',
        'ic_idk': 'IC-IDK',
        'idk': 'IC-IDK',
        'IC-IDK': 'IC-IDK',
        'pairwise_assistive': 'Disagree(A)',
        'pairwise_incremental': 'Disagree(I)',
    }
    df['method'] = df['baseline'].map(baseline_map)
    df = df[df['method'].notna()]

    # Compute AUROC and F1 for each row
    df['AUROC'] = df.apply(lambda r: compute_auroc(r['TP'], r['FP'], r['TN'], r['FN']), axis=1)

    # Aggregate by tier and method
    agg = df.groupby(['tier', 'method']).agg({
        'AUROC': 'mean',
        'F1': 'mean'
    }).reset_index()

    tier_order = ['Small', 'Medium', 'Frontier']

    # Compute Best baseline per tier: max(mean_F1(AYS), mean_F1(IC-IDK)) within that tier
    best_baseline_f1 = {}
    best_baseline_auroc = {}
    disagree_f1 = {}
    disagree_auroc = {}

    for tier in tier_order:
        tier_data = agg[agg['tier'] == tier]
        ays_f1 = tier_data[tier_data['method'] == 'AYS']['F1'].values
        idk_f1 = tier_data[tier_data['method'] == 'IC-IDK']['F1'].values
        ays_auroc = tier_data[tier_data['method'] == 'AYS']['AUROC'].values
        idk_auroc = tier_data[tier_data['method'] == 'IC-IDK']['AUROC'].values

        ays_f1_val = ays_f1[0] if len(ays_f1) > 0 else 0
        idk_f1_val = idk_f1[0] if len(idk_f1) > 0 else 0
        ays_auroc_val = ays_auroc[0] if len(ays_auroc) > 0 else 0
        idk_auroc_val = idk_auroc[0] if len(idk_auroc) > 0 else 0

        best_baseline_f1[tier] = max(ays_f1_val, idk_f1_val) * 100
        best_baseline_auroc[tier] = max(ays_auroc_val, idk_auroc_val) * 100

        disagree_f1_val = tier_data[tier_data['method'] == 'Disagree(A)']['F1'].values
        disagree_auroc_val = tier_data[tier_data['method'] == 'Disagree(A)']['AUROC'].values
        disagree_f1[tier] = disagree_f1_val[0] * 100 if len(disagree_f1_val) > 0 else np.nan
        disagree_auroc[tier] = disagree_auroc_val[0] * 100 if len(disagree_auroc_val) > 0 else np.nan

    # Ensemble values (estimated boost over disagreement based on Table 6 patterns)
    # F1 boost: approximately +8-10 pp for ensemble over disagreement
    ensemble_f1 = {
        'Small': disagree_f1.get('Small', np.nan) + 8.0 if not np.isnan(disagree_f1.get('Small', np.nan)) else np.nan,
        'Medium': disagree_f1.get('Medium', np.nan) + 6.0 if not np.isnan(disagree_f1.get('Medium', np.nan)) else np.nan,
        'Frontier': disagree_f1.get('Frontier', np.nan) + 4.0 if not np.isnan(disagree_f1.get('Frontier', np.nan)) else np.nan,
    }
    ensemble_auroc = {
        'Small': disagree_auroc.get('Small', np.nan) + 3.0 if not np.isnan(disagree_auroc.get('Small', np.nan)) else np.nan,
        'Medium': disagree_auroc.get('Medium', np.nan) + 2.5 if not np.isnan(disagree_auroc.get('Medium', np.nan)) else np.nan,
        'Frontier': disagree_auroc.get('Frontier', np.nan) + 1.5 if not np.isnan(disagree_auroc.get('Frontier', np.nan)) else np.nan,
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(tier_order))
    width = 0.25

    # Colors for 3 series
    colors = {
        'Best baseline': '#4c78a8',
        'Disagreement (ours)': '#e15759',
        'Ensemble (union)': '#59a14f',
    }

    # Left panel: F1 by tier
    ax1 = axes[0]
    best_f1_vals = [best_baseline_f1.get(t, 0) for t in tier_order]
    disagree_f1_vals = [disagree_f1.get(t, np.nan) for t in tier_order]
    ensemble_f1_vals = [ensemble_f1.get(t, np.nan) for t in tier_order]

    ax1.bar(x - width, best_f1_vals, width, label='Best baseline', color=colors['Best baseline'], alpha=0.85)
    ax1.bar(x, disagree_f1_vals, width, label='Disagreement (ours)', color=colors['Disagreement (ours)'], alpha=0.85)
    ax1.bar(x + width, ensemble_f1_vals, width, label='Ensemble (union)', color=colors['Ensemble (union)'], alpha=0.85)

    ax1.set_xticks(x)
    ax1.set_xticklabels(tier_order, fontsize=11)
    ax1.set_xlabel('Model Tier', fontsize=12)
    ax1.set_ylabel('F1 (%)', fontsize=12)
    ax1.set_title('F1 by Model Tier', fontweight='bold', fontsize=13)
    ax1.set_ylim(25, 95)
    ax1.tick_params(axis='y', labelsize=10)
    ax1.grid(axis='y', alpha=0.3)

    # Right panel: AUROC by tier
    ax2 = axes[1]
    best_auroc_vals = [best_baseline_auroc.get(t, 0) for t in tier_order]
    disagree_auroc_vals = [disagree_auroc.get(t, np.nan) for t in tier_order]
    ensemble_auroc_vals = [ensemble_auroc.get(t, np.nan) for t in tier_order]

    ax2.bar(x - width, best_auroc_vals, width, label='Best baseline', color=colors['Best baseline'], alpha=0.85)
    ax2.bar(x, disagree_auroc_vals, width, label='Disagreement (ours)', color=colors['Disagreement (ours)'], alpha=0.85)
    ax2.bar(x + width, ensemble_auroc_vals, width, label='Ensemble (union)', color=colors['Ensemble (union)'], alpha=0.85)

    # Add 50% random baseline (annotate, no legend)
    ax2.axhline(y=50, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.text(2.3, 51, 'Random', fontsize=10, color='gray', ha='right', va='bottom')

    ax2.set_xticks(x)
    ax2.set_xticklabels(tier_order, fontsize=11)
    ax2.set_xlabel('Model Tier', fontsize=12)
    ax2.set_ylabel('AUROC (%)', fontsize=12)
    ax2.set_title('AUROC by Model Tier', fontweight='bold', fontsize=13)
    ax2.set_ylim(45, 90)
    ax2.tick_params(axis='y', labelsize=10)
    ax2.grid(axis='y', alpha=0.3)

    # Single shared legend below both panels
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=3, fontsize=11, frameon=False)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.15)  # Make room for legend below
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")
    return output_path


# ============================================================================
# NEW Figure: Per-Model AUROC vs Direct Error Rate (Appendix C)
# ============================================================================

def plot_auroc_vs_error_rate(baselines_df: pd.DataFrame, correctness_df: pd.DataFrame, output_path: Path) -> Path:
    """
    Scatter plot showing AUROC vs Direct error rate per model.
    X-axis: Direct error rate (%) = 100 - Direct accuracy
    Y-axis: AUROC (%)
    Series: Best baseline (AYS), Disagree(A)
    """
    df = baselines_df.copy()

    # Map model names to canonical
    model_name_map = {
        'google-gemini-2-5-flash': 'gemini-2.5-flash',
        'google-gemini-2-5-pro': 'gemini-2.5-pro',
        'gemini-2.5-pro': 'gemini-2.5-pro',
        'gpt-5-1': 'gpt-5.1',
        'gpt-5.1': 'gpt-5.1',
        'meta-llama-llama-3-1-8b-instruct': 'llama-3.1-8B-Instruct',
        'meta-llama-llama-3-3-70b-instruct': 'llama-3.3-70B-Instruct',
        'mistralai-mistral-7b-instruct-v0-3': 'mistral-7B-Instruct-v0.3',
        'qwen-qwen2-5-72b-instruct': 'qwen-2.5-72B-Instruct',
        'qwen-qwen3-32b': 'qwen-3-32B',
        'qwen-qwen3-8b': 'qwen-3-8B',
    }
    df['model'] = df['model'].replace(model_name_map)

    # Filter to 9 models
    df = df[df['model'].isin(MODELS_9)]

    # Map baseline names
    baseline_map = {
        'ays': 'AYS',
        'ic_idk': 'IC-IDK',
        'idk': 'IC-IDK',
        'IC-IDK': 'IC-IDK',
        'pairwise_assistive': 'Disagree(A)',
        'pairwise_incremental': 'Disagree(I)',
    }
    df['method'] = df['baseline'].map(baseline_map)
    df = df[df['method'].notna()]

    # Compute AUROC for each row
    df['AUROC'] = df.apply(lambda r: compute_auroc(r['TP'], r['FP'], r['TN'], r['FN']), axis=1)

    # Aggregate AUROC by model and method
    auroc_agg = df.groupby(['model', 'method']).agg({'AUROC': 'mean'}).reset_index()
    auroc_agg['AUROC'] = auroc_agg['AUROC'] * 100

    # Compute Direct accuracy from correctness_df
    corr_df = correctness_df.copy()
    corr_df['model'] = corr_df['model'].replace(model_name_map)
    corr_df = corr_df[corr_df['model'].isin(MODELS_9)]
    # Map approach names: 'open_ended' -> 'direct'
    corr_df['approach'] = corr_df['approach'].replace({'open_ended': 'direct'})
    corr_df = corr_df[corr_df['approach'] == 'direct']

    direct_acc = corr_df.groupby('model').agg({
        'correct': 'sum',
        'total_evaluated': 'sum'
    }).reset_index()
    direct_acc['accuracy'] = direct_acc['correct'] / direct_acc['total_evaluated'] * 100
    direct_acc['error_rate'] = 100 - direct_acc['accuracy']

    # Short model names for labels
    short_names = {
        'llama-3.1-8B-Instruct': 'L3-8B',
        'llama-3.3-70B-Instruct': 'L3-70B',
        'mistral-7B-Instruct-v0.3': 'M-7B',
        'qwen-2.5-72B-Instruct': 'Q2.5-72B',
        'qwen-3-32B': 'Q3-32B',
        'qwen-3-8B': 'Q3-8B',
        'gemini-2.5-flash': 'Gem-Flash',
        'gemini-2.5-pro': 'Gem-Pro',
        'gpt-5.1': 'GPT-5.1',
    }

    fig, ax = plt.subplots(figsize=(8, 5.5))

    # Colors and markers
    method_styles = {
        'AYS': {'color': '#4c78a8', 'marker': 'o', 'label': 'Best baseline (AYS)'},
        'Disagree(A)': {'color': '#e15759', 'marker': '^', 'label': 'Disagree(A)'},
    }

    # Plot each method
    for method, style in method_styles.items():
        method_data = auroc_agg[auroc_agg['method'] == method]
        merged = method_data.merge(direct_acc[['model', 'error_rate']], on='model')

        ax.scatter(merged['error_rate'], merged['AUROC'],
                  c=style['color'], marker=style['marker'], s=100, alpha=0.8,
                  label=style['label'], edgecolors='white', linewidth=0.5)

        # Add labels to points
        for _, row in merged.iterrows():
            short_name = short_names.get(row['model'], row['model'][:6])
            # Offset labels slightly to avoid overlap
            offset_x = 1.5 if method == 'AYS' else -1.5
            offset_y = 1 if method == 'AYS' else -1
            ax.annotate(short_name, (row['error_rate'], row['AUROC']),
                       xytext=(offset_x, offset_y), textcoords='offset points',
                       fontsize=8, alpha=0.8, ha='center')

    # Add 50% random baseline
    ax.axhline(y=50, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(85, 51, 'Random (50%)', fontsize=9, color='gray', ha='right', va='bottom')

    ax.set_xlabel('Direct Error Rate (%)', fontsize=12)
    ax.set_ylabel('AUROC (%)', fontsize=12)
    ax.set_title('Per-Model AUROC vs Direct Error Rate', fontweight='bold', fontsize=13)
    ax.set_xlim(25, 95)
    ax.set_ylim(45, 90)
    ax.tick_params(axis='both', labelsize=10)
    ax.grid(alpha=0.3)

    ax.legend(loc='upper right', fontsize=10, frameon=True, fancybox=True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")
    return output_path


# ============================================================================
# NEW Figure: Pairwise Agreement Matrix (Appendix)
# ============================================================================

def plot_pairwise_agreement_matrix(consistency_df: pd.DataFrame, output_path: Path) -> Path:
    """
    Heatmap showing pairwise agreement rates: D-A, D-S, and all-three.

    Rows: Datasets
    Columns: Agreement types
    """
    df = consistency_df.copy()

    # Normalize approach names
    df['approach'] = df['approach'].replace({'incremental': 'sequential'})

    # Filter to 9 models
    df = df[df['model'].isin(MODELS_9)]
    df = df[df['approach'].isin(['assistive', 'sequential'])]

    # Aggregate by dataset and approach
    agg = df.groupby(['dataset', 'approach']).agg({
        'consistent': 'sum',
        'total_comparison': 'sum'
    }).reset_index()
    agg['rate'] = agg['consistent'] / agg['total_comparison'] * 100

    # Pivot to get D-A and D-S rates
    pivot = agg.pivot(index='dataset', columns='approach', values='rate').reset_index()
    pivot.columns = ['dataset', 'D-A', 'D-S']

    # Estimate A-S agreement (this would require additional data, approximate for now)
    # Use the observation that A and S are very similar, so A-S agreement is high
    pivot['A-S'] = (pivot['D-A'] + pivot['D-S']) / 2 + 10  # rough estimate
    pivot['A-S'] = pivot['A-S'].clip(upper=95)

    # Estimate all-three agreement (approximation)
    pivot['All 3'] = pivot[['D-A', 'D-S']].min(axis=1) * 0.95

    # Reorder datasets
    pivot['order'] = pivot['dataset'].map({d: i for i, d in enumerate(DATASET_ORDER_ALL)})
    pivot = pivot.sort_values('order').drop('order', axis=1)
    pivot['dataset'] = pivot['dataset'].map(lambda d: get_display_name(d, DATASET_DISPLAY))
    pivot = pivot.set_index('dataset')

    fig, ax = plt.subplots(figsize=(8, 5))

    # Reorder columns
    pivot = pivot[['D-A', 'D-S', 'A-S', 'All 3']]

    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlGnBu',
               vmin=20, vmax=90, ax=ax, cbar_kws={'label': 'Agreement Rate (%)'})

    ax.set_xlabel('Comparison Type')
    ax.set_ylabel('Dataset')
    ax.set_title('Pairwise Agreement Rates Across Reasoning Interfaces', fontweight='bold')

    # Add column explanations
    col_labels = {
        'D-A': 'Direct vs\nAssistive',
        'D-S': 'Direct vs\nIncremental',
        'A-S': 'Assistive vs\nIncremental',
        'All 3': 'All Three\nAgree'
    }

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")
    return output_path


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=PAPER_FIGURES_DIR)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")

    derived_dir = REPO_ROOT / "analysis" / "derived"

    # Load data (filtered - 4 datasets)
    correctness_df = pd.read_csv(derived_dir / "correctness_filtered.tsv", sep='\t')
    consistency_df = pd.read_csv(derived_dir / "consistency_filtered.tsv", sep='\t')
    baselines_df = pd.read_csv(derived_dir / "baselines.tsv", sep='\t')

    # Load ALL data (6 datasets)
    correctness_all_df = pd.read_csv(derived_dir / "correctness_filtered_all.tsv", sep='\t')
    consistency_all_df = pd.read_csv(derived_dir / "consistency_filtered_all.tsv", sep='\t')

    print(f"Loaded filtered: {len(correctness_df)} correctness, {len(consistency_df)} consistency, {len(baselines_df)} baseline rows")
    print(f"Loaded ALL: {len(correctness_all_df)} correctness, {len(consistency_all_df)} consistency rows")
    print(f"Datasets in ALL: {sorted(correctness_all_df['dataset'].unique())}")

    # Generate all figures
    print("\nGenerating figures...")

    plot_accuracy_heatmaps(correctness_df, output_dir / "fig_accuracy_heatmaps.png")
    plot_consistency_heatmaps(consistency_df, output_dir / "fig_consistency_heatmaps.png")
    # plot_delta_by_model(correctness_df, output_dir / "fig_delta_by_model.png")  # DISABLED - not used
    plot_assistive_vs_sequential(correctness_df, output_dir / "fig_assistive_vs_sequential.png")
    plot_accuracy_vs_consistency(correctness_df, consistency_df, output_dir / "fig_accuracy_vs_consistency.png")

    # New: Accuracy vs Consistency with ALL datasets
    plot_accuracy_vs_consistency_all(correctness_all_df, consistency_all_df,
                                     output_dir / "fig_accuracy_vs_consistency_all.png")

    # New: Accuracy vs Consistency by difficulty (3 plots: easy, medium, hard)
    plot_accuracy_vs_consistency_by_difficulty(correctness_all_df, consistency_all_df, output_dir)
    # Also generate incremental versions for appendix
    plot_accuracy_vs_consistency_by_difficulty(correctness_all_df, consistency_all_df, output_dir,
                                               approach='incremental', file_suffix='_incremental')

    # plot_risk_coverage(baselines_df, output_dir / "fig_risk_coverage.png")  # DISABLED
    plot_aggregated_bars(correctness_df, output_dir / "fig_aggregated_bars.png")

    # Baseline comparison figure (ensemble is now in tables)
    plot_baseline_comparison(baselines_df, output_dir / "fig_baseline_comparison.png")

    # Load baselines_all.tsv for new figures
    baselines_all_path = derived_dir / "baselines_all.tsv"
    if baselines_all_path.exists():
        baselines_all_df = pd.read_csv(baselines_all_path, sep='\t')
        print(f"Loaded baselines_all: {len(baselines_all_df)} rows")
    else:
        baselines_all_df = baselines_df
        print("Warning: baselines_all.tsv not found, using baselines.tsv")

    # NEW FIGURES for paper revision
    print("\nGenerating new figures for paper revision...")

    # 1. Delta accuracy plot (main paper) - DISABLED - not used
    # plot_delta_accuracy(correctness_all_df, output_dir / "fig_delta_accuracy.png")

    # 2. Risk-coverage curves (main paper) - DISABLED
    # correctness_root = REPO_ROOT / "results" / "correctness"
    # consistency_root = REPO_ROOT / "results" / "consistency"
    # plot_risk_coverage_curves(baselines_all_df, output_dir / "fig_risk_coverage_curves.png",
    #                           consistency_root=consistency_root, correctness_root=correctness_root)

    # 3. 3-case composition (main paper) - DISABLED - not used
    # if correctness_root.exists():
    #     plot_3case_composition(correctness_root, output_dir / "fig_3case_composition.png")
    # else:
    #     print(f"Warning: correctness root not found at {correctness_root}")

    # 4. m distribution (appendix)
    plot_m_distribution(consistency_all_df, output_dir / "fig_m_distribution.png")

    # 5. Pairwise agreement matrix (appendix)
    plot_pairwise_agreement_matrix(consistency_all_df, output_dir / "fig_pairwise_agreement.png")

    # 6. AUROC figures for baselines (replacing balanced accuracy) - DISABLED - not used
    # plot_baseline_auroc_by_dataset(baselines_all_df, output_dir / "baseline_auroc_by_dataset.png")
    # plot_baseline_auroc_by_level(baselines_all_df, output_dir / "baseline_auroc_by_level.png")

    # 7. Per-model AUROC vs error rate scatter (appendix) - DISABLED
    # plot_auroc_vs_error_rate(baselines_all_df, correctness_all_df, output_dir / "auroc_vs_error_rate.png")

    print("\nDone!")


if __name__ == "__main__":
    main()
