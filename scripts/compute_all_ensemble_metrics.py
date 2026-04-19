#!/usr/bin/env python3
"""Compute missing ensemble metrics for all models across all datasets."""

import json
import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DERIVED_DIR = BASE_DIR / "analysis" / "derived"

# Standard datasets with JSONL format
STANDARD_DATASETS = ['bamboogle', 'crag', 'hotpotqa', 'mintaka']

# Models in standard dataset directories
STANDARD_MODELS = [
    'google-gemini-2-5-flash',
    'google-gemini-2-5-pro',
    'gpt-5-1',
    'meta-llama-llama-3-1-8b-instruct',
    'meta-llama-llama-3-3-70b-instruct',
    'mistralai-mistral-7b-instruct-v0-3',
    'qwen-qwen2-5-72b-instruct',
    'qwen-qwen3-32b',
    'qwen-qwen3-8b',
]

# Models we already have ensemble data for (skip these)
EXISTING_ENSEMBLE_MODELS = ['gpt-5-1', 'meta-llama-llama-3-3-70b-instruct', 'qwen-qwen3-8b']


def compute_auroc(tp, fp, tn, fn):
    """Compute AUROC from confusion matrix."""
    p = tp + fn  # actual positives
    n = fp + tn  # actual negatives
    if p == 0 or n == 0:
        return np.nan
    auroc = (tp * tn + 0.5 * (tp * fp + fn * tn)) / (p * n)
    return auroc


def compute_metrics(tp, fp, tn, fn, total):
    """Compute all metrics from confusion matrix."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    auroc = compute_auroc(tp, fp, tn, fn)

    # Coverage = proportion of questions answered (not abstained)
    abstained = tp + fp
    answered = tn + fn
    coverage = answered / total if total > 0 else 0

    # Acc@Cov = accuracy among answered questions
    acc_at_cov = tn / answered if answered > 0 else 0

    # Overall_Acc = (correct answered) / total
    overall_acc = tn / total if total > 0 else 0

    # Decision_Acc = (TP + TN) / total
    decision_acc = (tp + tn) / total if total > 0 else 0

    return {
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'Coverage': coverage,
        'Acc@Cov': acc_at_cov,
        'Overall_Acc': overall_acc,
        'Decision_Acc': decision_acc,
        'AUROC': auroc
    }


def load_standard_baselines(dataset, model):
    """Load baseline data from JSONL file for standard datasets."""
    baselines_dir = BASE_DIR / 'results' / 'baselines' / dataset / model

    # Find the baselines_full.jsonl file
    jsonl_files = list(baselines_dir.glob('*_baselines_full.jsonl'))
    if not jsonl_files:
        print(f"  No JSONL file found for {dataset}/{model}")
        return None

    data = []
    with open(jsonl_files[0], 'r') as f:
        for line in f:
            data.append(json.loads(line))

    return data


def compute_ensemble_for_standard(dataset, model):
    """Compute ensemble metrics for a standard dataset/model pair."""
    data = load_standard_baselines(dataset, model)
    if data is None:
        return None, None

    # Compute ensemble metrics
    results = {
        'ensemble_assistive': {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0},
        'ensemble_incremental': {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}
    }

    for item in data:
        baselines = item.get('baselines', {})

        ays = baselines.get('ays', {})
        pairwise_a = baselines.get('pairwise_assistive', {})
        pairwise_i = baselines.get('pairwise_incremental', {})

        ays_accepted = ays.get('accepted', True)
        pa_accepted = pairwise_a.get('accepted', True)
        pi_accepted = pairwise_i.get('accepted', True)

        # Correctness is based on the open-ended (direct) answer
        # Use ays.correct since that reflects open_ended correctness
        correct = ays.get('correct', 0) == 1

        # Ensemble(A) = abstain if AYS abstains OR Disagree(A) abstains
        ens_a_abstains = (not ays_accepted) or (not pa_accepted)

        # Ensemble(I) = abstain if AYS abstains OR Disagree(I) abstains
        ens_i_abstains = (not ays_accepted) or (not pi_accepted)

        # Update confusion matrix for ensemble_assistive
        if ens_a_abstains:
            if not correct:
                results['ensemble_assistive']['TP'] += 1  # Correctly abstained (error detected)
            else:
                results['ensemble_assistive']['FP'] += 1  # Wrongly abstained
        else:
            if correct:
                results['ensemble_assistive']['TN'] += 1  # Correctly answered
            else:
                results['ensemble_assistive']['FN'] += 1  # Missed error

        # Update confusion matrix for ensemble_incremental
        if ens_i_abstains:
            if not correct:
                results['ensemble_incremental']['TP'] += 1
            else:
                results['ensemble_incremental']['FP'] += 1
        else:
            if correct:
                results['ensemble_incremental']['TN'] += 1
            else:
                results['ensemble_incremental']['FN'] += 1

    total = len(data)

    ens_a_metrics = compute_metrics(
        results['ensemble_assistive']['TP'],
        results['ensemble_assistive']['FP'],
        results['ensemble_assistive']['TN'],
        results['ensemble_assistive']['FN'],
        total
    )
    ens_a_metrics.update(results['ensemble_assistive'])

    ens_i_metrics = compute_metrics(
        results['ensemble_incremental']['TP'],
        results['ensemble_incremental']['FP'],
        results['ensemble_incremental']['TN'],
        results['ensemble_incremental']['FN'],
        total
    )
    ens_i_metrics.update(results['ensemble_incremental'])

    return ens_a_metrics, ens_i_metrics


def main():
    all_results = []

    # Process standard datasets for models that don't have ensemble data yet
    for dataset in STANDARD_DATASETS:
        print(f"\nProcessing {dataset}...")

        for model in STANDARD_MODELS:
            # Skip models that already have ensemble data
            if model in EXISTING_ENSEMBLE_MODELS:
                print(f"  Skipping {model} (already has ensemble data)")
                continue

            print(f"  Computing ensemble for {model}...")

            ens_a, ens_i = compute_ensemble_for_standard(dataset, model)

            if ens_a is not None:
                all_results.append({
                    'dataset': dataset,
                    'model': model,
                    'baseline': 'ensemble_assistive',
                    **ens_a
                })

            if ens_i is not None:
                all_results.append({
                    'dataset': dataset,
                    'model': model,
                    'baseline': 'ensemble_incremental',
                    **ens_i
                })

    # Create DataFrame
    if all_results:
        df = pd.DataFrame(all_results)

        # Reorder columns to match baselines_all.tsv
        cols = ['dataset', 'model', 'baseline', 'TP', 'FP', 'TN', 'FN',
                'Precision', 'Recall', 'F1', 'Coverage', 'Acc@Cov',
                'Overall_Acc', 'Decision_Acc', 'AUROC']
        df = df[cols]

        print(f"\nComputed {len(df)} new ensemble rows")

        # Load existing data
        existing_df = pd.read_csv(DERIVED_DIR / 'baselines_all.tsv', sep='\t')

        # Remove any existing entries for these model/dataset/baseline combinations
        for _, row in df.iterrows():
            mask = ~((existing_df['dataset'] == row['dataset']) &
                     (existing_df['model'] == row['model']) &
                     (existing_df['baseline'] == row['baseline']))
            existing_df = existing_df[mask]

        # Append new data
        combined = pd.concat([existing_df, df], ignore_index=True)

        # Save
        combined.to_csv(DERIVED_DIR / 'baselines_all.tsv', sep='\t', index=False)
        print(f"Updated baselines_all.tsv (total rows: {len(combined)})")

        # Print summary
        print("\nNew ensemble entries added:")
        print(df[['dataset', 'model', 'baseline', 'F1', 'AUROC']].to_string())
    else:
        print("No new ensemble data computed")


if __name__ == '__main__':
    main()
