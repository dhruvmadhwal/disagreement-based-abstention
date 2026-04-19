"""Configuration constants for Mintaka baseline runs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence, Tuple

from generate.mintaka.config import (
    DEFAULT_DATASET_PATH,
    DEFAULT_RESULTS_ROOT as GENERATION_RESULTS_ROOT,
    REFERENCE_DATE,
    REPO_ROOT,
    TEMPERATURE,
    TOP_K,
    TOP_P,
)

PAIRWISE_TARGET_REGIMES: Tuple[str, ...] = ("assistive", "incremental", "model_generated")
PAIRWISE_BASELINES: Tuple[str, ...] = tuple(f"pairwise_{regime}" for regime in PAIRWISE_TARGET_REGIMES)

# Baseline names are kept snake_case to mirror generation regime naming.
BASELINES: Tuple[str, ...] = (
    "ays",
    "idk",
    "ic_idk",
    "self_consistency",
    *PAIRWISE_BASELINES,
)

# Default run excludes plain IDK unless explicitly requested.
DEFAULT_BASELINES: Tuple[str, ...] = ("ays", "ic_idk", "pairwise_assistive", "pairwise_incremental", "pairwise_model_generated")

BASELINE_FILENAMES: Dict[str, str] = {
    "ays": "mintaka_ays.json",
    "idk": "mintaka_idk.json",
    "ic_idk": "mintaka_ic_idk.json",
    "self_consistency": "mintaka_self_consistency.json",
    "pairwise_assistive": "mintaka_pairwise_assistive.json",
    "pairwise_incremental": "mintaka_pairwise_incremental.json",
    "pairwise_model_generated": "mintaka_pairwise_model_generated.json",
}

DEFAULT_COMBINED_FILENAME = "mintaka_baselines_full.jsonl"

# Directory layout mirrors generation outputs but nests under results/baselines.
BASELINE_RESULTS_ROOT = REPO_ROOT / "results" / "baselines" / "mintaka"
CORRECTNESS_RESULTS_ROOT = REPO_ROOT / "results" / "correctness" / "mintaka"
CONSISTENCY_RESULTS_ROOT = REPO_ROOT / "results" / "consistency" / "mintaka"

# Default hyperparameters pulled from generation config for consistency.
DEFAULT_TEMPERATURE = TEMPERATURE
DEFAULT_TOP_P = TOP_P
DEFAULT_TOP_K = TOP_K

DEFAULT_SELF_CONSISTENCY_ZERO_TEMPERATURE = 0.0
DEFAULT_SELF_CONSISTENCY_ZERO_SAMPLES = 1
DEFAULT_SELF_CONSISTENCY_SAMPLES = 7
DEFAULT_SELF_CONSISTENCY_MIN_VOTES = 4
DEFAULT_SELF_CONSISTENCY_TEMPERATURE = 0.7

DEFAULT_IC_IDK_K = 15
DEFAULT_IC_IDK_D = 4
DEFAULT_PAIRWISE_TEMPERATURE = 0.0
DEFAULT_PAIRWISE_TOP_P = 0.95
DEFAULT_PAIRWISE_TOP_K = 1


def ensure_baseline_dir(model_slug: str) -> Path:
    """Create and return the results directory for the provided model slug."""
    target = BASELINE_RESULTS_ROOT / model_slug
    target.mkdir(parents=True, exist_ok=True)
    return target


__all__: Sequence[str] = [
    "BASELINES",
    "BASELINE_FILENAMES",
    "BASELINE_RESULTS_ROOT",
    "CONSISTENCY_RESULTS_ROOT",
    "CORRECTNESS_RESULTS_ROOT",
    "DEFAULT_COMBINED_FILENAME",
    "DEFAULT_DATASET_PATH",
    "DEFAULT_IC_IDK_D",
    "DEFAULT_IC_IDK_K",
    "DEFAULT_PAIRWISE_TEMPERATURE",
    "DEFAULT_PAIRWISE_TOP_K",
    "DEFAULT_PAIRWISE_TOP_P",
    "DEFAULT_SELF_CONSISTENCY_MIN_VOTES",
    "DEFAULT_SELF_CONSISTENCY_SAMPLES",
    "DEFAULT_SELF_CONSISTENCY_TEMPERATURE",
    "DEFAULT_SELF_CONSISTENCY_ZERO_SAMPLES",
    "DEFAULT_SELF_CONSISTENCY_ZERO_TEMPERATURE",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TOP_K",
    "DEFAULT_TOP_P",
    "GENERATION_RESULTS_ROOT",
    "PAIRWISE_BASELINES",
    "PAIRWISE_TARGET_REGIMES",
    "REFERENCE_DATE",
    "REPO_ROOT",
    "ensure_baseline_dir",
]
