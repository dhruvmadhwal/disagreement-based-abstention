"""Configuration constants shared by CRAG generation scripts."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, Set


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = REPO_ROOT / "data" / "processed" / "crag" / "crag_dsl.json"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results" / "generation" / "crag"

REFERENCE_DATE = "2022-10-04"
TEMPERATURE = 0.0
TOP_P = 0.95
TOP_K = 64
DEFAULT_CHECKPOINT_INTERVAL = 10

GEMINI_MODEL_CHOICES: Set[str] = {
    "gemini",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-pro",
}

PRIMARY_MODEL_CHOICES = [
    "gpt5",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "qwen3-8b",
    "qwen3-32b",
    "qwen-3-32b",
    "llama-3.1-8b-instruct",
    "gemma-3-4b-it",
    "mistralai/mistral-7b-instruct-v0.3",
    "llama-3.3-70b-instruct",
    "qwen-2.5-72b-instruct",
]

MODEL_PRESETS: Dict[str, str] = {
    "gpt5": "gpt-5.1",
    "gemini": "google/gemini-2.5-flash",
    "gemini-2.5-flash": "google/gemini-2.5-flash",
    "gemini-2.5-pro": "google/gemini-2.5-pro",
    "gemini-pro": "google/gemini-2.5-pro",
    "qwen": "Qwen/Qwen3-8B",
    "qwen3-8b": "Qwen/Qwen3-8B",
    "qwen-3-8b": "Qwen/Qwen3-8B",
    "qwen3-32b": "Qwen/Qwen3-32B",
    "qwen-3-32b": "Qwen/Qwen3-32B",
    "qwen-2.5-72b-instruct": "Qwen/Qwen2.5-72B-Instruct",
    "qwen2.5-72b-instruct": "Qwen/Qwen2.5-72B-Instruct",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "llama-3.3-70b-instruct": "meta-llama/Llama-3.3-70B-Instruct",
    "gemma": "google/gemma-3-4b-it",
    "gemma-3-4b-it": "google/gemma-3-4b-it",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/mistral-7b-instruct-v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
}


def normalize_model_choice(choice: str) -> str:
    """Normalize user-provided model choices for preset lookup."""
    if not choice:
        return ""
    normalized = choice.strip().lower()
    normalized = normalized.replace("_", "-")
    normalized = re.sub(r"\s+", "-", normalized)
    return normalized


def make_model_slug(model_choice: str | None, model_name: str | None) -> str:
    """Create filesystem-friendly slug for storing CRAG outputs."""
    base = model_name or model_choice or ""
    slug = re.sub(r"[^a-z0-9]+", "-", base.lower()).strip("-")
    if not slug and model_choice:
        slug = re.sub(r"[^a-z0-9]+", "-", model_choice.lower()).strip("-")
    return slug or "model"


def ensure_results_dir(model_slug: str) -> Path:
    """Create and return the results directory for the provided model slug."""
    target = DEFAULT_RESULTS_ROOT / model_slug
    target.mkdir(parents=True, exist_ok=True)
    return target


__all__ = [
    "DEFAULT_CHECKPOINT_INTERVAL",
    "DEFAULT_DATASET_PATH",
    "DEFAULT_RESULTS_ROOT",
    "GEMINI_MODEL_CHOICES",
    "MODEL_PRESETS",
    "PRIMARY_MODEL_CHOICES",
    "REFERENCE_DATE",
    "REPO_ROOT",
    "TEMPERATURE",
    "TOP_K",
    "TOP_P",
    "ensure_results_dir",
    "make_model_slug",
    "normalize_model_choice",
]
