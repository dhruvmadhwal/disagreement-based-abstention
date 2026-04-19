"""Shared configuration for decomposition equivalence runs."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]

DATASET_SLUGS: List[str] = [
    "bamboogle",
    "crag",
    "hotpotqa",
    "mintaka",
]

DATASET_FILES: Dict[str, Path] = {
    slug: REPO_ROOT / "data" / "processed" / slug / f"{slug}_dsl.json"
    for slug in DATASET_SLUGS
}

RESULTS_ROOT = REPO_ROOT / "decomposition_equivalence" / "results"
DSL_OUTPUT_ROOT = RESULTS_ROOT / "dsl"
EVAL_OUTPUT_ROOT = RESULTS_ROOT / "eval"
SUMMARY_OUTPUT_ROOT = RESULTS_ROOT / "summary"
BATCH_OUTPUT_ROOT = RESULTS_ROOT / "batch"

DEFAULT_GEN_TEMPERATURE = float(os.environ.get("DECOMP_EQUIV_GEN_TEMPERATURE", "0.1"))
DEFAULT_GEN_TOP_P = float(os.environ.get("DECOMP_EQUIV_GEN_TOP_P", "0.95"))
DEFAULT_GEN_TOP_K = int(os.environ.get("DECOMP_EQUIV_GEN_TOP_K", "64"))
DEFAULT_GEN_MAX_TOKENS = int(os.environ.get("DECOMP_EQUIV_GEN_MAX_TOKENS", "2048"))

DEFAULT_JUDGE_MODEL = os.environ.get("DECOMP_EQUIV_JUDGE_MODEL", "google/gemini-2.5-flash")
DEFAULT_JUDGE_TEMPERATURE = float(os.environ.get("DECOMP_EQUIV_JUDGE_TEMPERATURE", "0.1"))
DEFAULT_JUDGE_MAX_TOKENS = int(os.environ.get("DECOMP_EQUIV_JUDGE_MAX_TOKENS", "2048"))

DEFAULT_VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
DEFAULT_VLLM_API_KEY = os.environ.get("VLLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or "EMPTY"

# Mirror generation presets so model slugs resolve to the same model names.
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


def slugify(text: str) -> str:
    """Create a filesystem-friendly slug from a model name or choice."""
    if not text:
        return "model"
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "model"


__all__ = [
    "REPO_ROOT",
    "DATASET_SLUGS",
    "DATASET_FILES",
    "RESULTS_ROOT",
    "DSL_OUTPUT_ROOT",
    "EVAL_OUTPUT_ROOT",
    "SUMMARY_OUTPUT_ROOT",
    "BATCH_OUTPUT_ROOT",
    "DEFAULT_GEN_TEMPERATURE",
    "DEFAULT_GEN_TOP_P",
    "DEFAULT_GEN_TOP_K",
    "DEFAULT_GEN_MAX_TOKENS",
    "DEFAULT_JUDGE_MODEL",
    "DEFAULT_JUDGE_TEMPERATURE",
    "DEFAULT_JUDGE_MAX_TOKENS",
    "DEFAULT_VLLM_BASE_URL",
    "DEFAULT_VLLM_API_KEY",
    "MODEL_PRESETS",
    "normalize_model_choice",
    "slugify",
]
