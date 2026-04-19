"""Per-dataset evaluation spec.

The four paper datasets differ only in:
  - the dataset string used in result paths and filename prefixes;
  - the per-dataset prompt module under prompts/{correctness,consistency}/<ds>/.

Everything else (judge model, temperature, regime list, error policy) is shared
across the four datasets after the C10 reconciliation.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class EvaluationSpec:
    """Per-dataset descriptors used by both correctness and consistency judges."""

    name: str  # "bamboogle" | "crag" | "hotpotqa" | "mintaka"

    @property
    def dsl_path(self) -> Path:
        return REPO_ROOT / "data" / "processed" / self.name / f"{self.name}_dsl.json"

    @property
    def generation_dir(self) -> Path:
        return REPO_ROOT / "results" / "generation" / self.name

    @property
    def correctness_dir(self) -> Path:
        return REPO_ROOT / "results" / "correctness" / self.name

    @property
    def consistency_dir(self) -> Path:
        return REPO_ROOT / "results" / "consistency" / self.name

    @property
    def regime_filename_map(self) -> Dict[str, str]:
        return {
            "open_ended": f"{self.name}_open_ended.json",
            "assistive": f"{self.name}_assistive.json",
            "incremental": f"{self.name}_sequential.json",
            "model_generated": f"{self.name}_model_generated.json",
        }

    @property
    def regime_choices(self) -> List[str]:
        return sorted(self.regime_filename_map.keys())

    @property
    def correctness_prompt_module(self) -> ModuleType:
        return import_module(f"prompts.correctness.{self.name}.correctness_prompt")

    @property
    def consistency_prompt_module(self) -> ModuleType:
        return import_module(f"prompts.consistency.{self.name}.comparison_prompt")


SPECS: Dict[str, EvaluationSpec] = {
    name: EvaluationSpec(name=name)
    for name in ("bamboogle", "crag", "hotpotqa", "mintaka")
}


def get_spec(name: str) -> EvaluationSpec:
    if name not in SPECS:
        raise ValueError(
            f"Unknown evaluation dataset: {name!r}; valid: {sorted(SPECS)}"
        )
    return SPECS[name]


__all__ = ["EvaluationSpec", "SPECS", "REPO_ROOT", "get_spec"]
