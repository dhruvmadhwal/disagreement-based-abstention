#!/usr/bin/env python3
"""Generate CRAG assistive regime outputs using shared utilities."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from generate.crag.cli import (
    add_shared_generation_args,
    ensure_results_dir,
    make_model_slug,
    setup_model_from_args,
)
from generate.crag.config import REFERENCE_DATE, TEMPERATURE
from generate.crag.pipeline import CragGenerationConfig, CragGenerationPipeline, load_examples

REGIME = ("assistive",)
REGIME_KEY = "assistive"
DEFAULT_FILENAME = "crag_assistive.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CRAG assistive reasoning outputs")
    add_shared_generation_args(parser)
    return parser.parse_args()


def resolve_output_path(args: argparse.Namespace, model_slug: str) -> Path:
    if args.output is not None:
        return Path(args.output)
    output_dir = ensure_results_dir(model_slug)
    return output_dir / DEFAULT_FILENAME


def main() -> None:
    args = parse_args()
    examples = load_examples(args.dataset_file, args.limit)
    print(f"Loaded {len(examples)} CRAG examples from {args.dataset_file}")

    with setup_model_from_args(args) as (model, launching_server):
        model_slug = make_model_slug(args.model_choice, args.model_name)
        target_path = resolve_output_path(args, model_slug)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Selected model preset: {args.model_choice}")
        if getattr(args, "use_vertex", False):
            print("Using Vertex AI endpoint for Gemini model.")
        elif launching_server:
            print(f"Using locally launched vLLM server at {args.vllm_host}:{args.vllm_port}")
        else:
            print(f"Connecting to vLLM server at {args.vllm_base_url}")

        config = CragGenerationConfig(
            model=model,
            model_name=args.model_name,
            regimes=REGIME,
            dataset_file=args.dataset_file,
            limit=args.limit,
            resume=args.resume,
            output_dir=target_path.parent,
            combined_output_path=None,
            reference_date=REFERENCE_DATE,
            temperature=TEMPERATURE,
            selfask_agent=None,
            regime_output_paths={REGIME_KEY: target_path},
            examples=examples,
        )
        pipeline = CragGenerationPipeline(config)
        new_records = pipeline.run()
        print(f"\nSaved {new_records} assistive records to {target_path}")


if __name__ == "__main__":
    main()
