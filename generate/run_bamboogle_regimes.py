#!/usr/bin/env python3
"""Run Bamboogle examples across four reasoning regimes with a shared model."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from generate.bamboogle.cli import (  # noqa: E402
    add_shared_generation_args,
    ensure_results_dir,
    make_model_slug,
    setup_model_from_args,
)
from generate.bamboogle.config import REFERENCE_DATE, TEMPERATURE  # noqa: E402
from generate.bamboogle.pipeline import (  # noqa: E402
    BamboogleGenerationConfig,
    BamboogleGenerationPipeline,
    load_examples,
)
from generate.model_generated_plan.agents.self_ask import SelfAskAgent  # noqa: E402

from utils.model_interface import _TOKEN_LOG_PATH  # noqa: E402

REGIMES = ("open_ended", "assistive", "incremental", "model_generated")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Bamboogle across four reasoning regimes with vLLM-backed models"
    )
    add_shared_generation_args(parser)
    parser.add_argument(
        "--max-selfask-steps",
        type=int,
        default=6,
        help="Maximum subquestions for model-generated plan",
    )
    parser.add_argument(
        "--skip-batch",
        action="store_true",
        help="Skip GPT-5.1 batch path (force synchronous for all regimes)",
    )
    return parser.parse_args()


def summarize_token_usage(example_ids: list[str]) -> dict | None:
    """Print per-question and average token usage from token_usage_log.jsonl."""
    log_path = REPO_ROOT / _TOKEN_LOG_PATH
    if not log_path.exists():
        print("Token log not found; skipping token summary.")
        return None

    example_ids = [eid for eid in example_ids if eid]
    if not example_ids:
        print("No example ids to summarize; skipping token summary.")
        return None

    regimes = ("open_ended", "assistive", "incremental", "model_generated")
    per = {eid: {r: [0, 0] for r in regimes} for eid in example_ids}

    with log_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = __import__("json").loads(line)
            except Exception:
                continue
            meta = rec.get("meta", {})
            ex = str(meta.get("example_id", ""))
            reg = meta.get("regime")
            if ex in per and reg in regimes:
                per[ex][reg][0] += rec.get("prompt_tokens", 0)
                per[ex][reg][1] += rec.get("completion_tokens", 0)

    print("\nToken usage per question:")
    for ex in example_ids:
        print(f"  question {ex}:")
        for reg in regimes:
            pin, pout = per[ex][reg]
            print(f"    {reg:14s} {pin} in, {pout} out")

    print("\nBamboogle averages (over these questions):")
    averages = {}
    for reg in regimes:
        total_in = sum(per[ex][reg][0] for ex in example_ids)
        total_out = sum(per[ex][reg][1] for ex in example_ids)
        avg_in = total_in / len(example_ids)
        avg_out = total_out / len(example_ids)
        print(f"  {reg:14s} {avg_in:.1f} in, {avg_out:.1f} out")
        averages[reg] = {"avg_in": avg_in, "avg_out": avg_out}

    return {"per_question": per, "averages": averages}


def main() -> None:
    args = parse_args()
    examples = load_examples(args.dataset_file, args.limit)
    print(f"Loaded {len(examples)} Bamboogle examples from {args.dataset_file}")

    with setup_model_from_args(args) as (model, launching_server):
        model_slug = make_model_slug(args.model_choice, args.model_name)
        output_dir = ensure_results_dir(model_slug)
        default_output = output_dir / "bamboogle_regimes_full.jsonl"
        output_path = Path(args.output) if args.output else default_output
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Selected model preset: {args.model_choice}")
        if getattr(args, "use_vertex", False):
            print("Using Vertex AI endpoint for Gemini model.")
        elif getattr(args, "use_gpt5", False):
            print("Using OpenAI GPT-5.1 Chat Completions.")
        elif launching_server:
            print(f"Using locally launched vLLM server at {args.vllm_host}:{args.vllm_port}")
        else:
            print(f"Connecting to vLLM server at {args.vllm_base_url}")

        print(f"Model: {args.model_name}")
        print(f"Output file: {output_path}")

        selfask_agent = SelfAskAgent(
            model,
            max_steps=args.max_selfask_steps,
            temperature=TEMPERATURE,
            dataset="bamboogle",
            model_name=args.model_name,
            default_usage_meta={"regime": "model_generated"},
        )

        config = BamboogleGenerationConfig(
            model=model,
            model_name=args.model_name,
            regimes=REGIMES,
            dataset_file=args.dataset_file,
            limit=args.limit,
            resume=args.resume,
            output_dir=output_dir,
            combined_output_path=output_path,
            reference_date=REFERENCE_DATE,
            temperature=TEMPERATURE,
            selfask_agent=selfask_agent,
            examples=examples,
            skip_batch=args.skip_batch,
        )
        pipeline = BamboogleGenerationPipeline(config)
        new_records = pipeline.run()
        print(f"\nSaved {new_records} new evaluation records to {output_path}")

    summary = summarize_token_usage([str(ex.get("id", "")) for ex in examples])
    if summary:
        summary_path = ensure_results_dir(make_model_slug(args.model_choice, args.model_name)) / "token_summary.json"
        import json
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Token summary written to {summary_path}")


if __name__ == "__main__":
    main()
