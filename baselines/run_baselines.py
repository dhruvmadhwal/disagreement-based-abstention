#!/usr/bin/env python3
"""Run baseline detectors (AYS, IDK, IC-IDK, self-consistency) across datasets."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
from typing import Dict, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from baselines.crag import config as crag_config  # noqa: E402
from baselines.crag.pipeline import CragBaselineConfig, CragBaselinePipeline  # noqa: E402
from baselines.hotpotqa import config as hotpotqa_config  # noqa: E402
from baselines.hotpotqa.pipeline import HotpotQABaselineConfig, HotpotQABaselinePipeline  # noqa: E402
from baselines.mintaka import config as mintaka_config  # noqa: E402
from baselines.mintaka.pipeline import MintakaBaselineConfig, MintakaBaselinePipeline  # noqa: E402
from generate.crag import cli as crag_cli  # noqa: E402
from generate.hotpotqa import cli as hotpotqa_cli  # noqa: E402
from generate.mintaka import cli as mintaka_cli  # noqa: E402

# Optional bamboogle support (not always present in all checkouts)
_BAMBOOGLE_AVAILABLE = True
try:
    from baselines.bamboogle import config as bamboogle_config  # type: ignore  # noqa: E402
    from baselines.bamboogle.pipeline import (  # type: ignore  # noqa: E402
        BamboogleBaselineConfig,
        BamboogleBaselinePipeline,
    )
    from generate.bamboogle import cli as bamboogle_cli  # type: ignore  # noqa: E402
except ImportError:
    _BAMBOOGLE_AVAILABLE = False


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    cli: object
    config: object
    pipeline_config_cls: object
    pipeline_cls: object
    ic_idk_raw_default: Path
    ic_idk_build_script: Path


DATASET_SPECS: Dict[str, DatasetSpec] = {
    "mintaka": DatasetSpec(
        key="mintaka",
        cli=mintaka_cli,
        config=mintaka_config,
        pipeline_config_cls=MintakaBaselineConfig,
        pipeline_cls=MintakaBaselinePipeline,
        ic_idk_raw_default=Path("data/processed/mintaka/mintaka_icidk_raw_100.json"),
        ic_idk_build_script=REPO_ROOT / "baselines" / "mintaka" / "build_ic_idk_pool.py",
    ),
    "crag": DatasetSpec(
        key="crag",
        cli=crag_cli,
        config=crag_config,
        pipeline_config_cls=CragBaselineConfig,
        pipeline_cls=CragBaselinePipeline,
        ic_idk_raw_default=Path("data/processed/crag/crag_icidk_raw_100.json"),
        ic_idk_build_script=REPO_ROOT / "baselines" / "crag" / "build_ic_idk_pool.py",
    ),
    "hotpotqa": DatasetSpec(
        key="hotpotqa",
        cli=hotpotqa_cli,
        config=hotpotqa_config,
        pipeline_config_cls=HotpotQABaselineConfig,
        pipeline_cls=HotpotQABaselinePipeline,
        ic_idk_raw_default=Path("data/processed/hotpotqa/hotpot_icidk_raw_100.json"),
        ic_idk_build_script=REPO_ROOT / "baselines" / "hotpotqa" / "build_ic_idk_pool.py",
    ),
}

if _BAMBOOGLE_AVAILABLE:
    DATASET_SPECS["bamboogle"] = DatasetSpec(
        key="bamboogle",
        cli=bamboogle_cli,
        config=bamboogle_config,
        pipeline_config_cls=BamboogleBaselineConfig,
        pipeline_cls=BamboogleBaselinePipeline,
        ic_idk_raw_default=Path("data/processed/bamboogle/bamboogle_dsl.json"),
        ic_idk_build_script=REPO_ROOT / "baselines" / "bamboogle" / "build_ic_idk_pool.py",
    )


def parse_args(argv: list[str] | None = None) -> Tuple[argparse.Namespace, DatasetSpec]:
    dataset_parser = argparse.ArgumentParser(add_help=False)
    dataset_parser.add_argument(
        "--dataset",
        choices=DATASET_SPECS.keys(),
        default="mintaka",
        help="Target dataset for baseline detectors.",
    )
    dataset_args, _ = dataset_parser.parse_known_args(argv)
    spec = DATASET_SPECS[dataset_args.dataset]

    parser = argparse.ArgumentParser(
        description="Run baseline detectors using a shared model (defaults to Gemini 2.5 Flash)."
    )
    parser.add_argument(
        "--dataset",
        choices=DATASET_SPECS.keys(),
        default=dataset_args.dataset,
        help="Dataset to evaluate (bamboogle, mintaka, crag, hotpotqa).",
    )
    spec.cli.add_shared_generation_args(parser)
    parser.add_argument(
        "--baselines",
        nargs="+",
        choices=spec.config.BASELINES,
        default=list(spec.config.DEFAULT_BASELINES),
        help="Subset of baselines to run (default excludes plain IDK).",
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Skip Gemini correctness judging (only produce raw baseline outputs).",
    )
    parser.add_argument(
        "--ic-idk-k",
        type=int,
        default=spec.config.DEFAULT_IC_IDK_K,
        help=f"Total IC-IDK demos to include (default: {spec.config.DEFAULT_IC_IDK_K}).",
    )
    parser.add_argument(
        "--ic-idk-d",
        type=int,
        default=spec.config.DEFAULT_IC_IDK_D,
        help=f"Number of IDK demos inside IC-IDK (default: {spec.config.DEFAULT_IC_IDK_D}).",
    )
    parser.add_argument(
        "--ic-idk-prompt-file",
        type=Path,
        default=None,
        help="Optional path to held-out IC-IDK prompt examples (outside the eval set).",
    )
    parser.add_argument(
        "--self-consistency-samples",
        type=int,
        default=spec.config.DEFAULT_SELF_CONSISTENCY_SAMPLES,
        help=(
            "Total samples for self-consistency voting (default: "
            f"{spec.config.DEFAULT_SELF_CONSISTENCY_SAMPLES}; 1 at T=0, the rest at the high-temp setting)."
        ),
    )
    parser.add_argument(
        "--self-consistency-temperature",
        type=float,
        default=spec.config.DEFAULT_SELF_CONSISTENCY_TEMPERATURE,
        help=(
            "High-temperature setting for self-consistency / self-reflection sampling "
            f"(default: {spec.config.DEFAULT_SELF_CONSISTENCY_TEMPERATURE})."
        ),
    )
    parser.add_argument(
        "--self-consistency-zero-samples",
        type=int,
        default=spec.config.DEFAULT_SELF_CONSISTENCY_ZERO_SAMPLES,
        help=(
            "Number of deterministic (T=0) samples to include in the self-consistency vote "
            f"(default: {spec.config.DEFAULT_SELF_CONSISTENCY_ZERO_SAMPLES})."
        ),
    )
    parser.add_argument(
        "--self-consistency-zero-temperature",
        type=float,
        default=spec.config.DEFAULT_SELF_CONSISTENCY_ZERO_TEMPERATURE,
        help=(
            "Temperature used for the deterministic self-consistency/self-reflection sample "
            f"(default: {spec.config.DEFAULT_SELF_CONSISTENCY_ZERO_TEMPERATURE})."
        ),
    )
    parser.add_argument(
        "--pairwise-temperature",
        type=float,
        default=spec.config.DEFAULT_PAIRWISE_TEMPERATURE,
        help=f"Temperature for pairwise consistency judging (default: {spec.config.DEFAULT_PAIRWISE_TEMPERATURE}).",
    )
    parser.add_argument(
        "--pairwise-top-p",
        type=float,
        default=spec.config.DEFAULT_PAIRWISE_TOP_P,
        help=f"Top-p for pairwise consistency judging (default: {spec.config.DEFAULT_PAIRWISE_TOP_P}).",
    )
    parser.add_argument(
        "--pairwise-top-k",
        type=int,
        default=spec.config.DEFAULT_PAIRWISE_TOP_K,
        help=f"Top-k for pairwise consistency judging (default: {spec.config.DEFAULT_PAIRWISE_TOP_K}).",
    )
    parser.add_argument(
        "--skip-batch",
        action="store_true",
        help="Skip GPT-5.1 batch path (force synchronous generation).",
    )
    parser.add_argument(
        "--ic-idk-raw-dataset",
        type=Path,
        default=spec.ic_idk_raw_default,
        help=f"Raw {spec.key} pool source for auto-building IC-IDK demos when none is provided.",
    )
    parser.add_argument(
        "--ic-idk-pool-start",
        type=int,
        default=0,
        help="Start index for held-out IC-IDK pool examples (default: 0).",
    )
    parser.add_argument(
        "--ic-idk-pool-size",
        type=int,
        default=100,
        help="Number of held-out examples to score when auto-building the IC-IDK pool (default: 100 curated examples).",
    )
    parser.add_argument(
        "--no-auto-ic-idk-pool",
        action="store_true",
        help="Skip auto-generating the IC-IDK pool when missing.",
    )
    parser.add_argument(
        "--ic-idk-seed",
        type=int,
        default=13,
        help="Seed for IC-IDK pool generation and demo shuffling (default: 13).",
    )
    parser.add_argument(
        "--no-model",
        action="store_true",
        help="Skip model initialization (use only cached results, no model calls).",
    )
    args = parser.parse_args(argv)
    return args, DATASET_SPECS[args.dataset]


def ensure_ic_idk_pool_path(
    args: argparse.Namespace,
    spec: DatasetSpec,
    model_slug: str,
) -> Path | None:
    if args.ic_idk_prompt_file:
        if not args.ic_idk_prompt_file.exists():
            raise FileNotFoundError(f"IC-IDK prompt file {args.ic_idk_prompt_file} not found.")
        return args.ic_idk_prompt_file

    internal_pool_root = Path("results") / "baselines" / spec.key / "ic_idk_pools"
    default_pool_root = getattr(spec.config, "IC_IDK_POOL_ROOT", internal_pool_root)
    default_pool = default_pool_root / f"{model_slug}.json"
    external_pool = default_pool_root != internal_pool_root
    if default_pool.exists() or args.no_auto_ic_idk_pool or external_pool:
        return default_pool if default_pool.exists() else None

    build_script = spec.ic_idk_build_script
    if not build_script.exists():
        raise FileNotFoundError(f"IC-IDK build script not found at {build_script}")

    cmd = [
        sys.executable,
        str(build_script),
        "--model-choice",
        args.model_choice,
        "--model-name",
        args.model_name,
        "--raw-dataset",
        str(args.ic_idk_raw_dataset),
        "--pool-start",
        str(args.ic_idk_pool_start),
        "--pool-size",
        str(args.ic_idk_pool_size),
        "--pool-output",
        str(default_pool),
        "--seed",
        str(args.ic_idk_seed),
    ]
    if not getattr(args, "use_vertex", False):
        cmd.append("--no-launch-vllm-server")
        base_url = args.vllm_base_url
        if not base_url:
            host = getattr(args, "vllm_host", None) or "127.0.0.1"
            port = getattr(args, "vllm_port", None) or 8000
            base_url = f"http://{host}:{port}/v1"
        cmd.extend(["--vllm-base-url", base_url])
        if args.vllm_api_key:
            cmd.extend(["--vllm-api-key", args.vllm_api_key])
    print("Auto-building IC-IDK pool:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return default_pool


def main(argv: list[str] | None = None) -> None:
    args, spec = parse_args(argv)
    baselines = args.baselines or list(spec.config.BASELINES)

    def run_pipeline(model, launching_server=False):
        # Always resolve model choice to get correct model_name for slug computation
        # (needed for --no-model to match existing directory names)
        spec.cli._resolve_model_choice(args)
        model_slug = spec.cli.make_model_slug(args.model_choice, args.model_name)
        output_dir = spec.config.ensure_baseline_dir(model_slug)
        combined_output = Path(args.output) if args.output else output_dir / spec.config.DEFAULT_COMBINED_FILENAME
        combined_output.parent.mkdir(parents=True, exist_ok=True)

        print(f"Dataset: {args.dataset}")
        print(f"Selected model preset: {args.model_choice}")
        if args.no_model:
            print("Skipping model initialization (--no-model). Using cached results only.")
        elif getattr(args, "use_vertex", False):
            print("Using Vertex AI endpoint for Gemini model.")
        elif args.model_choice and args.model_choice.lower() == "gpt5":
            print("Using OpenAI GPT-5.1 (batch-capable).")
        elif launching_server:
            print(f"Using locally launched vLLM server at {args.vllm_host}:{args.vllm_port}")
        else:
            print(f"Connecting to vLLM server at {args.vllm_base_url}")

        print(f"Model: {args.model_name}")
        print(f"Output directory: {output_dir}")
        if args.output:
            print(f"Combined output file: {combined_output}")

        config = spec.pipeline_config_cls(
            model=model,
            model_name=args.model_name,
            model_choice=args.model_choice,
            baselines=baselines,
            dataset_file=args.dataset_file,
            limit=args.limit,
            resume=args.resume,
            output_dir=spec.config.BASELINE_RESULTS_ROOT,
            combined_output_path=combined_output,
            reference_date=spec.config.REFERENCE_DATE,
            temperature=spec.config.DEFAULT_TEMPERATURE,
            top_p=spec.config.DEFAULT_TOP_P,
            top_k=spec.config.DEFAULT_TOP_K,
            self_consistency_samples=args.self_consistency_samples,
            self_consistency_temperature=args.self_consistency_temperature,
            self_consistency_zero_samples=args.self_consistency_zero_samples,
            self_consistency_zero_temperature=args.self_consistency_zero_temperature,
            ic_idk_k=args.ic_idk_k,
            ic_idk_d=args.ic_idk_d,
            ic_idk_prompt_file=ensure_ic_idk_pool_path(args, spec, model_slug)
            if "ic_idk" in baselines and not args.no_model
            else None,
            run_judge=not args.skip_judge and not args.no_model,
            pairwise_temperature=args.pairwise_temperature,
            pairwise_top_p=args.pairwise_top_p,
            pairwise_top_k=args.pairwise_top_k,
            skip_batch=args.skip_batch,
        )
        pipeline = spec.pipeline_cls(config)
        new_records = pipeline.run()
        print(f"\nSaved {new_records} new baseline records to {output_dir}")

    if args.no_model:
        # Skip model initialization entirely
        run_pipeline(None, launching_server=False)
    else:
        with spec.cli.setup_model_from_args(args) as (model, launching_server):
            run_pipeline(model, launching_server)


if __name__ == "__main__":
    main()
