#!/usr/bin/env python3
"""Single dispatcher for the multi-hop QA consistency pipeline.

Subcommands:
  generate   - generate answers for a dataset+model across the four reasoning
               regimes (open_ended, assistive, incremental, model_generated).
  evaluate   - run consistency (DBA-A) and optionally correctness judges for
               a dataset+model against generated answers.
  baseline   - run AYS / IC-IDK / DBA-A / Self-Consistency baselines for a
               dataset+model.
  decomp-gen - generate model-produced DSL decompositions across datasets
               (appendix ablation).
  decomp-eval- judge model-produced decompositions against gold DSL.

All subcommands accept `--dataset {bamboogle,crag,hotpotqa,mintaka}` where
applicable and pass through remaining flags to the underlying per-dataset
pipeline; the dispatcher only resolves which module to invoke.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATASETS = ("bamboogle", "crag", "hotpotqa", "mintaka")


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------

def _cmd_generate(args: argparse.Namespace, passthrough: list[str]) -> int:
    """Run the four-regime generation pipeline for one dataset + model."""
    script = REPO_ROOT / "generate" / f"run_{args.dataset}_regimes.py"
    if not script.exists():
        sys.exit(f"Generation runner not found: {script}")
    cmd = [sys.executable, str(script), *passthrough]
    return subprocess.run(cmd).returncode


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

def _cmd_evaluate(args: argparse.Namespace, passthrough: list[str]) -> int:
    """Run consistency + optional correctness judges for one dataset + model."""
    from evaluation.consistency import (
        DEFAULT_TARGET_REGIMES as DEFAULT_CONS_REGIMES,
        run_comparisons,
    )
    from evaluation.correctness import REGIME_CHOICES as ALL_CORR_REGIMES
    from evaluation.correctness import run_correctness_batch
    from evaluation.specs import get_spec

    if not args.skip_consistency:
        regimes = args.consistency_regimes or DEFAULT_CONS_REGIMES
        run_comparisons(
            args.dataset,
            args.model,
            regimes,
            args.consistency_limit,
            resume=args.resume,
        )
    else:
        print("Skipping consistency evaluation (use without --skip-consistency to enable).")

    if args.run_correctness:
        regimes = args.correctness_regimes or ALL_CORR_REGIMES
        dsl_path = Path(args.dsl).resolve() if args.dsl else get_spec(args.dataset).dsl_path
        run_correctness_batch(
            args.dataset,
            args.model,
            regimes,
            dsl_path,
            resume=args.resume,
        )
    else:
        print("Skipping correctness evaluation (use --run-correctness to enable).")
    return 0


def _call_with_optional_resume(fn, *args, resume: bool) -> None:
    """Call `fn(*args)`, passing `resume=` only if the function accepts it."""
    sig = inspect.signature(fn)
    if "resume" in sig.parameters:
        fn(*args, resume=resume)
    else:
        fn(*args)


# ---------------------------------------------------------------------------
# baseline
# ---------------------------------------------------------------------------

def _cmd_baseline(args: argparse.Namespace, passthrough: list[str]) -> int:
    """Run the AYS / IC-IDK / DBA-A / Self-Consistency baseline pipeline."""
    script = REPO_ROOT / "baselines" / "run_baselines.py"
    cmd = [sys.executable, str(script), "--dataset", args.dataset, *passthrough]
    return subprocess.run(cmd).returncode


# ---------------------------------------------------------------------------
# decomp-gen / decomp-eval
# ---------------------------------------------------------------------------

def _cmd_decomp_gen(args: argparse.Namespace, passthrough: list[str]) -> int:
    """Generate model-produced DSL decompositions for all datasets."""
    from decomposition_equivalence.config import DATASET_SLUGS

    datasets = args.datasets or DATASET_SLUGS
    script = REPO_ROOT / "decomposition_equivalence" / "generate_dsls.py"
    for dataset in datasets:
        cmd = [sys.executable, str(script), "--dataset", dataset, "--model-slug", args.model_slug]
        if args.model_name:
            cmd += ["--model-name", args.model_name]
        cmd += passthrough
        print(f"\n=== Generating DSLs: {dataset} ({args.model_slug}) ===")
        subprocess.run(cmd, check=True)
    return 0


def _cmd_decomp_eval(args: argparse.Namespace, passthrough: list[str]) -> int:
    """Judge model-produced DSLs against gold DSL."""
    from decomposition_equivalence.config import DATASET_SLUGS, DSL_OUTPUT_ROOT

    datasets = args.datasets or DATASET_SLUGS
    script = REPO_ROOT / "decomposition_equivalence" / "evaluate_decomposition.py"
    found = False
    for dataset in datasets:
        dataset_dir = DSL_OUTPUT_ROOT / dataset
        if not dataset_dir.exists():
            continue
        for path in sorted(dataset_dir.glob("*.json")):
            slug = path.stem
            if args.model_slug and slug != args.model_slug:
                continue
            found = True
            cmd = [
                sys.executable, str(script),
                "--dataset", dataset,
                "--model-slug", slug,
                "--model-dsl", str(path),
                *passthrough,
            ]
            print(f"\n=== Evaluating DSLs: {dataset} ({slug}) ===")
            subprocess.run(cmd, check=True)
    if not found:
        print("No DSL generations found to evaluate.")
    return 0


# ---------------------------------------------------------------------------
# argparse wiring
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="task", required=True)

    # generate
    p_gen = sub.add_parser("generate", help="Generate answers across four regimes.")
    p_gen.add_argument("--dataset", required=True, choices=DATASETS)

    # evaluate
    p_eval = sub.add_parser("evaluate", help="Run consistency / correctness judges.")
    p_eval.add_argument("--dataset", required=True, choices=DATASETS)
    p_eval.add_argument("--model", required=True, help="Model slug matching folder under results/generation/<dataset>/.")
    p_eval.add_argument("--skip-consistency", action="store_true")
    p_eval.add_argument("--consistency-regimes", nargs="+", default=None)
    p_eval.add_argument("--consistency-limit", type=int, default=None)
    p_eval.add_argument("--run-correctness", action="store_true")
    p_eval.add_argument("--correctness-regimes", nargs="+", default=None)
    p_eval.add_argument("--dsl", type=str, default=None, help="Gold DSL path (default: per-dataset DEFAULT_DSL).")
    p_eval.add_argument("--resume", action="store_true")

    # baseline
    p_base = sub.add_parser("baseline", help="Run abstention baselines.")
    p_base.add_argument("--dataset", required=True, choices=DATASETS)

    # decomp-gen
    p_dg = sub.add_parser("decomp-gen", help="Generate model DSLs for all datasets.")
    p_dg.add_argument("--model-slug", required=True)
    p_dg.add_argument("--model-name", type=str, default=None)
    p_dg.add_argument("--datasets", nargs="+", default=None)

    # decomp-eval
    p_de = sub.add_parser("decomp-eval", help="Judge model DSLs vs gold DSLs.")
    p_de.add_argument("--model-slug", type=str, default=None, help="Filter to one slug.")
    p_de.add_argument("--datasets", nargs="+", default=None)

    return parser


HANDLERS = {
    "generate": _cmd_generate,
    "evaluate": _cmd_evaluate,
    "baseline": _cmd_baseline,
    "decomp-gen": _cmd_decomp_gen,
    "decomp-eval": _cmd_decomp_eval,
}


def main() -> int:
    parser = _build_parser()
    args, passthrough = parser.parse_known_args()
    return HANDLERS[args.task](args, passthrough)


if __name__ == "__main__":
    raise SystemExit(main())
