#!/usr/bin/env python3
"""Build IC-IDK prompt pools (15 demos: 11 correct, 4 IDK) using local/Vertex models."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
repo_str = str(REPO_ROOT)
if repo_str in sys.path:
    sys.path.remove(repo_str)
sys.path.insert(0, repo_str)

from baselines.mintaka.judge import CorrectnessJudge  # noqa: E402
from generate.mintaka.cli import add_shared_generation_args, setup_model_from_args  # noqa: E402
from generate.mintaka.config import make_model_slug  # noqa: E402
from prompts.mintaka.prompts import create_open_ended_system_prompt, create_open_ended_user_prompt
from generate.mintaka.pipeline import parse_open_ended_response, strip_think_tags
from generate.mintaka.config import TEMPERATURE, TOP_K, TOP_P


def load_raw(path: Path, start: int, limit: int | None) -> List[dict]:
    data = json.loads(path.read_text())
    start_idx = max(0, start)
    end_idx = start_idx + limit if limit else None
    return data[start_idx:end_idx]


def collect_ic_idk_pool(
    raw_examples: List[dict],
    model,
    judge: CorrectnessJudge,
    *,
    k_correct: int,
    k_incorrect: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """Iterate sequentially and stop once k_correct + k_incorrect are collected."""

    sys_prompt = create_open_ended_system_prompt()
    selected_correct: List[Dict[str, object]] = []
    selected_incorrect: List[Dict[str, object]] = []
    processed = 0
    rng = random.Random(seed)
    total_candidates = len(raw_examples)

    for ex in raw_examples:
        if len(selected_correct) >= k_correct and len(selected_incorrect) >= k_incorrect:
            break
        qid = str(ex.get("id") or ex.get("qid") or "")
        question = str(ex.get("question", "")).strip()
        gold = str(ex.get("answer", "")).strip()
        if not question:
            continue

        raw = model.generate_answer(
            question,
            system=sys_prompt,
            user=create_open_ended_user_prompt(question),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        processed += 1
        cleaned = strip_think_tags(raw)
        prediction = parse_open_ended_response(cleaned)
        judge_result = judge.score(question, gold, prediction)
        is_correct = judge_result.correct == 1
        status = "correct" if is_correct else "incorrect"
        print(
            f"[scored {processed}/{total_candidates}] id={qid} {status} | "
            f"collected {len(selected_correct)}/{k_correct} correct, "
            f"{len(selected_incorrect)}/{k_incorrect} incorrect"
        )
        record = {
            "id": qid,
            "question": question,
            "gold": gold,
            "ic_idk_is_correct": is_correct,
            "ic_idk_answer": gold if is_correct else "I don't know",
            "raw_response": cleaned,
            "judge_reasoning": judge_result.reasoning,
            "judge_raw_response": judge_result.raw_response,
        }
        if is_correct and len(selected_correct) < k_correct:
            selected_correct.append(record)
        elif not is_correct and len(selected_incorrect) < k_incorrect:
            selected_incorrect.append(record)

    selected = selected_correct + selected_incorrect
    rng.shuffle(selected)
    summary = {
        "total_scored": processed,
        "selected_correct": len(selected_correct),
        "selected_incorrect": len(selected_incorrect),
        "target_correct": k_correct,
        "target_incorrect": k_incorrect,
        "early_stop": processed < len(raw_examples)
        and len(selected_correct) >= k_correct
        and len(selected_incorrect) >= k_incorrect,
    }
    return selected, summary


def save_output(path: Path, selected: List[Dict[str, object]], summary: Dict[str, object]) -> None:
    payload = {"summary": summary, "records": selected}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate IC-IDK prompt pools (11 correct + 4 IDK).")
    add_shared_generation_args(parser)
    parser.add_argument(
        "--raw-dataset",
        type=Path,
        default=Path("data/processed/mintaka/mintaka_icidk_raw_100.json"),
        help="Raw Mintaka data for IC-IDK pool (default: curated 100 held-out examples).",
    )
    parser.add_argument(
        "--pool-output",
        type=Path,
        default=None,
        help="Override output path for the IC-IDK pool (default: results/baselines/mintaka/ic_idk_pools/<model_slug>.json).",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--k-correct", type=int, default=11)
    parser.add_argument("--k-incorrect", type=int, default=4)
    parser.add_argument(
        "--pool-start",
        type=int,
        default=0,
        help="Offset into --raw-dataset to start building the prompt pool (default: 0).",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=300,
        help="Number of held-out examples to score for IC-IDK prompt construction (default: 300).",
    )
    args = parser.parse_args()

    pool_size = args.limit or args.pool_size
    raw_examples = load_raw(args.raw_dataset, args.pool_start, pool_size)
    print(
        f"Building IC-IDK pool from {len(raw_examples)} held-out examples "
        f"(start={args.pool_start}, target={args.k_correct} correct/{args.k_incorrect} incorrect)."
    )

    with setup_model_from_args(args) as (model, _):
        judge = CorrectnessJudge()
        selected, summary = collect_ic_idk_pool(
            raw_examples,
            model,
            judge,
            k_correct=args.k_correct,
            k_incorrect=args.k_incorrect,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            seed=args.seed,
        )

    model_slug = make_model_slug(args.model_choice, args.model_name)
    default_output = Path("results/baselines/mintaka/ic_idk_pools") / f"{model_slug}.json"
    output_path = args.pool_output or default_output

    summary = {
        "dataset": str(args.raw_dataset),
        "model_slug": model_slug,
        "pool_start": args.pool_start,
        "pool_size": pool_size,
        **summary,
    }
    save_output(output_path, selected, summary)

    if summary.get("selected_correct", 0) < args.k_correct or summary.get("selected_incorrect", 0) < args.k_incorrect:
        print(
            f"Warning: requested {args.k_correct} correct / {args.k_incorrect} incorrect; "
            f"got {summary.get('selected_correct', 0)} / {summary.get('selected_incorrect', 0)}."
        )
    print(f"Saved IC-IDK pool to {output_path}")


if __name__ == "__main__":
    main()
