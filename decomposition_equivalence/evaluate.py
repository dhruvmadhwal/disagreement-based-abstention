#!/usr/bin/env python3
"""Evaluate DSL decompositions against gold using Gemini judge."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from utils.model_interface import GeminiVertexModel

from decomposition_equivalence.config import (
    DATASET_FILES,
    DATASET_SLUGS,
    DSL_OUTPUT_ROOT,
    EVAL_OUTPUT_ROOT,
    SUMMARY_OUTPUT_ROOT,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_JUDGE_TEMPERATURE,
    DEFAULT_JUDGE_MAX_TOKENS,
    slugify,
)
from decomposition_equivalence.dsl_utils import parse_dsl_hops, clean_dsl_text
from decomposition_equivalence.prompts import (
    build_judge_system_prompt,
    build_judge_user_prompt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DSL decompositions vs gold")
    parser.add_argument("--dataset", required=True, choices=DATASET_SLUGS)
    parser.add_argument("--model-slug", required=False, default=None)
    parser.add_argument("--generated", type=Path, default=None, help="Path to generated DSL JSON")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--judge-model", type=str, default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--judge-temperature", type=float, default=DEFAULT_JUDGE_TEMPERATURE)
    parser.add_argument("--judge-max-tokens", type=int, default=DEFAULT_JUDGE_MAX_TOKENS)
    parser.add_argument("--sleep", type=float, default=0.3)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2, ensure_ascii=False)


def safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    # fallback: grab first {...}
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        return None
    return None


def load_gold(dataset: str) -> Dict[str, Dict[str, Any]]:
    data = load_json(DATASET_FILES[dataset])
    return {str(item["id"]): item for item in data if isinstance(item, dict) and item.get("id") is not None}


def load_generated(path: Path) -> Dict[str, Dict[str, Any]]:
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in generated DSL file: {path}")
    return {str(item.get("id")): item for item in data if isinstance(item, dict) and item.get("id") is not None}


def load_existing(output_path: Path) -> Tuple[List[Dict[str, Any]], set[str]]:
    if not output_path.exists():
        return [], set()
    try:
        data = load_json(output_path)
    except Exception:
        return [], set()
    if isinstance(data, dict) and "results" in data:
        records = data.get("results", [])
    elif isinstance(data, list):
        records = data
    else:
        records = []
    processed = {str(item.get("id")) for item in records if isinstance(item, dict) and item.get("id") is not None}
    return records, processed


def compute_precision_recall_f1(matched_model: int, matched_gold: int, model_count: int, gold_count: int) -> Dict[str, float]:
    precision = matched_model / model_count if model_count else 0.0
    recall = matched_gold / gold_count if gold_count else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    hop_diffs = [r.get("hop_count_diff") for r in records if isinstance(r.get("hop_count_diff"), (int, float))]
    hop_ratios = [r.get("hop_count_ratio") for r in records if isinstance(r.get("hop_count_ratio"), (int, float))]
    precs = [r.get("precision") for r in records if isinstance(r.get("precision"), (int, float))]
    recalls = [r.get("recall") for r in records if isinstance(r.get("recall"), (int, float))]
    f1s = [r.get("f1") for r in records if isinstance(r.get("f1"), (int, float))]
    equivalences = [r.get("equivalent_final") for r in records if isinstance(r.get("equivalent_final"), int)]

    def _mean(values: List[float]) -> Optional[float]:
        return statistics.mean(values) if values else None

    return {
        "count": len(records),
        "avg_hop_diff": _mean(hop_diffs),
        "avg_hop_ratio": _mean(hop_ratios),
        "avg_precision": _mean(precs),
        "avg_recall": _mean(recalls),
        "avg_f1": _mean(f1s),
        "equivalence_rate": (
            sum(1 for v in equivalences if v == 1) / len(equivalences)
            if equivalences else None
        ),
    }


def judge_one(
    judge: GeminiVertexModel,
    dataset: str,
    question: str,
    gold_dsl: str,
    model_dsl: str,
    gold_hops: List[str],
    model_hops: List[str],
    temperature: float,
    max_tokens: int,
    debug: bool,
) -> Dict[str, Any]:
    system_prompt = build_judge_system_prompt(dataset)
    user_prompt = build_judge_user_prompt(dataset, question, gold_dsl, model_dsl, gold_hops, model_hops)
    if debug:
        print("\n=== JUDGE INPUT ===")
        print(user_prompt)
    response = judge.generate_answer(
        "",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    parsed = safe_json_loads(response.strip())
    if parsed is None:
        parsed = {
            "equivalent_final": -1,
            "gold_hops_covered": [],
            "model_hops_covered": [],
            "coverage_pairs": [],
            "notes": "failed_to_parse",
            "raw_response": response.strip(),
        }
    else:
        parsed["raw_response"] = response.strip()
    return parsed


def main() -> None:
    args = parse_args()

    model_slug = args.model_slug
    if model_slug is None and args.generated is not None:
        model_slug = slugify(args.generated.stem)
    if model_slug is None:
        model_slug = "model"

    if args.generated is None:
        generated_path = DSL_OUTPUT_ROOT / args.dataset / f"{model_slug}.json"
    else:
        generated_path = args.generated

    if args.output is None:
        output_path = EVAL_OUTPUT_ROOT / args.dataset / f"{model_slug}.json"
    else:
        output_path = args.output

    if args.summary is None:
        summary_path = SUMMARY_OUTPUT_ROOT / args.dataset / f"{model_slug}.json"
    else:
        summary_path = args.summary

    gold = load_gold(args.dataset)
    generated = load_generated(generated_path)

    common_ids = sorted(set(gold.keys()) & set(generated.keys()))
    if args.limit is not None:
        common_ids = common_ids[: args.limit]

    existing_records: List[Dict[str, Any]] = []
    processed_ids: set[str] = set()
    if args.resume:
        existing_records, processed_ids = load_existing(output_path)
        if existing_records:
            print(f"Resuming from {output_path} with {len(processed_ids)} records")

    records = list(existing_records)

    judge = GeminiVertexModel(model_name=args.judge_model)

    for idx, qid in enumerate(common_ids, 1):
        if args.resume and qid in processed_ids:
            continue
        gold_item = gold[qid]
        gen_item = generated[qid]
        question = gold_item.get("question", gen_item.get("question", ""))
        gold_dsl = clean_dsl_text(str(gold_item.get("dsl", "")))
        model_dsl = clean_dsl_text(str(gen_item.get("dsl", gen_item.get("dsl_string", ""))))

        gold_hops = parse_dsl_hops(gold_dsl)
        model_hops = parse_dsl_hops(model_dsl)

        judge_result = judge_one(
            judge,
            args.dataset,
            question,
            gold_dsl,
            model_dsl,
            gold_hops,
            model_hops,
            args.judge_temperature,
            args.judge_max_tokens,
            args.debug,
        )

        gold_count = len(gold_hops)
        model_count = len(model_hops)

        gold_covered = judge_result.get("gold_hops_covered", []) or []
        model_covered = judge_result.get("model_hops_covered", []) or []

        matched_gold = len({val for i in gold_covered if (val := _safe_int(i)) is not None})
        matched_model = len({val for i in model_covered if (val := _safe_int(i)) is not None})

        metrics = compute_precision_recall_f1(matched_model, matched_gold, model_count, gold_count)

        record = {
            "id": qid,
            "question": question,
            "gold_hops": gold_hops,
            "model_hops": model_hops,
            "gold_hop_count": gold_count,
            "model_hop_count": model_count,
            "hop_count_diff": model_count - gold_count,
            "hop_count_ratio": (model_count / gold_count) if gold_count else None,
            "equivalent_final": judge_result.get("equivalent_final", -1),
            "gold_hops_covered": gold_covered,
            "model_hops_covered": model_covered,
            "coverage_pairs": judge_result.get("coverage_pairs", []),
            "judge_notes": judge_result.get("notes"),
            "judge_raw": judge_result.get("raw_response"),
            "matched_gold": matched_gold,
            "matched_model": matched_model,
            "model_name": gen_item.get("model"),
            "model_slug": gen_item.get("model_slug", model_slug),
            **metrics,
        }
        records.append(record)

        if idx % 10 == 0:
            save_json(output_path, {"results": records})

        if args.sleep:
            time.sleep(args.sleep)

    summary = summarize(records)
    save_json(output_path, {"results": records, "summary": summary})
    save_json(summary_path, summary)

    print(f"Saved {len(records)} evaluation records to {output_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
