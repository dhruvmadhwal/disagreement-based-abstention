#!/usr/bin/env python3
"""Evaluate decomposition equivalence between gold and model-generated DSLs."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from utils.model_interface import GeminiVertexModel, VertexBatchHelper

from decomposition_equivalence.config import (
    DATASET_FILES,
    DATASET_SLUGS,
    DSL_OUTPUT_ROOT,
    EVAL_OUTPUT_ROOT,
    SUMMARY_OUTPUT_ROOT,
    BATCH_OUTPUT_ROOT,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_JUDGE_TEMPERATURE,
    DEFAULT_JUDGE_MAX_TOKENS,
)
from decomposition_equivalence.dsl_utils import parse_dsl_hops
from decomposition_equivalence.prompts import (
    build_judge_system_prompt,
    build_judge_user_prompt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate decomposition equivalence for a dataset")
    parser.add_argument("--dataset", required=True, choices=DATASET_SLUGS)
    parser.add_argument("--model-slug", required=True, help="Slug used for model DSL output")
    parser.add_argument("--model-dsl", type=Path, default=None, help="Override model DSL JSON path")
    parser.add_argument("--gold-dsl", type=Path, default=None, help="Override gold DSL JSON path")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--judge-model", type=str, default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--judge-temperature", type=float, default=DEFAULT_JUDGE_TEMPERATURE)
    parser.add_argument("--judge-max-tokens", type=int, default=DEFAULT_JUDGE_MAX_TOKENS)
    parser.add_argument("--batch", action="store_true", help="Use Vertex batch prediction for Gemini judge.")
    parser.add_argument("--batch-fetch", action="store_true", help="Fetch and parse batch output instead of submitting.")
    parser.add_argument("--batch-input-uri", type=str, default=None, help="GCS URI for batch input JSONL.")
    parser.add_argument("--batch-output-uri", type=str, default=None, help="GCS URI prefix for batch output.")
    parser.add_argument("--batch-upload", action="store_true", help="Upload local batch JSONL to GCS using gsutil.")
    parser.add_argument("--batch-index", type=Path, default=None, help="Path to batch index JSON (optional).")
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2, ensure_ascii=False)


def load_existing(path: Path) -> Tuple[List[Dict[str, Any]], set[str]]:
    if not path.exists():
        return [], set()
    try:
        data = load_json(path)
    except Exception:
        return [], set()
    if not isinstance(data, list):
        return [], set()
    processed = {str(item.get("id")) for item in data if isinstance(item, dict)}
    return data, processed


def parse_json_response(raw: str) -> Dict[str, Any]:
    if not raw:
        return {}
    raw = raw.strip()
    try:
        return json.loads(raw)
    except Exception:
        # Attempt to salvage JSON object from within text.
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(raw[start : end + 1])
            except Exception:
                return {}
    return {}


def normalize_index_list(indices: Any, max_len: int) -> List[int]:
    if not isinstance(indices, list):
        return []
    cleaned: List[int] = []
    for item in indices:
        try:
            idx = int(item)
        except Exception:
            continue
        if 1 <= idx <= max_len and idx not in cleaned:
            cleaned.append(idx)
    return cleaned


def compute_metrics(gold_count: int, model_count: int, matches: int) -> Dict[str, float]:
    precision = matches / model_count if model_count else 0.0
    recall = matches / gold_count if gold_count else 0.0
    if precision + recall:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main() -> None:
    args = parse_args()

    gold_path = args.gold_dsl or DATASET_FILES[args.dataset]
    model_path = args.model_dsl or (DSL_OUTPUT_ROOT / args.dataset / f"{args.model_slug}.json")

    if not model_path.exists():
        raise FileNotFoundError(f"Model DSL file not found: {model_path}")

    output_path = args.output or (EVAL_OUTPUT_ROOT / args.dataset / f"{args.model_slug}.json")
    summary_path = args.summary or (SUMMARY_OUTPUT_ROOT / args.dataset / f"{args.model_slug}.json")

    gold_data = load_json(Path(gold_path))
    model_data = load_json(Path(model_path))

    gold_lookup = {str(item.get("id")): item for item in gold_data if isinstance(item, dict)}
    model_lookup = {str(item.get("id")): item for item in model_data if isinstance(item, dict)}

    common_ids = sorted(set(gold_lookup) & set(model_lookup))
    if args.limit is not None:
        common_ids = common_ids[: args.limit]

    existing: List[Dict[str, Any]] = []
    completed: set[str] = set()
    if args.resume:
        existing, completed = load_existing(output_path)
        if existing:
            print(f"Resuming from {output_path}: {len(completed)} examples")

    results = list(existing)
    pending_ids = [example_id for example_id in common_ids if example_id not in completed]

    if args.batch or args.batch_fetch:
        batch_dir = BATCH_OUTPUT_ROOT / args.dataset
        batch_dir.mkdir(parents=True, exist_ok=True)
        batch_input_path = batch_dir / f"{args.model_slug}.jsonl"
        index_path = args.batch_index or (batch_dir / f"{args.model_slug}.index.json")

        if args.batch_fetch:
            if not args.batch_output_uri:
                raise ValueError("--batch-output-uri is required for --batch-fetch.")
            if not index_path.exists():
                raise FileNotFoundError(f"Batch index not found: {index_path}")
            with index_path.open("r", encoding="utf-8") as fp:
                batch_index = json.load(fp)
            if not isinstance(batch_index, list):
                raise ValueError(f"Batch index must be a list: {index_path}")

            output_files = VertexBatchHelper.list_gcs_jsonl(args.batch_output_uri)
            if not output_files:
                raise RuntimeError(f"No batch output JSONL files found under {args.batch_output_uri}")
            output_lines: List[str] = []
            for uri in output_files:
                output_lines.extend(VertexBatchHelper.read_gcs_jsonl(uri))

            if not output_lines:
                raise RuntimeError("Batch output files were empty.")

            for idx, line in enumerate(output_lines):
                if idx >= len(batch_index):
                    break
                example_id = str(batch_index[idx])
                if args.resume and example_id in completed:
                    continue
                gold_item = gold_lookup.get(example_id)
                model_item = model_lookup.get(example_id)
                if not gold_item or not model_item:
                    continue

                question = gold_item.get("question", "")
                gold_dsl = str(gold_item.get("dsl", ""))
                model_dsl = str(model_item.get("dsl", model_item.get("dsl_string", "")))

                gold_hops = parse_dsl_hops(gold_dsl)
                model_hops = parse_dsl_hops(model_dsl)

                raw = line
                parsed = parse_json_response(VertexBatchHelper.extract_text(json.loads(line)))

                gold_count = len(gold_hops)
                model_count = len(model_hops)

                matches_raw = parsed.get("matches")
                if isinstance(matches_raw, (int, float, str)):
                    try:
                        matches = int(float(matches_raw))
                    except Exception:
                        matches = 0
                else:
                    matches = 0
                matches = max(0, min(matches, gold_count, model_count))

                metrics = compute_metrics(gold_count, model_count, matches)
                hop_ratio = (model_count / gold_count) if gold_count else None
                hop_diff = model_count - gold_count

                record = {
                    "id": example_id,
                    "question": question,
                    "gold_dsl": gold_dsl,
                    "model_dsl": model_dsl,
                    "gold_hops": gold_hops,
                    "model_hops": model_hops,
                    "gold_hop_count": gold_count,
                    "model_hop_count": model_count,
                    "hop_ratio": hop_ratio,
                    "hop_diff": hop_diff,
                    "judge_raw": raw,
                    "judge_parsed": parsed,
                    "matches": matches,
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "equivalent_final": parsed.get("equivalent_final"),
                }
                results.append(record)

            save_json(output_path, results)
        else:
            if not args.batch_input_uri or not args.batch_output_uri:
                raise ValueError("--batch-input-uri and --batch-output-uri are required for --batch.")
            if not pending_ids:
                print("No pending examples to batch.")
                return
            system_prompt = build_judge_system_prompt(args.dataset)
            batch_lines: List[Dict[str, Any]] = []
            batch_index: List[str] = []
            for example_id in pending_ids:
                gold_item = gold_lookup[example_id]
                model_item = model_lookup[example_id]

                question = gold_item.get("question", "")
                gold_dsl = str(gold_item.get("dsl", ""))
                model_dsl = str(model_item.get("dsl", model_item.get("dsl_string", "")))

                gold_hops = parse_dsl_hops(gold_dsl)
                model_hops = parse_dsl_hops(model_dsl)

                user_prompt = build_judge_user_prompt(
                    args.dataset,
                    question,
                    gold_dsl,
                    model_dsl,
                    gold_hops,
                    model_hops,
                )

                batch_lines.append(
                    VertexBatchHelper.build_request(
                        user_prompt,
                        system_prompt,
                        temperature=args.judge_temperature,
                        max_tokens=args.judge_max_tokens,
                    )
                )
                batch_index.append(example_id)

            VertexBatchHelper.write_jsonl(batch_input_path, batch_lines)
            index_path.parent.mkdir(parents=True, exist_ok=True)
            with index_path.open("w", encoding="utf-8") as fp:
                json.dump(batch_index, fp, indent=2)

            if args.batch_upload:
                VertexBatchHelper.upload_to_gcs(batch_input_path, args.batch_input_uri)

            display_name = f"decomp-equiv-{args.dataset}-{args.model_slug}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
            job_name = VertexBatchHelper.submit_batch_job(
                args.judge_model,
                args.batch_input_uri,
                args.batch_output_uri,
                display_name,
            )
            print(f"Submitted batch job: {job_name}")
            print(f"Batch input: {batch_input_path}")
            print(f"Batch index: {index_path}")
            return

    system_prompt = build_judge_system_prompt(args.dataset)
    judge = GeminiVertexModel(model_name=args.judge_model)

    try:
        from tqdm import tqdm  # type: ignore
        iterator = tqdm(common_ids, desc=f"{args.dataset}:{args.model_slug}", unit="ex")
    except Exception:
        iterator = common_ids

    for idx, example_id in enumerate(iterator, 1):
        if args.resume and example_id in completed:
            continue

        gold_item = gold_lookup[example_id]
        model_item = model_lookup[example_id]

        question = gold_item.get("question", "")
        gold_dsl = str(gold_item.get("dsl", ""))
        model_dsl = str(model_item.get("dsl", model_item.get("dsl_string", "")))

        gold_hops = parse_dsl_hops(gold_dsl)
        model_hops = parse_dsl_hops(model_dsl)

        user_prompt = build_judge_user_prompt(
            args.dataset,
            question,
            gold_dsl,
            model_dsl,
            gold_hops,
            model_hops,
        )

        if args.debug:
            print("\n=== JUDGE PROMPT ===")
            print(user_prompt)

        try:
            raw = judge.generate_answer(
                user_prompt,
                system=system_prompt,
                temperature=args.judge_temperature,
                max_tokens=args.judge_max_tokens,
                usage_meta={"example_id": example_id, "regime": "decomposition_equivalence"},
            )
        except Exception as exc:
            raw = ""
            parsed: Dict[str, Any] = {"error": str(exc)}
        else:
            parsed = parse_json_response(raw)

        gold_count = len(gold_hops)
        model_count = len(model_hops)

        matches_raw = parsed.get("matches")
        if isinstance(matches_raw, (int, float, str)):
            try:
                matches = int(float(matches_raw))
            except Exception:
                matches = 0
        else:
            matches = 0

        # Backward-compat: infer matches from old judge format if present.
        if matches <= 0 and "gold_hops_covered" in parsed:
            gold_cov = normalize_index_list(parsed.get("gold_hops_covered"), gold_count)
            matches = len(gold_cov)

        matches = max(0, min(matches, gold_count, model_count))
        metrics = compute_metrics(gold_count, model_count, matches)
        hop_ratio = (model_count / gold_count) if gold_count else None
        hop_diff = model_count - gold_count

        record = {
            "id": example_id,
            "question": question,
            "gold_dsl": gold_dsl,
            "model_dsl": model_dsl,
            "gold_hops": gold_hops,
            "model_hops": model_hops,
            "gold_hop_count": gold_count,
            "model_hop_count": model_count,
            "hop_ratio": hop_ratio,
            "hop_diff": hop_diff,
            "judge_raw": raw,
            "judge_parsed": parsed,
            "matches": matches,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "equivalent_final": parsed.get("equivalent_final"),
        }

        results.append(record)

        if idx % 10 == 0:
            save_json(output_path, results)

        if args.sleep:
            time.sleep(args.sleep)

    save_json(output_path, results)

    # Summary metrics
    total = len(results)
    if total == 0:
        summary = {"total": 0}
        save_json(summary_path, summary)
        print(f"No results to summarize. Wrote {summary_path}")
        return

    def _mean(values: List[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    hop_ratios = [r["hop_ratio"] for r in results if r.get("hop_ratio") is not None]
    hop_diffs = [r.get("hop_diff", 0.0) for r in results]

    precisions = [r.get("precision", 0.0) for r in results]
    recalls = [r.get("recall", 0.0) for r in results]
    f1s = [r.get("f1", 0.0) for r in results]

    eq_vals = [r.get("equivalent_final") for r in results]
    eq_counts = {
        "equivalent": sum(1 for v in eq_vals if v == 1),
        "not_equivalent": sum(1 for v in eq_vals if v == 0),
        "uncertain": sum(1 for v in eq_vals if v == -1),
    }

    summary = {
        "total": total,
        "average_gold_hops": _mean([r.get("gold_hop_count", 0) for r in results]),
        "average_model_hops": _mean([r.get("model_hop_count", 0) for r in results]),
        "average_hop_ratio": _mean([float(r) for r in hop_ratios]) if hop_ratios else 0.0,
        "average_hop_diff": _mean([float(v) for v in hop_diffs]),
        "precision_macro": _mean(precisions),
        "recall_macro": _mean(recalls),
        "f1_macro": _mean(f1s),
        "equivalence_counts": eq_counts,
        "equivalence_rate": eq_counts["equivalent"] / total,
    }

    save_json(summary_path, summary)
    print(f"Saved {len(results)} results to {output_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
