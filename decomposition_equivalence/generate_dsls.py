#!/usr/bin/env python3
"""Generate DSL decompositions for a dataset using a specified model."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from utils.model_interface import GeminiVertexModel, GPT51Model, VLLMModel

from decomposition_equivalence.config import (
    DATASET_FILES,
    DATASET_SLUGS,
    DSL_OUTPUT_ROOT,
    DEFAULT_GEN_MAX_TOKENS,
    DEFAULT_GEN_TEMPERATURE,
    DEFAULT_GEN_TOP_K,
    DEFAULT_GEN_TOP_P,
    DEFAULT_VLLM_API_KEY,
    DEFAULT_VLLM_BASE_URL,
    MODEL_PRESETS,
    normalize_model_choice,
    slugify,
)
from decomposition_equivalence.prompts import (
    build_generation_system_prompt,
    build_generation_user_prompt,
)


GENERATION_FILE_CANDIDATES = [
    "{dataset}_sequential.json",
    "{dataset}_open_ended.json",
    "{dataset}_assistive.json",
    "{dataset}_model_generated.json",
    "{dataset}_regimes_full.jsonl",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate DSL decompositions for a dataset")
    parser.add_argument("--dataset", required=True, choices=DATASET_SLUGS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--model-slug", type=str, default=None)
    parser.add_argument("--use-vertex", action="store_true", help="Use Vertex (Gemini) backend")
    parser.add_argument("--use-gpt5", action="store_true", help="Use OpenAI GPT-5.x backend")
    parser.add_argument("--vllm-base-url", type=str, default=DEFAULT_VLLM_BASE_URL)
    parser.add_argument("--vllm-api-key", type=str, default=DEFAULT_VLLM_API_KEY)
    parser.add_argument("--from-generation", action="store_true", help="Infer model name from results/generation")
    parser.add_argument("--generation-dir", type=Path, default=None, help="Override generation dir when inferring model")
    parser.add_argument("--temperature", type=float, default=DEFAULT_GEN_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_GEN_TOP_P)
    parser.add_argument("--top-k", type=int, default=DEFAULT_GEN_TOP_K)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_GEN_MAX_TOKENS)
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


def load_dataset(dataset: str, limit: Optional[int]) -> List[Dict[str, Any]]:
    dataset_path = DATASET_FILES[dataset]
    data = load_json(dataset_path)
    if limit is not None:
        data = data[:limit]
    return data


def resolve_generation_dir(dataset: str, model_slug: str, override: Optional[Path]) -> Path:
    if override is not None:
        return override
    return REPO_ROOT / "results" / "generation" / dataset / model_slug


def infer_model_name(dataset: str, model_slug: str, generation_dir: Path) -> Optional[str]:
    if not generation_dir.exists():
        return None
    candidates = [
        generation_dir / filename.format(dataset=dataset)
        for filename in GENERATION_FILE_CANDIDATES
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            if candidate.suffix == ".jsonl":
                with candidate.open("r", encoding="utf-8") as fp:
                    line = fp.readline().strip()
                    if not line:
                        continue
                    record = json.loads(line)
            else:
                payload = load_json(candidate)
                record = payload[0] if isinstance(payload, list) and payload else payload
        except Exception:
            continue
        if isinstance(record, dict):
            model = record.get("model") or record.get("model_name") or record.get("model_choice")
            if model:
                return str(model)
    return None


def resolve_model_name_from_slug(model_slug: str) -> Optional[str]:
    if not model_slug:
        return None
    normalized = normalize_model_choice(model_slug)
    if normalized in MODEL_PRESETS:
        return MODEL_PRESETS[normalized]
    if model_slug in MODEL_PRESETS:
        return MODEL_PRESETS[model_slug]
    for preset_name in MODEL_PRESETS.values():
        if slugify(preset_name) == model_slug:
            return preset_name
    return None


def build_model(args: argparse.Namespace):
    if args.use_vertex:
        if not args.model_name:
            raise ValueError("--model-name is required for --use-vertex")
        return GeminiVertexModel(model_name=args.model_name)
    if args.use_gpt5:
        return GPT51Model(model_name=args.model_name or "gpt-5.1")
    if not args.model_name:
        raise ValueError("--model-name is required for vLLM backend")
    return VLLMModel(
        model_name=args.model_name,
        base_url=args.vllm_base_url,
        api_key=args.vllm_api_key,
    )


def load_existing(output_path: Path) -> Tuple[List[Dict[str, Any]], set[str]]:
    if not output_path.exists():
        return [], set()
    try:
        data = load_json(output_path)
    except Exception:
        return [], set()
    if not isinstance(data, list):
        return [], set()
    processed = {str(item.get("id")) for item in data if isinstance(item, dict) and item.get("id") is not None}
    return data, processed


def main() -> None:
    args = parse_args()

    model_slug = args.model_slug
    if not model_slug and args.model_name:
        model_slug = slugify(args.model_name)
    if not model_slug:
        model_slug = "model"

    if args.from_generation:
        generation_dir = resolve_generation_dir(args.dataset, model_slug, args.generation_dir)
        inferred = infer_model_name(args.dataset, model_slug, generation_dir)
        if inferred:
            args.model_name = inferred
        elif not args.model_name:
            args.model_name = resolve_model_name_from_slug(model_slug) or model_slug
    elif not args.model_name:
        args.model_name = resolve_model_name_from_slug(model_slug) or model_slug

    if args.model_name and not args.use_vertex and not args.use_gpt5:
        lowered = args.model_name.lower()
        if "gemini" in lowered:
            args.use_vertex = True
        elif lowered.startswith("gpt-5") or lowered.startswith("gpt5"):
            args.use_gpt5 = True

    if args.output is None:
        output_path = DSL_OUTPUT_ROOT / args.dataset / f"{model_slug}.json"
    else:
        output_path = args.output

    dataset = load_dataset(args.dataset, args.limit)
    system_prompt = build_generation_system_prompt(args.dataset)

    model = build_model(args)

    existing: List[Dict[str, Any]] = []
    processed_ids: set[str] = set()
    if args.resume:
        existing, processed_ids = load_existing(output_path)
        if existing:
            print(f"Resuming from {output_path} with {len(processed_ids)} records")

    results = list(existing)

    try:
        from tqdm import tqdm  # type: ignore
        iterator = tqdm(dataset, desc=f"{args.dataset}:{model_slug}", unit="ex")
    except Exception:
        iterator = dataset

    for idx, item in enumerate(iterator, 1):
        example_id = str(item.get("id"))
        if args.resume and example_id in processed_ids:
            continue

        question = item.get("question", "")
        if not question:
            continue

        user_prompt = build_generation_user_prompt(question)
        if args.debug:
            print("\n=== DSL GENERATION ===")
            print(f"ID: {example_id}")
            print(f"Question: {question}")

        try:
            dsl_output = model.generate_answer(
                question,
                system=system_prompt,
                user=user_prompt,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_tokens=args.max_tokens,
                usage_meta={"example_id": example_id, "regime": "dsl_generation"},
            ).strip()
            success = bool(dsl_output)
            record = {
                "id": example_id,
                "question": question,
                "dsl": dsl_output,
                "model": args.model_name,
                "model_slug": model_slug,
                "success": success,
            }
            results.append(record)
        except Exception as exc:
            record = {
                "id": example_id,
                "question": question,
                "dsl": "",
                "model": args.model_name,
                "model_slug": model_slug,
                "success": False,
                "error": str(exc),
            }
            results.append(record)

        if idx % 10 == 0:
            save_json(output_path, results)

        if args.sleep:
            time.sleep(args.sleep)

    save_json(output_path, results)
    print(f"Saved {len(results)} records to {output_path}")


if __name__ == "__main__":
    main()
