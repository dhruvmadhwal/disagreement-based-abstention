#!/usr/bin/env python3
"""
Convert CRAG closed-book questions to DSL format using qa_model calls.

Mirrors the bamboogle conversion flow but pulls questions from the processed
closed-book top-300 set and writes DSL to data/processed/crag/crag_dsl.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from utils.model_interface import GeminiModel  # noqa: E402

DEFAULT_INPUT = Path("data/processed/crag/crag_closed_book_top300.jsonl")
DEFAULT_OUTPUT = Path("data/processed/crag/crag_dsl.json")
DEFAULT_MODEL_NAME = os.environ.get("DSL_MODEL_NAME", "google/gemini-2.5-flash")
model: GeminiModel | None = None

DSL_SYSTEM_PROMPT = """You convert a multi-hop natural-language question into a sequence of intermediate questions written in a Python-flavored DSL using:

def qa_model(question: str, **kwargs) -> str: ...

Rules:
1. Output only DSL lines, no explanations or comments.
2. Use numbered variables: answer_1, answer_2, … (sequential, no gaps).
3. All left-hand sides must be typed exactly as str.
4. Keep questions minimal and well-formed.
5. Do not invent data.
6. CRITICAL: The FINAL qa_model call must directly answer the original question when its placeholders are resolved.
7. For comparisons (e.g., "Is X bigger than Y?"), gather both values first, then compare.
8. These questions are designed for multi-hop reasoning, so you should put together a good decomposition that serves as a plan.

Examples:

# Q1. Where was the guitarist from R.E.M. born?
# Final step asks "Where was {answer_1} born?" which answers the original question.
answer_1: str = qa_model("Who is the guitarist for R.E.M.?")
answer_2: str = qa_model("Where was {answer_1} born?", answer_1=answer_1)

# Q2. What is the population of the country where the Spanish Steps are located?
# Final step asks for population of the country, answering the original question.
answer_1: str = qa_model("In which country are the Spanish Steps located?")
answer_2: str = qa_model("What is the population of {answer_1}?", answer_1=answer_1)

# Q3. Who directed the highest-grossing film of 2019?
# Final step asks who directed the film, answering the original question.
answer_1: str = qa_model("What was the highest-grossing film of 2019?")
answer_2: str = qa_model("Who directed {answer_1}?", answer_1=answer_1)

# Q4. Who distributed the movie based on the Harry Potter book released in 2004?
# Three hops needed: book -> movie -> distributor. Final step answers the original.
answer_1: str = qa_model("Which Harry Potter book was released in 2004?")
answer_2: str = qa_model("What is the film adaptation of {answer_1}?", answer_1=answer_1)
answer_3: str = qa_model("Who distributed {answer_2}?", answer_2=answer_2)
"""

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_json(data: Any, file_path: Path | str) -> None:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2, ensure_ascii=False)


def load_existing_results(path: Path) -> Tuple[List[Dict[str, Any]], set[str]]:
    if not path.exists():
        return [], set()
    try:
        with path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
        if not isinstance(data, list):
            raise ValueError("Existing DSL file must contain a list")
    except Exception as exc:
        print(f"Error loading existing DSL file ({path}): {exc}")
        return [], set()
    processed_ids = {
        str(item.get("id"))
        for item in data
        if isinstance(item, dict) and item.get("id")
    }
    return data, processed_ids


def convert_question_to_dsl(question: str, *, debug: bool = False) -> str:
    """Convert a single question to DSL format using the shared system prompt."""
    if model is None:
        raise RuntimeError("Model is not initialized. Call from main().")
    user_message = f"Q: {question}"
    if debug:
        print(f"Converting question: {question}")

    messages = [
        {"role": "system", "content": DSL_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    dsl_output = model.generate_answer(
        "",
        messages=messages,
        temperature=0.1,
        top_p=0.95,
        top_k=64,
        max_tokens=2048,
    ).strip()
    if debug:
        print(f"DSL Output:\n{dsl_output}")
    return dsl_output


def process_crag_questions(
    input_file: Path,
    output_file: Path,
    *,
    limit: int | None = None,
    sleep: float = 0.0,
    debug: bool = False,
    resume: bool = True,
    model_name: str = DEFAULT_MODEL_NAME,
) -> int:
    """Process CRAG closed-book questions and convert to DSL."""
    global model
    model = GeminiModel(model_name)
    print(f"Loading questions from {input_file}")
    dataset = load_jsonl(input_file)
    if limit:
        dataset = dataset[:limit]
        print(f"Processing first {limit} questions for testing")

    if resume:
        existing_results, processed_ids = load_existing_results(output_file)
        if existing_results:
            print(f"Resuming from {output_file} with {len(existing_results)} items.")
    else:
        existing_results, processed_ids = [], set()
        if output_file.exists():
            print(f"Ignoring existing file {output_file} because resume=False.")

    results = list(existing_results)
    to_process = [
        item for item in dataset if str(item.get("interaction_id")) not in processed_ids
    ]
    print(f"Converting {len(to_process)} questions to DSL format...")

    for idx, item in enumerate(to_process, start=1):
        qid = str(item.get("interaction_id") or f"crag_{idx}")
        question = item.get("query") or ""
        answer = item.get("answer") or ""

        if not question.strip():
            print(f"Skipping empty question for id={qid}")
            continue

        print(f"[{idx}/{len(to_process)}] {qid}")
        try:
            dsl_output = convert_question_to_dsl(question, debug=debug)
        except Exception as exc:
            print(f"Error converting {qid}: {exc}")
            dsl_output = ""

        results.append(
            {
                "id": qid,
                "question": question,
                "answer": answer,
                "dsl": dsl_output,
                "success": bool(dsl_output.strip()),
            }
        )

        if sleep:
            time.sleep(sleep)

        if idx % 10 == 0:
            save_json(results, output_file)

    save_json(results, output_file)
    print(f"Wrote {len(results)} items to {output_file}")
    return len(results)


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert CRAG closed-book questions to DSL.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input JSONL file.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output DSL JSON file.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for debugging.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between requests.")
    parser.add_argument("--debug", action="store_true", help="Print model responses for debugging.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME, help="Model name (default: google/gemini-2.5-flash).")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from existing output (default: True). Use --no-resume to regenerate all.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Alias for --no-resume (kept for backward compatibility).",
    )
    return parser.parse_args(argv)


def main(argv: List[str]) -> None:
    args = parse_args(argv)
    resume = args.resume and not args.overwrite
    process_crag_questions(
        args.input,
        args.output,
        limit=args.limit,
        sleep=args.sleep,
        debug=args.debug,
        resume=resume,
        model_name=args.model,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
