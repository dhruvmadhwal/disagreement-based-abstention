#!/usr/bin/env python3
"""
Convert Bamboogle questions to DSL format using qa_model calls.

This script mirrors the Mintaka conversion flow but reads questions from the
local raw dump at `data/raw/bamboogle/bamboogle_dev.json` and writes the results
to `data/processed/bamboogle/bamboogle_dsl.json` by default.
"""

from __future__ import annotations

import argparse
import hashlib
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

# ─────────── Configuration ───────────
DEFAULT_MODEL_NAME = os.environ.get("DSL_MODEL_NAME", "google/gemini-2.5-pro")

DEFAULT_INPUT = Path("data/raw/bamboogle/bamboogle_dev.json")
DEFAULT_OUTPUT = Path("data/processed/bamboogle/bamboogle_dsl.json")

# Lazy: instantiated by main() so importing the module does not require Vertex creds.
model: GeminiModel | None = None

DSL_SYSTEM_PROMPT = """You convert a single natural-language question into a sequence of intermediate questions written in a Python-flavored DSL using:

def qa_model(question: str, **kwargs) -> str: ...

Rules

Output only DSL lines, no explanations.

Use numbered variables: answer_1, answer_2, … (sequential, no gaps).

All left-hand sides must be typed exactly as str. No lists, dicts, ints, or floats.

Keep questions minimal and well-formed 

Do not invent data

Examples:

# Q1. Where was the guitarist from R.E.M. born?
answer_1: str = qa_model("Who is the guitarist for the band R.E.M.?")
answer_2: str = qa_model("Where was {answer_1} born?", answer_1=answer_1)

# Q2. What is the population of the country where the 'Spanish Steps' are located?
answer_1: str = qa_model("In which country are the 'Spanish Steps' located?")
answer_2: str = qa_model("What is the population of {answer_1}?", answer_1=answer_1)

# Q3. What is the fourth main dungeon in the first Zelda game for the N64?
answer_1: str = qa_model("What is the first Zelda game released for the Nintendo 64?")
answer_2: str = qa_model("What is the fourth main dungeon in '{answer_1}'?", answer_1=answer_1)

# Q4. Who is the protagonist in the eighth Final Fantasy game?
answer_1: str = qa_model("What is the title of the eighth Final Fantasy game?")
answer_2: str = qa_model("Who is the main protagonist of '{answer_1}'?", answer_1=answer_1)

# Q5. How tall is the actor who played Deadpool?
answer_1: str = qa_model("Who played Deadpool in the films?")
answer_2: str = qa_model("How tall is {answer_1}?", answer_1=answer_1)

# Q6. Who starred in the movie based on *Fear and Loathing in Las Vegas*?
answer_1: str = qa_model("What is the film adaptation of the book 'Fear and Loathing in Las Vegas'?")
answer_2: str = qa_model("Who starred in '{answer_1}'?", answer_1=answer_1)

# Q7. When did the author who wrote *Dubliners* die?
answer_1: str = qa_model("Who wrote the book 'Dubliners'?")
answer_2: str = qa_model("When did {answer_1} die?", answer_1=answer_1)

# Q8. How many kids does the lead actor of *Troy* have?
answer_1: str = qa_model("Who was the lead actor in the film 'Troy' (2004)?")
answer_2: str = qa_model("How many children does {answer_1} have?", answer_1=answer_1)

# Q9. Who was the first wife of Queen Elizabeth II's eldest son?
answer_1: str = qa_model("Who is the eldest son of Queen Elizabeth II?")
answer_2: str = qa_model("Who was the first wife of {answer_1}?", answer_1=answer_1)

# Q10. Who distributed the movie that's based on the Harry Potter book released in 2004?
answer_1: str = qa_model("Which Harry Potter book was released in 2004?")
answer_2: str = qa_model("What is the title of the film based on '{answer_1}'?", answer_1=answer_1)
answer_3: str = qa_model("Who distributed the film '{answer_2}'?", answer_2=answer_2)"""


def save_json(data: Any, file_path: Path | str) -> None:
    """Persist JSON data, creating parent directories if needed."""
    path = Path(file_path)
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2, ensure_ascii=False)


def load_json_list(file_path: Path | str) -> List[Dict[str, Any]]:
    """Load a JSON list from disk."""
    path = Path(file_path)
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {file_path}, found {type(data)}")
    return data


def load_existing_results(file_path: Path | str) -> Tuple[List[Dict[str, Any]], set[str]]:
    """Load prior conversion results for resuming the run."""
    path = Path(file_path)
    if not path.exists():
        return [], set()

    try:
        with path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
        if not isinstance(data, list):
            raise ValueError("Existing DSL file must contain a list")
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"Error loading existing DSL file ({file_path}): {exc}")
        return [], set()

    processed_ids = {
        str(item.get("id"))
        for item in data
        if isinstance(item, dict) and item.get("id")
    }
    return data, processed_ids


def convert_question_to_dsl(question: str, debug: bool = False) -> str:
    """Convert a single question to DSL format using the shared system prompt."""

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


def format_bamboogle_id(question: str, index: int) -> str:
    """Create a stable identifier for each Bamboogle question."""
    normalized = question.strip().lower() or f"bamboogle_{index}"
    digest = hashlib.md5(normalized.encode("utf-8")).hexdigest()[:8]
    return f"bamboogle_{index + 1:04d}_{digest}"


def prepare_bamboogle_records(
    input_file: Path | str, limit: int | None
) -> List[Dict[str, str]]:
    """Load and normalize Bamboogle questions from the raw JSON."""
    print(f"Loading Bamboogle questions from {input_file} ...")
    raw_records = load_json_list(input_file)

    if limit is not None:
        limit = min(limit, len(raw_records))
        raw_records = raw_records[:limit]
        print(f"Limiting to first {limit} questions for this run.")

    records: List[Dict[str, str]] = []
    for idx, row in enumerate(raw_records):
        question = (row.get("question") or "").strip()
        answer = row.get("answer", "")
        if isinstance(answer, str):
            answer = answer.strip()
        qid = str(row.get("id") or "").strip() or format_bamboogle_id(question, idx)
        records.append(
            {
                "id": qid,
                "question": question,
                "answer": answer,
            }
        )
    return records


def process_bamboogle_questions(
    input_file: Path | str,
    output_file: Path | str,
    limit: int | None = None,
    debug: bool = False,
    resume: bool = True,
) -> None:
    """Run DSL generation for the specified Bamboogle questions file."""

    records = prepare_bamboogle_records(input_file, limit)
    print(f"Loaded {len(records)} total questions.")

    if resume:
        existing_results, processed_ids = load_existing_results(output_file)
        results = existing_results.copy()
    else:
        existing_results, processed_ids = [], set()
        results = []
        if Path(output_file).exists():
            print(f"Ignoring existing output file {output_file} because resume=False.")

    remaining = [record for record in records if record["id"] not in processed_ids]
    print(f"Converting {len(remaining)} remaining questions to DSL format...")

    for idx, item in enumerate(remaining):
        question = item["question"]
        qid = item["id"]
        answer = item["answer"]

        if not question:
            print(f"Warning: Empty question for item {qid}")
            continue

        print(f"Processing {idx + 1}/{len(remaining)}: {qid}")

        try:
            dsl_output = convert_question_to_dsl(question, debug=debug)
            result = {
                "id": qid,
                "question": question,
                "answer": answer,
                "dsl": dsl_output,
                "success": bool(dsl_output.strip()),
            }
        except Exception as exc:  # pragma: no cover - network errors are runtime issues
            print(f"Error processing item {qid}: {exc}")
            result = {
                "id": qid,
                "question": question,
                "answer": answer,
                "dsl": "",
                "success": False,
                "error": str(exc),
            }

        results.append(result)

        if (idx + 1) % 10 == 0 or idx == len(remaining) - 1:
            save_json(results, output_file)
            print(
                f"  Progress: {idx + 1}/{len(remaining)} items processed - "
                f"Saved to {output_file}"
            )

        if idx < len(remaining) - 1:
            time.sleep(0.5)

    successful = sum(1 for r in results if r.get("success"))
    failed = len(results) - successful

    print("\n✅ Conversion complete!")
    print(f"📁 Output saved to: {output_file}")
    print("📊 Results:")
    print(f"  - Successful conversions: {successful}")
    print(f"  - Failed conversions: {failed}")
    print(f"  - Total questions: {len(results)}")

    if failed:
        print("\n❌ Failed items:")
        for record in results:
            if not record.get("success"):
                print(f"  - {record.get('id')}: {record.get('error', 'Unknown error')}")


def main() -> None:
    global model
    if model is None:
        model = GeminiModel(DEFAULT_MODEL_NAME)
    parser = argparse.ArgumentParser(
        description="Convert Bamboogle questions to DSL format."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Path to the raw Bamboogle JSON file (default: data/raw/bamboogle/bamboogle_dev.json).",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Path to write the DSL JSON output.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional limit on number of questions to process.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print each question and generated DSL.",
    )
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

    args = parser.parse_args()

    try:
        resume = args.resume and not args.overwrite
        process_bamboogle_questions(
            input_file=args.input,
            output_file=args.output,
            limit=args.limit,
            debug=args.debug,
            resume=resume,
        )
    except KeyboardInterrupt:
        print("\n⏹️  Process interrupted by user")
    except Exception as exc:
        print(f"❌ Error: {exc}")


if __name__ == "__main__":
    main()
