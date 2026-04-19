#!/usr/bin/env python3
"""
Convert HotpotQA questions to DSL format using qa_model calls.

Similar to convert_crag_to_dsl.py but for HotpotQA dataset.
Includes optional sanity checks to filter out bad DSLs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from utils.model_interface import GeminiModel

DEFAULT_INPUT = Path("data/raw/hotpotqa/hotpotqa_dev_300.json")
DEFAULT_OUTPUT = Path("data/processed/hotpotqa/hotpotqa_dsl.json")
DEFAULT_MODEL_NAME = os.environ.get("DSL_MODEL_NAME", "google/gemini-2.5-flash")
model: GeminiModel | None = None

DSL_SYSTEM_PROMPT = """You convert a single natural-language question into a sequence of intermediate questions written in a Python-flavored DSL using:

def qa_model(question: str, **kwargs) -> str: ...

Rules

Output only DSL lines, no explanations.

Use numbered variables: answer_1, answer_2, ... (sequential, no gaps).

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


def load_json(path: Path) -> List[Dict[str, Any]]:
    """Load a JSON file (list or convert dict values to list)."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return list(data.values())


def save_json(data: Any, file_path: Path | str) -> None:
    """Save data to a JSON file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2, ensure_ascii=False)


def load_existing_results(path: Path) -> Tuple[List[Dict[str, Any]], set[str]]:
    """Load existing results for resume functionality."""
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


def validate_dsl(dsl: str) -> Tuple[bool, List[str]]:
    """
    Validate that a DSL is well-formed and multi-hop.

    Returns (is_valid, list_of_issues)
    """
    import re

    issues = []

    if not dsl or not dsl.strip():
        issues.append("Empty DSL")
        return False, issues

    # Count qa_model calls
    num_calls = len(re.findall(r"qa_model\s*\(", dsl))

    if num_calls == 0:
        issues.append("No qa_model calls")
        return False, issues

    if num_calls == 1:
        issues.append("One-hop (only 1 qa_model call)")
        return False, issues

    # Check for dependencies
    lines = dsl.split("\n")
    defined_vars = set()
    has_dep = False

    for line in lines:
        def_match = re.match(r"(answer_\d+)\s*:\s*str\s*=", line)
        if def_match:
            var_name = def_match.group(1)
            for prev_var in defined_vars:
                if f"{{{prev_var}}}" in line or f"{prev_var}={prev_var}" in line:
                    has_dep = True
                    break
            defined_vars.add(var_name)

    if not has_dep:
        issues.append("No dependency between steps (no-compose)")
        return False, issues

    # Check for truncation (basic)
    last_line = lines[-1].strip() if lines else ""
    if last_line.endswith(",") or last_line.endswith("(") or last_line.endswith("="):
        issues.append("Truncated DSL")
        return False, issues

    return True, issues


def process_hotpotqa_questions(
    input_file: Path,
    output_file: Path,
    *,
    limit: int | None = None,
    sleep: float = 0.0,
    debug: bool = False,
    resume: bool = True,
    model_name: str = DEFAULT_MODEL_NAME,
    validate: bool = False,
) -> int:
    """Process HotpotQA questions and convert to DSL."""
    global model
    model = GeminiModel(model_name)
    print(f"Loading questions from {input_file}")
    dataset = load_json(input_file)
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
        item for item in dataset
        if str(item.get("id", item.get("_id", ""))) not in processed_ids
    ]
    print(f"Converting {len(to_process)} questions to DSL format...")

    for idx, item in enumerate(to_process, start=1):
        qid = str(item.get("id", item.get("_id", f"hotpotqa_{idx}")))
        question = item.get("question", "")
        answer = item.get("answer", "")

        if not question.strip():
            print(f"Skipping empty question for id={qid}")
            continue

        print(f"[{idx}/{len(to_process)}] {qid}")
        try:
            dsl_output = convert_question_to_dsl(question, debug=debug)
        except Exception as exc:
            print(f"Error converting {qid}: {exc}")
            dsl_output = ""

        result = {
            "id": qid,
            "question": question,
            "answer": answer,
            "dsl": dsl_output,
            "success": bool(dsl_output.strip()),
        }

        # Optional validation
        if validate and dsl_output:
            is_valid, issues = validate_dsl(dsl_output)
            result["dsl_valid"] = is_valid
            result["dsl_issues"] = issues

        results.append(result)

        if sleep:
            time.sleep(sleep)

        if idx % 10 == 0:
            save_json(results, output_file)

    save_json(results, output_file)
    print(f"Wrote {len(results)} items to {output_file}")
    return len(results)


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert HotpotQA questions to DSL.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input JSON file.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output DSL JSON file.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for debugging.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between requests.")
    parser.add_argument("--debug", action="store_true", help="Print model responses for debugging.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME, help="Model name.")
    parser.add_argument("--overwrite", action="store_true", help="Ignore existing output and regenerate all.")
    parser.add_argument("--validate", action="store_true", help="Validate DSLs and mark issues.")
    return parser.parse_args(argv)


def main(argv: List[str]) -> None:
    args = parse_args(argv)
    process_hotpotqa_questions(
        args.input,
        args.output,
        limit=args.limit,
        sleep=args.sleep,
        debug=args.debug,
        resume=not args.overwrite,
        model_name=args.model,
        validate=args.validate,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
