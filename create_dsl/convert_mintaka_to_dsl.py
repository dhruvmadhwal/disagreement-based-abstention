#!/usr/bin/env python3
"""
Convert Mintaka questions to DSL format using qa_model calls.

This script takes questions from mintaka_multihop_300.json and converts them 
to a linear DSL that uses only qa_model(...) calls following the specified format.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from utils.model_interface import GeminiModel  # noqa: E402

DEFAULT_MODEL_NAME = os.environ.get("DSL_MODEL_NAME", "google/gemini-2.5-pro")
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

def load_json_file(file_path: str) -> List[Dict[str, Any]]:
    """Load data from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data: Any, file_path: str):
    """Save data to a JSON file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def convert_question_to_dsl(question: str, debug: bool = False) -> str:
    """Convert a single question to DSL format using the system prompt."""
    
    user_message = f"Q: {question}"
    
    if debug:
        print(f"Converting question: {question}")
    
    try:
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
        
    except Exception as e:
        print(f"Error converting question to DSL: {e}")
        return ""

def process_mintaka_questions(
    input_file: str,
    output_file: str,
    limit: int = None,
    debug: bool = False,
    resume: bool = True,
):
    """Process all questions from mintaka_multihop_300.json and convert to DSL."""
    
    print(f"Loading questions from {input_file}...")
    mintaka_data = load_json_file(input_file)
    
    if limit:
        mintaka_data = mintaka_data[:limit]
        print(f"Processing first {limit} questions for testing")
    
    # Check if output file exists to resume processing
    existing_results = []
    processed_ids = set()
    
    if resume and os.path.exists(output_file):
        try:
            existing_results = load_json_file(output_file)
            processed_ids = {str(item.get("id")) for item in existing_results}
            print(f"Found existing output file with {len(existing_results)} items. Will resume processing.")
        except Exception as e:
            print(f"Error loading existing results: {e}")
            existing_results = []
    elif os.path.exists(output_file) and not resume:
        print(f"Ignoring existing output file {output_file} because resume=False.")
    
    results = existing_results.copy()
    
    # Count items that need processing
    remaining_items = [item for item in mintaka_data 
                     if str(item.get("id", item.get("qid", f"item_{mintaka_data.index(item)}"))) not in processed_ids]
    
    print(f"Converting {len(remaining_items)} questions to DSL format...")
    
    for i, item in enumerate(remaining_items):
        # Get required fields with appropriate fallbacks for Mintaka format
        question = item.get("question", "")
        item_id = item.get("id", item.get("qid", f"item_{i}"))
        answer = item.get("answer", item.get("answers", ""))
        
        # Handle different answer formats
        if isinstance(answer, list) and answer:
            answer = answer[0] if len(answer) == 1 else str(answer)
        
        if not question:
            print(f"Warning: Empty question for item {item_id}")
            continue
            
        print(f"Processing {i+1}/{len(remaining_items)}: {item_id}")
        
        try:
            dsl_output = convert_question_to_dsl(question, debug=debug)
            
            # Only include the fields we need
            result = {
                "id": item_id,
                "question": question,
                "answer": answer,
                "dsl": dsl_output,
                "success": bool(dsl_output.strip())
            }
                    
            results.append(result)
            
        except Exception as e:
            print(f"Error processing item {item_id}: {e}")
            error_result = {
                "id": item_id,
                "question": question,
                "answer": answer,
                "dsl": "",
                "success": False,
                "error": str(e)
            }
            results.append(error_result)
        
        # Add small delay to avoid rate limiting
        if i < len(remaining_items) - 1:
            time.sleep(0.5)
        
        # Save the main file every 10 items
        if (i + 1) % 10 == 0 or i == len(remaining_items) - 1:
            save_json(results, output_file)
            print(f"  Progress: {i+1}/{len(remaining_items)} items processed - Saved to {output_file}")
    
    # Print summary
    successful = len([r for r in results if r.get("success", False)])
    failed = len(results) - successful
    
    print(f"\n✅ Conversion complete!")
    print(f"📁 Output saved to: {output_file}")
    print(f"📊 Results:")
    print(f"  - Successful conversions: {successful}")
    print(f"  - Failed conversions: {failed}")
    print(f"  - Total questions: {len(results)}")
    
    if failed > 0:
        print(f"\n❌ Failed items:")
        for result in results:
            if not result.get("success", False):
                print(f"  - {result['id']}: {result.get('error', 'Unknown error')}")

def main():
    import argparse

    global model
    if model is None:
        model = GeminiModel(DEFAULT_MODEL_NAME)

    parser = argparse.ArgumentParser(description='Convert Mintaka questions to DSL format')
    parser.add_argument('--input', default='mintaka_multihop_300.json', 
                       help='Input file with questions')
    parser.add_argument('--output', default='mintaka_with_dsl.json',
                       help='Output file for DSL results')
    parser.add_argument('--limit', type=int, help='Limit number of questions to process')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument(
        '--resume',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Resume from existing output (default: True). Use --no-resume to regenerate all.',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Alias for --no-resume (kept for backward compatibility).',
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found!")
        print("Make sure mintaka_multihop_300.json exists in the current directory.")
        return
    
    try:
        resume = args.resume and not args.overwrite
        process_mintaka_questions(args.input, args.output, args.limit, args.debug, resume=resume)
    except KeyboardInterrupt:
        print("\n⏹️  Process interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
