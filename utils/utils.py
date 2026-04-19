"""
Utility functions for QA consistency evaluation pipelines.
"""

import os
import json
import hashlib
import random
import numpy as np
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# Import model classes
from .model_interface import GPT51Model, LlamaModel, QwenModel, GemmaModel, OLMoModel

try:
    from .model_interface import VLLMModel
except ImportError:
    VLLMModel = None


def normalize_answer(answer: str) -> str:
    """Normalize answers for comparison (lower-case strip)."""
    if not isinstance(answer, str):
        answer = str(answer)
    return answer.lower().strip()


def cache_key(dataset_id: str, mode: str, model: str, temp: float) -> str:
    """Generate cache key for completions."""
    key_str = f"{dataset_id}_{mode}_{model}_{temp}"
    return hashlib.md5(key_str.encode()).hexdigest()


def load_cache(cache_file: str) -> Dict[str, Any]:
    """Load completion cache from file."""
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load cache file {cache_file}: {e}")
    return {}


def save_cache(cache: Dict[str, Any], cache_file: str):
    """Save completion cache to file."""
    try:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'w') as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save cache file {cache_file}: {e}")


def generate_output_filename(prefix: str, dataset: str, model_choice: str, setup: Optional[str] = None) -> str:
    """Generate output filename with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    parts = [prefix, timestamp, dataset, model_choice]
    if setup:
        parts.append(setup)
    # Use .jsonl for line-delimited JSON
    return f"../results/{'_'.join(parts)}.jsonl"


def get_example_indices(data_loader, max_examples: int, random_seed: Optional[int] = None) -> List[int]:
    """Get example indices, either sequential or random based on random_seed."""
    total_examples = len(data_loader)
    num_examples = min(max_examples, total_examples)
    
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
        indices = list(range(total_examples))
        random.shuffle(indices)
        return indices[:num_examples]
    else:
        return list(range(num_examples))


def convert_numpy_floats_to_python_floats(data: Any) -> Any:
    """
    Recursively convert numpy.float32 instances to Python float in a nested structure.
    """
    if isinstance(data, list):
        return [convert_numpy_floats_to_python_floats(item) for item in data]
    elif isinstance(data, dict):
        return {key: convert_numpy_floats_to_python_floats(value) for key, value in data.items()}
    elif isinstance(data, (np.float32, np.float64, np.floating)):
        return float(data)
    return data


def initialize_model(args: argparse.Namespace) -> Tuple[Any, str]:
    """Initialize the model based on command-line arguments."""
    model = None
    model_name_for_cache = args.model_choice

    use_vllm = getattr(args, 'use_vllm', False)
    tensor_parallel_size = getattr(args, 'tensor_parallel_size', 1)

    if use_vllm:
        model_map = {
            'llama': args.llama_model_name,
            'qwen': args.qwen_model_name,
            'gemma': args.gemma_model_name,
            'olmo': args.olmo_model_name,
            'mistral': args.mistral_model_name,
        }
        model_name_for_vllm = model_map.get(args.model_choice)
        if not model_name_for_vllm:
            raise ValueError(f"vLLM is not configured for model choice: {args.model_choice}")

        base_url = getattr(args, "vllm_base_url", os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"))
        api_key = getattr(args, "vllm_api_key", os.environ.get("VLLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or "EMPTY")

        print(f"Connecting to vLLM server at {base_url}")
        print(f"Using model: {model_name_for_vllm}")
        model = VLLMModel(
            model_name=model_name_for_vllm,
            tensor_parallel_size=tensor_parallel_size,
            base_url=base_url,
            api_key=api_key,
        )
        if tensor_parallel_size != 1:
            print("Note: tensor_parallel_size is ignored when connecting to an external vLLM server.")
        model_name_for_cache = model_name_for_vllm

    elif args.model_choice == 'gpt5':
        model = GPT51Model(api_key=os.getenv("OPENAI_API_KEY"))
        model_name_for_cache = model.model_name
    elif args.model_choice == 'gemini':
        from model_interface import GeminiModel
        model = GeminiModel(model_name=args.gemini_model_name)
        model_name_for_cache = args.gemini_model_name
    else:
        model_map = {
            'llama': (LlamaModel, args.llama_model_name),
            'qwen': (QwenModel, args.qwen_model_name),
            'gemma': (GemmaModel, args.gemma_model_name),
            'olmo': (OLMoModel, args.olmo_model_name),
        }
        if args.model_choice in model_map:
            ModelClass, model_name = model_map[args.model_choice]
            model = ModelClass(model_name=model_name, load_in_8bit=True)
            model_name_for_cache = model_name
    
    if model is None:
        # Check for unsupported models with specific error messages
        if args.model_choice == 'mistral':
            raise NotImplementedError("Mistral model is not yet implemented. Please use another model choice.")
        else:
            raise ValueError(f"Unsupported model choice: {args.model_choice}")
        
    if hasattr(model, 'model_name'):
        model_name_for_cache = model.model_name

    return model, model_name_for_cache


def setup_directories():
    """Create necessary output and cache directories."""
    os.makedirs("../results", exist_ok=True)
    os.makedirs("../cache", exist_ok=True)


def print_experiment_header(title: str, description: str, args: argparse.Namespace):
    """Print standardized experiment header."""
    print("=" * 80)
    print(title)
    print("=" * 80)
    print(description)
    print()
    print("Experimental Setup:")
    
    # This will be customized by each script
    yield  # Allow caller to print specific setup details
    
    print()
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Model: {args.model_choice.upper()}")
    
    # Print model-specific details
    if args.model_choice == 'qwen':
        print(f"Qwen Model: {args.qwen_model_name}")
    elif args.model_choice == 'llama':
        print(f"Llama Model: {args.llama_model_name}")
    elif args.model_choice == 'gemma':
        print(f"Gemma Model: {args.gemma_model_name}")
    elif args.model_choice == 'gemini':
        print(f"Gemini Model: {args.gemini_model_name}")
    elif args.model_choice == 'olmo':
        print(f"OLMo Model: {args.olmo_model_name}")
    elif args.model_choice == 'mistral':
        print(f"Mistral Model: {args.mistral_model_name}")
    
    if getattr(args, 'use_vllm', False):
        print("Inference engine: vLLM")
        if hasattr(args, 'tensor_parallel_size'):
            print(f"Tensor Parallel Size: {args.tensor_parallel_size}")
    else:
        print("Inference engine: Hugging Face Transformers")
        
    print(f"Temperature: {args.temperature}")
    print(f"Max examples: {args.max_examples}")
    
    if hasattr(args, 'max_selfask_steps'):
        print(f"Max Self-Ask steps: {args.max_selfask_steps}")
    
    if hasattr(args, 'output_file') and args.output_file:
        print(f"Output file: {args.output_file}")
    
    print("=" * 80)
    print()


def initialize_data_loader(args: argparse.Namespace):
    """Initialize and load the dataset."""
    from data_loader import UnifiedDataLoader
    
    data_loader = UnifiedDataLoader(args.dataset)
    data_loader.load_dataset(split=args.split)
    
    print(f"Total examples in {args.dataset} {args.split} split: {len(data_loader)}")
    print("=" * 80)
    
    if len(data_loader) == 0:
        raise ValueError(f"No data loaded from {args.dataset} {args.split} split")
    
    return data_loader


def extract_gold_subquestions(example: Any, data_loader: Any) -> List[str]:
    """Extracts gold subquestions from an example, handling different dataset structures."""
    if not hasattr(data_loader, 'dataset_name'):
        return []

    if data_loader.dataset_name == "fanoutqa":
        if hasattr(example, 'decomposition') and example.decomposition:
            # Flatten the entire decomposition tree into a list of questions
            all_questions = []
            queue = list(example.decomposition)
            while queue:
                subq = queue.pop(0)
                all_questions.append(subq.question)
                if subq.decomposition:
                    queue.extend(subq.decomposition)
            return all_questions
        return []
        
    elif data_loader.dataset_name == "musique":
        # Correctly extract structured subquestions using the data loader's method
        structured_subqs = data_loader.get_subquestions_structured(example) or []
        return [sq.get("text", "") for sq in structured_subqs if sq.get("text")]
        
    # Default for other datasets that might have subquestions in a simple list
    if hasattr(data_loader, 'get_subquestions_structured'):
        try:
            structured_subqs = data_loader.get_subquestions_structured(example) or []
            return [sq.get("text", "") for sq in structured_subqs if sq.get("text")]
        except (AttributeError, KeyError):
            return []
            
    return []


def print_sampling_info(args: argparse.Namespace):
    """Print information about sampling strategy."""
    if args.random_seed is not None:
        print(f"Using random sampling with seed {args.random_seed}")
    else:
        print("Using sequential sampling")


def save_results_jsonl(results: List[Dict], output_file: str):
    """Save results as JSONL format."""
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')


def print_final_summary(results: List[Dict], output_file: str, cache_file: str):
    """Print final summary footer."""
    print("=" * 80)
    print(f"Results saved to: {output_file}")
    if cache_file:
        print(f"Cache saved to: {cache_file}")
    print("=" * 80)


def add_common_args(parser: argparse.ArgumentParser):
    """Add common command-line arguments to the parser."""
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset to use (e.g., 'fanoutqa', 'musique')")
    parser.add_argument("--model_choice", type=str, required=True, choices=['gpt5', 'llama', 'qwen', 'gemma', 'gemini', 'olmo', 'mistral'],
                        help="Model to use for making the choice")
    parser.add_argument("--use_vllm", action='store_true', help="Use VLLM for inference if available")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of GPUs for tensor parallelism with VLLM")
    parser.add_argument(
        "--vllm_base_url",
        type=str,
        default=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
        help="Base URL for the shared vLLM OpenAI-compatible server",
    )
    parser.add_argument(
        "--vllm_api_key",
        type=str,
        default=os.environ.get("VLLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or "EMPTY",
        help="API key for the vLLM server (defaults to VLLM_API_KEY/OPENAI_API_KEY env vars)",
    )

    # Model name arguments
    parser.add_argument(
        "--qwen_model_name",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Name of the Qwen model to use (e.g., Qwen/Qwen3-8B, Qwen/Qwen3-32B)",
    )
    parser.add_argument("--llama_model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Name of the Llama model to use")
    parser.add_argument("--gemma_model_name", type=str, default="google/gemma-3-4b-it", help="Name of the Gemma model to use")
    parser.add_argument("--gemini_model_name", type=str, default="google/gemini-2.5-flash", help="Name of the Gemini model to use")
    parser.add_argument("--olmo_model_name", type=str, default="allenai/OLMo-2-1124-7B", help="Name of the OLMo model to use")
    parser.add_argument("--mistral_model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.3", help="Name of the Mistral model to use")

    parser.add_argument("--max_examples", type=int, default=400, help="Maximum number of examples to evaluate")
    parser.add_argument('--split', type=str, default='dev', help='Dataset split to use')
    parser.add_argument('--temperature', type=float, default=0.2, help='Temperature for model generation')
    parser.add_argument('--max_selfask_steps', type=int, default=12, help='Maximum steps for Self-Ask agent')
    parser.add_argument('--random_seed', type=int, default=None, help='Random seed for random sampling')
    parser.add_argument('--output_file', type=str, default=None, help='Path to save the results (auto-generated if not specified)')
    parser.add_argument('--cache_file', type=str, default=None, help='Path to cache file for completions')


def fix_qwen3_spacing(text: str) -> str:
    """
    Fix the specific spacing issue from Qwen3-8B/32B with vLLM where every character
    is separated by spaces: "* * A n s w e r : * *" -> "**Answer:**"
    Also handles: "< t h i n k >" -> "<think>", "K a m a l a H a r r i s" -> "Kamala Harris"

    Strategy: Use comma as reliable word boundary, capitalize as heuristic word boundary.
    """
    import re

    if not text:
        return ""

    # Check if this looks like the Qwen3 spacing issue
    # Pattern: multiple single characters separated by spaces
    if not re.search(r'[A-Za-z]\s+[A-Za-z]\s+[A-Za-z]', text):
        return text

    # Commas are reliable word boundaries - replace with marker
    text = text.replace(',', ' ⟪COMMA⟫ ')

    # Capital letters often indicate new words (heuristic)
    # Insert marker before capital letter if preceded by lowercase
    text = re.sub(r'([a-z])\s+([A-Z])', r'\1 ⟪WORD⟫ \2', text)

    words = text.split()
    result = []
    current_word = ""

    for word in words:
        if word == '⟪COMMA⟫':
            # Comma boundary - flush and add comma
            if current_word:
                result.append(current_word)
                current_word = ""
            result.append(',')
        elif word == '⟪WORD⟫':
            # Word boundary - flush current word
            if current_word:
                result.append(current_word)
                current_word = ""
        elif len(word) == 1 and (word.isalnum() or word in '.,;:!?*#-<>/[](){}\'\"'):
            # Single character - accumulate
            current_word += word
        else:
            # Multi-character word - flush current and add this
            if current_word:
                result.append(current_word)
                current_word = ""
            result.append(word)

    # Don't forget the last accumulated word
    if current_word:
        result.append(current_word)

    return ' '.join(result)


def clean_model_output(text: str) -> str:
    """
    Clean model output by removing thinking tags and other artifacts.
    Handles both normal and spaced thinking tags (Qwen3 vLLM issue).
    """
    import re

    if not text:
        return ""

    # FIRST: Fix Qwen3 spacing issue if present (before removing tags)
    # This handles cases where thinking tags have spaces: "< t h i n k >"
    text = fix_qwen3_spacing(text)

    # THEN: Remove all thinking tag variations and content
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<think/>', '', text, flags=re.DOTALL)
    text = re.sub(r'<think>.*', '', text, flags=re.DOTALL)
    text = re.sub(r'</think>', '', text, flags=re.DOTALL)
    
    # Remove any content that looks like reasoning/thinking
    text = re.sub(r'Let me think.*?(?=Answer:|$)', '', text, flags=re.DOTALL|re.IGNORECASE)
    text = re.sub(r'First,.*?(?=Answer:|$)', '', text, flags=re.DOTALL|re.IGNORECASE)
    text = re.sub(r'Okay,.*?(?=Answer:|$)', '', text, flags=re.DOTALL|re.IGNORECASE)
    text = re.sub(r'So,.*?(?=Answer:|$)', '', text, flags=re.DOTALL|re.IGNORECASE)
    
    # Remove instruction tags
    text = re.sub(r'\[INST\].*?\[/INST\]', ' ', text, flags=re.DOTALL)
    text = re.sub(r'\||\[INST\]|\[/INST\]|\[/s\]', ' ', text)
    
    # Remove obvious phrases
    obvious_phrases = [
        "Answer:", "The answer is:", "Final answer:", "The final answer is:",
        "Based on the context:", "According to the context:",
        "Looking at the information provided:", "Let me approach this",
        "I need to", "First, I", "To answer this"
    ]
    for phrase in obvious_phrases:
        text = re.sub(re.escape(phrase), "", text, flags=re.IGNORECASE)
    
    # Extract content after "Answer:" if it exists
    answer_match = re.search(r'Answer:\s*(.+)', text, flags=re.IGNORECASE|re.DOTALL)
    if answer_match:
        text = answer_match.group(1)
    
    # Basic whitespace cleanup
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove leading/trailing punctuation if extraneous
    text = text.strip('.,;:!?-_ ')
    
    return text
