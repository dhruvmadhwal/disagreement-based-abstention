"""CLI helpers shared across CRAG generation scripts."""

from __future__ import annotations

import argparse
import os
import shlex
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Tuple

from utils.model_interface import BaseModel, GeminiVertexModel, VLLMModel, GPT51Model
from utils.vllm_server import launch_vllm_server

from .config import (
    DEFAULT_DATASET_PATH,
    GEMINI_MODEL_CHOICES,
    MODEL_PRESETS,
    PRIMARY_MODEL_CHOICES,
    ensure_results_dir,
    make_model_slug,
    normalize_model_choice,
)


def add_shared_generation_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add dataset/model arguments common to every CRAG generator."""
    parser.add_argument("--limit", type=int, default=None, help="Number of examples to process")
    parser.add_argument(
        "--dataset-file",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to CRAG DSL dataset",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override the default output file path",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Model name to load with vLLM / Vertex (defaults per --model-choice)",
    )
    parser.add_argument(
        "--model-choice",
        default="qwen3-8b",  # qwen3-32b also supported
        help=(
            "Preset model choice (default: %(default)s). "
            "Supported presets include: " + ", ".join(PRIMARY_MODEL_CHOICES) + ". "
            "Legacy aliases like 'qwen', 'llama', 'gemma', and 'mistral' remain available."
        ),
    )
    parser.add_argument("--llama", action="store_true", help="Shortcut for selecting the llama preset")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallelism for vLLM")
    parser.add_argument("--no-launch-vllm-server", action="store_true", help="Skip launching a local vLLM server")
    parser.add_argument("--vllm-host", default=os.environ.get("VLLM_HOST", "127.0.0.1"), help="Host for the local vLLM server")
    parser.add_argument("--vllm-port", type=int, default=int(os.environ.get("VLLM_PORT", 8000)), help="Port for the local vLLM server")
    parser.add_argument(
        "--vllm-base-url",
        default=None,
        help="Base URL for an existing vLLM OpenAI-compatible server (overrides host/port)",
    )
    parser.add_argument(
        "--vllm-api-key",
        default=None,
        help="API key for the vLLM server (defaults to VLLM_API_KEY/OPENAI_API_KEY env vars)",
    )
    parser.add_argument(
        "--vllm-chat-template",
        default=os.environ.get("VLLM_CHAT_TEMPLATE"),
        help="Optional chat template path passed to vLLM when launching locally",
    )
    parser.add_argument(
        "--vllm-extra-args",
        default=os.environ.get("VLLM_EXTRA_ARGS"),
        help="Additional arguments for vLLM serve when launching locally (quoted string)",
    )
    parser.add_argument(
        "--vllm-startup-timeout",
        type=float,
        default=float(os.environ.get("VLLM_STARTUP_TIMEOUT", 180.0)),
        help="Seconds to wait for vLLM server startup when launching locally",
    )
    parser.add_argument(
        "--vllm-max-model-len",
        type=int,
        default=(int(os.environ["VLLM_MAX_MODEL_LEN"]) if os.environ.get("VLLM_MAX_MODEL_LEN") else None),
        help="Override max model length passed to vLLM serve when launching locally",
    )
    parser.add_argument(
        "--vllm-gpu-mem-utilization",
        type=float,
        default=(float(os.environ["VLLM_GPU_MEMORY_UTILIZATION"]) if os.environ.get("VLLM_GPU_MEMORY_UTILIZATION") else None),
        help="Override gpu_memory_utilization passed to vLLM serve when launching locally",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from existing output/regime files")
    return parser


def _resolve_model_choice(args: argparse.Namespace) -> None:
    if args.llama:
        args.model_choice = "llama"

    normalized_choice = normalize_model_choice(args.model_choice or "")
    preset_name = MODEL_PRESETS.get(normalized_choice)

    if preset_name is None and args.model_name is None:
        raise ValueError(
            f"Unsupported model choice '{args.model_choice}'. "
            f"Supported presets: {', '.join(PRIMARY_MODEL_CHOICES)} "
            f"(plus legacy aliases like qwen/llama/gemma/mistral)."
        )

    if args.model_name is None and preset_name:
        args.model_name = preset_name

    if preset_name is not None:
        args.model_choice = normalized_choice or args.model_choice
    else:
        args.model_choice = args.model_choice or "custom"

    args.use_vertex = normalized_choice in GEMINI_MODEL_CHOICES
    args.use_gpt5 = normalized_choice == "gpt5"
    if args.use_vertex:
        args.no_launch_vllm_server = True
    if args.use_gpt5:
        args.no_launch_vllm_server = True

    if args.vllm_api_key is None and not args.use_vertex and not args.use_gpt5:
        args.vllm_api_key = os.environ.get("VLLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or "EMPTY"

    if args.vllm_max_model_len is None:
        model_lower = (args.model_name or "").lower()
        args.vllm_max_model_len = 4096 if "olmo" in model_lower else 8192

    if args.vllm_gpu_mem_utilization is None:
        args.vllm_gpu_mem_utilization = 0.8


def _build_model(args: argparse.Namespace) -> BaseModel:
    if getattr(args, "use_vertex", False):
        return GeminiVertexModel(model_name=args.model_name)
    if getattr(args, "use_gpt5", False):
        return GPT51Model(api_key=os.environ.get("OPENAI_API_KEY"), model_name=args.model_name or "gpt-5.1")
    return VLLMModel(
        model_name=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        base_url=args.vllm_base_url,
        api_key=args.vllm_api_key,
    )


@contextmanager
def setup_model_from_args(args: argparse.Namespace) -> Iterator[Tuple[BaseModel, bool]]:
    """Resolve presets, launch vLLM if needed, and yield a usable model."""
    _resolve_model_choice(args)
    launching_server = not (args.no_launch_vllm_server or getattr(args, "use_vertex", False) or getattr(args, "use_gpt5", False))

    if launching_server:
        args.vllm_base_url = f"http://{args.vllm_host}:{args.vllm_port}/v1"
        extra_args = shlex.split(args.vllm_extra_args) if args.vllm_extra_args else None
    else:
        if args.vllm_base_url is None and not getattr(args, "use_vertex", False):
            args.vllm_base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
        extra_args = None

    if launching_server:
        with launch_vllm_server(
            args.model_name,
            host=args.vllm_host,
            port=args.vllm_port,
            tensor_parallel_size=args.tensor_parallel_size,
            chat_template=args.vllm_chat_template,
            extra_args=extra_args,
            env_overrides=None,
            startup_timeout=args.vllm_startup_timeout,
            max_model_len=args.vllm_max_model_len,
            gpu_memory_utilization=args.vllm_gpu_mem_utilization,
        ):
            yield _build_model(args), True
    else:
        yield _build_model(args), False


__all__ = [
    "add_shared_generation_args",
    "ensure_results_dir",
    "make_model_slug",
    "setup_model_from_args",
]
