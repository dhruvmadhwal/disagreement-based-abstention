"""
Utility helpers for managing a local vLLM OpenAI-compatible server lifecycle.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from contextlib import contextmanager
from typing import Dict, Iterable, Iterator, Optional, Union
from urllib.error import URLError
from urllib.request import urlopen


def _wait_for_server(base_url: str, timeout: float = 120.0, process: Optional[subprocess.Popen] = None) -> None:
    """Poll the vLLM server until it responds or timeout is reached."""
    deadline = time.time() + timeout
    probe_url = base_url.rstrip("/") + "/models"

    while time.time() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError("vLLM server process exited unexpectedly during startup.")
        try:
            with urlopen(probe_url, timeout=5):
                return
        except URLError:
            time.sleep(1.0)
        except Exception:
            time.sleep(1.0)
    raise RuntimeError(f"Timed out waiting for vLLM server at {probe_url}")


@contextmanager
def launch_vllm_server(
    model_name: str,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    tensor_parallel_size: int = 1,
    chat_template: Optional[str] = None,
    extra_args: Optional[Iterable[str]] = None,
    env_overrides: Optional[Dict[str, str]] = None,
    startup_timeout: float = 120.0,
    max_model_len: Optional[int] = None,
    gpu_memory_utilization: Optional[Union[int, float]] = None,
) -> Iterator[str]:
    """
    Launch a local vLLM server as a subprocess, yielding the base URL.

    The server is terminated automatically when the context exits.
    """

    cmd = [
        "vllm",
        "serve",
        model_name,
        "--host",
        host,
        "--port",
        str(port),
    ]
    if tensor_parallel_size and tensor_parallel_size > 1:
        cmd.extend(["--tensor-parallel-size", str(tensor_parallel_size)])
    if chat_template:
        cmd.extend(["--chat-template", chat_template])
    if extra_args:
        cmd.extend(list(extra_args))
    if max_model_len:
        cmd.extend(["--max-model-len", str(max_model_len)])
    if gpu_memory_utilization:
        cmd.extend(["--gpu-memory-utilization", str(gpu_memory_utilization)])

    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)

    process = subprocess.Popen(cmd, env=env)
    base_url = f"http://{host}:{port}/v1"

    try:
        _wait_for_server(base_url, timeout=startup_timeout, process=process)
        yield base_url
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        # Give the OS a short moment to release the socket
        time.sleep(1.0)
