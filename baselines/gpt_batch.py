"""Helpers for GPT-5.1 batch execution in baseline pipelines."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from utils.model_interface import (
    GPT51Model,
    _log_usage,
    parse_gpt51_batch_output_with_usage,
)


def _parse_custom_id(batch_line: str) -> Optional[str]:
    try:
        payload = json.loads(batch_line)
    except json.JSONDecodeError:
        return None
    cid = payload.get("custom_id")
    return str(cid) if cid is not None else None


def _update_max_completion_tokens(batch_line: str, max_completion_tokens: int) -> Optional[str]:
    """Return an updated JSONL line with a larger max_completion_tokens (or None on parse failure)."""
    try:
        payload = json.loads(batch_line)
    except json.JSONDecodeError:
        return None
    body = payload.get("body")
    if not isinstance(body, dict):
        return None
    body["max_completion_tokens"] = int(max_completion_tokens)
    payload["body"] = body
    return json.dumps(payload)


def run_gpt51_batch(
    *,
    model: GPT51Model,
    lines: Iterable[str],
    output_dir: Path,
    job_tag: str,
    poll_interval: float = 5.0,
    usage_context: str = "baselines-batch",
    retry_empty_completions: bool = True,
    retry_max_completion_tokens: int = 4096,
) -> Dict[str, Dict[str, Any]]:
    """Submit a GPT-5.1 batch job and return {custom_id: {content, usage, model}}."""
    batch_lines = [line for line in lines if line]
    if not batch_lines:
        return {}

    sent_ids = {_parse_custom_id(line) for line in batch_lines}
    sent_ids.discard(None)

    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    input_path = output_dir / f"gpt5_{job_tag}_input_{stamp}.jsonl"
    input_path.write_text("\n".join(batch_lines), encoding="utf-8")

    client = model.client
    with input_path.open("rb") as f:
        input_file = client.files.create(file=f, purpose="batch")
    batch_job = model.create_batch_job(input_file_id=input_file.id)
    print(f"Submitted GPT-5.1 batch job {batch_job.id} ({job_tag}, {len(batch_lines)} requests)")

    import time

    while True:
        status = client.batches.retrieve(batch_job.id)
        if status.status in ("completed", "failed", "cancelled"):
            print(f"Batch {batch_job.id} status: {status.status}")
            break
        time.sleep(poll_interval)

    if status.status != "completed":
        return {}

    out_file = client.files.content(status.output_file_id)
    out_path = output_dir / f"gpt5_{job_tag}_output_{stamp}.jsonl"
    out_path.write_text(out_file.text, encoding="utf-8")
    print(f"GPT-5.1 batch output saved to {out_path}")

    detailed = parse_gpt51_batch_output_with_usage(out_file.text)

    # Log usage for the initial attempt.
    for custom_id, payload in detailed.items():
        usage = payload.get("usage", {}) or {}
        model_name = payload.get("model")
        parts = str(custom_id).split("::")
        meta = {
            "context": usage_context,
            "example_id": parts[0] if parts else None,
            "baseline": parts[1] if len(parts) > 1 else None,
            "attempt": "initial",
        }
        _log_usage(
            model_name=model_name,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
            meta=meta,
        )

    if retry_empty_completions and retry_max_completion_tokens:
        empty_ids = [
            cid
            for cid, payload in detailed.items()
            if not str((payload or {}).get("content") or "").strip()
        ]
        missing_ids = sorted(set(sent_ids) - set(detailed.keys()))
        retry_ids = sorted(set(empty_ids) | set(missing_ids))
        if retry_ids:
            id_to_line: Dict[str, str] = {}
            for line in batch_lines:
                cid = _parse_custom_id(line)
                if cid:
                    id_to_line[cid] = line
            retry_lines = []
            for cid in retry_ids:
                original = id_to_line.get(cid)
                if not original:
                    continue
                updated = _update_max_completion_tokens(original, retry_max_completion_tokens)
                if updated:
                    retry_lines.append(updated)
            if retry_lines:
                retry_stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
                retry_input = output_dir / f"gpt5_{job_tag}_retry_input_{retry_stamp}.jsonl"
                retry_input.write_text("\n".join(retry_lines), encoding="utf-8")
                with retry_input.open("rb") as f:
                    retry_file = client.files.create(file=f, purpose="batch")
                retry_job = model.create_batch_job(input_file_id=retry_file.id)
                print(
                    f"Retrying {len(retry_lines)} empty/missing completions with max_completion_tokens={retry_max_completion_tokens}: {retry_job.id}"
                )
                import time

                while True:
                    retry_status = client.batches.retrieve(retry_job.id)
                    if retry_status.status in ("completed", "failed", "cancelled"):
                        print(f"Batch {retry_job.id} status: {retry_status.status}")
                        break
                    time.sleep(poll_interval)
                if retry_status.status == "completed":
                    retry_out_file = client.files.content(retry_status.output_file_id)
                    retry_out_path = output_dir / f"gpt5_{job_tag}_retry_output_{retry_stamp}.jsonl"
                    retry_out_path.write_text(retry_out_file.text, encoding="utf-8")
                    print(f"GPT-5.1 batch retry output saved to {retry_out_path}")
                    retry_detailed = parse_gpt51_batch_output_with_usage(retry_out_file.text)
                    # Log usage for retry attempts separately (the initial attempt was already logged).
                    for custom_id, payload in retry_detailed.items():
                        usage = payload.get("usage", {}) or {}
                        model_name = payload.get("model")
                        parts = str(custom_id).split("::")
                        meta = {
                            "context": usage_context,
                            "example_id": parts[0] if parts else None,
                            "baseline": parts[1] if len(parts) > 1 else None,
                            "attempt": "retry",
                        }
                        _log_usage(
                            model_name=model_name,
                            prompt_tokens=usage.get("prompt_tokens"),
                            completion_tokens=usage.get("completion_tokens"),
                            total_tokens=usage.get("total_tokens"),
                            meta=meta,
                        )
                    for cid, payload in retry_detailed.items():
                        content = str((payload or {}).get("content") or "").strip()
                        if content:
                            detailed[cid] = payload
    return detailed


__all__ = ["run_gpt51_batch"]
