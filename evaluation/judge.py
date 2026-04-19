"""Shared Vertex AI client + JSON-parsing helpers for the LLM judges."""

from __future__ import annotations

import json
import os
from typing import Any, Dict

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def build_vertex_client(task_label: str = "evaluation") -> OpenAI:
    """Build a Vertex-backed OpenAI-compatible client.

    Reads project + location from `VERTEX_PROJECT_ID`/`VERTEX_LOCATION`
    (falling back to `GOOGLE_CLOUD_PROJECT`/`GOOGLE_CLOUD_REGION`/"global");
    refreshes ADC credentials. `task_label` is only used to phrase the error
    message when project resolution fails.
    """
    # Lazy imports so importing this module does not pull in google.auth
    # (the unit-test sweep imports it without ADC available).
    from google.auth import default
    import google.auth.transport.requests

    location = os.environ.get(
        "VERTEX_LOCATION", os.environ.get("GOOGLE_CLOUD_REGION", "global")
    )
    project_id = os.environ.get(
        "VERTEX_PROJECT_ID", os.environ.get("GOOGLE_CLOUD_PROJECT")
    )
    if not project_id:
        raise RuntimeError(
            f"Set VERTEX_PROJECT_ID or GOOGLE_CLOUD_PROJECT before running {task_label}."
        )

    credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    credentials.refresh(google.auth.transport.requests.Request())

    return OpenAI(
        base_url=(
            f"https://aiplatform.googleapis.com/v1/projects/{project_id}"
            f"/locations/{location}/endpoints/openapi"
        ),
        api_key=credentials.token,
    )


def strip_markdown_json(text: str) -> str:
    """Strip leading ``` / ```json fences from a model response."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    return text


def parse_judge_json(response_text: str) -> Dict[str, Any]:
    """Parse a JSON block from a judge response, tolerating Markdown fences."""
    return json.loads(strip_markdown_json(response_text))


__all__ = ["build_vertex_client", "strip_markdown_json", "parse_judge_json"]
