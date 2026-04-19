"""Gemini-specific logprob helper for the confidence baseline."""

from __future__ import annotations

import os
from typing import List, Optional, Tuple


def _build_contents(system_prompt: Optional[str], user_prompt: str):
    """Vertex logprob endpoint rejects system role; fold system+user into one user message."""
    try:
        from google.genai import types
    except Exception:
        return None
    combined = user_prompt if not system_prompt else f"{system_prompt}\n\n{user_prompt}"
    return [types.Content(role="user", parts=[types.Part(text=combined)])]


def fetch_gemini_logprobs(
    *,
    question: str,
    system_prompt: Optional[str],
    user_prompt: str,
    model_name: str,
    temperature: float = 0.0,
    top_p: float = 0.95,
    top_k: int = 64,
    max_output_tokens: int = 256,
    logprobs_k: int = 5,
) -> Tuple[str, Optional[List[float]]]:
    """Generate with Gemini on Vertex and return (text, chosen_token_logprobs)."""
    try:
        from google import genai
        from google.genai import types
    except Exception as exc:  # pragma: no cover - optional dep
        raise RuntimeError("google-genai not installed; cannot fetch logprobs.") from exc

    project_id = os.environ.get("VERTEX_PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = os.environ.get("VERTEX_LOCATION") or os.environ.get("GOOGLE_CLOUD_REGION") or "global"
    if not project_id:
        raise RuntimeError("Set VERTEX_PROJECT_ID or GOOGLE_CLOUD_PROJECT to enable Gemini logprobs.")

    client = genai.Client(vertexai=True, project=project_id, location=location)

    contents = _build_contents(system_prompt, user_prompt)
    if contents is None:
        raise RuntimeError("Failed to build Gemini contents; google-genai types not available.")

    response = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=types.GenerateContentConfig(
            temperature=float(temperature),
            top_p=top_p,
            top_k=top_k,
            max_output_tokens=max_output_tokens,
            response_logprobs=True,
            logprobs=logprobs_k,
        ),
    )
    if not response.candidates:
        return "", None
    cand = response.candidates[0]
    text_parts = [getattr(part, "text", "") for part in getattr(cand.content, "parts", [])]
    text = "".join(text_parts).strip()

    logprobs: List[float] = []
    lp_result = getattr(cand, "logprobs_result", None)
    chosen = getattr(lp_result, "chosen_candidates", None)
    if chosen:
        try:
            for token_info in chosen:
                lp = getattr(token_info, "log_probability", None)
                if lp is not None:
                    logprobs.append(float(lp))
        except Exception:
            logprobs = []

    return text, logprobs or None


__all__ = ["fetch_gemini_logprobs"]
