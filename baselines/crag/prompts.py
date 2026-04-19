"""Prompt builders for CRAG baselines."""

from __future__ import annotations

from prompts.crag.prompts import (
    create_open_ended_system_prompt,
    create_open_ended_user_prompt,
)

from .config import REFERENCE_DATE

IDK_INSTRUCTION = "If you are not sure you know the answer, answer with 'I don't know' only."
AYS_FOLLOW_UP = "Are you sure regarding the correctness of your claim? Please answer with Yes or No."


def build_idk_user_prompt(question: str) -> str:
    return f"{create_open_ended_user_prompt(question)}\n\n{IDK_INSTRUCTION}"


def build_ays_user_prompt(question: str, answer: str) -> str:
    return (
        f"Question: {question.strip()}\n"
        f"Claim: {answer.strip()}\n\n"
        f"{AYS_FOLLOW_UP}"
    )


def build_self_consistency_prompt(question: str, reference_date: str = REFERENCE_DATE) -> tuple[str, str]:
    system = create_open_ended_system_prompt(reference_date=reference_date)
    user = create_open_ended_user_prompt(question)
    return system, user


__all__ = [
    "AYS_FOLLOW_UP",
    "IDK_INSTRUCTION",
    "build_ays_user_prompt",
    "build_idk_user_prompt",
    "build_self_consistency_prompt",
]
