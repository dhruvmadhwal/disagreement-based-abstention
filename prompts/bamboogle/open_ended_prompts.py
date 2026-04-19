"""Backward-compatible wrappers for Bamboogle open-ended prompts."""

from prompts.bamboogle.prompts import (
    create_open_ended_system_prompt,
    create_open_ended_user_prompt,
)

create_prompt = create_open_ended_system_prompt
create_user_prompt = create_open_ended_user_prompt

__all__ = ["create_prompt", "create_user_prompt"]
