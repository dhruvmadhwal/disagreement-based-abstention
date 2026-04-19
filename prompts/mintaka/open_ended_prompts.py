"""Backward-compatible wrappers for Mintaka open-ended prompts."""

from prompts.mintaka.prompts import (
    create_open_ended_system_prompt,
    create_open_ended_user_prompt,
)

create_prompt = create_open_ended_system_prompt
create_user_prompt = create_open_ended_user_prompt

__all__ = ["create_prompt", "create_user_prompt"]
