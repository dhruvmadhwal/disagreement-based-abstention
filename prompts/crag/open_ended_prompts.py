"""Backward-compatible wrappers for CRAG open-ended prompts."""

from prompts.crag.prompts import (
    create_open_ended_system_prompt,
    create_open_ended_user_prompt,
)

create_prompt = create_open_ended_system_prompt
create_user_prompt = create_open_ended_user_prompt

__all__ = ["create_prompt", "create_user_prompt"]
