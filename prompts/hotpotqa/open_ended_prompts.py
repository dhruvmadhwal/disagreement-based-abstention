"""Backward-compatible wrappers for HotpotQA open-ended prompts."""

from prompts.hotpotqa.prompts import (
    create_open_ended_system_prompt,
    create_open_ended_user_prompt,
)

create_prompt = create_open_ended_system_prompt
create_user_prompt = create_open_ended_user_prompt

__all__ = ["create_prompt", "create_user_prompt"]
