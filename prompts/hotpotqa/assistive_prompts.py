"""Backward-compatible wrappers for HotpotQA assistive prompts."""

from prompts.hotpotqa.prompts import (
    create_assistive_system_prompt,
    create_assistive_user_prompt,
)

create_system_prompt = create_assistive_system_prompt
create_user_prompt = create_assistive_user_prompt

__all__ = ["create_system_prompt", "create_user_prompt"]
