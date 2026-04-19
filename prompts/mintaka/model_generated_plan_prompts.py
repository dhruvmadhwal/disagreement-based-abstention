"""Backward-compatible wrappers for Mintaka model-generated plan prompts."""

from prompts.mintaka.prompts import (
    create_model_generated_base_prompt,
    create_model_generated_direct_prompt,
)

create_base_system_prompt = create_model_generated_base_prompt
create_direct_answer_prompt = create_model_generated_direct_prompt

__all__ = ["create_base_system_prompt", "create_direct_answer_prompt"]
