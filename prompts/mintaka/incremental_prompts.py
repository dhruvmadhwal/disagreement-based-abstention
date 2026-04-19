"""Backward-compatible wrappers for Mintaka incremental prompts."""

from prompts.mintaka.prompts import (
    create_incremental_aggregation_system_prompt,
    create_incremental_subquestion_system_prompt,
)

create_subquestion_system_prompt = create_incremental_subquestion_system_prompt
create_aggregation_system_prompt = create_incremental_aggregation_system_prompt

__all__ = ["create_subquestion_system_prompt", "create_aggregation_system_prompt"]
