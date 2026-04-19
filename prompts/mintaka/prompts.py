"""Unified Mintaka prompt builders shared across all generation regimes."""

from __future__ import annotations

from typing import List, Optional

from prompts.common import (
    format_assistive_examples,
    format_open_ended_examples,
    join_sections,
    shared_guidelines,
)
from prompts.mintaka.fewshots import assistive_examples, open_ended_examples


def create_open_ended_system_prompt(
    reference_date: str = "2022-10-04",
    *,
    include_examples: bool = True,
    max_examples: int = 12,
) -> str:
    """Create the system prompt for Mintaka direct QA generation."""

    header = (
        "You are a precise question answering assistant.\n"
        "Always answer the question; if unsure, provide your best single answer.\n"
        "Do not expose hidden reasoning or use <think> tags."
    )
    blocks: List[str] = [
        header,
        shared_guidelines(reference_date),
        "OUTPUT:\nanswer:\n<one line with only the normalized final answer>",
    ]
    if include_examples:
        examples = open_ended_examples()[:max_examples]
        blocks.append(format_open_ended_examples(examples))
    return join_sections(blocks)


def create_open_ended_user_prompt(question: str) -> str:
    """User prompt for Mintaka direct QA."""
    return f"Q: {question.strip()}"


def create_assistive_system_prompt(
    reference_date: str = "2022-10-04",
    *,
    include_examples: bool = True,
    max_examples: int = 8,
) -> str:
    """Create the system prompt for Mintaka DSL execution."""

    intro = (
        shared_guidelines(reference_date)
        + "\n\nYou will be given a short Python-flavored DSL and must execute it step by step to produce intermediate answers."
    )
    signature = "Function signature (conceptual):\ndef qa_model(question: str, **kwargs) -> str"
    form_block = (
        "Each line has the form:\n"
        "answer_k: TYPE = qa_model(\"QUESTION WITH {placeholders}\", **kwargs)\n"
        "OR\n"
        "answer_k: TYPE = qa_model(\"QUESTION\")"
    )
    rules = (
        "EXECUTION RULES\n"
        "- Order: Execute strictly in sequence (answer_1, answer_2, …). Later answers may reference earlier ones.\n"
        "- Outputs: Every answer_k MUST be a string.\n"
        '- Multi-item requests: If a step requires multiple items or aligned sets, return ONE string with items separated by semicolons in order (e.g., "A; B; C").\n'
        "- Concision: values only—no extra prose; no terminal period.\n"
        "- Always answer; if unsure, provide your best answer based on widely known information as of the reference date.\n"
        "- Do not include <think> tags or hidden reasoning—output only the requested values."
    )
    output_fmt = (
        "OUTPUT FORMAT (MANDATORY)\n\n"
        "Your entire reply MUST be a single valid JSON object:\n\n"
        "{\n"
        '  "answer_1": "<value>",\n'
        '  "answer_2": "<value>",\n'
        "  ...\n"
        "}\n"
        "Do not add prefixes, explanations, or extra text before or after the JSON."
    )

    lines = [intro, form_block, signature, rules, output_fmt]
    if include_examples:
        examples = assistive_examples()[:max_examples]
        lines.append(format_assistive_examples(examples))
    return join_sections(lines)


def create_assistive_user_prompt(dsl: str) -> str:
    """User prompt that carries Mintaka DSL instructions."""
    return f"\nDSL to execute:\n{dsl}\n"


def create_incremental_subquestion_system_prompt(reference_date: str = "2022-10-04") -> str:
    """System prompt for Mintaka incremental subquestions."""
    rules = (
        "Rules:\n"
        "- Answer concisely (prefer 1–3 words when natural).\n"
        "- No extra prose.\n"
        "- Always answer; if unsure, provide your best single answer.\n"
        "- Never output <think> or similar tags; respond directly with the answer text."
    )
    multi_answer = "Multi-item answers: return one string with items separated by semicolons (A; B; C)."
    output = "Output:\n[Only the answer text]"
    return join_sections([shared_guidelines(reference_date), rules, multi_answer, output])


def create_incremental_aggregation_system_prompt(reference_date: str = "2022-10-04") -> str:
    """System prompt for Mintaka incremental aggregation."""
    intro = (
        shared_guidelines(reference_date)
        + "\n"
        "You will be given a reasoning chain (sequence of sub-questions and answers) that builds up to a complex answer."
    )
    rules = (
        "Rules:\n"
        "- Use ONLY the provided chain to answer the original question.\n"
        "- Answer concisely (prefer 1–3 words when natural).\n"
        "- Always answer; if unsure, provide your best single answer.\n"
        "- Do not include <think> tags or hidden reasoning in the final output."
    )
    output = (
        "Output (exactly):\n"
        "answer:\n<final answer>\n\n"
        "explanation:\n<one brief line describing how the chain leads to the answer>"
    )
    return join_sections([intro, rules, output])


def create_model_generated_base_prompt(
    context_paragraphs: Optional[str] = None,
    *,
    dataset: str = "mintaka",
    reference_date: str = "2022-10-04",
) -> str:
    """Base prompt shared across Mintaka self-ask steps."""
    del dataset  # reserved for future specialization
    preface = (
        "You are a precise question answering assistant.\n"
        "Always answer; if unsure, provide your best single answer.\n"
        "Do not expose hidden reasoning or use <think> tags."
    )
    blocks: List[str] = [preface, shared_guidelines(reference_date)]
    if context_paragraphs:
        blocks.append("Context:\n" + str(context_paragraphs).strip())
    return join_sections(blocks) + "\n"


def create_model_generated_direct_prompt(
    question: str,
    *,
    context_paragraphs: Optional[str] = None,
    dataset: str = "mintaka",
    reference_date: str = "2022-10-04",
) -> str:
    """Prompt used for direct answers inside the model-generated plan regime."""
    base = create_model_generated_base_prompt(
        context_paragraphs=context_paragraphs,
        dataset=dataset,
        reference_date=reference_date,
    )
    return f"{base}\nQuestion: {question}\nAnswer:"


__all__ = [
    "create_open_ended_system_prompt",
    "create_open_ended_user_prompt",
    "create_assistive_system_prompt",
    "create_assistive_user_prompt",
    "create_incremental_subquestion_system_prompt",
    "create_incremental_aggregation_system_prompt",
    "create_model_generated_base_prompt",
    "create_model_generated_direct_prompt",
]
