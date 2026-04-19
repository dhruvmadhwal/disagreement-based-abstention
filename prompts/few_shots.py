"""Centralized few-shot examples shared across prompt builders."""

from __future__ import annotations

from typing import Any, Dict, List


def _base_plan_examples() -> List[Dict[str, Any]]:
    """Normalized version of the shared multi-hop examples."""
    return [
        {
            "question": "Which river flows through the capital city of the country whose constitution was drafted under B. R. Ambedkar's leadership?",
            "steps": [
                {
                    "question": "Which country's constitution did B. R. Ambedkar help draft as its chief architect?",
                    "answer": "India",
                },
                {
                    "question": "What is the capital city of this country?",
                    "answer": "New Delhi",
                },
                {
                    "question": "Which major river runs through the capital city?",
                    "answer": "The Yamuna River",
                },
            ],
            "final_answer": "The Yamuna River",
        },
        {
            "question": "Which mountain range borders the city that hosts Chile's Constitutional Court?",
            "steps": [
                {
                    "question": "Which country is home to the Constitutional Court in question?",
                    "answer": "Chile",
                },
                {
                    "question": "What city hosts the country's Constitutional Court?",
                    "answer": "Santiago",
                },
                {
                    "question": "Which mountain range rises along the eastern edge of the city?",
                    "answer": "The Andes Mountains",
                },
            ],
            "final_answer": "The Andes Mountains",
        },
        {
            "question": "Who won the Chinese Super League earlier, Shanghai Shenhua or Guangzhou Evergrande?",
            "steps": [
                {
                    "question": "When did Shanghai Shenhua first win China's top-flight league (later named the Chinese Super League)?",
                    "answer": "They won the national top division in 1995 (during the Jia-A era).",
                },
                {
                    "question": "When did Guangzhou Evergrande first win the Chinese Super League?",
                    "answer": "Their first CSL title came in 2011.",
                },
                {
                    "question": "Comparing those championship years, which club secured a league title earlier?",
                    "answer": "Shanghai Shenhua’s 1995 title predates Guangzhou Evergrande’s 2011 win.",
                },
            ],
            "final_answer": "Shanghai Shenhua",
        },
    ]


def format_planning_examples(max_examples: int = 3) -> str:
    """Format decomposition few-shots based on the shared examples."""
    examples = _base_plan_examples()[:max_examples]
    if not examples:
        return ""

    lines: List[str] = [
        "Decompose each question into a numbered list of concrete, answerable subquestions. Each line should capture one lookup.",
        "",
    ]
    for idx, example in enumerate(examples, 1):
        lines.append(f"Example {idx}:")
        lines.append(f"Question: {example['question']}")
        lines.append("Decomposition:")
        for sub_idx, step in enumerate(example["steps"], 1):
            lines.append(f"{sub_idx}. {step['question']}")
        lines.append("")
    return "\n".join(lines).strip()


def format_subquestion_examples(max_examples: int = 5) -> str:
    """Format concise QA few-shots for individual subquestions."""
    flat_steps: List[Dict[str, str]] = []
    for example in _base_plan_examples():
        flat_steps.extend(example["steps"])

    examples = flat_steps[:max_examples]
    if not examples:
        return ""

    lines: List[str] = [
        "Answer each subquestion with only the direct fact—no explanations, lists, or extra prose.",
        "",
    ]
    for idx, example in enumerate(examples, 1):
        lines.append(f"Example {idx}:")
        lines.append(f"Q: {example['question']}")
        lines.append(f"A: {example['answer']}")
        lines.append("")
    return "\n".join(lines).strip()


def format_model_generated_plan_examples(max_examples: int = 3) -> str:
    """
    Format the full reasoning chains for the aggregation stage.

    Args:
        max_examples: Maximum number of examples to include.

    Returns:
        A formatted string ready to append to prompts, or an empty string if no examples exist.
    """
    examples = _base_plan_examples()[:max_examples]
    if not examples:
        return ""

    lines: List[str] = [
        "Use only the provided reasoning chain—do not invent missing steps or assume extra propagation between answers.",
        "Each example shows how the final answer cites specific subquestion answers.",
        "",
    ]
    for idx, example in enumerate(examples, 1):
        lines.append(f"Example {idx}:")
        lines.append(f"Question: {example['question']}")
        lines.append("Reasoning chain:")
        for step_idx, step in enumerate(example["steps"], 1):
            lines.append(f"- Step {step_idx}: {step['question']}")
            lines.append(f"  Answer: {step['answer']}")
        lines.append(f"Final Answer: {example['final_answer']}")
        lines.append("")
    return "\n".join(lines).strip()


__all__ = [
    "format_model_generated_plan_examples",
    "format_planning_examples",
    "format_subquestion_examples",
]
