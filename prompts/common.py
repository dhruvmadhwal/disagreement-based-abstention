from typing import List, Dict, Any


def shared_guidelines(reference_date: str) -> str:
    return f"""
You are a precise question answering assistant.
Your goal is to produce a concise, normalized final answer.

GENERAL GUIDELINES
- Always return your best single answer even if uncertain.
- Do not include <think> tags, internal thoughts, or hidden reasoning in your output.
- Keep prose minimal—prefer the shortest phrasing that still answers the question.

REFERENCE DATE
- Treat all time-relative terms (e.g., "now", "currently") and age calculations as referring to {reference_date} unless the question specifies another date.

NORMALIZATION (make answers easy to compare)
- Dates: if day known → YYYY-MM-DD; if only month+year → YYYY-MM; if only year → YYYY. Use this order unless stated otherwise in the question.
- Units: prefer symbols (km, m, ft, in, kg, mi², km²). Keep one consistent unit; don't add conversions unless asked.
- Multi-item strings: separate with semicolons exactly like "A; B; C" (single space after each semicolon; no trailing semicolon).
- Names: prefer canonical full names (e.g., "Thomas Edison" not "Edison") unless a shorter form is standard (e.g., "NBA").
""".strip()


def join_sections(sections: List[str]) -> str:
    return "\n\n".join([s for s in sections if s and s.strip()])


def format_open_ended_examples(examples: List[Dict[str, str]]) -> str:
    lines: List[str] = ["EXAMPLES:"]
    for idx, ex in enumerate(examples, start=1):
        lines.append(f"Examples {idx}:")
        lines.append(f"Q: {ex['q']}")
        lines.append("answer:")
        lines.append(ex['answer'])
        lines.append("")
    return "\n".join(lines).rstrip()


def format_assistive_examples(examples: List[Dict[str, Any]]) -> str:
    lines: List[str] = ["Few-shot examples"]
    for idx, ex in enumerate(examples, start=1):
        lines.append("")
        lines.append(f"Example {idx}")
        lines.append("DSL:")
        for dsl_line in ex.get('dsl_lines', []):
            lines.append(dsl_line)
        lines.append("")
        lines.append("Output:")
        lines.append("intermediate_answers:")
        answers = ex.get('answers', {})
        # Pretty-print answers as JSON-like block
        lines.append("{")
        for i, (k, v) in enumerate(answers.items()):
            comma = "," if i < len(answers) - 1 else ""
            lines.append(f"  \"{k}\": \"{v}\"{comma}")
        lines.append("}")
    return "\n".join(lines).rstrip()
