"""Few-shot examples for HotpotQA correctness evaluation."""

FEWSHOT_EXAMPLES = [
    {
        "question": "Who wrote the novel that inspired Blade Runner?",
        "gold": "Philip K. Dick",
        "prediction": "Blade Runner is based on Philip K. Dick's work.",
        "label": 1,
        "reasoning": "Same fact, extra wording allowed.",
    },
    {
        "question": "What city is the birthplace of Elon Musk?",
        "gold": "Pretoria",
        "prediction": "Elon Musk was born in Johannesburg.",
        "label": 0,
        "reasoning": "Contradiction on birthplace (Pretoria vs Johannesburg).",
    },
    {
        "question": "Approximately how many official languages does Switzerland have?",
        "gold": "4",
        "prediction": "Switzerland recognizes four official languages.",
        "label": 1,
        "reasoning": "Numeric word vs digit normalization.",
    },
    {
        "question": "Which scientist developed the theory of relativity?",
        "gold": "Albert Einstein",
        "prediction": "I can't answer that.",
        "label": -1,
        "reasoning": "Refusal / not meaningful.",
    },
]


def format_fewshots_for_prompt() -> str:
    blocks = ["──────────────────────────── EXAMPLES ────────────────────────────"]
    for idx, example in enumerate(FEWSHOT_EXAMPLES, start=1):
        block = (
            f"Example {idx}\n"
            f"  question = \"{example['question']}\"\n"
            f"  gold     = \"{example['gold']}\"\n"
            f"  prediction = \"{example['prediction']}\"\n\n"
            f"  → correct: {example['label']}\n"
            f"    reasoning: {example['reasoning']}"
        )
        blocks.append(block)
    return "\n\n".join(blocks)
