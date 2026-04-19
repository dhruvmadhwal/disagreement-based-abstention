"""Few-shot examples for Bamboogle correctness evaluation."""

FEWSHOT_EXAMPLES = [
    {
        "question": "Which city is the most populous in the United States?",
        "gold": "New York City",
        "prediction": "The most populous U.S. city is NYC.",
        "label": 1,
        "reasoning": "Alias normalization: 'NYC' ≡ 'New York City'; no contradiction.",
    },
    {
        "question": "What is the capital of France?",
        "gold": "Paris",
        "prediction": "The capital of France is Marseille, not Paris.",
        "label": 0,
        "reasoning": "Prediction explicitly negates the gold fact (Marseille vs Paris).",
    },
    {
        "question": "Approximately how many survivors were there?",
        "gold": "10",
        "prediction": "There were about ten survivors.",
        "label": 1,
        "reasoning": "Word vs digit normalization; 'about' still within tolerance, no contradiction.",
    },
    {
        "question": "Who is the fifth president of the United States?",
        "gold": "James Monroe",
        "prediction": "I don't know.",
        "label": -1,
        "reasoning": "Refusal / not meaningful – does not provide the requested fact.",
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
