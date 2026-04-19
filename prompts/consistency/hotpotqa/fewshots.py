"""Few-shot bank for HotpotQA consistency comparisons."""

FEWSHOT_EXAMPLES = [
    {
        "question": "Which river flows through Paris?",
        "answer_a": "Seine",
        "answer_b": "the river Seine",
        "equivalent": 1,
        "explanation": "Same entity; article + casing ignored.",
    },
    {
        "question": "Where was Elon Musk born?",
        "answer_a": "Pretoria, South Africa",
        "answer_b": "Johannesburg",
        "equivalent": 0,
        "explanation": "Contradiction on birthplace city.",
    },
    {
        "question": "Name two official languages spoken in Switzerland.",
        "answer_a": "German; French",
        "answer_b": "French; German",
        "equivalent": 1,
        "explanation": "Set comparison ignores order.",
    },
    {
        "question": "Who composed the score for Inception?",
        "answer_a": "I am not able to answer that.",
        "answer_b": "Hans Zimmer",
        "equivalent": -1,
        "explanation": "Answer A is a refusal -> not meaningful.",
    },
    {
        "question": "What year was the director of \"E.T. the Extra-Terrestrial\" born?",
        "answer_a": "1946",
        "answer_b": "December 18, 1946",
        "equivalent": 1,
        "explanation": "Year matches; A omits the day/month but does not contradict B.",
    },
    {
        "question": "What nationality was the author of the novel that the film \"Blade Runner\" is based on?",
        "answer_a": "American",
        "answer_b": "U.S.",
        "equivalent": 1,
        "explanation": "Demonym vs country abbreviation referring to the same nationality.",
    },
    {
        "question": "How many studio albums did the lead singer of Queen release in his solo career?",
        "answer_a": "Two",
        "answer_b": "2",
        "equivalent": 1,
        "explanation": "Numeric vs spelled-out integer; identical value.",
    },
    {
        "question": "Which two awards did the actor who played the Joker in 'The Dark Knight' win posthumously for that role?",
        "answer_a": "Academy Award; BAFTA",
        "answer_b": "BAFTA Award for Best Supporting Actor; Oscar for Best Supporting Actor",
        "equivalent": 1,
        "explanation": "Same two awards; A uses informal naming, B uses formal naming.",
    },
    {
        "question": "What is the capital of the country whose flag was designed by Francisco Miranda?",
        "answer_a": "Bogotá",
        "answer_b": "Caracas",
        "equivalent": 0,
        "explanation": "Different capitals; A confuses Colombia for Venezuela.",
    },
    {
        "question": "Who founded the company that owns the social network LinkedIn?",
        "answer_a": "Bill Gates and Paul Allen",
        "answer_b": "Reid Hoffman",
        "equivalent": 0,
        "explanation": "Different founders; A names Microsoft's founders, but the question is about LinkedIn.",
    },
    {
        "question": "Which philosopher influenced the political theory of the third U.S. president?",
        "answer_a": "I cannot determine that without more context.",
        "answer_b": "John Locke",
        "equivalent": -1,
        "explanation": "Answer A refuses; not a meaningful comparison.",
    },
]


def format_fewshots_for_prompt() -> str:
    """Return the fewshot bank as a string for prompt inclusion."""

    blocks = ["──────────────────────────── EXAMPLES ────────────────────────────"]
    for idx, example in enumerate(FEWSHOT_EXAMPLES, start=1):
        snippet = (
            f"Example {idx}\n"
            f"question = \"{example['question']}\"\n"
            f"answer_a = \"{example['answer_a']}\"\n"
            f"answer_b = \"{example['answer_b']}\"\n\n"
            f"→ equivalent: {example['equivalent']}\n"
            f"reasoning: {example['explanation']}"
        )
        blocks.append(snippet)
    return "\n\n".join(blocks)
