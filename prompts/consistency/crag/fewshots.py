"""Few-shot bank for CRAG consistency comparisons."""

FEWSHOT_EXAMPLES = [
    {
        "question": "Which genre of music does the singer of \"Blank Space\" and \"Wildest Dreams\" sing?",
        "answer_a": "Pop and Country",
        "answer_b": "country; pop",
        "equivalent": 1,
        "explanation": "Unordered list normalization; separators normalized; same two unique items.",
    },
    {
        "question": "When did the author who wrote A Farewell to Arms die?",
        "answer_a": "July 2nd, 1961",
        "answer_b": "2-Jul-1961",
        "equivalent": 1,
        "explanation": "Date-format normalization; exact same calendar date.",
    },
    {
        "question": "Where was the author who wrote The Great Gatsby born?",
        "answer_a": "Saint Paul, MN",
        "answer_b": "Saint Paul, Minnesota",
        "equivalent": 1,
        "explanation": "Abbreviation expansion (\"MN\" ≡ \"Minnesota\"); same location.",
    },
    {
        "question": "How many square miles is the capital of Alaska?",
        "answer_a": "3,255 mi²",
        "answer_b": "3255 square miles",
        "equivalent": 1,
        "explanation": "Unit spelling/symbol normalization and thousands-separator ignored.",
    },
    {
        "question": "Exactly how old is the oldest river in the United States?",
        "answer_a": "260 million to 325 million years ago",
        "answer_b": "300 million years",
        "equivalent": 1,
        "explanation": "Range vs point—single value lies within stated range.",
    },
    {
        "question": "How tall is the actor who plays John Wick?",
        "answer_a": "He is six foot one inch tall.",
        "answer_b": "6 ft 1 in",
        "equivalent": 1,
        "explanation": "Feet–inches notation normalization; same height.",
    },
    {
        "question": "How rich is the singer of Cant Touch This?",
        "answer_a": "$2 million",
        "answer_b": "USD 2,000,000",
        "equivalent": 1,
        "explanation": "Currency symbol/name normalization and scale words; same value.",
    },
    {
        "question": "When was the fifth U.S. president born?",
        "answer_a": "April 28, 1758",
        "answer_b": "1758-04-28",
        "equivalent": 1,
        "explanation": "Date-format normalization; exact same calendar date.",
    },
    {
        "question": "What is the population of the largest city in the country south of Canada?",
        "answer_a": "8.5 million residents in New York City",
        "answer_b": "8,500,000",
        "equivalent": 1,
        "explanation": "Extra descriptive text allowed; numeric formatting normalized; same value.",
    },
    {
        "question": "How many people live in the city of Shanghai?",
        "answer_a": "29,868,000 per census 2024",
        "answer_b": "28,517,000 per census 2022",
        "equivalent": 1,
        "explanation": "Within 5 percentage-point tolerance due to natural drift over time.",
    },
    {
        "question": "Who is the current president of China?",
        "answer_a": "To find out who the president of China is, you can check the latest news or official sources.",
        "answer_b": "Xi Jinping.",
        "equivalent": -1,
        "explanation": "Not meaningful answer; Answer A does not provide the fact.",
    },
    {
        "question": "Who is the fifth president of the United States?",
        "answer_a": "John Quincy Adams",
        "answer_b": "I don't know.",
        "equivalent": -1,
        "explanation": "Not meaningful answer; Answer B refuses the question.",
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
