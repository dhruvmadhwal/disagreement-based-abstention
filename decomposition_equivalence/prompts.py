"""Prompt templates for decomposition equivalence generation + judging."""

from __future__ import annotations

from typing import List

_GOLD_DSL_SYSTEM_PROMPT = """You convert a single natural-language question into a sequence of intermediate questions written in a Python-flavored DSL using:

def qa_model(question: str, **kwargs) -> str: ...

Rules

Output only DSL lines, no explanations.

Use numbered variables: answer_1, answer_2, … (sequential, no gaps).

All left-hand sides must be typed exactly as str. No lists, dicts, ints, or floats.

Keep questions minimal and well-formed 

Do not invent data

Examples:

# Q1. Where was the guitarist from R.E.M. born?
answer_1: str = qa_model("Who is the guitarist for the band R.E.M.?")
answer_2: str = qa_model("Where was {answer_1} born?", answer_1=answer_1)

# Q2. What is the population of the country where the 'Spanish Steps' are located?
answer_1: str = qa_model("In which country are the 'Spanish Steps' located?")
answer_2: str = qa_model("What is the population of {answer_1}?", answer_1=answer_1)

# Q3. What is the fourth main dungeon in the first Zelda game for the N64?
answer_1: str = qa_model("What is the first Zelda game released for the Nintendo 64?")
answer_2: str = qa_model("What is the fourth main dungeon in '{answer_1}'?", answer_1=answer_1)

# Q4. Who is the protagonist in the eighth Final Fantasy game?
answer_1: str = qa_model("What is the title of the eighth Final Fantasy game?")
answer_2: str = qa_model("Who is the main protagonist of '{answer_1}'?", answer_1=answer_1)

# Q5. How tall is the actor who played Deadpool?
answer_1: str = qa_model("Who played Deadpool in the films?")
answer_2: str = qa_model("How tall is {answer_1}?", answer_1=answer_1)

# Q6. Who starred in the movie based on *Fear and Loathing in Las Vegas*?
answer_1: str = qa_model("What is the film adaptation of the book 'Fear and Loathing in Las Vegas'?")
answer_2: str = qa_model("Who starred in '{answer_1}'?", answer_1=answer_1)

# Q7. When did the author who wrote *Dubliners* die?
answer_1: str = qa_model("Who wrote the book 'Dubliners'?")
answer_2: str = qa_model("When did {answer_1} die?", answer_1=answer_1)

# Q8. How many kids does the lead actor of *Troy* have?
answer_1: str = qa_model("Who was the lead actor in the film 'Troy' (2004)?")
answer_2: str = qa_model("How many children does {answer_1} have?", answer_1=answer_1)

# Q9. Who was the first wife of Queen Elizabeth II's eldest son?
answer_1: str = qa_model("Who is the eldest son of Queen Elizabeth II?")
answer_2: str = qa_model("Who was the first wife of {answer_1}?", answer_1=answer_1)

# Q10. Who distributed the movie that's based on the Harry Potter book released in 2004?
answer_1: str = qa_model("Which Harry Potter book was released in 2004?")
answer_2: str = qa_model("What is the title of the film based on '{answer_1}'?", answer_1=answer_1)
answer_3: str = qa_model("Who distributed the film '{answer_2}'?", answer_2=answer_2)"""

def build_generation_system_prompt(dataset: str) -> str:
    """Return the system prompt for DSL generation for a dataset."""
    return _GOLD_DSL_SYSTEM_PROMPT


def build_generation_user_prompt(question: str) -> str:
    return f"Q: {question}".strip()


def build_judge_system_prompt(dataset: str) -> str:
    """Return the system prompt for DSL equivalence judging."""
    base = (
        "You are a careful evaluator of question decompositions. "
        "Focus on semantic coverage, not exact wording.\n\n"
        "Output must be strict JSON only (no markdown, no commentary， no explanation)."
    )
    return base


def build_judge_user_prompt(
    dataset: str,
    question: str,
    gold_dsl: str,
    model_dsl: str,
    gold_hops: List[str],
    model_hops: List[str],
) -> str:
    """Build the user prompt for the LLM judge."""
    gold_lines = [line.strip() for line in gold_dsl.splitlines() if line.strip()]
    model_lines = [line.strip() for line in model_dsl.splitlines() if line.strip()]
    gold_block = "\n".join([f"- {line}" for line in gold_lines]) or "(none)"
    model_block = "\n".join([f"- {line}" for line in model_lines]) or "(none)"

    instructions = (
        "Task:\n"
        "1) Decide if the two decompositions are semantically equivalent, that is, if they would lead to the same final answer. "
        "Given the question. Return equivalent_final as 1 (yes) or 0 (no).\n"
        "2) Count how many gold hops are covered by some hop in the model decomposition.\n"
        "Coverage means the hop asks for the same intermediate information (allow paraphrase and minor scope differences).\n\n"
        "Return JSON with the following fields:\n"
        "- equivalent_final: 0 or 1\n"
        "- matches: integer (# gold hops covered by some model hop)\n"
    )

    return (
        f"{instructions}\n"
        f"Dataset: {dataset}\n"
        f"Question: {question}\n\n"
        "Gold DSL:\n"
        f"{gold_dsl.strip()}\n\n"
        "Gold DSL lines:\n"
        f"{gold_block}\n\n"
        "Model DSL:\n"
        f"{model_dsl.strip()}\n\n"
        "Model DSL lines:\n"
        f"{model_block}\n\n"
    )


__all__ = [
    "build_generation_system_prompt",
    "build_generation_user_prompt",
    "build_judge_system_prompt",
    "build_judge_user_prompt",
]
