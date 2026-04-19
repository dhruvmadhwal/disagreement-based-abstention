"""Lightweight DSL sanity checks used for dataset filtering."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

QA_CALL_RE = re.compile(r"qa_model\(", re.IGNORECASE)
QA_STR_RE = re.compile(r"qa_model\(\s*(\"(?:[^\"\\\\]|\\\\.)*\"|'(?:[^'\\\\]|\\\\.)*')", re.IGNORECASE)
ANSWER_REF_RE = re.compile(r"\banswer_(\d+)\b")

WH_WORDS = ("what", "who", "when", "where", "which", "how", "whom", "whose")
YES_NO = {"yes", "no"}


@dataclass(frozen=True)
class DslSanityResult:
    num_calls: int
    truncated: bool
    one_hop: bool
    no_decompose: bool
    yes_no_misframed: bool
    scope_mismatch: bool
    last_question: str | None

    @property
    def sane(self) -> bool:
        return (
            not self.truncated
            and self.num_calls >= 2
            and not self.no_decompose
            and not self.yes_no_misframed
            and not self.scope_mismatch
        )


def normalize_answer(answer: Any) -> str:
    if answer is None:
        return ""
    if isinstance(answer, list):
        if not answer:
            return ""
        if len(answer) == 1:
            answer = answer[0]
        else:
            answer = "; ".join(str(item) for item in answer)
    text = str(answer).strip().lower()
    return text.strip(".!? ")


def _extract_questions(dsl: str) -> Tuple[List[str], int, bool]:
    if not dsl:
        return [], 0, True
    num_calls = len(QA_CALL_RE.findall(dsl))
    matches = list(QA_STR_RE.finditer(dsl))
    questions: List[str] = []
    for match in matches:
        literal = match.group(1)
        try:
            questions.append(ast.literal_eval(literal))
        except Exception:
            questions.append(literal.strip("\"'"))
    truncated = num_calls > len(matches)
    return questions, num_calls, truncated


def detect_no_decompose(dsl: str) -> bool:
    lines = [line for line in dsl.splitlines() if "qa_model(" in line]
    if len(lines) < 2:
        return False
    for idx, line in enumerate(lines, start=1):
        assigned = idx
        match = re.search(r"\banswer_(\d+)\s*:", line)
        if match:
            try:
                assigned = int(match.group(1))
            except ValueError:
                assigned = idx
        refs = [int(num) for num in ANSWER_REF_RE.findall(line) if num.isdigit()]
        refs = [ref for ref in refs if ref != assigned]
        if any(ref < assigned for ref in refs):
            return False
    return True


def detect_yes_no_misframed(answer: Any, dsl: str) -> Tuple[bool, str | None]:
    if normalize_answer(answer) not in YES_NO:
        return False, None
    questions, _, _ = _extract_questions(dsl)
    if not questions:
        return False, None
    last = questions[-1].strip().lower()
    last = re.sub(r"^[^a-z]+", "", last)
    for word in WH_WORDS:
        if last == word or last.startswith(f"{word} "):
            return True, questions[-1]
    return False, questions[-1]


def detect_scope_mismatch(question: str, dsl: str) -> bool:
    question_lower = (question or "").lower()
    dsl_lower = (dsl or "").lower()
    if any(token in question_lower for token in ("2022-12", "december 2022", "dec 2022")):
        if any(token in dsl_lower for token in ("2022-2023", "2022-23", "2022/23", "season")):
            return True
    return False


def analyze_dsl(question: str, answer: Any, dsl: str) -> DslSanityResult:
    questions, num_calls, truncated = _extract_questions(dsl)
    one_hop = num_calls == 1
    no_decompose = detect_no_decompose(dsl)
    yes_no_misframed, last_question = detect_yes_no_misframed(answer, dsl)
    scope_mismatch = detect_scope_mismatch(question, dsl)
    return DslSanityResult(
        num_calls=num_calls,
        truncated=truncated,
        one_hop=one_hop,
        no_decompose=no_decompose,
        yes_no_misframed=yes_no_misframed,
        scope_mismatch=scope_mismatch,
        last_question=last_question,
    )


def summarize_failures(results: List[DslSanityResult]) -> Dict[str, int]:
    summary = {
        "total": len(results),
        "truncated": 0,
        "one_hop": 0,
        "no_decompose": 0,
        "yes_no_misframed": 0,
        "scope_mismatch": 0,
        "sane": 0,
    }
    for res in results:
        if res.truncated:
            summary["truncated"] += 1
        if res.one_hop:
            summary["one_hop"] += 1
        if res.no_decompose:
            summary["no_decompose"] += 1
        if res.yes_no_misframed:
            summary["yes_no_misframed"] += 1
        if res.scope_mismatch:
            summary["scope_mismatch"] += 1
        if res.sane:
            summary["sane"] += 1
    return summary
