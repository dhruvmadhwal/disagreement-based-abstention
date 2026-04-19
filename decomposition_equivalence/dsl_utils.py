"""Utilities for parsing DSL strings into hop lists."""

from __future__ import annotations

import ast
import re
from typing import List, Optional


_CODE_FENCE_RE = re.compile(r"^```+", re.IGNORECASE)


def clean_dsl_text(text: str) -> str:
    if not text:
        return ""
    lines = []
    for raw in str(text).splitlines():
        if _CODE_FENCE_RE.match(raw.strip()):
            continue
        lines.append(raw)
    return "\n".join(lines).strip()


def _joinedstr_to_template(node: ast.JoinedStr) -> str:
    parts: List[str] = []
    for value in node.values:
        if isinstance(value, ast.Constant):
            parts.append(str(value.value))
        elif isinstance(value, ast.FormattedValue):
            try:
                expr = ast.unparse(value.value)
            except Exception:
                expr = "..."
            parts.append("{" + expr + "}")
    return "".join(parts)


def _extract_question_from_line(line: str) -> Optional[str]:
    if "qa_model" not in line:
        return None

    # Strip leading type annotation for parsing.
    cleaned = re.sub(r"^\s*answer_\d+\s*:\s*str\s*=\s*", "x = ", line)

    try:
        node = ast.parse(cleaned)
        if not node.body:
            return None
        stmt = node.body[0]
        if isinstance(stmt, ast.Assign):
            call = stmt.value
        elif isinstance(stmt, ast.AnnAssign):
            call = stmt.value
        else:
            return None
        if not isinstance(call, ast.Call):
            return None
        if not call.args:
            return None
        arg0 = call.args[0]
        if isinstance(arg0, ast.Constant) and isinstance(arg0.value, str):
            return arg0.value
        if isinstance(arg0, ast.JoinedStr):
            return _joinedstr_to_template(arg0)
    except Exception:
        pass

    # Fallback: regex to capture the first quoted string literal.
    match = re.search(r"qa_model\s*\(\s*('([^'\\]|\\.)*'|\"([^\"\\]|\\.)*\")", line)
    if not match:
        return None
    literal = match.group(1)
    try:
        value = ast.literal_eval(literal)
        if isinstance(value, str):
            return value
    except Exception:
        return None
    return None


def parse_dsl_hops(dsl_text: str) -> List[str]:
    cleaned = clean_dsl_text(dsl_text)
    hops: List[str] = []
    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        # Remove inline comments.
        if "#" in line:
            line = line.split("#", 1)[0].strip()
        if "qa_model" not in line:
            continue
        question = _extract_question_from_line(line)
        if question:
            hops.append(question.strip())
    return hops


__all__ = ["clean_dsl_text", "parse_dsl_hops"]
