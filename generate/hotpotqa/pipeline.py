"""Shared HotpotQA regime execution utilities."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from prompts.hotpotqa.prompts import (
    create_assistive_system_prompt,
    create_assistive_user_prompt,
    create_incremental_aggregation_system_prompt,
    create_incremental_subquestion_system_prompt,
    create_open_ended_system_prompt,
    create_open_ended_user_prompt,
)
from utils.model_interface import (
    BaseModel,
    GPT51Model,
    build_gpt51_batch_line,
    parse_gpt51_batch_output_with_usage,
    _log_usage,
)

from .config import (
    DEFAULT_CHECKPOINT_INTERVAL,
    DEFAULT_DATASET_PATH,
    DEFAULT_RESULTS_ROOT,
    REFERENCE_DATE,
    TEMPERATURE,
    TOP_K,
    TOP_P,
)


REGIME_FILENAMES = {
    "open_ended": "hotpotqa_open_ended.json",
    "assistive": "hotpotqa_assistive.json",
    "incremental": "hotpotqa_sequential.json",
    "model_generated": "hotpotqa_model_generated.json",
}

REGIME_ORDER = tuple(REGIME_FILENAMES.keys())


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks and standalone think tags."""
    if not text:
        return text
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"</?think>", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def load_examples(dataset_file: Path, limit: Optional[int]) -> List[Dict[str, object]]:
    data = json.loads(dataset_file.read_text())
    return data[:limit] if limit else data


def parse_open_ended_response(text: str) -> str:
    answer, section = "", None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.lower().startswith("answer:"):
            section = "answer"
            answer = line.split(":", 1)[1].strip()
        elif section == "answer" and line:
            answer = f"{answer} {line}".strip()
    if not answer:
        answer = text.strip()
    return answer.strip()


def _sanitize_json_candidate(text: str) -> Optional[str]:
    """Return a cleaned JSON string extracted from the given text."""
    if not text:
        return None

    cleaned = text.strip()
    if not cleaned:
        return None

    # If the response includes fenced code blocks, extract the interior.
    if "```" in cleaned:
        lines = cleaned.splitlines()
        inside_block = False
        block_lines: List[str] = []
        for raw_line in lines:
            line = raw_line.strip()
            if line.startswith("```"):
                inside_block = not inside_block
                continue
            if inside_block:
                block_lines.append(raw_line)
        if block_lines:
            cleaned = "\n".join(block_lines).strip()
        else:
            cleaned = cleaned.replace("```", "")

    # Trim to the first/last braces if present.
    if "{" in cleaned and "}" in cleaned:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            cleaned = cleaned[start : end + 1]
    else:
        return None

    sanitized = (
        cleaned.replace("True", "true")
        .replace("False", "false")
        .replace("None", "null")
    ).strip()

    return sanitized if sanitized else None


def _unwrap_intermediate_answers(intermediate: Dict[str, object]) -> Dict[str, str]:
    """Unwrap nested intermediate_answers structure if present.

    Some models return {"intermediate_answers": {"answer_1": ..., "answer_2": ...}}
    instead of {"answer_1": ..., "answer_2": ...}. This function normalizes both formats.
    """
    if not intermediate or not isinstance(intermediate, dict):
        return intermediate or {}

    # Check for nested intermediate_answers key
    if "intermediate_answers" in intermediate:
        nested = intermediate["intermediate_answers"]
        if isinstance(nested, dict) and not nested.get("error"):
            # Verify the nested dict has answer_N keys or values
            if any(k.startswith("answer_") for k in nested.keys()) or nested:
                return nested

    return intermediate


def parse_assistive_response(text: str) -> Tuple[str, Dict[str, str], str]:
    explanation, json_lines, section = "", [], None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.lower().startswith("explanation:"):
            section = "explanation"
            explanation = line.split(":", 1)[1].strip()
        elif line.lower().startswith("intermediate_answers:"):
            section = "json"
            json_lines = []
        elif section == "explanation" and line:
            explanation = f"{explanation} {line}".strip()
        elif section == "json" and line:
            json_lines.append(line)

    intermediate: Dict[str, str] = {}
    json_blob = "\n".join(json_lines) if json_lines else ""
    if not json_blob:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            json_blob = text[start : end + 1]

    if json_blob:
        sanitized = json_blob.replace("True", "true").replace("False", "false")
        try:
            intermediate = json.loads(sanitized)
        except json.JSONDecodeError:
            intermediate = {"error": "malformed_json", "raw": json_blob}

    # Attempt additional sanitization strategies similar to MuSiQue parsing.
    if not intermediate or ("error" in intermediate and intermediate.get("error") == "malformed_json"):
        candidates: List[str] = []
        # First try the full response.
        sanitized_full = _sanitize_json_candidate(text)
        if sanitized_full:
            candidates.append(sanitized_full)
        # Then the extracted blob from above.
        sanitized_blob = _sanitize_json_candidate(json_blob) if json_blob else None
        if sanitized_blob:
            candidates.append(sanitized_blob)
        # Finally, the joined json_lines if present.
        if json_lines:
            sanitized_lines = _sanitize_json_candidate("\n".join(json_lines))
            if sanitized_lines:
                candidates.append(sanitized_lines)

        parse_error: Optional[Exception] = None
        for candidate in candidates:
            try:
                intermediate = json.loads(candidate)
                parse_error = None
                break
            except json.JSONDecodeError as exc:
                parse_error = exc

        if parse_error is not None or not intermediate:
            raw_payload = candidates[-1] if candidates else (json_blob or text)
            intermediate = {
                "error": "malformed_json",
                "raw": raw_payload,
                "error_detail": str(parse_error) if parse_error else "unable_to_parse",
            }

    final_answer = extract_final_answer(intermediate)
    return explanation.strip(), intermediate, final_answer.strip()


def extract_final_answer(intermediate: Dict[str, str]) -> str:
    if not intermediate or "error" in intermediate:
        return ""
    keys = [key for key in intermediate if key.startswith("answer_") and key.split("_")[-1].isdigit()]
    if not keys:
        values = list(intermediate.values())
        return str(values[-1]) if values else ""
    keys.sort(key=lambda item: int(item.split("_")[1]))
    return str(intermediate[keys[-1]])


def parse_final_response(text: str) -> Tuple[str, str]:
    answer = parse_open_ended_response(text)
    if answer:
        return answer.strip(), ""

    match_answer = re.search(r"(?:final\s*answer|answer)\s*[:：]\s*(.+)", text, flags=re.IGNORECASE)
    if match_answer:
        answer = match_answer.group(1).strip()
        remainder = text[match_answer.end() :]
        match_expl = re.search(r"explanation\s*[:：]\s*(.+)", remainder, flags=re.IGNORECASE)
        explanation = match_expl.group(1).strip() if match_expl else remainder.strip()
        return answer, explanation

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        answer = lines[0]
        explanation = " ".join(lines[1:]).strip()
        return answer, explanation

    return "", ""


def parse_dsl(dsl_text: str) -> List[Tuple[str, str]]:
    steps: List[Tuple[str, str]] = []
    for raw_line in dsl_text.strip().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        var_match = re.match(r"(\w+):\s*\w+\s*=", line)
        if not var_match:
            continue
        var_name = var_match.group(1)
        qa_match = re.search(r"qa_model\((.*)\)", line)
        if not qa_match:
            continue
        args_text = qa_match.group(1)
        string_match = re.match(r'"((?:[^"\\]|\\.)*)"\s*(?:,|$)', args_text)
        if not string_match:
            continue
        question = string_match.group(1)
        question = question.encode("utf-8").decode("unicode_escape")
        steps.append((var_name, question))
    return steps


def replace_placeholders(question: str, answers: Dict[str, str]) -> str:
    updated = question
    for key, value in answers.items():
        updated = updated.replace(f"{{{key}}}", value)
    return updated


def split_multi_answers(answer_text: str) -> List[str]:
    """Split semicolon-separated answers into individual pieces."""
    if answer_text is None:
        return [""]
    if ";" not in answer_text:
        return [answer_text.strip()]
    parts = [part.strip() for part in answer_text.split(";") if part.strip()]
    if len(parts) > 12:
        parts = parts[:12]  # truncate instead of erroring
    return parts or [answer_text.strip()]


def join_multi_answers(answers: Sequence[str]) -> str:
    """Join multiple answers with a normalized semicolon separator."""
    return "; ".join(part.strip() for part in answers if part.strip())


class HotpotQARegimeRunner:
    """Execute each HotpotQA reasoning regime using a shared model."""

    def __init__(
        self,
        model: BaseModel,
        *,
        reference_date: str = REFERENCE_DATE,
        temperature: float = TEMPERATURE,
        top_p: float = TOP_P,
        top_k: int = TOP_K,
        selfask_agent: Optional[object] = None,
    ):
        self.model = model
        self.reference_date = reference_date
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.selfask_agent = selfask_agent

    def run_open_ended(self, question: str, **kwargs) -> Dict[str, object]:
        system_prompt = create_open_ended_system_prompt(reference_date=self.reference_date)
        user_prompt = create_open_ended_user_prompt(question)
        raw = self.model.generate_answer(
            question,
            system=system_prompt,
            user=user_prompt,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            **kwargs,
        )
        cleaned = strip_think_tags(raw)
        answer = parse_open_ended_response(cleaned)
        return {"answer": answer, "raw_completion": cleaned}

    def run_assistive(self, dsl: str, **kwargs) -> Dict[str, object]:
        if not dsl.strip():
            return {"final_answer": "", "explanation": "", "intermediate_answers": {}, "raw_completion": ""}
        system_prompt = create_assistive_system_prompt(reference_date=self.reference_date)
        user_prompt = create_assistive_user_prompt(dsl)
        raw = self.model.generate_answer(
            dsl,
            system=system_prompt,
            user=user_prompt,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            **kwargs,
        )
        cleaned = strip_think_tags(raw)
        explanation, intermediate, final_answer = parse_assistive_response(cleaned)
        return {
            "final_answer": final_answer,
            "explanation": explanation,
            "intermediate_answers": intermediate,
            "raw_completion": cleaned,
        }

    def run_incremental(self, question: str, dsl: str, **kwargs) -> Dict[str, object]:
        steps = parse_dsl(dsl)
        if not steps:
            return {
                "final_answer": "",
                "explanation": "",
                "intermediate_answers": {},
                "qa_chain": [],
                "aggregation_raw": "",
            }

        sub_system = create_incremental_subquestion_system_prompt(reference_date=self.reference_date)
        answers: Dict[str, str] = {}
        qa_chain: List[Dict[str, object]] = []

        overflow_detected = False
        overflow_message = ""

        for var_name, template in steps:
            placeholder_vars = re.findall(r"{(\w+)}", template)
            multi_answer_needed = any(
                (placeholder in answers and ";" in answers[placeholder])
                for placeholder in placeholder_vars
            )

            if multi_answer_needed:
                multi_vars: Dict[str, List[str]] = {}
                for placeholder in placeholder_vars:
                    if placeholder in answers:
                        try:
                            parts = split_multi_answers(answers[placeholder])
                        except ValueError as exc:
                            print(
                                f"⚠️ Too many items for placeholder '{placeholder}' "
                                f"in question '{template}'. Marking as unanswerable: {exc}"
                            )
                            overflow_message = (
                                "Suppressed multi-answer expansion: model produced more than 12 items."
                            )
                            answers[placeholder] = ""
                            for item in qa_chain:
                                if item.get("variable") == placeholder:
                                    item["answer"] = ""
                                    item["raw_completion"] = ""
                            overflow_detected = True
                            parts = []
                        if len(parts) > 1:
                            multi_vars[placeholder] = parts
                if not multi_vars:
                    multi_answer_needed = False

            if multi_answer_needed:
                first_multi_var = next(iter(multi_vars))
                individual_values = multi_vars[first_multi_var]
                resolved_questions: List[str] = []
                individual_answers: List[str] = []

                for value in individual_values:
                    temp_answers = dict(answers)
                    temp_answers[first_multi_var] = value
                    resolved = replace_placeholders(template, temp_answers)
                    sub_raw = self.model.generate_answer(
                        resolved,
                        system=sub_system,
                        user=resolved,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        top_k=self.top_k,
                        **kwargs,
                    )
                    sub_clean = strip_think_tags(sub_raw)
                    if ";" in sub_clean:
                        try:
                            split_multi_answers(sub_clean)
                        except ValueError as exc:
                            print(
                                f"⚠️ Subquestion produced too many answers for '{resolved}'. "
                                f"Marking as empty: {exc}"
                            )
                            overflow_message = (
                                "Suppressed multi-answer expansion: model produced more than 12 items."
                            )
                            overflow_detected = True
                            sub_clean = ""
                    resolved_questions.append(resolved)
                    individual_answers.append(sub_clean)

                combined_answer = join_multi_answers(individual_answers) or ""
                answers[var_name] = combined_answer
                qa_chain.append(
                    {
                        "variable": var_name,
                        "question": template,
                        "answer": combined_answer,
                        "raw_completion": combined_answer,
                        "multi_answer": True,
                        "resolved_questions": resolved_questions,
                        "individual_answers": individual_answers,
                    }
                )
            else:
                resolved = replace_placeholders(template, answers)
                sub_raw = self.model.generate_answer(
                    resolved,
                    system=sub_system,
                    user=resolved,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    **kwargs,
                )
                sub_clean = strip_think_tags(sub_raw)
                if ";" in sub_clean:
                    try:
                        split_multi_answers(sub_clean)
                    except ValueError as exc:
                        print(
                            f"⚠️ Answer contained too many items for '{resolved}'. "
                            f"Marking as empty: {exc}"
                        )
                        overflow_message = (
                            "Suppressed multi-answer expansion: model produced more than 12 items."
                        )
                        overflow_detected = True
                        sub_clean = ""
                answers[var_name] = sub_clean
                qa_chain.append(
                    {
                        "variable": var_name,
                        "question": resolved,
                        "answer": sub_clean,
                        "raw_completion": sub_clean,
                        "multi_answer": False,
                    }
                )

        reasoning_lines = []
        for item in qa_chain:
            if item.get("multi_answer"):
                reasoning_lines.append(f"- {item['question']} → {item['answer']} (multi-answer)")
            else:
                reasoning_lines.append(f"- {item['question']} → {item['answer']}")
        if overflow_detected:
            return {
                "final_answer": "",
                "explanation": overflow_message,
                "intermediate_answers": answers,
                "qa_chain": qa_chain,
                "aggregation_raw": "",
            }

        aggregation_prompt = (
            "Based on the following reasoning chain:\n\n"
            + "\n".join(reasoning_lines)
            + f"\n\nNow answer the original question: {question}\n"
            + "Return only the final answer. Do not include any explanation."
        )

        agg_system = create_incremental_aggregation_system_prompt(reference_date=self.reference_date)
        aggregation_raw = self.model.generate_answer(
            aggregation_prompt,
            system=agg_system,
            user=aggregation_prompt,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            **kwargs,
        )
        aggregation_clean = strip_think_tags(aggregation_raw)
        final_answer, final_explanation = parse_final_response(aggregation_clean)

        return {
            "final_answer": (final_answer or extract_final_answer(answers)).strip(),
            "explanation": final_explanation,
            "intermediate_answers": answers,
            "qa_chain": qa_chain,
            "aggregation_raw": aggregation_clean,
        }

    def run_model_generated(self, question: str, **kwargs) -> Dict[str, object]:
        if not self.selfask_agent:
            raise RuntimeError("SelfAsk agent not provided but model_generated regime requested.")
        chain, final_answer = self.selfask_agent.generate_full_decomposition(
            question,
            usage_meta=kwargs.get("usage_meta"),
        )
        qa_chain = [
            {"step": step.step_number if hasattr(step, "step_number") else idx + 1, "question": step.question, "answer": step.answer}
            for idx, step in enumerate(chain)
        ]
        return {"final_answer": final_answer, "qa_chain": qa_chain}


@dataclass
class HotpotQAGenerationConfig:
    model: BaseModel
    model_name: str
    regimes: Sequence[str]
    dataset_file: Path = DEFAULT_DATASET_PATH
    limit: Optional[int] = None
    resume: bool = False
    output_dir: Path = DEFAULT_RESULTS_ROOT
    checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL
    combined_output_path: Optional[Path] = None
    reference_date: str = REFERENCE_DATE
    temperature: float = TEMPERATURE
    top_p: float = TOP_P
    top_k: int = TOP_K
    selfask_agent: Optional[object] = None
    regime_output_paths: Optional[Dict[str, Path]] = None
    examples: Optional[List[Dict[str, object]]] = None
    skip_batch: bool = False
    batch_only_ids: Optional[List[str]] = None


class HotpotQAGenerationPipeline:
    """Coordinate dataset iteration, regime execution, and result persistence."""

    def __init__(self, config: HotpotQAGenerationConfig):
        self.config = config
        ordered = []
        for regime in REGIME_ORDER:
            if regime in config.regimes:
                ordered.append(regime)
        extras = [r for r in config.regimes if r not in ordered]
        self.regimes: List[str] = ordered + extras
        if not self.regimes:
            raise ValueError("At least one regime must be specified.")

        self.runner = HotpotQARegimeRunner(
            config.model,
            reference_date=config.reference_date,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            selfask_agent=config.selfask_agent,
        )

        self.output_dir = config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.combined_output_path = config.combined_output_path
        if self.combined_output_path:
            self.combined_output_path.parent.mkdir(parents=True, exist_ok=True)
            if self.combined_output_path.exists() and not config.resume:
                self.combined_output_path.unlink()

        self.regime_records: Dict[str, List[Dict[str, object]]] = {}
        self._load_existing_records()
        self.processed_ids = self._load_processed_ids()

    # ------------------------------------------------------------------ helpers
    def _load_existing_records(self) -> None:
        for regime in self.regimes:
            path = self._regime_path(regime)
            if self.config.resume and path.exists():
                try:
                    data = json.loads(path.read_text())
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Failed to parse existing regime file {path}: {exc}") from exc
                if not isinstance(data, list):
                    raise ValueError(f"Expected list data in {path}, found {type(data).__name__}")
                self.regime_records[regime] = data
            else:
                self.regime_records[regime] = []

    def _load_processed_ids(self) -> set[str]:
        if not self.config.resume:
            return set()
        # If skipping batch on GPT-5.1, only consider regimes that will run now.
        if self.config.skip_batch and isinstance(self.config.model, GPT51Model):
            regimes_to_check = [r for r in self.regimes if r not in ("open_ended", "assistive")]
            intersection: Optional[set[str]] = None
            for regime in regimes_to_check:
                records = self.regime_records.get(regime, [])
                current_ids = {str(item.get("id")) for item in records if item.get("id") is not None}
                if intersection is None:
                    intersection = current_ids
                else:
                    intersection &= current_ids
            return intersection or set()

        if self.combined_output_path and self.combined_output_path.exists():
            ids: set[str] = set()
            with self.combined_output_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    record_id = record.get("id")
                    if record_id:
                        ids.add(str(record_id))
            if ids:
                return ids

        intersection: Optional[set[str]] = None
        for records in self.regime_records.values():
            current_ids = {str(item.get("id")) for item in records if item.get("id") is not None}
            if intersection is None:
                intersection = current_ids
            else:
                intersection &= current_ids
        return intersection or set()

    def _upsert_regime_record(self, regime: str, record: Dict[str, object]) -> None:
        records = self.regime_records.setdefault(regime, [])
        record_id = str(record.get("id"))
        for idx, existing in enumerate(records):
            if str(existing.get("id")) == record_id:
                records[idx] = record
                return
        records.append(record)

    # ------------------------------------------------------------------- pipeline
    def run(self) -> int:
        examples = self.config.examples or load_examples(self.config.dataset_file, self.config.limit)
        if not examples:
            raise ValueError(f"No examples found in {self.config.dataset_file}")

        example_lookup = {
            str(example.get("id", f"example_{idx}")): example
            for idx, example in enumerate(examples, start=1)
        }

        processed_ids = self.processed_ids
        resume_offset = len(processed_ids)

        total_examples = len(examples)
        new_records = 0

        # ------------------------------------------------------------------
        # For GPT-5.1: run open_ended + assistive via Batch (independent).
        # Sequential (incremental/model_generated) remains synchronous.
        # ------------------------------------------------------------------
        batch_results: Dict[str, str] = {}
        is_gpt5 = isinstance(self.config.model, GPT51Model)
        batch_only_ids = [str(x) for x in (self.config.batch_only_ids or [])]
        open_lookup = {str(item.get("id")): item for item in self.regime_records.get("open_ended", [])}
        assist_lookup = {str(item.get("id")): item for item in self.regime_records.get("assistive", [])}
        if is_gpt5 and not getattr(self.config, "skip_batch", False):
            batch_lines: List[str] = []
            batch_examples = examples
            if batch_only_ids:
                batch_examples = [example_lookup[eid] for eid in batch_only_ids if eid in example_lookup]
            for idx, example in enumerate(batch_examples, start=1):
                example_id = str(example.get("id", f"example_{idx}"))
                if not batch_only_ids and example_id in processed_ids:
                    continue
                question = str(example.get("question", "")).strip()
                dsl = str(example.get("dsl", ""))

                open_system = create_open_ended_system_prompt(reference_date=self.config.reference_date)
                open_user = create_open_ended_user_prompt(question)
                batch_lines.append(
                    build_gpt51_batch_line(
                        custom_id=f"{example_id}::open_ended",
                        question=question,
                        reasoning_effort="medium",
                        max_completion_tokens=2048,
                        messages=[
                            {"role": "system", "content": open_system},
                            {"role": "user", "content": open_user},
                        ],
                    )
                )

                if dsl.strip():
                    assist_system = create_assistive_system_prompt(reference_date=self.config.reference_date)
                    assist_user = create_assistive_user_prompt(dsl)
                    batch_lines.append(
                        build_gpt51_batch_line(
                            custom_id=f"{example_id}::assistive",
                            question=dsl,
                            reasoning_effort="medium",
                            max_completion_tokens=2048,
                            messages=[
                                {"role": "system", "content": assist_system},
                                {"role": "user", "content": assist_user},
                            ],
                        )
                    )

            if batch_lines:
                batch_filename = f"gpt5_batch_input_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.jsonl"
                batch_path = self.output_dir / batch_filename
                batch_path.write_text("\n".join(batch_lines), encoding="utf-8")
                print(f"GPT-5.1 batch input saved to {batch_path}")

                client = self.config.model.client  # type: ignore[attr-defined]
                input_file = client.files.create(file=open(batch_path, "rb"), purpose="batch")
                batch_job = self.config.model.create_batch_job(input_file_id=input_file.id)  # type: ignore[attr-defined]
                print(f"Submitted GPT-5.1 batch job {batch_job.id} with {len(batch_lines)} requests")

                import time
                while True:
                    status = client.batches.retrieve(batch_job.id)
                    if status.status in ("completed", "failed", "cancelled"):
                        print(f"Batch status: {status.status}")
                        break
                    time.sleep(5)

                if status.status == "completed":
                    out_file = client.files.content(status.output_file_id)
                    out_name = f"gpt5_batch_output_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.jsonl"
                    out_path = self.output_dir / out_name
                    out_path.write_text(out_file.text, encoding="utf-8")
                    print(f"GPT-5.1 batch output saved to {out_path}")

                    detailed = parse_gpt51_batch_output_with_usage(out_file.text)
                    for cid, payload in detailed.items():
                        usage = payload.get("usage", {}) or {}
                        ex_id, reg = (cid.split("::", 1) + [""])[:2]
                        _log_usage(
                            model_name=payload.get("model", self.config.model_name),
                            prompt_tokens=usage.get("prompt_tokens"),
                            completion_tokens=usage.get("completion_tokens"),
                            total_tokens=usage.get("total_tokens"),
                            meta={"context": "generation-batch", "regime": reg, "example_id": ex_id},
                        )
                    batch_results = {cid: payload.get("content", "") for cid, payload in detailed.items()}
                else:
                    print("⚠️ Batch did not complete successfully; falling back to empty batch results.")

            if batch_only_ids and batch_results:
                timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                for example_id in batch_only_ids:
                    if example_id not in example_lookup:
                        continue
                    example = example_lookup[example_id]
                    question = str(example.get("question", ""))
                    dsl = str(example.get("dsl", ""))
                    expected_answer = example.get("answer", "")
                    # open_ended
                    open_raw = batch_results.get(f"{example_id}::open_ended", "")
                    open_clean = strip_think_tags(open_raw)
                    open_answer = parse_open_ended_response(open_clean)
                    self._upsert_regime_record(
                        "open_ended",
                        {
                            "id": example_id,
                            "question": question,
                            "expected_answer": expected_answer,
                            "answer": open_answer.strip(),
                            "raw_response": open_clean,
                            "success": bool(open_answer),
                            "model": self.config.model_name,
                            "timestamp": timestamp,
                        },
                    )
                    # assistive
                    assist_raw = batch_results.get(f"{example_id}::assistive", "")
                    assist_clean = strip_think_tags(assist_raw)
                    explanation, intermediate, final_answer = parse_assistive_response(assist_clean)
                    steps = parse_dsl(dsl) if dsl else []
                    intermediate_questions = [template for _, template in steps]
                    self._upsert_regime_record(
                        "assistive",
                        {
                            "id": example_id,
                            "model": self.config.model_name,
                            "question": question,
                            "dsl_string": dsl,
                            "final_answer": final_answer.strip(),
                            "explanation": explanation,
                            "intermediate_answers": intermediate,
                            "gold_intermediate_answers": [],
                            "raw_response": assist_clean,
                            "success": bool(final_answer),
                            "timestamp": timestamp,
                            "intermediate_questions": intermediate_questions,
                        },
                    )
                self._flush_regime_files()
                return len(batch_only_ids)
        for idx, example in enumerate(examples, start=1):
            question = str(example.get("question", "")).strip()
            dsl = str(example.get("dsl", ""))
            example_id = str(example.get("id", f"example_{idx}"))
            if example_id in processed_ids:
                continue

            progress_index = resume_offset + new_records + 1
            print(f"\n[{progress_index}/{total_examples}] {example_id}: {question}")

            timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            steps = parse_dsl(dsl) if dsl else []
            intermediate_questions = [template for _, template in steps]

            regime_payloads: Dict[str, Dict[str, object]] = {}
            for regime in self.regimes:
                if is_gpt5 and getattr(self.config, "skip_batch", False) and regime in ("open_ended", "assistive"):
                    # Skip batch-backed regimes when resuming sync-only.
                    continue
                if regime == "open_ended":
                    if is_gpt5 and batch_results:
                        raw = batch_results.get(f"{example_id}::open_ended", "")
                        cleaned = strip_think_tags(raw)
                        answer = parse_open_ended_response(cleaned)
                        regime_payloads[regime] = {"answer": answer, "raw_completion": cleaned}
                    else:
                        regime_payloads[regime] = self.runner.run_open_ended(
                            question,
                            usage_meta={"example_id": example_id, "regime": regime},
                        )
                elif regime == "assistive":
                    if is_gpt5 and batch_results:
                        raw = batch_results.get(f"{example_id}::assistive", "")
                        cleaned = strip_think_tags(raw)
                        explanation, intermediate, final_answer = parse_assistive_response(cleaned)
                        regime_payloads[regime] = {
                            "final_answer": final_answer,
                            "explanation": explanation,
                            "intermediate_answers": intermediate,
                            "raw_completion": cleaned,
                        }
                    else:
                        regime_payloads[regime] = self.runner.run_assistive(
                            dsl,
                            usage_meta={"example_id": example_id, "regime": regime},
                        )
                elif regime == "incremental":
                    regime_payloads[regime] = self.runner.run_incremental(
                        question,
                        dsl,
                        usage_meta={"example_id": example_id, "regime": regime},
                    )
                elif regime == "model_generated":
                    regime_payloads[regime] = self.runner.run_model_generated(
                        question,
                        usage_meta={"example_id": example_id, "regime": regime},
                    )
                else:
                    raise ValueError(f"Unsupported regime '{regime}' requested.")

            combined_payloads = dict(regime_payloads)
            if is_gpt5 and getattr(self.config, "skip_batch", False):
                if example_id in open_lookup:
                    open_record = open_lookup[example_id]
                    combined_payloads["open_ended"] = {
                        "answer": open_record.get("answer", ""),
                        "raw_completion": open_record.get("raw_response", ""),
                    }
                if example_id in assist_lookup:
                    assist_record = assist_lookup[example_id]
                    combined_payloads["assistive"] = {
                        "final_answer": assist_record.get("final_answer", ""),
                        "explanation": assist_record.get("explanation", ""),
                        "intermediate_answers": assist_record.get("intermediate_answers", {}),
                        "raw_completion": assist_record.get("raw_response", ""),
                    }

            append_payloads = regime_payloads
            if is_gpt5 and getattr(self.config, "skip_batch", False):
                append_payloads = {k: v for k, v in regime_payloads.items() if k not in ("open_ended", "assistive")}

            self._append_regime_records(
                example,
                append_payloads,
                timestamp=timestamp,
                intermediate_questions=intermediate_questions,
            )
            self._write_combined_record(example, combined_payloads)

            new_records += 1
            if new_records % self.config.checkpoint_interval == 0:
                self._flush_regime_files()

        self._flush_regime_files()
        if self.config.resume and new_records:
            print(f"Total records available after resume: {resume_offset + new_records}")
        return new_records

    # -------------------------------------------------------------- record utils
    def _append_regime_records(
        self,
        example: Dict[str, object],
        regime_payloads: Dict[str, Dict[str, object]],
        *,
        timestamp: str,
        intermediate_questions: List[str],
    ) -> None:
        example_id = str(example.get("id"))
        question = str(example.get("question", ""))
        dsl = str(example.get("dsl", ""))
        expected_answer = example.get("answer", "")

        if "open_ended" in regime_payloads:
            payload = regime_payloads["open_ended"]
            self.regime_records["open_ended"].append(
                {
                    "id": example_id,
                    "question": question,
                    "expected_answer": expected_answer,
                    "answer": payload.get("answer", "").strip(),
                    "raw_response": payload.get("raw_completion", ""),
                    "success": bool(payload.get("answer")),
                    "model": self.config.model_name,
                    "timestamp": timestamp,
                }
            )

        if "assistive" in regime_payloads:
            payload = regime_payloads["assistive"]
            self.regime_records["assistive"].append(
                {
                    "id": example_id,
                    "model": self.config.model_name,
                    "question": question,
                    "dsl_string": dsl,
                    "final_answer": payload.get("final_answer", "").strip(),
                    "explanation": payload.get("explanation", ""),
                    "intermediate_answers": payload.get("intermediate_answers", {}),
                    "gold_intermediate_answers": [],
                    "raw_response": payload.get("raw_completion", ""),
                    "success": bool(payload.get("final_answer")),
                    "timestamp": timestamp,
                    "intermediate_questions": intermediate_questions,
                }
            )

        if "incremental" in regime_payloads:
            payload = regime_payloads["incremental"]
            intermediate_answers_dict = payload.get("intermediate_answers", {})
            qa_chain_entries = []
            for item in payload.get("qa_chain", []):
                qa_chain_entries.append(
                    {
                        "question": item.get("question", ""),
                        "answer": item.get("answer", ""),
                        "variable": item.get("variable"),
                        "multi_answer": bool(item.get("multi_answer")),
                        "resolved_questions": item.get("resolved_questions", []),
                        "individual_answers": item.get("individual_answers", []),
                    }
                )
            self.regime_records["incremental"].append(
                {
                    "id": example_id,
                    "question": question,
                    "dsl_string": dsl,
                    "final_answer": payload.get("final_answer", "").strip(),
                    "intermediate_answers": intermediate_answers_dict,
                    "raw_response": payload.get("aggregation_raw", ""),
                    "qa_chain": qa_chain_entries,
                    "success": bool(payload.get("final_answer")),
                    "model": self.config.model_name,
                    "timestamp": timestamp,
                    "intermediate_questions": intermediate_questions,
                }
            )

        if "model_generated" in regime_payloads:
            payload = regime_payloads["model_generated"]
            self.regime_records["model_generated"].append(
                {
                    "id": example_id,
                    "question": question,
                    "final_answer": payload.get("final_answer", "").strip(),
                    "qa_chain": payload.get("qa_chain", []),
                    "model": self.config.model_name,
                    "timestamp": timestamp,
                    "success": bool(payload.get("final_answer")),
                }
            )

    def _write_combined_record(self, example: Dict[str, object], regime_payloads: Dict[str, Dict[str, object]]) -> None:
        if not self.combined_output_path:
            return
        record = {
            "id": example.get("id"),
            "question": example.get("question"),
            "gold_answer": example.get("answer"),
            "regimes": regime_payloads,
        }
        with self.combined_output_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _regime_path(self, regime: str) -> Path:
        overrides = self.config.regime_output_paths or {}
        path = overrides.get(regime)
        if path is not None:
            return Path(path)
        filename = REGIME_FILENAMES.get(regime)
        if not filename:
            raise ValueError(f"No filename registered for regime '{regime}'")
        return self.output_dir / filename

    def _flush_regime_files(self) -> None:
        for regime in self.regime_records:
            path = self._regime_path(regime)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as f:
                json.dump(self.regime_records[regime], f, indent=2)


__all__ = [
    "HotpotQAGenerationConfig",
    "HotpotQAGenerationPipeline",
    "HotpotQARegimeRunner",
    "load_examples",
    "parse_dsl",
    "strip_think_tags",
]
