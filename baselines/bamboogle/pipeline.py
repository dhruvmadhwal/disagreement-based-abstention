"""Pipeline for running Bamboogle baselines across the dataset."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import random
from typing import Dict, List, Optional, Sequence, Tuple

from generate.bamboogle.config import make_model_slug
from generate.bamboogle.pipeline import (
    REGIME_FILENAMES,
    load_examples,
    parse_open_ended_response,
    strip_think_tags,
)
from prompts.bamboogle.prompts import create_open_ended_system_prompt, create_open_ended_user_prompt
from prompts.consistency.bamboogle.comparison_prompt import (
    create_consistency_system_prompt,
    create_consistency_user_prompt,
)
from utils.model_interface import BaseModel, GPT51Model, build_gpt51_batch_line
from baselines.gpt_batch import run_gpt51_batch

from .config import (
    BASELINE_FILENAMES,
    BASELINE_RESULTS_ROOT,
    BASELINES,
    DEFAULT_BASELINES,
    DEFAULT_COMBINED_FILENAME,
    DEFAULT_DATASET_PATH,
    DEFAULT_IC_IDK_D,
    DEFAULT_IC_IDK_K,
    DEFAULT_SELF_CONSISTENCY_MIN_VOTES,
    DEFAULT_SELF_CONSISTENCY_SAMPLES,
    DEFAULT_SELF_CONSISTENCY_TEMPERATURE,
    DEFAULT_SELF_CONSISTENCY_ZERO_SAMPLES,
    DEFAULT_SELF_CONSISTENCY_ZERO_TEMPERATURE,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    DEFAULT_PAIRWISE_TEMPERATURE,
    DEFAULT_PAIRWISE_TOP_K,
    DEFAULT_PAIRWISE_TOP_P,
    IC_IDK_POOL_ROOT,
    PAIRWISE_TARGET_REGIMES,
    REFERENCE_DATE,
    ensure_baseline_dir,
)
from .judge import CorrectnessJudge, JudgeResult
from .prompts import (
    build_ays_user_prompt,
    build_idk_user_prompt,
)
from .utils import (
    DemoExample,
    build_ic_idk_demos,
    compute_metrics,
    extract_final_answer,
    is_idk,
    load_consistency_lookup,
    load_correctness_lookup,
    load_generation_lookup,
    normalize_text,
)


@dataclass
class BamboogleBaselineConfig:
    model: BaseModel
    model_name: str
    baselines: Sequence[str] = DEFAULT_BASELINES
    model_choice: Optional[str] = None
    dataset_file: Path = DEFAULT_DATASET_PATH
    limit: Optional[int] = None
    resume: bool = False
    output_dir: Path = BASELINE_RESULTS_ROOT
    combined_output_path: Optional[Path] = None
    reference_date: str = REFERENCE_DATE
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    top_k: int = DEFAULT_TOP_K
    self_consistency_samples: int = DEFAULT_SELF_CONSISTENCY_SAMPLES
    self_consistency_min_votes: int = DEFAULT_SELF_CONSISTENCY_MIN_VOTES
    self_consistency_temperature: float = DEFAULT_SELF_CONSISTENCY_TEMPERATURE
    self_consistency_zero_temperature: float = DEFAULT_SELF_CONSISTENCY_ZERO_TEMPERATURE
    self_consistency_zero_samples: int = DEFAULT_SELF_CONSISTENCY_ZERO_SAMPLES
    ic_idk_k: int = DEFAULT_IC_IDK_K
    ic_idk_d: int = DEFAULT_IC_IDK_D
    ic_idk_seed: int = 13
    ic_idk_prompt_file: Optional[Path] = None
    run_judge: bool = True
    judge: Optional[CorrectnessJudge] = None
    pairwise_temperature: float = DEFAULT_PAIRWISE_TEMPERATURE
    pairwise_top_p: float = DEFAULT_PAIRWISE_TOP_P
    pairwise_top_k: int = DEFAULT_PAIRWISE_TOP_K
    skip_batch: bool = False


def _parse_equivalence(text: str) -> Tuple[int, str]:
    equivalent = 0
    reasoning = "No reasoning provided"
    import re
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        lowered = line.lower()
        match = re.search(r"(equivalent|not meaningful)\s*[:=]\s*(-?1|0)", line, flags=re.IGNORECASE)
        if match:
            parsed = int(match.group(2))
            if match.group(1).lower().startswith("not meaningful"):
                parsed = -1
            if parsed in (-1, 0, 1):
                equivalent = parsed
            continue
        if "reasoning" in lowered or "explanation" in lowered:
            if ":" in line:
                extracted = line.split(":", 1)[1].strip()
                if extracted:
                    reasoning = extracted
    return equivalent, reasoning


class BamboogleBaselineRunner:
    """Implements each baseline using a shared model instance."""

    def __init__(self, model: BaseModel, *, reference_date: str = REFERENCE_DATE):
        self.model = model
        self.reference_date = reference_date

    def _sample_open_ended_answers(
        self,
        question: str,
        *,
        temperature_schedule: Sequence[float],
        top_p: float,
        top_k: int,
    ) -> Tuple[List[Dict[str, object]], List[str], List[str]]:
        """Sample answers for a question using the open-ended prompt."""
        system_prompt = create_open_ended_system_prompt(reference_date=self.reference_date)
        user_prompt = create_open_ended_user_prompt(question)
        samples: List[Dict[str, object]] = []
        raw_answers: List[str] = []
        normalized: List[str] = []

        temps = list(temperature_schedule) or [0.0]
        for idx, temp in enumerate(temps, 1):
            print(f"  [Sample {idx}/{len(temps)}] T={temp}")
            import time
            start_time = time.time()
            raw = self.model.generate_answer(
                question,
                system=system_prompt,
                user=user_prompt,
                temperature=temp,
                top_p=top_p,
                top_k=top_k,
            )
            elapsed = time.time() - start_time
            cleaned = strip_think_tags(raw)
            final = parse_open_ended_response(cleaned) or extract_final_answer(cleaned)
            print(f"  [Sample {idx}/{len(temps)}] Completed in {elapsed:.2f}s: '{final[:50] if final else '(empty)'}...'")
            raw_answers.append(final)
            normalized.append(normalize_text(final))
            samples.append({"raw_response": cleaned, "final_answer": final, "temperature": temp})

        return samples, raw_answers, normalized

    # --------------------------- individual baselines ---------------------------
    def run_ays(self, question: str, base_answer: str) -> Dict[str, object]:
        if not base_answer:
            return {
                "accepted": False,
                "prediction": "",
                "raw_response": "",
                "reason": "missing_base_answer",
            }
        system_prompt = "You are verifying whether a claim is correct. Respond with a single word: Yes or No."
        user_prompt = build_ays_user_prompt(question, base_answer)
        print(f"  [AYS] Calling model with question='{question[:50]}...', answer='{base_answer}'")
        import time
        start_time = time.time()
        raw = self.model.generate_answer(
            question,
            system=system_prompt,
            user=user_prompt,
            temperature=0.0,
            top_p=0.01,
            top_k=1,
        )
        elapsed = time.time() - start_time
        print(f"  [AYS] Model responded in {elapsed:.2f}s: '{raw[:100] if raw else '(empty)'}...'")
        cleaned = strip_think_tags(raw)
        decision_text = normalize_text(cleaned).lower()
        accepted = decision_text.startswith("yes")
        return {
            "accepted": bool(accepted),
            "prediction": base_answer.strip(),
            "raw_response": cleaned,
        }

    def run_idk(self, question: str, *, temperature: float = 0.0, top_p: float = 0.95, top_k: int = 64) -> Dict[str, object]:
        system_prompt = create_open_ended_system_prompt(reference_date=self.reference_date)
        user_prompt = build_idk_user_prompt(question)
        raw = self.model.generate_answer(
            question,
            system=system_prompt,
            user=user_prompt,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        cleaned = strip_think_tags(raw)
        prediction = parse_open_ended_response(cleaned)
        accepted = not is_idk(prediction)
        return {
            "accepted": bool(accepted),
            "prediction": prediction.strip(),
            "raw_response": cleaned,
        }

    def run_ic_idk(
        self,
        question: str,
        demos: Sequence[DemoExample],
        *,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 64,
    ) -> Dict[str, object]:
        demo_lines: List[str] = []
        k = len(demos)
        d = sum(1 for demo in demos if is_idk(demo.answer))
        demo_lines.append(
            f"You will see {k} demonstrations. {d} of them are labeled with \"I don't know\" (model failures), "
            f"and the remaining {max(0, k - d)} provide gold answers. Follow the same pattern: if unsure, reply exactly \"I don't know\" and do not guess."
        )
        demo_lines.append("")
        for demo in demos:
            demo_lines.append(f"Q: {demo.question}")
            demo_lines.append(f"A: {demo.answer}")
            demo_lines.append("")
        prompt = "\n".join(demo_lines) + f"\nQ: {question.strip()}\nA: "
        system_prompt = create_open_ended_system_prompt(reference_date=self.reference_date, include_examples=False)
        print(f"  [IC-IDK] Calling model with {len(demos)} demos, prompt length={len(prompt)} chars")
        import time
        start_time = time.time()
        raw = self.model.generate_answer(
            question,
            system=system_prompt,
            user=prompt,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        elapsed = time.time() - start_time
        print(f"  [IC-IDK] Model responded in {elapsed:.2f}s")
        cleaned = strip_think_tags(raw)
        # Mirror open-ended correctness extraction: take the parsed answer if present,
        # otherwise fall back to the cleaned text.
        parsed = parse_open_ended_response(cleaned)
        prediction = (parsed or extract_final_answer(cleaned) or cleaned).strip() or "I don't know"
        accepted = not is_idk(prediction)
        return {
            "accepted": bool(accepted),
            "prediction": prediction,
            "raw_response": cleaned,
            "demos_used": len(demos),
        }

    def run_self_consistency(
        self,
        question: str,
        *,
        n: int,
        min_votes: int,
        temperature: float,
        top_p: float,
        top_k: int,
        temperature_schedule: Optional[Sequence[float]] = None,
    ) -> Dict[str, object]:
        temps = list(temperature_schedule) if temperature_schedule else [temperature] * max(1, n)
        samples, _, normalized_answers = self._sample_open_ended_answers(
            question,
            temperature_schedule=temps,
            top_p=top_p,
            top_k=top_k,
        )
        idk_in_chain = any(is_idk(ans) or "don't know" in normalize_text(ans).lower() for ans in normalized_answers)

        # Let the generation model perform the majority vote over its own samples.
        filtered = [a for a in normalized_answers if not is_idk(a)]
        counts = Counter(filtered)
        prompt_lines: List[str] = [
            "You will pick the majority answer from the candidate list below.",
            f"Only choose an answer if it appears at least {min_votes} times. Otherwise, reply exactly \"I don't know\".",
            f"Question: {question.strip()}",
            "Candidates:",
        ]
        for idx, ans in enumerate(normalized_answers, start=1):
            fallback = ans if ans else "I don't know"
            prompt_lines.append(f"- {idx}. {fallback}")
        prompt_lines.append("Final answer (short):")
        agg_prompt = "\n".join(prompt_lines)
        agg_raw = self.model.generate_answer(
            question,
            system=None,
            user=agg_prompt,
            temperature=0.0,
            top_p=0.01,
            top_k=1,
        )
        final_answer = extract_final_answer(strip_think_tags(agg_raw)).strip() or "I don't know"
        accepted = not is_idk(final_answer) and "don't know" not in normalize_text(final_answer).lower() and not idk_in_chain
        return {
            "accepted": accepted,
            "prediction": final_answer.strip(),
            "raw_samples": samples,
            "vote_counts": dict(counts),
            "aggregator_raw_response": agg_raw,
            "temperature_schedule": temps,
        }

    def run_pairwise_consistency(
        self,
        question: str,
        base_answer: str,
        other_answer: str,
        *,
        comparison_regime: str,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> Dict[str, object]:
        if not base_answer or not other_answer:
            return {
                "accepted": False,
                "prediction": "",
                "raw_response": "",
                "reason": "missing_answer",
                "equivalent": 0,
                "comparison_regime": comparison_regime,
            }
        system_prompt = create_consistency_system_prompt()
        user_prompt = create_consistency_user_prompt(question, base_answer, other_answer)
        raw = self.model.generate_answer(
            question,
            system=system_prompt,
            user=user_prompt,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        cleaned = strip_think_tags(raw)
        equivalent, reasoning = _parse_equivalence(cleaned)
        accepted = equivalent == 1
        prediction = base_answer if accepted else "I don't know"
        return {
            "accepted": accepted,
            "prediction": prediction.strip(),
            "raw_response": cleaned,
            "equivalent": equivalent,
            "reasoning": reasoning,
            "comparison_regime": comparison_regime,
        }


class BamboogleBaselinePipeline:
    """Coordinate dataset iteration, baseline execution, and result persistence."""

    def __init__(self, config: BamboogleBaselineConfig):
        self.config = config
        ordered: List[str] = []
        for baseline in BASELINES:
            if baseline in config.baselines:
                ordered.append(baseline)
        extras = [b for b in config.baselines if b not in ordered]
        self.baselines: List[str] = ordered + extras
        if not self.baselines:
            raise ValueError("At least one baseline must be specified.")

        self.model_slug = make_model_slug(config.model_choice, config.model_name)
        if config.output_dir == BASELINE_RESULTS_ROOT:
            self.output_dir = ensure_baseline_dir(self.model_slug)
        else:
            self.output_dir = Path(config.output_dir) / self.model_slug
            self.output_dir.mkdir(parents=True, exist_ok=True)
        if config.combined_output_path is None:
            self.combined_output_path = self.output_dir / DEFAULT_COMBINED_FILENAME
        else:
            self.combined_output_path = config.combined_output_path
        if self.combined_output_path.exists() and not config.resume:
            self.combined_output_path.unlink()

        self.runner = BamboogleBaselineRunner(config.model, reference_date=config.reference_date)
        self.judge = config.judge or (CorrectnessJudge() if config.run_judge else None)

        self.baseline_records: Dict[str, List[Dict[str, object]]] = {b: [] for b in self.baselines}
        self._load_existing_records()

        self.examples = load_examples(config.dataset_file, config.limit)
        self.example_lookup = {str(ex.get("id")): ex for ex in self.examples if ex.get("id") is not None}
        # Use a dedicated IC-IDK prompt pool when provided (preferred to avoid reusing eval questions).
        default_pool = IC_IDK_POOL_ROOT / f"{self.model_slug}.json"
        ic_idk_source = config.ic_idk_prompt_file or (default_pool if default_pool.exists() else config.dataset_file)
        if config.ic_idk_prompt_file and not ic_idk_source.exists():
            raise FileNotFoundError(
                f"IC-IDK prompt file {ic_idk_source} not found. Provide 15 held-out examples outside the eval set."
            )
        if not config.ic_idk_prompt_file and not default_pool.exists():
            print(
                "IC-IDK prompt pool not found; defaulting to the evaluation dataset. "
                f"Provide a Mintaka pool at {default_pool} or override via --ic-idk-prompt-file."
            )
        if ic_idk_source.suffix.lower() == ".json":
            raw_demo_payload = json.loads(ic_idk_source.read_text())
            if isinstance(raw_demo_payload, dict) and "records" in raw_demo_payload:
                self.demo_examples = raw_demo_payload.get("records", [])
            else:
                self.demo_examples = raw_demo_payload if isinstance(raw_demo_payload, list) else []
        else:
            self.demo_examples = load_examples(ic_idk_source, None)
        self.generation_lookup = load_generation_lookup(
            self.model_slug,
            REGIME_FILENAMES.keys(),
        )
        self.correctness_lookup = load_correctness_lookup(self.model_slug)
        self.consistency_lookup = load_consistency_lookup(self.model_slug)
        self.ic_idk_demos: List[DemoExample] = build_ic_idk_demos(
            self.demo_examples,
            self.generation_lookup.get("open_ended", {}),
            self.correctness_lookup,
            k=config.ic_idk_k,
            d=config.ic_idk_d,
            seed=config.ic_idk_seed,
        )

        self.self_consistency_schedule = self._build_self_consistency_schedule()
        self.processed_ids = self._load_processed_ids()

    # ------------------------------------------------------------------- pipeline
    def run(self) -> int:
        if isinstance(self.config.model, GPT51Model) and not self.config.skip_batch:
            return self._run_gpt_batch()
        if not self.examples:
            raise ValueError(f"No examples found in {self.config.dataset_file}")

        # Always rebuild combined JSONL from scratch (individual JSON files are source of truth)
        if self.combined_output_path.exists():
            self.combined_output_path.unlink()

        new_records = 0
        reused_records = 0
        total = len(self.examples)
        for idx, example in enumerate(self.examples, start=1):
            example_id = str(example.get("id", f"example_{idx}"))
            question = str(example.get("question", "")).strip()
            gold = str(example.get("answer", "")).strip()
            timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

            # Count how many baselines need to be run vs reused
            baselines_to_run = [b for b in self.baselines if example_id not in self.baseline_id_lookup.get(b, {})]
            baselines_to_reuse = [b for b in self.baselines if example_id in self.baseline_id_lookup.get(b, {})]

            if baselines_to_run:
                print(f"\n[{new_records + 1}/{total}] {example_id}: {question}")
                print(f"  Running {len(baselines_to_run)} baseline(s): {', '.join(baselines_to_run)}")
                if baselines_to_reuse:
                    print(f"  Reusing {len(baselines_to_reuse)} baseline(s): {', '.join(baselines_to_reuse)}")

            baseline_payloads: Dict[str, Dict[str, object]] = {}
            skip_example = False
            for baseline in self.baselines:
                # Check if we already have a record for this baseline
                existing_record = self.baseline_id_lookup.get(baseline, {}).get(example_id)
                if existing_record is not None:
                    baseline_payloads[baseline] = existing_record
                    continue

                # Run the baseline
                payload = self._run_single_baseline(baseline, example_id, question, gold)
                if payload == "__SKIP__":
                    # Skip this baseline only (e.g., missing consistency cache)
                    continue
                if payload is None:
                    print(f"  [SKIPPING] Example {example_id} due to judge error - will retry on resume")
                    skip_example = True
                    break
                payload["timestamp"] = timestamp
                payload["baseline"] = baseline
                payload["id"] = example_id
                payload["question"] = question
                payload["gold_answer"] = gold
                baseline_payloads[baseline] = payload
                self.baseline_records[baseline].append(payload)
                # Add to lookup so we don't re-run if there's an error later
                self.baseline_id_lookup[baseline][example_id] = payload

            if skip_example:
                continue

            self._write_combined_record(example, baseline_payloads, timestamp=timestamp)
            if baselines_to_run:
                new_records += 1
            else:
                reused_records += 1

        self._write_baseline_files()
        print(f"\n[Summary] New: {new_records}, Reused: {reused_records}, Total: {new_records + reused_records}")
        return new_records

    # ------------------------------------------------------------ internal utils
    def _run_single_baseline(self, baseline: str, example_id: str, question: str, gold: str) -> Optional[Dict[str, object]]:
        print(f"  Running baseline: {baseline}")
        if baseline == "ays":
            if self.config.model is None:
                return "__SKIP__"
            base_answer = self._lookup_answer(example_id, regime="open_ended")
            result = self.runner.run_ays(question, base_answer)
        elif baseline == "idk":
            if self.config.model is None:
                return "__SKIP__"
            result = self.runner.run_idk(
                question,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
            )
        elif baseline == "ic_idk":
            if self.config.model is None:
                return "__SKIP__"
            demos = [demo for demo in self.ic_idk_demos if getattr(demo, "id", None) != example_id]
            rng = random.Random(f"{self.config.ic_idk_seed}-{example_id}")
            rng.shuffle(demos)
            if len(demos) > self.config.ic_idk_k:
                demos = demos[: self.config.ic_idk_k]
            result = self.runner.run_ic_idk(
                question,
                demos=demos,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
            )
        elif baseline == "self_consistency":
            if self.config.model is None:
                return "__SKIP__"
            result = self.runner.run_self_consistency(
                question,
                n=len(self.self_consistency_schedule),
                min_votes=self.config.self_consistency_min_votes,
                temperature=self.config.self_consistency_temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                temperature_schedule=self.self_consistency_schedule,
            )
        elif baseline.startswith("pairwise_"):
            target_regime = baseline.split("pairwise_", 1)[1]
            if target_regime not in PAIRWISE_TARGET_REGIMES:
                raise ValueError(f"Unsupported pairwise target regime '{target_regime}'.")
            base_answer = self._lookup_answer(example_id, regime="open_ended")
            # Require existing consistency results - skip if missing
            cached = self.consistency_lookup.get(target_regime, {}).get(example_id)
            if cached is None:
                print(f"  [SKIP] No consistency cache for {baseline} on {example_id}")
                return "__SKIP__"
            equivalent = cached.get("equivalent", 0)
            accepted = equivalent == 1
            result = {
                "accepted": accepted,
                "prediction": base_answer if accepted else "I don't know",
                "raw_response": cached.get("raw_response", "consistency_reused"),
                "equivalent": equivalent,
                "reasoning": cached.get("reasoning", "consistency_reused"),
                "comparison_regime": target_regime,
            }
        else:
            raise ValueError(f"Unsupported baseline '{baseline}' requested.")

        return self._apply_correctness(baseline, example_id, question, gold, result)

    def _apply_correctness(
        self,
        baseline: str,
        example_id: str,
        question: str,
        gold: str,
        result: Dict[str, object],
    ) -> Dict[str, object]:
        """Attach correctness fields to a baseline payload."""
        correctness_info = self.correctness_lookup.get(example_id, {}) or {}
        lookup_correct = None
        try:
            val = correctness_info.get("correct")
            if val is not None:
                lookup_correct = int(val)
        except Exception:
            lookup_correct = None
        if lookup_correct not in (-1, 0, 1):
            lookup_correct = None

        accepted = bool(result.get("accepted"))
        prediction = str(result.get("prediction") or "").strip()
        if baseline.startswith("pairwise_"):
            result["correct"] = lookup_correct if lookup_correct is not None else 0
            result["judge_reasoning"] = correctness_info.get("reasoning", "open_ended_correctness_reused")
            if accepted:
                result["judge_raw_response"] = "open_ended_correctness_reused"
        else:
            if lookup_correct is not None:
                result["correct"] = lookup_correct
                result["judge_reasoning"] = correctness_info.get("reasoning", "open_ended_correctness_reused")
                result["judge_raw_response"] = "open_ended_correctness_reused"
            elif self.judge:
                judge_result = self.judge.score(question, gold, prediction)
                if judge_result.error:
                    print(f"  [ERROR] Judge API failed: {judge_result.reasoning}")
                    return None  # Signal to skip this record
                result["correct"] = judge_result.correct
                result["judge_reasoning"] = judge_result.reasoning
                result["judge_raw_response"] = judge_result.raw_response
            else:
                result["correct"] = None
                result["judge_reasoning"] = "skipped"
            if not accepted and "judge_reasoning" not in result:
                result["judge_reasoning"] = "Rejected or abstained"
        return result

    def _run_gpt_batch(self) -> int:
        if not self.examples:
            raise ValueError(f"No examples found in {self.config.dataset_file}")

        pending_examples = [ex for ex in self.examples if str(ex.get("id")) not in self.processed_ids]
        if not pending_examples:
            return 0

        baseline_payloads: Dict[str, Dict[str, Dict[str, object]]] = {}
        for ex in pending_examples:
            baseline_payloads[str(ex.get("id"))] = {}

        batch_lines: List[str] = []
        sample_lines: List[str] = []

        open_system = create_open_ended_system_prompt(reference_date=self.config.reference_date)
        open_system_no_examples = create_open_ended_system_prompt(
            reference_date=self.config.reference_date,
            include_examples=False,
        )
        for example in pending_examples:
            example_id = str(example.get("id"))
            question = str(example.get("question", "")).strip()

            if "ays" in self.baselines:
                base_answer = self._lookup_answer(example_id, regime="open_ended")
                if base_answer:
                    system_prompt = "You are verifying whether a claim is correct. Respond with a single word: Yes or No."
                    user_prompt = build_ays_user_prompt(question, base_answer)
                    batch_lines.append(
                        build_gpt51_batch_line(
                            custom_id=f"{example_id}::ays",
                            question=question,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt},
                            ],
                        )
                    )
                else:
                    baseline_payloads[example_id]["ays"] = {
                        "accepted": False,
                        "prediction": "",
                        "raw_response": "",
                        "reason": "missing_base_answer",
                    }

            if "idk" in self.baselines:
                user_prompt = build_idk_user_prompt(question)
                batch_lines.append(
                    build_gpt51_batch_line(
                        custom_id=f"{example_id}::idk",
                        question=question,
                        messages=[
                            {"role": "system", "content": open_system},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                    )
                )

            if "ic_idk" in self.baselines:
                demos = [demo for demo in self.ic_idk_demos if getattr(demo, "id", None) != example_id]
                rng = random.Random(f"{self.config.ic_idk_seed}-{example_id}")
                rng.shuffle(demos)
                if len(demos) > self.config.ic_idk_k:
                    demos = demos[: self.config.ic_idk_k]
                demo_lines: List[str] = []
                k = len(demos)
                d = sum(1 for demo in demos if is_idk(demo.answer))
                demo_lines.append(
                    f"You will see {k} demonstrations. {d} of them are labeled with \"I don't know\" (model failures), "
                    f"and the remaining {max(0, k - d)} provide gold answers. Follow the same pattern: if unsure, reply exactly \"I don't know\" and do not guess."
                )
                demo_lines.append("")
                for demo in demos:
                    demo_lines.append(f"Q: {demo.question}")
                    demo_lines.append(f"A: {demo.answer}")
                    demo_lines.append("")
                prompt = "\n".join(demo_lines) + f"\nQ: {question}\nA: "
                batch_lines.append(
                    build_gpt51_batch_line(
                        custom_id=f"{example_id}::ic_idk",
                        question=question,
                        messages=[
                            {"role": "system", "content": open_system_no_examples},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                    )
                )

            for target_regime in PAIRWISE_TARGET_REGIMES:
                baseline_name = f"pairwise_{target_regime}"
                if baseline_name not in self.baselines:
                    continue
                base_answer = self._lookup_answer(example_id, regime="open_ended")
                other_answer = self._lookup_answer(example_id, regime=target_regime)
                if not base_answer or not other_answer:
                    baseline_payloads[example_id][baseline_name] = {
                        "accepted": False,
                        "prediction": "",
                        "raw_response": "",
                        "reason": "missing_answer",
                        "equivalent": 0,
                        "comparison_regime": target_regime,
                    }
                    continue
                system_prompt = create_consistency_system_prompt()
                user_prompt = create_consistency_user_prompt(question, base_answer, other_answer)
                batch_lines.append(
                    build_gpt51_batch_line(
                        custom_id=f"{example_id}::pairwise::{target_regime}",
                        question=question,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=self.config.pairwise_temperature,
                        top_p=self.config.pairwise_top_p,
                    )
                )

            if "self_consistency" in self.baselines:
                temps = self.self_consistency_schedule
                user_prompt = create_open_ended_user_prompt(question)
                for idx, temp in enumerate(temps, start=1):
                    sample_lines.append(
                        build_gpt51_batch_line(
                            custom_id=f"{example_id}::self_consistency_sample::{idx}",
                            question=question,
                            messages=[
                                {"role": "system", "content": open_system},
                                {"role": "user", "content": user_prompt},
                            ],
                            temperature=temp,
                            top_p=self.config.top_p,
                        )
                    )

        batch_results = run_gpt51_batch(
            model=self.config.model,
            lines=batch_lines,
            output_dir=self.output_dir,
            job_tag="baselines",
        )
        sample_results = run_gpt51_batch(
            model=self.config.model,
            lines=sample_lines,
            output_dir=self.output_dir,
            job_tag="self_consistency_samples",
        )

        agg_lines: List[str] = []
        if "self_consistency" in self.baselines:
            for example in pending_examples:
                example_id = str(example.get("id"))
                question = str(example.get("question", "")).strip()
                temps = self.self_consistency_schedule
                normalized_answers: List[str] = []
                for idx, _temp in enumerate(temps, start=1):
                    raw = sample_results.get(f"{example_id}::self_consistency_sample::{idx}", {}).get("content", "")
                    cleaned = strip_think_tags(raw)
                    parsed = parse_open_ended_response(cleaned)
                    final = (parsed or extract_final_answer(cleaned) or cleaned).strip()
                    normalized_answers.append(normalize_text(final))
                prompt_lines: List[str] = [
                    "You will pick the majority answer from the candidate list below.",
                    f"Only choose an answer if it appears at least {self.config.self_consistency_min_votes} times. Otherwise, reply exactly \"I don't know\".",
                    f"Question: {question.strip()}",
                    "Candidates:",
                ]
                for idx, ans in enumerate(normalized_answers, start=1):
                    fallback = ans if ans else "I don't know"
                    prompt_lines.append(f"- {idx}. {fallback}")
                prompt_lines.append("Final answer (short):")
                agg_prompt = "\n".join(prompt_lines)
                agg_lines.append(
                    build_gpt51_batch_line(
                        custom_id=f"{example_id}::self_consistency_agg",
                        question=question,
                        messages=[{"role": "user", "content": agg_prompt}],
                        temperature=0.0,
                        top_p=0.01,
                    )
                )

        agg_results = run_gpt51_batch(
            model=self.config.model,
            lines=agg_lines,
            output_dir=self.output_dir,
            job_tag="self_consistency_agg",
        )

        for example in pending_examples:
            example_id = str(example.get("id"))
            question = str(example.get("question", "")).strip()
            gold = str(example.get("answer", "")).strip()
            timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

            if "ays" in self.baselines and "ays" not in baseline_payloads[example_id]:
                raw = batch_results.get(f"{example_id}::ays", {}).get("content", "")
                cleaned = strip_think_tags(raw)
                decision_text = normalize_text(cleaned).lower()
                accepted = decision_text.startswith("yes")
                result = {
                    "accepted": bool(accepted),
                    "prediction": self._lookup_answer(example_id, regime="open_ended").strip(),
                    "raw_response": cleaned,
                }
                baseline_payloads[example_id]["ays"] = self._apply_correctness(
                    "ays", example_id, question, gold, result
                )

            if "idk" in self.baselines:
                raw = batch_results.get(f"{example_id}::idk", {}).get("content", "")
                cleaned = strip_think_tags(raw)
                prediction = parse_open_ended_response(cleaned)
                accepted = not is_idk(prediction)
                result = {
                    "accepted": bool(accepted),
                    "prediction": prediction.strip() if prediction else "",
                    "raw_response": cleaned,
                }
                baseline_payloads[example_id]["idk"] = self._apply_correctness(
                    "idk", example_id, question, gold, result
                )

            if "ic_idk" in self.baselines:
                raw = batch_results.get(f"{example_id}::ic_idk", {}).get("content", "")
                cleaned = strip_think_tags(raw)
                parsed = parse_open_ended_response(cleaned)
                prediction = (parsed or extract_final_answer(cleaned) or cleaned).strip() or "I don't know"
                accepted = not is_idk(prediction)
                result = {
                    "accepted": bool(accepted),
                    "prediction": prediction,
                    "raw_response": cleaned,
                    "demos_used": self.config.ic_idk_k,
                }
                baseline_payloads[example_id]["ic_idk"] = self._apply_correctness(
                    "ic_idk", example_id, question, gold, result
                )

            for target_regime in PAIRWISE_TARGET_REGIMES:
                baseline_name = f"pairwise_{target_regime}"
                if baseline_name not in self.baselines:
                    continue
                if baseline_name in baseline_payloads[example_id]:
                    continue
                raw = batch_results.get(f"{example_id}::pairwise::{target_regime}", {}).get("content", "")
                cleaned = strip_think_tags(raw)
                equivalent, reasoning = _parse_equivalence(cleaned)
                accepted = equivalent == 1
                prediction = self._lookup_answer(example_id, regime="open_ended") if accepted else "I don't know"
                result = {
                    "accepted": accepted,
                    "prediction": prediction.strip() if prediction else "",
                    "raw_response": cleaned,
                    "equivalent": equivalent,
                    "reasoning": reasoning,
                    "comparison_regime": target_regime,
                }
                baseline_payloads[example_id][baseline_name] = self._apply_correctness(
                    baseline_name, example_id, question, gold, result
                )

            if "self_consistency" in self.baselines:
                samples: List[Dict[str, object]] = []
                normalized_answers: List[str] = []
                temps = self.self_consistency_schedule
                for idx, temp in enumerate(temps, start=1):
                    raw = sample_results.get(f"{example_id}::self_consistency_sample::{idx}", {}).get("content", "")
                    cleaned = strip_think_tags(raw)
                    parsed = parse_open_ended_response(cleaned)
                    final = (parsed or extract_final_answer(cleaned) or cleaned).strip()
                    samples.append({"raw_response": cleaned, "final_answer": final, "temperature": temp})
                    normalized_answers.append(normalize_text(final))

                idk_in_chain = any(is_idk(ans) or "don't know" in normalize_text(ans).lower() for ans in normalized_answers)
                filtered = [a for a in normalized_answers if not is_idk(a)]
                counts = Counter(filtered)
                agg_raw = agg_results.get(f"{example_id}::self_consistency_agg", {}).get("content", "")
                final_answer = extract_final_answer(strip_think_tags(agg_raw)).strip() or "I don't know"
                accepted = (
                    not is_idk(final_answer)
                    and "don't know" not in normalize_text(final_answer).lower()
                    and not idk_in_chain
                )
                result = {
                    "accepted": accepted,
                    "prediction": final_answer.strip(),
                    "raw_samples": samples,
                    "vote_counts": dict(counts),
                    "aggregator_raw_response": agg_raw,
                    "temperature_schedule": temps,
                }
                baseline_payloads[example_id]["self_consistency"] = self._apply_correctness(
                    "self_consistency", example_id, question, gold, result
                )

            for baseline, payload in list(baseline_payloads[example_id].items()):
                if "correct" not in payload:
                    baseline_payloads[example_id][baseline] = self._apply_correctness(
                        baseline, example_id, question, gold, payload
                    )

            for baseline, payload in baseline_payloads[example_id].items():
                payload["timestamp"] = timestamp
                payload["baseline"] = baseline
                payload["id"] = example_id
                payload["question"] = question
                payload["gold_answer"] = gold
                self.baseline_records[baseline].append(payload)

            self._write_combined_record(example, baseline_payloads[example_id], timestamp=timestamp)

        self._write_baseline_files()
        return len(pending_examples)

    def _lookup_answer(self, example_id: str, *, regime: str) -> str:
        lookup = self.generation_lookup.get(regime, {})
        entry = lookup.get(example_id, {})
        if not entry:
            return ""
        return (
            entry.get("final_answer")
            or entry.get("answer")
            or entry.get("prediction")
            or entry.get("prediction_1")
            or entry.get("prediction_2")
            or ""
        )

    def _write_combined_record(self, example: Dict[str, object], baseline_payloads: Dict[str, Dict[str, object]], *, timestamp: str) -> None:
        """Append a flattened JSONL row for the unified combined file."""
        row = {"id": example.get("id"), "question": example.get("question"), "gold_answer": example.get("answer"), "timestamp": timestamp, "baselines": {}}
        for name, payload in baseline_payloads.items():
            row["baselines"][name] = payload
        with self.combined_output_path.open("a") as f:
            f.write(json.dumps(row))
            f.write("\n")

    def _write_baseline_files(self) -> None:
        for baseline, records in self.baseline_records.items():
            summary = compute_metrics(records)
            payload = {"summary": summary, "records": records}
            path = self.output_dir / BASELINE_FILENAMES.get(baseline, f"{baseline}.json")
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

    def _load_existing_records(self) -> None:
        """Load existing records from individual baseline JSON files and build lookup."""
        self.baseline_id_lookup: Dict[str, Dict[str, Dict[str, object]]] = {b: {} for b in self.baselines}
        if not self.config.resume:
            return
        for baseline in self.baselines:
            path = self.output_dir / BASELINE_FILENAMES.get(baseline, f"{baseline}.json")
            if not path.exists():
                continue
            try:
                payload = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue
            records = payload.get("records") or []
            if isinstance(records, list):
                self.baseline_records[baseline].extend(records)
                # Build lookup by example_id for per-baseline resume
                for record in records:
                    rid = str(record.get("id", ""))
                    if rid:
                        self.baseline_id_lookup[baseline][rid] = record

    def _load_processed_ids(self) -> set[str]:
        ids = set()
        if self.combined_output_path.exists() and self.config.resume:
            with self.combined_output_path.open() as f:
                for line in f:
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    ex_id = str(payload.get("id"))
                    if ex_id:
                        ids.add(ex_id)
        return ids

    def _build_self_consistency_schedule(self) -> List[float]:
        """Create the per-sample temperature schedule for self-consistency."""
        temps: List[float] = []
        zero_samples = max(0, self.config.self_consistency_zero_samples)
        temps.extend([self.config.self_consistency_zero_temperature] * zero_samples)
        remaining = max(0, self.config.self_consistency_samples - len(temps))
        temps.extend([self.config.self_consistency_temperature] * remaining)
        return temps


__all__ = [
    "BamboogleBaselineConfig",
    "BamboogleBaselinePipeline",
    "BamboogleBaselineRunner",
]
