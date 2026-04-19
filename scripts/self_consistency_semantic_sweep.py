#!/usr/bin/env python3
"""
Post-hoc semantic self-consistency analysis (reviewer W2).

Given existing self-consistency baseline outputs (7 samples per question),
cluster the sampled answers by semantic equivalence using a Gemini judge,
then:
  - define m = max cluster size among the samples
  - treat score_error = n_samples - m as a classifier score (higher => more likely incorrect)
  - compute proper AUROC (threshold-agnostic)
  - sweep thresholds over m to pick best F1 and report the fixed 4/7 point

Default framing matches the repo baseline framing: "positive" = the direct/open-ended answer is incorrect.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openai import OpenAI  # noqa: E402

from evaluation.correctness.bamboogle_correctness import build_vertex_client  # noqa: E402
from prompts.consistency.bamboogle.comparison_prompt import (  # noqa: E402
    create_consistency_system_prompt as create_bamboogle_consistency_system_prompt,
    create_consistency_user_prompt as create_bamboogle_consistency_user_prompt,
)
from prompts.consistency.mintaka.comparison_prompt import (  # noqa: E402
    create_consistency_system_prompt as create_mintaka_consistency_system_prompt,
    create_consistency_user_prompt as create_mintaka_consistency_user_prompt,
)


IDK_TOKENS = (
    "i don't know",
    "i do not know",
    "i dont know",
    "idk",
)


def normalize_for_key(text: str) -> str:
    """Normalize answer text for cache keys and deduping."""
    if text is None:
        return ""
    cleaned = str(text).strip().lower()
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.strip(" \n\t\r.,;:!?\"'`")
    return cleaned


def is_idk(text: Optional[str]) -> bool:
    if text is None:
        return False
    lowered = normalize_for_key(text)
    return any(lowered == token or lowered.startswith(f"{token}.") or lowered.startswith(f"{token},") for token in IDK_TOKENS)


def parse_equivalence(text: str) -> Tuple[int, str]:
    """Parse the 2-line equivalence format used across consistency prompts."""
    equivalent = 0
    reasoning = ""
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        lowered = line.lower()
        match = re.search(r"(equivalent|not meaningful)\s*[:=]\s*(-?1|0|1)", line, flags=re.IGNORECASE)
        if match:
            parsed = int(match.group(2))
            if match.group(1).lower().startswith("not meaningful"):
                parsed = -1
            if parsed in (-1, 0, 1):
                equivalent = parsed
            continue
        if ("reasoning" in lowered or "explanation" in lowered) and ":" in line:
            reasoning = line.split(":", 1)[1].strip()
    return equivalent, reasoning or "No reasoning provided"


class UnionFind:
    def __init__(self, items: Sequence[str]):
        self.parent: Dict[str, str] = {item: item for item in items}
        self.rank: Dict[str, int] = {item: 0 for item in items}

    def find(self, item: str) -> str:
        p = self.parent.get(item, item)
        if p != item:
            self.parent[item] = self.find(p)
        return self.parent.get(item, item)

    def union(self, a: str, b: str) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        rank_a = self.rank.get(ra, 0)
        rank_b = self.rank.get(rb, 0)
        if rank_a < rank_b:
            self.parent[ra] = rb
        elif rank_a > rank_b:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] = rank_a + 1


def auroc_rank(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Compute ROC AUC via rank statistics (Mann-Whitney U), tie-aware."""
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length")
    n = len(labels)
    if n == 0:
        return float("nan")
    n_pos = sum(1 for y in labels if y == 1)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    indexed = list(enumerate(scores))
    indexed.sort(key=lambda x: x[1])
    ranks = [0.0] * n

    i = 0
    rank = 1
    while i < n:
        j = i
        while j + 1 < n and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (rank + (rank + (j - i))) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        rank += (j - i + 1)
        i = j + 1

    rank_sum_pos = sum(ranks[idx] for idx, y in enumerate(labels) if y == 1)
    u_pos = rank_sum_pos - (n_pos * (n_pos + 1)) / 2.0
    return float(u_pos / (n_pos * n_neg))


def compute_detection_metrics(tp: int, fp: int, tn: int, fn: int) -> Dict[str, float]:
    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    coverage = (tn + fn) / total if total else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "coverage": coverage,
        "total": total,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


@dataclass(frozen=True)
class JudgeConfig:
    model: str
    temperature: float
    include_fewshots: bool


class EquivalenceJudge:
    def __init__(
        self,
        client: OpenAI,
        config: JudgeConfig,
        *,
        create_system_prompt: Callable[[bool], str],
        create_user_prompt: Callable[[str | None, str, str], str],
    ):
        self.client = client
        self.config = config
        self.system_prompt = create_system_prompt(config.include_fewshots)
        self.create_user_prompt = create_user_prompt

    def equivalent(self, *, question: str, answer_a: str, answer_b: str) -> Tuple[int, str, str]:
        user_prompt = self.create_user_prompt(question, answer_a, answer_b)
        response = self.client.chat.completions.create(
            model=self.config.model,
            temperature=float(self.config.temperature),
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        raw = response.choices[0].message.content or ""
        eq, reasoning = parse_equivalence(raw)
        return eq, reasoning, raw.strip()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def iter_self_consistency_records(path: Path) -> List[dict]:
    payload = load_json(path)
    records = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError(f"Unexpected payload format in {path}")
    return records


def cache_key(*, dataset: str, example_id: str, judge_model: str, a_key: str, b_key: str) -> str:
    left, right = sorted([a_key, b_key])
    return f"{dataset}|{example_id}|{judge_model}|{left}|{right}"


def load_cache(path: Path) -> Dict[str, int]:
    if not path.exists():
        return {}
    cached: Dict[str, int] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = obj.get("key")
            val = obj.get("equivalent")
            if isinstance(key, str) and isinstance(val, int):
                cached[key] = int(val)
    return cached


def append_cache(path: Path, *, key: str, equivalent: int, reasoning: str, raw: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "key": key,
        "equivalent": int(equivalent),
        "reasoning": reasoning,
        "raw_response": raw,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")


def semantic_max_cluster_size(
    *,
    dataset: str,
    example_id: str,
    question: str,
    answers: Sequence[str],
    judge: EquivalenceJudge,
    cache: Dict[str, int],
    cache_path: Optional[Path],
) -> int:
    # Exclude IDK-like answers
    non_idk = [a for a in answers if a and not is_idk(a)]
    if not non_idk:
        return 0

    # Dedup by normalized key; keep counts.
    counts: Dict[str, int] = {}
    reps: Dict[str, str] = {}
    for ans in non_idk:
        k = normalize_for_key(ans)
        if not k:
            continue
        counts[k] = counts.get(k, 0) + 1
        if k not in reps:
            reps[k] = ans
    keys = list(reps.keys())
    if not keys:
        return 0
    if len(keys) == 1:
        return counts[keys[0]]

    uf = UnionFind(keys)
    judge_model = judge.config.model
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a_key = keys[i]
            b_key = keys[j]
            if a_key == b_key:
                uf.union(a_key, b_key)
                continue
            k = cache_key(
                dataset=dataset,
                example_id=example_id,
                judge_model=judge_model,
                a_key=a_key,
                b_key=b_key,
            )
            eq_val = cache.get(k)
            if eq_val is None:
                try:
                    eq_val, reasoning, raw = judge.equivalent(
                        question=question,
                        answer_a=reps[a_key],
                        answer_b=reps[b_key],
                    )
                except Exception as exc:
                    # On API failures, default to non-equivalent (conservative for clustering).
                    eq_val, reasoning, raw = 0, f"API error: {exc}", str(exc)
                cache[k] = int(eq_val)
                if cache_path is not None:
                    append_cache(cache_path, key=k, equivalent=int(eq_val), reasoning=reasoning, raw=raw)
            if int(eq_val) == 1:
                uf.union(a_key, b_key)

    cluster_sizes: Dict[str, int] = {}
    for k in keys:
        root = uf.find(k)
        cluster_sizes[root] = cluster_sizes.get(root, 0) + counts.get(k, 0)
    return max(cluster_sizes.values()) if cluster_sizes else 0


def compute_for_threshold(records: Sequence[dict], *, t: int, m_by_id: Dict[str, int]) -> Dict[str, float]:
    tp = fp = tn = fn = 0
    for rec in records:
        example_id = str(rec.get("id"))
        m = int(m_by_id.get(example_id, 0))
        accepted = m >= t
        correct_raw = rec.get("correct")
        try:
            correct = int(correct_raw)
        except Exception:
            continue  # skip unlabeled
        is_correct = correct == 1
        if accepted:
            if is_correct:
                tn += 1
            else:
                fn += 1
        else:
            if is_correct:
                fp += 1
            else:
                tp += 1
    return compute_detection_metrics(tp, fp, tn, fn)


def dataset_prompt_fns(dataset: str) -> Tuple[Callable[[bool], str], Callable[[str | None, str, str], str]]:
    if dataset == "bamboogle":
        return create_bamboogle_consistency_system_prompt, create_bamboogle_consistency_user_prompt
    if dataset == "mintaka":
        return create_mintaka_consistency_system_prompt, create_mintaka_consistency_user_prompt
    raise ValueError(f"Unsupported dataset: {dataset}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Semantic self-consistency AUROC + threshold sweep (reviewer W2).")
    parser.add_argument(
        "--dataset",
        default="bamboogle",
        choices=["bamboogle", "mintaka"],
        help="Dataset key.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "qwen-qwen3-8b",
            "qwen-qwen3-32b",
            "meta-llama-llama-3-3-70b-instruct",
            "google-gemini-2-5-pro",
        ],
        help="Model slugs under results/baselines/<dataset>/",
    )
    parser.add_argument(
        "--judge-model",
        default="google/gemini-2.5-flash",
        help="Vertex judge model for semantic equivalence clustering.",
    )
    parser.add_argument("--judge-temperature", type=float, default=0.0, help="Judge temperature (default 0.0).")
    parser.add_argument("--no-fewshots", action="store_true", help="Disable few-shots in the equivalence prompt.")
    parser.add_argument(
        "--cache",
        type=Path,
        default=REPO_ROOT / "results" / "analysis" / "self_consistency_semantic_cache.jsonl",
        help="JSONL cache for pairwise equivalence decisions.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="TSV path to write per-model summary.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on records per model (debugging).")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    dataset = args.dataset
    output_path: Path = args.output or (REPO_ROOT / "results" / "analysis" / f"{dataset}_self_consistency_semantic_sweep.tsv")
    cache_path: Path = args.cache
    cache = load_cache(cache_path)

    client = build_vertex_client()
    create_system_prompt, create_user_prompt = dataset_prompt_fns(dataset)
    judge = EquivalenceJudge(
        client,
        JudgeConfig(
            model=args.judge_model,
            temperature=float(args.judge_temperature),
            include_fewshots=not args.no_fewshots,
        ),
        create_system_prompt=create_system_prompt,
        create_user_prompt=create_user_prompt,
    )

    rows: List[Dict[str, object]] = []
    for model_slug in args.models:
        baseline_path = REPO_ROOT / "results" / "baselines" / dataset / model_slug / f"{dataset}_self_consistency.json"
        if not baseline_path.exists():
            print(f"[WARN] Missing self-consistency file: {baseline_path}")
            continue

        records = iter_self_consistency_records(baseline_path)
        if args.limit is not None:
            records = records[: max(0, int(args.limit))]

        m_by_id: Dict[str, int] = {}
        labels: List[int] = []
        scores: List[float] = []

        for rec in records:
            example_id = str(rec.get("id"))
            question = str(rec.get("question", "")).strip()
            raw_samples = rec.get("raw_samples") or []
            answers = [str(s.get("final_answer") or "").strip() for s in raw_samples if isinstance(s, dict)]
            n_samples = len(answers)
            if n_samples == 0:
                continue

            m = semantic_max_cluster_size(
                dataset=dataset,
                example_id=example_id,
                question=question,
                answers=answers,
                judge=judge,
                cache=cache,
                cache_path=cache_path,
            )
            m_by_id[example_id] = int(m)

            correct_raw = rec.get("correct")
            try:
                correct = int(correct_raw)
            except Exception:
                continue
            y = 0 if correct == 1 else 1
            labels.append(int(y))
            scores.append(float(n_samples - m))

        auc = auroc_rank(labels, scores)
        max_n = 7
        if m_by_id:
            max_n = max(m_by_id.values())
            # Thresholds should be in [1..n_samples]; use 7 as the intended setting when possible.
            max_n = max(7, max_n)

        best_t = 1
        best = {"precision": 0.0, "recall": 0.0, "f1": -1.0, "coverage": 0.0}
        for t in range(1, max_n + 1):
            metrics = compute_for_threshold(records, t=t, m_by_id=m_by_id)
            if metrics["f1"] > best["f1"] or (metrics["f1"] == best["f1"] and metrics["coverage"] > best["coverage"]):
                best_t = t
                best = metrics

        t4 = compute_for_threshold(records, t=4, m_by_id=m_by_id)

        rows.append(
            {
                "dataset": dataset,
                "model": model_slug,
                "n_records": len(m_by_id),
                "n_pos_incorrect": sum(labels),
                "n_neg_correct": len(labels) - sum(labels),
                "auroc": auc,
                "best_t": best_t,
                "best_precision": best["precision"],
                "best_recall": best["recall"],
                "best_f1": best["f1"],
                "best_coverage": best["coverage"],
                "t4_precision": t4["precision"],
                "t4_recall": t4["recall"],
                "t4_f1": t4["f1"],
                "t4_coverage": t4["coverage"],
            }
        )

        auc_str = "nan" if math.isnan(auc) else f"{auc:.4f}"
        print(
            f"{dataset}/{model_slug}: n={len(m_by_id)} AUROC={auc_str} "
            f"best_t={best_t} F1={best['f1']:.4f} (P={best['precision']:.4f}, R={best['recall']:.4f}, cov={best['coverage']:.4f}) "
            f"t=4 F1={t4['f1']:.4f}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "dataset",
        "model",
        "n_records",
        "n_pos_incorrect",
        "n_neg_correct",
        "auroc",
        "best_t",
        "best_precision",
        "best_recall",
        "best_f1",
        "best_coverage",
        "t4_precision",
        "t4_recall",
        "t4_f1",
        "t4_coverage",
    ]
    with output_path.open("w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for row in rows:
            vals = []
            for key in header:
                val = row.get(key)
                if isinstance(val, float):
                    if math.isnan(val):
                        vals.append("nan")
                    else:
                        vals.append(f"{val:.6f}")
                else:
                    vals.append(str(val))
            f.write("\t".join(vals) + "\n")

    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
