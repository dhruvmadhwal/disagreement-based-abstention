#!/usr/bin/env python3
"""
Generate a presentable method-level efficiency comparison table (Assistive setting).

This is meant to help answer reviewer questions about DBA overhead (extra latency
and extra token usage) compared to baselines such as AYS / IC-IDK / Self-Consistency.

We report:
  - prompt (input) tokens
  - completion (output) tokens
  - multipliers vs open-ended, computed on a shared per-dataset comparison set

DBA-A (Assistive) here includes:
  - open-ended generation (1 call)
  - assistive generation (1 call)
  - consistency check between open vs assistive (1 call)

Optionally we also include DSL generation as an additional call (1 call).

Data sources:
  - GPT-5.1 generation tokens: results/token_usage_log.jsonl with meta.context == "generation-batch"
  - Baselines (AYS / IC-IDK / Self-Consistency) tokens: results/baselines/<dataset>/<model_slug>/gpt5_*_output_*.jsonl
  - Consistency-judge tokens (Gemini 2.5 Flash in our experiments): estimated from saved consistency outputs
      results/consistency/<dataset>/<model_slug>/*_assistive_vs_open_ended_consistency.json
    Note: these files do not include per-call usage metadata, so we estimate tokens via a simple chars/4 heuristic.

Outputs:
  - LaTeX: tables/efficiency_method_comparison_assistive.tex
  - Markdown: results/efficiency_method_comparison_assistive.md
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from decomposition_equivalence.prompts import (
    build_generation_system_prompt as build_dsl_system_prompt,
    build_generation_user_prompt as build_dsl_user_prompt,
)


DEFAULT_DATASETS: Tuple[str, ...] = ("bamboogle", "crag", "hotpotqa", "mintaka")

DISPLAY_DATASET: Dict[str, str] = {
    "bamboogle": "Bamboogle",
    "crag": "CRAG",
    "hotpotqa": "HotpotQA",
    "mintaka": "Mintaka",
}


def _fmt_int(x: float) -> str:
    return f"{int(round(x)):,}"


def _fmt_float(x: float, ndigits: int = 2) -> str:
    return f"{x:.{ndigits}f}"


def _safe_mean(values: Iterable[float]) -> Optional[float]:
    vals = list(values)
    return (sum(vals) / len(vals)) if vals else None


def _load_json_maybe(path: Path) -> Optional[object]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _iter_jsonl(path: Path) -> Iterable[dict]:
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    yield obj
    except FileNotFoundError:
        return


def load_dataset_ids(dataset: str) -> set[str]:
    """Load example ids from the processed DSL file for a dataset.

    This is the reference evaluation set (e.g., CRAG has 162 examples in `crag_dsl.json`).
    """
    path = Path("data/processed") / dataset / f"{dataset}_dsl.json"
    payload = _load_json_maybe(path)
    if not isinstance(payload, list):
        return set()
    ids: set[str] = set()
    for item in payload:
        if not isinstance(item, dict):
            continue
        ex_id = item.get("id")
        if ex_id is None:
            continue
        ids.add(str(ex_id))
    return ids


def load_generation_tokens_from_log(
    token_log: Path,
    *,
    model_name: str,
) -> Tuple[Dict[str, Tuple[int, int]], Dict[str, Tuple[int, int]]]:
    """Return (open_ended_by_id, assistive_by_id) for the given model.

Uses meta.context == "generation-batch" and keeps the latest timestamp per (id, regime).
"""
    latest: Dict[Tuple[str, str], Tuple[str, int, int]] = {}
    for rec in _iter_jsonl(token_log):
        if rec.get("model") != model_name:
            continue
        meta = rec.get("meta") or {}
        if not isinstance(meta, dict):
            continue
        if meta.get("context") != "generation-batch":
            continue
        regime = meta.get("regime")
        ex_id = meta.get("example_id")
        if not regime or ex_id is None:
            continue
        regime = str(regime)
        if regime not in {"open_ended", "assistive"}:
            continue
        ex_id = str(ex_id)
        ts = str(rec.get("timestamp") or "")
        try:
            pt = int(rec.get("prompt_tokens") or 0)
            ct = int(rec.get("completion_tokens") or 0)
        except Exception:
            continue
        key = (ex_id, regime)
        prev = latest.get(key)
        if prev is None or ts > prev[0]:
            latest[key] = (ts, pt, ct)
    open_by_id: Dict[str, Tuple[int, int]] = {}
    assist_by_id: Dict[str, Tuple[int, int]] = {}
    for (ex_id, regime), (_, pt, ct) in latest.items():
        if regime == "open_ended":
            open_by_id[ex_id] = (pt, ct)
        elif regime == "assistive":
            assist_by_id[ex_id] = (pt, ct)
    return open_by_id, assist_by_id


def _estimate_tokens(text: str) -> int:
    """Rough token estimate used when true usage metadata is unavailable."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def load_dsl_generation_tokens_from_log(
    token_log: Path,
    *,
    model_name: str,
) -> Dict[str, Tuple[int, int]]:
    """Return dsl_generation tokens by example id for the given model.

Uses meta.regime == "dsl_generation" and keeps the latest timestamp per id.
"""
    latest: Dict[str, Tuple[str, int, int]] = {}
    for rec in _iter_jsonl(token_log):
        if rec.get("model") != model_name:
            continue
        meta = rec.get("meta") or {}
        if not isinstance(meta, dict):
            continue
        if meta.get("regime") != "dsl_generation":
            continue
        ex_id = meta.get("example_id")
        if ex_id is None:
            continue
        ex_id = str(ex_id)
        ts = str(rec.get("timestamp") or "")
        try:
            pt = int(rec.get("prompt_tokens") or 0)
            ct = int(rec.get("completion_tokens") or 0)
        except Exception:
            continue
        prev = latest.get(ex_id)
        if prev is None or ts > prev[0]:
            latest[ex_id] = (ts, pt, ct)
    return {ex_id: (pt, ct) for ex_id, (_, pt, ct) in latest.items()}


def estimate_dsl_generation_usage(dataset: str) -> Dict[str, Tuple[int, int]]:
    """Estimate DSL-generation (prompt, completion) tokens from processed DSL strings.

    We treat one DSL conversion as:
      prompt = DSL system prompt + "Q: <question>"
      completion = emitted DSL program string
    """
    path = Path("data/processed") / dataset / f"{dataset}_dsl.json"
    payload = _load_json_maybe(path)
    if not isinstance(payload, list):
        return {}

    system_prompt = build_dsl_system_prompt(dataset)
    out: Dict[str, Tuple[int, int]] = {}
    for item in payload:
        if not isinstance(item, dict):
            continue
        ex_id = item.get("id")
        question = item.get("question") or ""
        dsl = item.get("dsl") or ""
        if ex_id is None:
            continue
        user_prompt = build_dsl_user_prompt(str(question))
        prompt = f"{system_prompt}\n\n{user_prompt}".strip()
        out[str(ex_id)] = (_estimate_tokens(prompt), _estimate_tokens(str(dsl)))
    return out


def estimate_consistency_judge_usage(dataset: str, *, model_slug: str) -> Dict[str, Tuple[int, int]]:
    """Estimate (prompt, completion) tokens for the saved consistency-judge comparisons."""
    if dataset == "bamboogle":
        from prompts.consistency.bamboogle.comparison_prompt import (  # type: ignore
            create_consistency_system_prompt,
            create_consistency_user_prompt,
        )
    elif dataset == "crag":
        from prompts.consistency.crag.comparison_prompt import (  # type: ignore
            create_consistency_system_prompt,
            create_consistency_user_prompt,
        )
    elif dataset == "hotpotqa":
        from prompts.consistency.hotpotqa.comparison_prompt import (  # type: ignore
            create_consistency_system_prompt,
            create_consistency_user_prompt,
        )
    elif dataset == "mintaka":
        from prompts.consistency.mintaka.comparison_prompt import (  # type: ignore
            create_consistency_system_prompt,
            create_consistency_user_prompt,
        )
    else:
        return {}

    path = (
        Path("results/consistency")
        / dataset
        / model_slug
        / f"{dataset}_assistive_vs_open_ended_consistency.json"
    )
    payload = _load_json_maybe(path)
    if not isinstance(payload, dict):
        return {}
    comparisons = payload.get("comparisons")
    if not isinstance(comparisons, list):
        return {}

    system_prompt = create_consistency_system_prompt()
    out: Dict[str, Tuple[int, int]] = {}
    for item in comparisons:
        if not isinstance(item, dict):
            continue
        ex_id = item.get("id")
        if ex_id is None:
            continue
        question = item.get("question")
        answer_a = item.get("answer_a") or ""
        answer_b = item.get("answer_b") or ""
        raw = item.get("raw_response") or ""
        user_prompt = create_consistency_user_prompt(question, str(answer_a), str(answer_b))
        prompt = f"{system_prompt}\n\n{user_prompt}".strip()
        out[str(ex_id)] = (_estimate_tokens(prompt), _estimate_tokens(str(raw)))
    return out


def load_openai_batch_outputs_with_usage(paths: Sequence[Path]) -> Dict[str, Tuple[int, int]]:
    """Parse OpenAI batch output JSONL(s) and return custom_id -> (prompt_tokens, completion_tokens).

Later files in `paths` (by sort order) override earlier entries for the same custom_id.
Only successful responses are kept.
"""
    out: Dict[str, Tuple[int, int]] = {}
    for path in sorted(paths):
        if not path.exists() or path.stat().st_size <= 0:
            continue
        for obj in _iter_jsonl(path):
            custom_id = obj.get("custom_id")
            response = obj.get("response")
            error = obj.get("error")
            if not custom_id or not isinstance(response, dict) or error:
                continue
            body = response.get("body") or {}
            if not isinstance(body, dict):
                continue
            usage = body.get("usage") or {}
            if not isinstance(usage, dict):
                continue
            pt = usage.get("prompt_tokens")
            ct = usage.get("completion_tokens")
            if pt is None or ct is None:
                continue
            try:
                out[str(custom_id)] = (int(pt), int(ct))
            except Exception:
                continue
    return out


def load_baseline_batch_usage(dataset: str, *, model_slug: str) -> Dict[str, Tuple[int, int]]:
    base_dir = Path("results/baselines") / dataset / model_slug
    patterns = (
        "gpt5_baselines_output_*.jsonl",
        "gpt5_baselines_repair_output_*.jsonl",
        "gpt5_baselines_retry_output_*.jsonl",
    )
    files: List[Path] = []
    for pat in patterns:
        files.extend(sorted(base_dir.glob(pat)))
    return load_openai_batch_outputs_with_usage(files)


def load_self_consistency_usage(dataset: str, *, model_slug: str) -> Dict[str, Tuple[int, int, int]]:
    """Return example_id -> (prompt_tokens_sum, completion_tokens_sum, calls) for self-consistency (if present)."""
    base_dir = Path("results/baselines") / dataset / model_slug
    patterns = (
        "gpt5_self_consistency_samples_output_*.jsonl",
        "gpt5_self_consistency_agg_output_*.jsonl",
    )
    files: List[Path] = []
    for pat in patterns:
        files.extend(sorted(base_dir.glob(pat)))
    by_custom = load_openai_batch_outputs_with_usage(files)
    by_ex: Dict[str, List[int]] = {}  # id -> [prompt_sum, completion_sum, calls]
    for custom_id, (pt, ct) in by_custom.items():
        ex_id = custom_id.split("::", 1)[0]
        acc = by_ex.setdefault(ex_id, [0, 0, 0])
        acc[0] += int(pt)
        acc[1] += int(ct)
        acc[2] += 1
    return {ex_id: (acc[0], acc[1], acc[2]) for ex_id, acc in by_ex.items()}


@dataclass(frozen=True)
class MethodStats:
    method: str
    n: int
    calls_per_ex: float
    prompt_tokens: float
    completion_tokens: float
    prompt_mult: float
    completion_mult: float
    total_mult: float


def compute_method_stats(
    *,
    common_ids: Sequence[str],
    open_by_id: Dict[str, Tuple[int, int]],
    assist_by_id: Dict[str, Tuple[int, int]],
    baseline_by_custom_id: Dict[str, Tuple[int, int]],
    judge_by_id: Dict[str, Tuple[int, int]],
    dsl_by_id: Dict[str, Tuple[int, int]],
    self_consistency_by_id: Dict[str, Tuple[int, int, int]],
    include_ic_idk: bool,
    include_self_consistency: bool,
) -> List[MethodStats]:
    ids = list(common_ids)
    if not ids:
        return []

    open_prompt = mean(open_by_id[ex][0] for ex in ids)
    open_completion = mean(open_by_id[ex][1] for ex in ids)
    open_total = open_prompt + open_completion
    if open_prompt <= 0 or open_completion <= 0 or open_total <= 0:
        return []

    def _stats_for_tokens(
        method: str,
        calls_per_ex: float,
        prompt_vals: Iterable[float],
        completion_vals: Iterable[float],
    ) -> MethodStats:
        p = mean(list(prompt_vals))
        c = mean(list(completion_vals))
        t = p + c
        return MethodStats(
            method=method,
            n=len(ids),
            calls_per_ex=float(calls_per_ex),
            prompt_tokens=p,
            completion_tokens=c,
            prompt_mult=p / open_prompt,
            completion_mult=c / open_completion,
            total_mult=t / open_total,
        )

    rows: List[MethodStats] = []

    # Open-ended (single pass)
    rows.append(
        _stats_for_tokens(
            "Open",
            1.0,
            (open_by_id[ex][0] for ex in ids),
            (open_by_id[ex][1] for ex in ids),
        )
    )

    # AYS: open + yes/no verification
    rows.append(
        _stats_for_tokens(
            "AYS (Open + verify)",
            2.0,
            (open_by_id[ex][0] + baseline_by_custom_id[f"{ex}::ays"][0] for ex in ids),
            (open_by_id[ex][1] + baseline_by_custom_id[f"{ex}::ays"][1] for ex in ids),
        )
    )

    # IC-IDK: one call with demonstrations
    if include_ic_idk:
        rows.append(
            _stats_for_tokens(
                "IC-IDK",
                1.0,
                (baseline_by_custom_id[f"{ex}::ic_idk"][0] for ex in ids),
                (baseline_by_custom_id[f"{ex}::ic_idk"][1] for ex in ids),
            )
        )

    # DBA-A: open + assistive + pairwise judge (open vs assistive)
    rows.append(
        _stats_for_tokens(
            "DBA-A (Open+Assist+Judge)",
            3.0,
            (
                open_by_id[ex][0]
                + assist_by_id[ex][0]
                + judge_by_id[ex][0]
                for ex in ids
            ),
            (
                open_by_id[ex][1]
                + assist_by_id[ex][1]
                + judge_by_id[ex][1]
                for ex in ids
            ),
        )
    )

    # DBA-A + DSL generation: add tokens from a dedicated DSL generation call.
    rows.append(
        _stats_for_tokens(
            "DBA-A + DSL gen",
            4.0,
            (
                open_by_id[ex][0]
                + assist_by_id[ex][0]
                + judge_by_id[ex][0]
                + dsl_by_id[ex][0]
                for ex in ids
            ),
            (
                open_by_id[ex][1]
                + assist_by_id[ex][1]
                + judge_by_id[ex][1]
                + dsl_by_id[ex][1]
                for ex in ids
            ),
        )
    )

    # Self-consistency: 7 samples + 1 aggregator in this repo (8 calls), where available.
    if include_self_consistency:
        calls = mean(self_consistency_by_id[ex][2] for ex in ids)
        rows.append(
            _stats_for_tokens(
                "Self-Consistency (7+1)",
                float(calls),
                (self_consistency_by_id[ex][0] for ex in ids),
                (self_consistency_by_id[ex][1] for ex in ids),
            )
        )

    return rows


def to_markdown(rows: Sequence[dict]) -> str:
    if not rows:
        return ""
    headers = list(rows[0].keys())
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        out.append("| " + " | ".join(str(r[h]) for h in headers) + " |")
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate method-level efficiency comparison table (Assistive).")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help="Datasets to include (default: bamboogle crag hotpotqa mintaka).",
    )
    parser.add_argument(
        "--token-log",
        type=Path,
        default=Path("results/token_usage_log.jsonl"),
        help="Token usage log path (default: results/token_usage_log.jsonl).",
    )
    parser.add_argument(
        "--gpt-model",
        type=str,
        default="gpt-5.1-2025-11-13",
        help="GPT-5.1 model name string as logged (default: gpt-5.1-2025-11-13).",
    )
    parser.add_argument(
        "--baseline-model-slug",
        type=str,
        default="gpt-5-1",
        help="Model slug directory for baseline batch outputs (default: gpt-5-1).",
    )
    parser.add_argument(
        "--consistency-model-slug",
        type=str,
        default="gpt-5-1",
        help="Model slug directory for saved Gemini-judged consistency outputs (default: gpt-5-1).",
    )
    parser.add_argument(
        "--judge-mode",
        choices=("consistency_estimate", "gpt_batch_pairwise"),
        default="consistency_estimate",
        help=(
            "How to source tokens for the DBA consistency judge: "
            "`consistency_estimate` (default; estimate from results/consistency/*) or "
            "`gpt_batch_pairwise` (use GPT baseline batch usage for ::pairwise::assistive)."
        ),
    )
    parser.add_argument(
        "--dsl-mode",
        choices=("estimate", "token_log"),
        default="estimate",
        help=(
            "How to source tokens for DSL generation: `estimate` (default; from DSL strings) or "
            "`token_log` (from results/token_usage_log.jsonl for --dsl-model)."
        ),
    )
    parser.add_argument(
        "--dsl-model",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Model name used for DSL generation token logs (default: Qwen/Qwen3-8B).",
    )
    parser.add_argument(
        "--out-tex",
        type=Path,
        default=Path("tables/efficiency_method_comparison_assistive.tex"),
        help="Output path for LaTeX table.",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("results/efficiency_method_comparison_assistive.md"),
        help="Output path for Markdown table.",
    )
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets if d.strip()]
    if not datasets:
        raise SystemExit("No datasets provided.")

    open_by_id, assist_by_id = load_generation_tokens_from_log(args.token_log, model_name=args.gpt_model)

    table_rows: List[dict] = []

    # Collect global (across all dataset comparison sets) for an overall summary.
    global_ids: List[str] = []
    global_meta: Dict[str, object] = {
        "baseline_by_custom_id": {},
        "self_consistency_by_id": {},
        "include_ic_idk": False,
        "include_self_consistency": False,
    }

    per_dataset_stats: Dict[str, List[MethodStats]] = {}

    for ds in datasets:
        baseline_by_custom_id = load_baseline_batch_usage(ds, model_slug=args.baseline_model_slug)
        self_consistency_by_id = load_self_consistency_usage(ds, model_slug=args.baseline_model_slug)

        ids = load_dataset_ids(ds)
        # Keep the reference set fixed to the processed DSL file for the dataset.

        if args.dsl_mode == "token_log":
            dsl_by_id = load_dsl_generation_tokens_from_log(args.token_log, model_name=args.dsl_model)
        else:
            dsl_by_id = estimate_dsl_generation_usage(ds)

        if args.judge_mode == "gpt_batch_pairwise":
            judge_by_id = {
                ex: baseline_by_custom_id[f"{ex}::pairwise::assistive"]
                for ex in ids
                if f"{ex}::pairwise::assistive" in baseline_by_custom_id
            }
        else:
            judge_by_id = estimate_consistency_judge_usage(ds, model_slug=args.consistency_model_slug)

        # Availability sets (on this dataset id universe).
        s_open = {ex for ex in ids if ex in open_by_id}
        s_assist = {ex for ex in ids if ex in assist_by_id}
        s_ays = {cid.split("::", 1)[0] for cid in baseline_by_custom_id.keys() if cid.endswith("::ays")}
        s_judge = set(judge_by_id.keys())
        s_ic = {cid.split("::", 1)[0] for cid in baseline_by_custom_id.keys() if cid.endswith("::ic_idk")}
        s_dsl = {ex for ex in ids if ex in dsl_by_id}
        s_sc = set(self_consistency_by_id.keys())

        include_ic = bool(s_ic)
        include_sc = bool(s_sc)

        # Shared comparison set to keep the Open reference consistent across rows.
        common = set(s_open) & set(s_assist) & set(s_ays) & set(s_judge) & set(s_dsl)
        if include_ic:
            common &= set(s_ic)
        if include_sc:
            common &= set(s_sc)
        common_ids = sorted(common)

        stats = compute_method_stats(
            common_ids=common_ids,
            open_by_id=open_by_id,
            assist_by_id=assist_by_id,
            baseline_by_custom_id=baseline_by_custom_id,
            judge_by_id=judge_by_id,
            dsl_by_id=dsl_by_id,
            self_consistency_by_id=self_consistency_by_id,
            include_ic_idk=include_ic,
            include_self_consistency=include_sc,
        )
        if not stats:
            continue
        per_dataset_stats[ds] = stats

        # Append formatted rows for markdown/latex.
        for row in stats:
            table_rows.append(
                {
                    "Dataset": DISPLAY_DATASET.get(ds, ds),
                    "Method": row.method,
                    "N": row.n,
                    "Calls/ex": _fmt_float(row.calls_per_ex, 1) if row.calls_per_ex % 1 else str(int(row.calls_per_ex)),
                    "Prompt": _fmt_int(row.prompt_tokens),
                    "Compl.": _fmt_int(row.completion_tokens),
                    "Prompt×": _fmt_float(row.prompt_mult),
                    "Compl.×": _fmt_float(row.completion_mult),
                    "Total×": _fmt_float(row.total_mult),
                }
            )

        # Build global aggregation using the same per-dataset comparison sets.
        global_ids.extend(common_ids)
        global_meta["baseline_by_custom_id"] = global_meta.get("baseline_by_custom_id") or {}
        global_meta["self_consistency_by_id"] = global_meta.get("self_consistency_by_id") or {}
        # We cannot safely merge per-dataset baseline maps without collisions across datasets.
        # Instead, recompute global rows later from the already materialized `table_rows`.
        # (We still add an overall summary row below using the per-dataset computed stats.)

    # Overall (weighted by dataset comparison-set sizes).
    overall_rows: List[MethodStats] = []
    # Aggregate per method across datasets (weighting by N).
    per_method_acc: Dict[str, List[MethodStats]] = {}
    for ds, stats in per_dataset_stats.items():
        for s in stats:
            per_method_acc.setdefault(s.method, []).append(s)

    for method, stats_list in per_method_acc.items():
        total_n = sum(s.n for s in stats_list)
        if total_n <= 0:
            continue

        prompt_sum = sum(s.prompt_tokens * s.n for s in stats_list)
        completion_sum = sum(s.completion_tokens * s.n for s in stats_list)
        open_prompt_sum = sum((s.prompt_tokens / s.prompt_mult) * s.n for s in stats_list)
        open_completion_sum = sum((s.completion_tokens / s.completion_mult) * s.n for s in stats_list)
        open_total_sum = open_prompt_sum + open_completion_sum
        method_total_sum = prompt_sum + completion_sum
        if open_prompt_sum <= 0 or open_completion_sum <= 0 or open_total_sum <= 0:
            continue

        calls_per_ex = sum(s.calls_per_ex * s.n for s in stats_list) / total_n
        prompt_avg = prompt_sum / total_n
        completion_avg = completion_sum / total_n
        prompt_mult = prompt_sum / open_prompt_sum
        completion_mult = completion_sum / open_completion_sum
        total_mult = method_total_sum / open_total_sum

        overall_rows.append(
            MethodStats(
                method=method,
                n=total_n,
                calls_per_ex=float(calls_per_ex),
                prompt_tokens=float(prompt_avg),
                completion_tokens=float(completion_avg),
                prompt_mult=float(prompt_mult),
                completion_mult=float(completion_mult),
                total_mult=float(total_mult),
            )
        )

    if overall_rows:
        table_rows.append(
            {
                "Dataset": "Overall",
                "Method": "",
                "N": "",
                "Calls/ex": "",
                "Prompt": "",
                "Compl.": "",
                "Prompt×": "",
                "Compl.×": "",
                "Total×": "",
            }
        )
        for row in overall_rows:
            table_rows.append(
                {
                    "Dataset": "Overall",
                    "Method": row.method,
                    "N": row.n,
                    "Calls/ex": _fmt_float(row.calls_per_ex, 1) if row.calls_per_ex % 1 else str(int(row.calls_per_ex)),
                    "Prompt": _fmt_int(row.prompt_tokens),
                    "Compl.": _fmt_int(row.completion_tokens),
                    "Prompt×": _fmt_float(row.prompt_mult),
                    "Compl.×": _fmt_float(row.completion_mult),
                    "Total×": _fmt_float(row.total_mult),
                }
            )

    judge_line = ""
    if args.judge_mode == "gpt_batch_pairwise":
        judge_line = (
            f"- DBA judge tokens: `{args.baseline_model_slug}` batch outputs under "
            f"`results/baselines/<dataset>/{args.baseline_model_slug}/` (custom_id `::pairwise::assistive`)\n"
        )
    else:
        judge_line = (
            f"- DBA judge tokens: estimated from `results/consistency/<dataset>/{args.consistency_model_slug}/*_assistive_vs_open_ended_consistency.json` "
            "(chars/4 heuristic; Gemini judge)\n"
        )

    dsl_line = ""
    if args.dsl_mode == "token_log":
        dsl_line = (
            f"- DSL generation tokens: `{args.dsl_model}` from `results/token_usage_log.jsonl` (regime=dsl_generation)\n"
        )
    else:
        dsl_line = (
            "- DSL generation tokens: estimated from processed `*_dsl.json` question+DSL strings (chars/4 heuristic)\n"
        )

    # Markdown
    md_header = (
        "# Method-level token/latency overhead (Assistive)\n\n"
        f"- Open/Assistive generation tokens: `{args.gpt_model}` from `results/token_usage_log.jsonl` (context=generation-batch)\n"
        f"- Baseline verification tokens (AYS / IC-IDK / Self-Consistency): `{args.baseline_model_slug}` batch outputs under `results/baselines/<dataset>/{args.baseline_model_slug}/`\n"
        + judge_line
        + dsl_line
        + "\nMultipliers are computed vs. Open on a shared per-dataset comparison set.\n\n"
    )
    md = md_header + to_markdown(table_rows)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(md, encoding="utf-8")

    # LaTeX
    latex_lines: List[str] = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Method-level token overhead in the Assistive setting. We report prompt (input) and completion (output) tokens separately. DBA-A includes Open+Assistive generation plus a consistency judge between the two outputs. The optional DBA-A+DSL row adds one DSL-generation call. Unless otherwise noted, tokens come from saved OpenAI usage. When usage metadata is unavailable (Gemini judge or DSL generation), we estimate tokens with a chars/4 heuristic. Multipliers are computed relative to Open on a shared per-dataset comparison set.}",
        "\\label{tab:efficiency_method_comparison_assistive}",
        "\\begin{tabular}{llrrrrrrr}",
        "\\toprule",
        "\\textbf{Dataset} & \\textbf{Method} & \\textbf{N} & \\textbf{Calls/ex} & \\textbf{Prompt} & \\textbf{Compl.} & \\textbf{Prompt$\\times$} & \\textbf{Compl.$\\times$} & \\textbf{Total$\\times$} \\\\",
        "\\midrule",
    ]
    for r in table_rows:
        if r["Dataset"] == "Overall" and r["Method"] == "":
            latex_lines.append("\\midrule")
            continue
        dataset = r["Dataset"]
        method = str(r["Method"]).replace("&", "\\&")
        latex_lines.append(
            f"{dataset} & {method} & {r['N']} & {r['Calls/ex']} & {r['Prompt']} & {r['Compl.']} & {r['Prompt×']} & {r['Compl.×']} & {r['Total×']} \\\\"
        )
    latex_lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    args.out_tex.parent.mkdir(parents=True, exist_ok=True)
    args.out_tex.write_text("\n".join(latex_lines), encoding="utf-8")

    print(f"Wrote {args.out_tex}")
    print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()
