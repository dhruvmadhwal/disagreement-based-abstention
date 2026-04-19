import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import streamlit as st


# --- Constants (Mintaka-focused inspector with selectable models/regimes) ---
REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = Path(os.environ.get("QA_BASE_DIR", os.environ.get("MINTAKA_BASE_DIR", str(REPO_ROOT))))
RESULTS_DIR = BASE_DIR / "results"
GENERATION_ROOT = RESULTS_DIR / "generation"
CONSISTENCY_ROOT = RESULTS_DIR / "consistency"
CORRECTNESS_ROOT = RESULTS_DIR / "correctness"
EVALUATION_ROOT = RESULTS_DIR / "evaluation"
DATA_PROCESSED_ROOT = BASE_DIR / "data/processed"

DEFAULT_DATASET = os.environ.get("QA_DATASET", "mintaka")
DEFAULT_MODEL_SLUG = os.environ.get(
    "QA_STREAMLIT_MODEL",
    os.environ.get("MINTAKA_STREAMLIT_MODEL", "gemini-2.5-flash"),
)

DATASET_LABELS = {
    "mintaka": "Mintaka",
    "hotpotqa": "HotpotQA",
    "musique": "MuSiQue",
}

LEGACY_DSL_PATHS = {
    "mintaka": BASE_DIR / "data/Mintaka/dsl/dsl-mintaka.json",
}
LEGACY_EVAL_CONSISTENCY = RESULTS_DIR / "evaluation/mintaka_consistency_sequential_vs_open_ended.json"
LEGACY_EVAL_CORRECTNESS = RESULTS_DIR / "evaluation/mintaka_correctness_sequential.json"

DEFAULT_REGIME_NAMES = ["open_ended", "assistive", "incremental", "model_generated"]
REGIME_FILE_TEMPLATES = {
    "open_ended": "{dataset}_open_ended.json",
    "assistive": "{dataset}_assistive.json",
    "incremental": "{dataset}_sequential.json",
    "model_generated": "{dataset}_model_generated.json",
}

REGIME_TITLES = {
    "open_ended": "Open-Ended Generation",
    "assistive": "Assistive (DSL Execution)",
    "incremental": "Incremental (Self-Ask)",
    "model_generated": "Model-Generated Plan",
}
REGIME_DISPLAY_ORDER = ["open_ended", "assistive", "incremental", "model_generated"]
JSONL_REGIME_KEY_MAP = {
    "sequential": "incremental",
}
INCREMENTAL_ALIASES = {"incremental", "sequential"}


def canonicalize_regime_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    slug = name.strip().lower().replace("-", "_")
    if slug in INCREMENTAL_ALIASES:
        return "incremental"
    if slug == "model_generated_plan":
        return "model_generated"
    return slug


def get_regime_title(regime: str) -> str:
    """Return a human readable label for a regime."""
    return REGIME_TITLES.get(regime, regime.replace("_", " ").title())


def resolve_regime_display_order(regime_names: Iterable[str]) -> List[str]:
    """Keep preferred ordering but append any extra regimes deterministically."""
    preferred = [name for name in REGIME_DISPLAY_ORDER if name in regime_names]
    extras = sorted(set(regime_names) - set(preferred))
    return preferred + extras


def collect_question_sources(dsl_idx: Dict[str, dict], model_payloads: Dict[str, dict]) -> List[Dict[str, dict]]:
    """Helper to gather every dictionary that might contain question text."""
    sources: List[Dict[str, dict]] = [dsl_idx]
    for payload in model_payloads.values():
        sources.extend(payload.get("regime_indices", {}).values())
    return sources


# --- Data loading ---
@st.cache_data(show_spinner=False)
def load_json(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def index_dsl_by_id(dsl_list: List[dict]) -> Dict[str, dict]:
    return {item.get("id"): item for item in dsl_list}


@st.cache_data(show_spinner=False)
def index_list_by_id(items: List[dict]) -> Dict[str, dict]:
    return {item.get("id"): item for item in items}


@st.cache_data(show_spinner=False)
def load_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def list_available_datasets() -> List[str]:
    if not GENERATION_ROOT.exists():
        return []
    return sorted([path.name for path in GENERATION_ROOT.iterdir() if path.is_dir()])


def list_available_model_slugs(dataset: str) -> List[str]:
    dataset_dir = GENERATION_ROOT / dataset
    if not dataset_dir.exists():
        return []
    return sorted([path.name for path in dataset_dir.iterdir() if path.is_dir()])


def get_dataset_label(dataset: str) -> str:
    return DATASET_LABELS.get(dataset, dataset.replace("_", " ").title())


def get_question_for_id(
    qid: str,
    sources: List[Dict[str, dict]],
) -> Optional[str]:
    for src in sources:
        node = src.get(qid) if src else None
        if node and isinstance(node, dict):
            q = node.get("question")
            if q:
                return q
    return None


def gather_all_ids(
    dsl_idx: Dict[str, dict],
    model_payloads: Dict[str, dict],
    cons_lookup: Optional[Dict[str, dict]],
    corr_lookup: Optional[Dict[str, dict]],
) -> List[str]:
    ids = set()
    ids.update(dsl_idx.keys())
    for payload in model_payloads.values():
        for idx in payload.get("regime_indices", {}).values():
            ids.update(idx.keys())
    if cons_lookup:
        ids.update(cons_lookup.keys())
    if corr_lookup:
        ids.update([k for k in corr_lookup.keys() if k and not str(k).startswith("_")])
    return sorted([i for i in ids if i])


def get_consistency_for_id(cons_lookup: Optional[Dict[str, dict]], qid: str) -> Optional[dict]:
    if not cons_lookup:
        return None
    return cons_lookup.get(qid)


def get_correctness_for_id(corr_lookup: Optional[Dict[str, dict]], qid: str) -> Optional[dict]:
    if not corr_lookup:
        return None
    if qid.startswith("_"):
        return None
    return corr_lookup.get(qid)


def sidebar_filters(
    all_ids: List[str],
    dsl_idx: Dict[str, dict],
    model_payloads: Dict[str, dict],
    cons_lookup: Optional[Dict[str, dict]],
    corr_lookup: Optional[Dict[str, dict]],
) -> Tuple[List[str], str, bool, bool, bool]:
    st.sidebar.header("Filters")

    search_text = st.sidebar.text_input("Search in question or ID")
    only_inconsistent = st.sidebar.checkbox("Only inconsistent (incremental vs open-ended)", value=False)
    only_incorrect = st.sidebar.checkbox("Only incorrect (incremental correctness)", value=False)
    present_everywhere = st.sidebar.checkbox("Present in all sources", value=False)

    question_sources = collect_question_sources(dsl_idx, model_payloads)

    def passes(qid: str) -> bool:
        # Presence
        if present_everywhere:
            if qid not in dsl_idx:
                return False
            cons_entry = get_consistency_for_id(cons_lookup, qid)
            corr_entry = get_correctness_for_id(corr_lookup, qid)
            if cons_entry is None or corr_entry is None:
                return False
            for payload in model_payloads.values():
                for idx in payload.get("regime_indices", {}).values():
                    if idx and qid not in idx:
                        return False

        # Search
        if search_text:
            q = get_question_for_id(qid, question_sources) or ""
            if search_text.lower() not in q.lower() and search_text.lower() not in qid.lower():
                return False

        # Consistency
        if only_inconsistent:
            c = get_consistency_for_id(cons_lookup, qid)
            if not c or c.get("equivalent") == 1:
                return False

        # Correctness
        if only_incorrect:
            k = get_correctness_for_id(corr_lookup, qid)
            if not k or k.get("correct") == 1:
                return False

        return True

    filtered = [qid for qid in all_ids if passes(qid)]
    return filtered, search_text, only_inconsistent, only_incorrect, present_everywhere

def question_selector(filtered_ids: List[str], dsl_idx: Dict[str, dict], model_payloads: Dict[str, dict]) -> Optional[str]:
    if "_sel_index" not in st.session_state:
        st.session_state._sel_index = 0

    # Clamp index if filter changed
    if st.session_state._sel_index >= len(filtered_ids):
        st.session_state._sel_index = 0

    # Build labels with short question preview
    labels = []
    question_sources = collect_question_sources(dsl_idx, model_payloads)
    for qid in filtered_ids:
        q = get_question_for_id(qid, question_sources) or ""
        preview = (q[:120] + "…") if len(q) > 120 else q
        labels.append(f"{qid} — {preview}")

    col_left, col_mid, col_right = st.columns([1, 6, 1])
    with col_left:
        if st.button("◀ Prev", disabled=len(filtered_ids) == 0):
            st.session_state._sel_index = (st.session_state._sel_index - 1) % max(len(filtered_ids), 1)

    with col_right:
        if st.button("Next ▶", disabled=len(filtered_ids) == 0):
            st.session_state._sel_index = (st.session_state._sel_index + 1) % max(len(filtered_ids), 1)

    selected = st.selectbox(
        "Select question",
        options=list(range(len(filtered_ids))),
        format_func=lambda i: labels[i] if i < len(labels) else "",
        index=st.session_state._sel_index if filtered_ids else 0,
        disabled=len(filtered_ids) == 0,
    )
    st.session_state._sel_index = selected

    return filtered_ids[selected] if filtered_ids else None


def render_dsl_section(qid: str, dsl_idx: Dict[str, dict]) -> None:
    st.markdown("**DSL (from dataset)**")
    dsl_item = dsl_idx.get(qid)
    if dsl_item:
        dsl_code = dsl_item.get("dsl") or ""
        if dsl_code:
            st.code(dsl_code, language="python")
        with st.expander("Intermediate answers and QA chain", expanded=False):
            st.json({
                "intermediate_answers": dsl_item.get("intermediate_answers"),
                "qa_chain": dsl_item.get("qa_chain"),
            })
    else:
        st.info("No DSL entry found for this ID in dataset file.")


def render_incremental_block(entry: Optional[dict]) -> None:
    st.markdown("**Incremental generation (answer)**")
    if entry:
        st.write(entry.get("final_answer") or entry.get("answer") or "—")
        if entry.get("explanation"):
            st.write(f"_Explanation:_ {entry.get('explanation')}")
        with st.expander("Incremental details"):
            st.json(entry)
    else:
        st.info("No incremental generation entry for this model.")


def render_open_ended_block(entry: Optional[dict]) -> None:
    st.markdown("**Open-ended generation**")
    if entry:
        st.write(entry.get("answer", "—"))
        if entry.get("explanation"):
            with st.expander("Explanation"):
                st.write(entry.get("explanation"))
        with st.expander("Open-ended raw entry"):
            st.json(entry)
    else:
        st.info("No open-ended generation entry.")


def render_assistive_block(entry: Optional[dict]) -> None:
    st.markdown("**Assistive (DSL execution)**")
    if entry:
        st.write(entry.get("final_answer", "—"))
        with st.expander("Assistive intermediate answers"):
            st.json(entry.get("intermediate_answers"))
    else:
        st.info("No assistive generation entry.")


def render_model_generated_block(entry: Optional[dict]) -> None:
    st.markdown("**Model-generated plan**")
    if entry:
        st.write(entry.get("final_answer", "—"))
        with st.expander("QA chain"):
            st.json(entry.get("qa_chain"))
    else:
        st.info("No model-generated plan entry.")


def render_generic_regime_block(regime: str, entry: Optional[dict]) -> None:
    st.markdown(f"**{get_regime_title(regime)}**")
    if entry:
        st.json(entry)
    else:
        st.info(f"No {get_regime_title(regime).lower()} entry.")


def gather_incremental_ids(regime_indices: Dict[str, Dict[str, dict]]) -> set:
    ids = set()
    for alias in INCREMENTAL_ALIASES:
        ids.update(regime_indices.get(alias, {}).keys())
    return ids


def compute_model_metrics(
    model_slug: str,
    payload: dict,
    consistency_lookup: Dict[str, dict],
    correctness_lookup: Dict[str, dict],
) -> dict:
    regime_indices: Dict[str, Dict[str, dict]] = payload.get("regime_indices", {})
    incremental_ids = gather_incremental_ids(regime_indices)
    open_ids = set(regime_indices.get("open_ended", {}).keys())
    comparison_ids = incremental_ids & open_ids

    evaluated_comp = 0
    equivalent = 0
    non_equivalent = 0
    for qid in comparison_ids:
        cons = consistency_lookup.get(qid) if consistency_lookup else None
        if cons is None:
            continue
        evaluated_comp += 1
        if cons.get("equivalent") == 1:
            equivalent += 1
        elif cons.get("equivalent") == 0:
            non_equivalent += 1

    eq_rate = (equivalent / evaluated_comp) if evaluated_comp else None

    corr_total = 0
    corr_correct = 0
    for qid in incremental_ids:
        corr = correctness_lookup.get(qid) if correctness_lookup else None
        if corr is None:
            continue
        corr_total += 1
        if corr.get("correct") == 1:
            corr_correct += 1

    return {
        "available_comparisons": len(comparison_ids),
        "evaluated_comparisons": evaluated_comp,
        "equivalent": equivalent,
        "non_equivalent": non_equivalent,
        "equivalence_rate": eq_rate,
        "inc_eval_total": corr_total,
        "inc_correct": corr_correct,
        "inc_total_records": len(incremental_ids),
    }


def render_model_metrics(
    model_slug: str,
    payload: dict,
    consistency_lookup: Dict[str, dict],
    correctness_lookup: Dict[str, dict],
) -> None:
    metrics = compute_model_metrics(model_slug, payload, consistency_lookup, correctness_lookup)
    col1, col2, col3, col4 = st.columns(4)

    total_display = (
        f"{metrics['evaluated_comparisons']}/{metrics['available_comparisons']}"
        if metrics["available_comparisons"]
        else str(metrics["evaluated_comparisons"])
    )
    with col1:
        st.metric("Total comparisons", value=total_display or "0")

    eq_rate = metrics["equivalence_rate"]
    with col2:
        if eq_rate is not None:
            st.metric("Equivalence rate", value=f"{eq_rate:.3f}")
        else:
            st.metric("Equivalence rate", value="—")

    inc_total = metrics["inc_eval_total"]
    inc_value = (
        f"{metrics['inc_correct']}/{inc_total}"
        if inc_total
        else ("0" if metrics["inc_total_records"] == 0 else f"0/{metrics['inc_total_records']}")
    )
    with col3:
        st.metric("Incremental correct", value=inc_value if inc_value else "—")

    with col4:
        if metrics["evaluated_comparisons"]:
            st.metric("Non-equivalent", value=f"{metrics['non_equivalent']}/{metrics['evaluated_comparisons']}")
        else:
            st.metric("Non-equivalent", value="—")


def render_model_panel(
    qid: str,
    model_slug: str,
    payload: dict,
) -> None:
    st.caption(f"Model run: {model_slug}")
    correctness_regime, correctness_lookup = get_preferred_correctness(payload)
    consistency_pair, consistency_bundle = get_preferred_consistency(payload)
    consistency_lookup = consistency_bundle.get("records", {})

    render_model_metrics(model_slug, payload, consistency_lookup, correctness_lookup)
    regime_indices: Dict[str, Dict[str, dict]] = payload.get("regime_indices", {})
    regime_entries = {name: idx.get(qid) for name, idx in regime_indices.items()}

    # Ensure default regimes are represented even if empty
    regime_names = set(regime_indices.keys()).union(REGIME_DISPLAY_ORDER)
    regime_order = resolve_regime_display_order(regime_names)

    incremental_entry = None
    for alias in INCREMENTAL_ALIASES:
        candidate = regime_entries.get(alias)
        if candidate:
            incremental_entry = candidate
            break

    left, right = st.columns(2)
    with left:
        render_incremental_block(incremental_entry)

    with right:
        for regime in regime_order:
            if regime in INCREMENTAL_ALIASES:
                continue
            entry = regime_entries.get(regime)
            if regime == "open_ended":
                render_open_ended_block(entry)
            elif regime == "assistive":
                render_assistive_block(entry)
            elif regime == "model_generated":
                render_model_generated_block(entry)
            else:
                render_generic_regime_block(regime, entry)

        # Catch-all for regimes discovered only via JSONL
        extra_regimes = [
            name for name in regime_entries.keys()
            if name not in regime_order and name not in INCREMENTAL_ALIASES
        ]
        for regime in sorted(extra_regimes):
            render_generic_regime_block(regime, regime_entries.get(regime))

    render_model_evaluation_details(
        qid,
        correctness_lookup,
        consistency_lookup,
        correctness_regime=correctness_regime,
        consistency_pair=consistency_pair,
    )


def render_model_evaluation_details(
    qid: str,
    correctness_lookup: Dict[str, dict],
    consistency_lookup: Dict[str, dict],
    correctness_regime: Optional[str] = None,
    consistency_pair: Optional[Tuple[str, str]] = None,
) -> None:
    st.markdown("**Evaluations**")
    corr = get_correctness_for_id(correctness_lookup, qid) if correctness_lookup else None
    cons = get_consistency_for_id(consistency_lookup, qid) if consistency_lookup else None

    corr_label = (
        f"{get_regime_title(correctness_regime)} correctness"
        if correctness_regime and correctness_regime in REGIME_TITLES
        else "Regime correctness"
    )
    if consistency_pair:
        regime_a, regime_b = consistency_pair
        cons_label = f"{get_regime_title(regime_a)} vs {get_regime_title(regime_b)}"
    else:
        cons_label = "Consistency"

    c1, c2 = st.columns(2)
    with c1:
        if corr is not None:
            st.metric(corr_label, value="✅" if corr.get("correct") == 1 else "❌")
        else:
            st.metric(corr_label, value="—")
    with c2:
        if cons is not None:
            st.metric(cons_label, value="✅" if cons.get("equivalent") == 1 else "❌")
        else:
            st.metric(cons_label, value="—")

    if corr is not None and corr.get("reasoning"):
        with st.expander("Correctness reasoning"):
            st.write(corr.get("reasoning"))

    if cons is not None:
        with st.expander("Consistency details"):
            st.json({
                "answer_a": cons.get("answer_a"),
                "answer_b": cons.get("answer_b"),
                "equivalent": cons.get("equivalent"),
                "reasoning": cons.get("reasoning"),
            })


def render_question_view(
    qid: str,
    dsl_idx: Dict[str, dict],
    model_payloads: Dict[str, dict],
):
    st.subheader(qid)
    question_sources = collect_question_sources(dsl_idx, model_payloads)
    question = get_question_for_id(qid, question_sources) or ""
    st.write(question)

    render_dsl_section(qid, dsl_idx)

    if not model_payloads:
        st.info("No model runs selected.")
        return

    render_model_tabs(qid, model_payloads)


def render_model_tabs(
    qid: str,
    model_payloads: Dict[str, dict],
) -> None:
    tab_slugs = list(model_payloads.keys())
    tab_labels = [f"Model: {slug}" for slug in tab_slugs]
    tabs = st.tabs(tab_labels)
    for tab, slug in zip(tabs, tab_slugs):
        with tab:
            render_model_panel(qid, slug, model_payloads[slug])


def resolve_dsl_dataset(dataset: str) -> List[dict]:
    processed_path = DATA_PROCESSED_ROOT / dataset / f"{dataset}_dsl.json"
    if processed_path.exists():
        return load_json(processed_path) or []
    legacy_path = LEGACY_DSL_PATHS.get(dataset)
    if legacy_path and legacy_path.exists():
        return load_json(legacy_path) or []
    return []


def find_jsonl_records(model_dir: Path, dataset: str) -> List[dict]:
    jsonl_files = sorted(model_dir.glob(f"{dataset}*_regimes*.jsonl"))
    records: List[dict] = []
    for file in jsonl_files:
        records.extend(load_jsonl(file))
    return records


def convert_regimes_from_jsonl(
    records: List[dict], model_slug: str, regime_names: Optional[Iterable[str]] = None
) -> Dict[str, List[dict]]:
    base_names = list(regime_names) if regime_names else list(DEFAULT_REGIME_NAMES)
    regime_lists = {regime: [] for regime in base_names}
    if not records:
        return regime_lists

    for rec in records:
        qid = rec.get("id")
        question = rec.get("question")
        regimes = rec.get("regimes", {})
        if not qid or not isinstance(regimes, dict):
            continue

        for raw_key, payload in regimes.items():
            canonical = JSONL_REGIME_KEY_MAP.get(raw_key, raw_key)
            if canonical not in regime_lists:
                regime_lists[canonical] = []
            if not isinstance(payload, dict):
                continue

            entry = {
                "id": qid,
                "question": question,
                "model": model_slug,
            }
            entry.update(payload)

            if "raw_response" not in entry and entry.get("raw_completion"):
                entry["raw_response"] = entry["raw_completion"]

            regime_lists[canonical].append(entry)

    return regime_lists


def parse_correctness_regime_from_name(filename: str, dataset: str) -> Optional[str]:
    prefix = f"{dataset}_"
    suffix = "_correctness.json"
    if not (filename.startswith(prefix) and filename.endswith(suffix)):
        return None
    return filename[len(prefix) : -len(suffix)]


def parse_consistency_pair_from_name(filename: str, dataset: str) -> Optional[Tuple[str, str]]:
    prefix = f"{dataset}_"
    suffix = "_consistency.json"
    if filename.startswith(prefix):
        body = filename[len(prefix) : -len(suffix)]
    else:
        body = filename
        if body.endswith(".json"):
            body = body[: -len(".json")]
        if body.endswith("_consistency"):
            body = body[: -len("_consistency")]
    if "_vs_" not in body:
        return None
    regime_a, regime_b = body.split("_vs_", 1)
    if not regime_a or not regime_b:
        return None
    return regime_a, regime_b


def load_correctness_lookup(path: Path) -> Dict[str, dict]:
    data = load_json(path) or {}
    if isinstance(data, dict):
        return {str(k): v for k, v in data.items() if isinstance(v, dict)}
    return {}


def load_consistency_bundle(path: Path) -> Dict[str, Any]:
    data = load_json(path) or {}
    comparisons = data.get("comparisons") or []
    summary = data.get("summary") or {}
    records = {str(item.get("id")): item for item in comparisons if item.get("id")}
    return {"summary": summary, "records": records}


def load_model_evaluations(dataset: str, model_slug: str) -> Dict[str, Any]:
    evaluations: Dict[str, Any] = {
        "correctness": {},
        "consistency": {},
    }

    corr_dir = CORRECTNESS_ROOT / dataset / model_slug
    if corr_dir.exists():
        for path in corr_dir.glob(f"{dataset}_*_correctness.json"):
            regime = parse_correctness_regime_from_name(path.name, dataset)
            canonical = canonicalize_regime_name(regime)
            if canonical:
                evaluations["correctness"][canonical] = load_correctness_lookup(path)

    cons_dir = CONSISTENCY_ROOT / dataset / model_slug
    if cons_dir.exists():
        for path in cons_dir.glob(f"{dataset}_*_vs_*_consistency.json"):
            pair = parse_consistency_pair_from_name(path.name, dataset)
            if pair:
                regime_a, regime_b = pair
                canonical_pair = (
                    canonicalize_regime_name(regime_a),
                    canonicalize_regime_name(regime_b),
                )
                if all(canonical_pair):
                    evaluations["consistency"][canonical_pair] = load_consistency_bundle(path)

    legacy_dir = EVALUATION_ROOT / dataset / model_slug
    if legacy_dir.exists():
        if not evaluations["correctness"]:
            for path in legacy_dir.glob(f"{dataset}_*_correctness.json"):
                regime = parse_correctness_regime_from_name(path.name, dataset)
                canonical = canonicalize_regime_name(regime)
                if canonical:
                    evaluations["correctness"][canonical] = load_correctness_lookup(path)
        if not evaluations["consistency"]:
            for path in legacy_dir.glob(f"{dataset}_*_vs_*_consistency.json"):
                pair = parse_consistency_pair_from_name(path.name, dataset)
                if pair:
                    regime_a, regime_b = pair
                    canonical_pair = (
                        canonicalize_regime_name(regime_a),
                        canonicalize_regime_name(regime_b),
                    )
                    if all(canonical_pair):
                        evaluations["consistency"][canonical_pair] = load_consistency_bundle(path)

    if dataset == "mintaka" and not evaluations["correctness"] and LEGACY_EVAL_CORRECTNESS.exists():
        evaluations["correctness"]["incremental"] = load_correctness_lookup(LEGACY_EVAL_CORRECTNESS)

    if dataset == "mintaka" and not evaluations["consistency"] and LEGACY_EVAL_CONSISTENCY.exists():
        evaluations["consistency"][("incremental", "open_ended")] = load_consistency_bundle(
            LEGACY_EVAL_CONSISTENCY
        )

    return evaluations


def aggregate_correctness_payloads(model_payloads: Dict[str, dict]) -> Dict[str, dict]:
    combined: Dict[str, dict] = {}
    for payload in model_payloads.values():
        correctness = payload.get("evaluations", {}).get("correctness", {})
        for lookup in correctness.values():
            for qid, record in lookup.items():
                prev = combined.get(qid)
                if not prev or (prev.get("correct") == 1 and record.get("correct") == 0):
                    combined[qid] = record
    return combined


def aggregate_consistency_payloads(model_payloads: Dict[str, dict]) -> Dict[str, dict]:
    combined: Dict[str, dict] = {}
    for payload in model_payloads.values():
        consistency = payload.get("evaluations", {}).get("consistency", {})
        for bundle in consistency.values():
            records = bundle.get("records", {})
            for qid, record in records.items():
                combined.setdefault(qid, record)
    return combined


def get_preferred_correctness(payload: dict) -> Tuple[Optional[str], Dict[str, dict]]:
    correctness = payload.get("evaluations", {}).get("correctness", {})
    if not correctness:
        return None, {}
    for key in ("incremental", "sequential"):
        if key in correctness:
            canonical = "incremental"
            return canonical, correctness[key]
    regime, lookup = next(iter(correctness.items()))
    return regime, lookup


def get_preferred_consistency(payload: dict) -> Tuple[Optional[Tuple[str, str]], Dict[str, Any]]:
    consistency = payload.get("evaluations", {}).get("consistency", {})
    if not consistency:
        return None, {"summary": {}, "records": {}}
    for candidate in [("incremental", "open_ended"), ("open_ended", "incremental")]:
        if candidate in consistency:
            return candidate, consistency[candidate]
    key, bundle = next(iter(consistency.items()))
    return key, bundle


def detect_regime_file_map(dataset: str, model_dir: Path) -> Dict[str, str]:
    regime_file_map = {
        regime: pattern.format(dataset=dataset)
        for regime, pattern in REGIME_FILE_TEMPLATES.items()
    }
    prefix = f"{dataset}_"
    for json_file in model_dir.glob(f"{dataset}_*.json"):
        stem = json_file.stem
        if stem.endswith("_legacy"):
            continue
        name = stem[len(prefix) :]
        canonical = canonicalize_regime_name(name)
        if canonical:
            regime_file_map[canonical] = json_file.name
    return regime_file_map


def load_regime_lists(dataset: str, model_slug: str, model_dir: Path) -> Tuple[Dict[str, List[dict]], Dict[str, Dict[str, dict]], Dict[str, str]]:
    regime_lists: Dict[str, List[dict]] = {}
    regime_indices: Dict[str, Dict[str, dict]] = {}
    regime_file_map = detect_regime_file_map(dataset, model_dir)
    missing_regimes: List[str] = []

    for regime, filename in regime_file_map.items():
        path = model_dir / filename
        data_list = load_json(path) or []
        regime_lists[regime] = data_list
        regime_indices[regime] = index_list_by_id(data_list) if data_list else {}
        if not data_list:
            missing_regimes.append(regime)

    if missing_regimes or not regime_lists:
        jsonl_records = find_jsonl_records(model_dir, dataset)
        fallback = convert_regimes_from_jsonl(jsonl_records, model_slug, regime_file_map.keys())
        for regime, records in fallback.items():
            if regime not in regime_lists or not regime_lists[regime]:
                regime_lists[regime] = records
                regime_indices[regime] = index_list_by_id(records)
            if regime not in regime_file_map and records:
                regime_file_map[regime] = f"{dataset}_{regime}.json"

    for regime in list(regime_lists.keys()):
        regime_indices.setdefault(regime, {})

    return regime_lists, regime_indices, regime_file_map


def warn_missing_regimes(
    model_slug: str,
    model_dir: Path,
    regime_lists: Dict[str, List[dict]],
    regime_file_map: Dict[str, str],
    dataset: str,
) -> None:
    for regime, records in regime_lists.items():
        if records:
            continue
        filename = regime_file_map.get(regime)
        expected_path = (model_dir / filename) if filename else None
        if expected_path and expected_path.exists():
            st.warning(
                f"[{model_slug}] Regime '{regime}' file exists but has no records.",
                icon="⚠️",
            )
        else:
            st.warning(
                f"[{model_slug}] No data for regime '{regime}' in {model_dir}. "
                f"Ensure JSON outputs or {dataset}_regimes*.jsonl are present.",
                icon="⚠️",
            )


def main():
    st.set_page_config(
        page_title="QA Pipeline Inspector",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("Multi-Hop QA Pipeline Inspector")

    available_datasets = list_available_datasets()
    if not available_datasets:
        st.error(f"No generation directories found under {GENERATION_ROOT}")
        return

    st.sidebar.title("Controls")
    dataset_index = 0
    if DEFAULT_DATASET in available_datasets:
        dataset_index = available_datasets.index(DEFAULT_DATASET)
    selected_dataset = st.sidebar.selectbox(
        "Dataset",
        options=available_datasets,
        index=dataset_index,
        format_func=get_dataset_label,
    )

    available_models = list_available_model_slugs(selected_dataset)
    if not available_models:
        st.error(
            f"No generation directories found for dataset '{selected_dataset}' under "
            f"{GENERATION_ROOT / selected_dataset}"
        )
        return

    default_selection = available_models.copy()
    if DEFAULT_MODEL_SLUG in available_models:
        available_models = sorted(available_models, key=lambda slug: (slug != DEFAULT_MODEL_SLUG, slug))

    selected_models = st.sidebar.multiselect(
        f"{get_dataset_label(selected_dataset)} model runs",
        options=available_models,
        default=default_selection,
    )
    if not selected_models:
        st.warning("Select at least one model run from the sidebar to inspect.")
        return

    st.caption(f"Dataset: {get_dataset_label(selected_dataset)} • Model runs: {', '.join(selected_models)}")

    # Load data
    dsl_dataset = resolve_dsl_dataset(selected_dataset)
    if not dsl_dataset:
        dsl_path = DATA_PROCESSED_ROOT / selected_dataset / f"{selected_dataset}_dsl.json"
        st.warning(
            f"Could not load DSL dataset for {get_dataset_label(selected_dataset)} from {dsl_path}.",
            icon="⚠️",
        )

    model_payloads: OrderedDict[str, dict] = OrderedDict()
    for model_slug in selected_models:
        model_dir = GENERATION_ROOT / selected_dataset / model_slug
        regime_lists, regime_indices, regime_file_map = load_regime_lists(selected_dataset, model_slug, model_dir)
        model_payloads[model_slug] = {
            "dir": model_dir,
            "regime_lists": regime_lists,
            "regime_indices": regime_indices,
            "regime_files": regime_file_map,
            "evaluations": load_model_evaluations(selected_dataset, model_slug),
        }
        warn_missing_regimes(model_slug, model_dir, regime_lists, regime_file_map, selected_dataset)

    cons_lookup = aggregate_consistency_payloads(model_payloads)
    corr_lookup = aggregate_correctness_payloads(model_payloads)

    # Indexes
    dsl_idx = index_dsl_by_id(dsl_dataset)

    # Filters
    all_ids = gather_all_ids(dsl_idx, model_payloads, cons_lookup, corr_lookup)
    filtered_ids, *_ = sidebar_filters(all_ids, dsl_idx, model_payloads, cons_lookup, corr_lookup)

    st.write(f"Showing {len(filtered_ids)} of {len(all_ids)} questions")

    # Selector + detail view
    selected_id = question_selector(filtered_ids, dsl_idx, model_payloads)
    if selected_id:
        render_question_view(selected_id, dsl_idx, model_payloads)
    else:
        st.info("No questions to display. Adjust filters or ensure files exist.")


if __name__ == "__main__":
    main()
