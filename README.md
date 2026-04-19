# Decomposed Prompting Does Not Fix Knowledge Gaps, But Helps Models
Say “I Don’t Know”

Code and data for **"Decomposed Prompting Does Not Fix Knowledge Gaps,
But Helps Models Say 'I Don't Know'"** [ACL 2026 Findings].
We probe LLM reliability on multi-hop QA by eliciting the
same question under three execution interfaces — **Direct / Assistive /
Incremental** — and treating disagreement across interfaces as a signal
of fragile grounding. The repository reproduces the paper's accuracy,
consistency, failure-mode, and selective-answering (DBA-A) results on
Bamboogle, CRAG, HotpotQA, and Mintaka.

---

## Repository layout

```
.
├── data/
│   ├── raw/              Original downloads, per dataset.
│   └── processed/        Filtered, DSL-annotated gold set (*_dsl.json).
├── prompts/              Few-shots + judge prompts, per dataset.
├── generate/             Answer-generation pipelines (4 regimes).
│   ├── {bamboogle,crag,hotpotqa,mintaka}/   Per-dataset config + pipeline.
│   ├── {open_ended,assistive,sequential,model_generated_plan}/  Interface shims.
│   └── run_<dataset>_regimes.py             All 4 regimes in one call.
├── evaluation/
│   ├── correctness/      LLM-judged correctness (Gemini 2.5 Flash).
│   └── consistency/      DBA-A: pairwise answer agreement across regimes.
├── baselines/            AYS, IC-IDK, DBA-A, Self-Consistency abstention.
├── decomposition_equivalence/   Model-produced DSL vs gold (appendix F).
├── create_dsl/           Convert raw datasets to DSL decompositions.
├── analyze/              Paper figure + table generators.
│   ├── figures/          PNG outputs.
│   └── derived/          TSVs/JSONs consumed by figure generators.
├── scripts/
│   ├── run.py            Single entry point for generate/evaluate/baseline.
│   ├── compute_all_ensemble_metrics.py
│   ├── generate_efficiency_{method_comparison,overhead}_table.py
│   ├── generate_appendix_baseline_tables.py
│   ├── self_consistency_semantic_sweep.py
│   └── format_self_consistency_semantic_sweep_table.py
├── tables/               LaTeX tables shipped with the paper.
├── results/
│   ├── generation/<dataset>/<model>/<regime>.json
│   ├── correctness/<dataset>/<model>/<dataset>_<regime>_correctness.json
│   ├── consistency/<dataset>/<model>/<dataset>_<regime>_vs_open_ended_consistency.json
│   └── baselines/<dataset>/<model>/*.json
├── utils/                Model clients (OpenAI, Vertex AI, vLLM) + DSL sanity.
└── requirements.txt
```

---

## Setup

```bash
git clone <repo-url> multi-hop-qa-consistency && cd $_
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**API credentials**. Export the keys you need:

```bash
OPENAI_API_KEY=...           # GPT-5.1 generation, AYS/self-consistency judges
GOOGLE_APPLICATION_CREDENTIALS=path/to/vertex.json   # Gemini 2.5 Flash/Pro judges
```

Local open-weights models (Qwen, Llama, Gemma, Mistral) are served via
[vLLM](https://github.com/vllm-project/vllm); generators launch the
server automatically.

---

## Quick tour

Every paper pipeline stage is accessible through **one** dispatcher:

```bash
python scripts/run.py {generate|evaluate|baseline|decomp-gen|decomp-eval} --dataset <name> ...
```

### 1. Generate answers across the four reasoning regimes

```bash
python scripts/run.py generate --dataset hotpotqa \
  --model-choice qwen-3-8b --model-name Qwen/Qwen3-8B --resume
```

Writes `results/generation/hotpotqa/<model_slug>/hotpotqa_{open_ended,
assistive,sequential,model_generated}.json` plus a combined JSONL and
`token_summary.json`.

Supported datasets: `bamboogle | crag | hotpotqa | mintaka`.
Supported model presets (see `generate/<dataset>/config.py`): `gpt5.1`,
`gemini-2.5-flash`, `gemini-2.5-pro`, `qwen-3-8b`, `qwen-3-32b`,
`qwen-2.5-72b-instruct`, `llama-3.1-8b-instruct`,
`llama-3.3-70b-instruct`, `mistralai/mistral-7b-instruct-v0.3`,
`gemma-3-4b-it`.

### 2. Evaluate generations

```bash
python scripts/run.py evaluate --dataset hotpotqa \
  --model <model_slug> --run-correctness --resume
```

Consistency (DBA-A judge) runs by default for each non-Direct regime
vs Direct; `--run-correctness` also scores each regime against gold
via the Gemini judge. Outputs land in
`results/{correctness,consistency}/<dataset>/<model>/`.

### 3. Abstention baselines

```bash
python scripts/run.py baseline --dataset hotpotqa \
  --model <model_slug>
```

Runs AYS, IC-IDK, Self-Consistency, and DBA-A abstention. Outputs land
in `results/baselines/<dataset>/<model>/`. DBA-A reuses the consistency
judgments from step 2.

### 4. Decomposition-equivalence ablation (appendix F)

```bash
python scripts/run.py decomp-gen  --model-slug qwen-3-8b \
    --model-name Qwen/Qwen3-8B
python scripts/run.py decomp-eval --model-slug qwen-3-8b
```

Compares model-produced DSL decompositions against the gold DSL across
all four datasets; writes to
`decomposition_equivalence/results/{dsl,eval,summary}/`.

---

## Reproducing the paper

The `results/` tree in this release already has the judge and baseline
outputs for every paper model, so re-running the figure/table
generators is sufficient:

```bash
# Figures — main-paper set (writes to ../Consistency-in-Multi-Hop-QA/latex/figures/)
python analyze/generate_paper_figures.py

# Figures — summary/heatmap set (writes to analyze/figures/)
python analyze/generate_summary_figures.py

# Tables — main-paper set
python analyze/generate_tables.py

# Appendix baseline tables
python scripts/generate_appendix_baseline_tables.py

# Efficiency tables (cost / token / latency)
python scripts/generate_efficiency_overhead_table.py
python scripts/generate_efficiency_method_comparison_table.py

# Self-consistency semantic-variant sweep table (Bamboogle)
python scripts/self_consistency_semantic_sweep.py             # -> TSV
python scripts/format_self_consistency_semantic_sweep_table.py   # -> .tex

# (Re)compute ensemble metrics into analyze/derived/baselines_all.tsv
python scripts/compute_all_ensemble_metrics.py
```

### Paper artifact → script map

| Paper artifact | Script |
|---|---|
| Fig. 1 — accuracy / consistency by model × regime | `analyze/generate_paper_figures.py::plot_accuracy_heatmaps` / `plot_consistency_heatmaps` |
| Fig. 2 — P(correct \| m) vs agreement score | `analyze/generate_paper_figures.py::plot_accuracy_vs_consistency` |
| Fig. 3 — risk–coverage curves | `analyze/generate_paper_figures.py::plot_baseline_comparison` |
| Table 1 — accuracy by model / regime | `analyze/generate_tables.py::generate_accuracy_table` |
| Table 2 — consistency by model / regime | `analyze/generate_tables.py::generate_consistency_table` |
| Table 3 — decomposition equivalence | `analyze/decomposition_equivalence_summary.py` |
| Table 4 — efficiency / cost | `scripts/generate_efficiency_method_comparison_table.py` |
| Table 5 — baseline PRF | `analyze/generate_tables.py::generate_baseline_prf_table` |
| Appendix A2 — pairwise regime agreement | `analyze/generate_summary_figures.py::_plot_benchmark_approach_heatmaps` |
| Appendix baseline tables | `scripts/generate_appendix_baseline_tables.py` |
| Appendix F — decomp-equivalence breakdown | `analyze/decomposition_equivalence_summary.py` |

---

## DSL workflow (if regenerating from raw)

The paper reports on a *filtered* gold DSL subset (single-hop,
time-dependent, and semantically ambiguous questions removed). The
already-filtered DSLs live in `data/processed/<dataset>/<dataset>_dsl.json`
and are sufficient to reproduce everything above. To rebuild from raw:

```bash
python create_dsl/convert_<dataset>_to_dsl.py   # raw -> processed
```

Filter rules live in `utils/dsl_sanity.py`.

---

## Models evaluated

Ten LLMs spanning three providers:

- **OpenAI**: GPT-5.1
- **Google Vertex AI**: Gemini 2.5 Flash, Gemini 2.5 Pro, Gemma 3 4B IT
- **Open-weights (vLLM)**: Qwen 3 8B, Qwen 3 32B, Qwen 2.5 72B Instruct,
  Llama 3.1 8B Instruct, Llama 3.3 70B Instruct, Mistral 7B Instruct v0.3

---

## Citation

Preprint: <https://arxiv.org/abs/2602.04853>

```bibtex
@inproceedings{madhwal2026decomposed,
  title     = {Decomposed Prompting Does Not Fix Knowledge Gaps,
               But Helps Models Say "{I} Don't Know"},
  author    = {Madhwal, Dhruv and Zhang, Lyuxin David and
               Roth, Dan and Wolfson, Tomer and Gupta, Vivek},
  year      = {2026},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2026},
  url       = {https://arxiv.org/abs/2602.04853},
}
```

---

## License

Code released under the MIT License. Dataset licenses follow the
upstream authors (Bamboogle, CRAG, HotpotQA, Mintaka) — see
`data/raw/<dataset>/LICENSE` where applicable.
