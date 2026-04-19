# Decomposition Equivalence

This folder builds DSL decompositions for multiple models and evaluates them against the gold DSLs using Gemini 2.5 Flash as an LLM judge.

## What it does

- **DSL generation** per dataset + model (same few-shot prompt as used to build gold DSLs).
- **LLM judging** with Gemini 2.5 Flash: semantic equivalence + hop coverage count.
- **Manual metrics** computed from judge output: hop count ratio/diff, precision/recall/F1.

## Inputs

- **Gold DSLs**: `data/processed/<dataset>/<dataset>_dsl.json` (Gemini 2.5 Flash gold).
- **Model DSLs**: written to `decomposition_equivalence/results/dsl/<dataset>/<model_slug>.json`.

## Generate DSLs

Generate DSLs for a dataset/model:

```bash
python decomposition_equivalence/generate_dsls.py \
  --dataset musique \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --model-slug llama-3.1-8B-Instruct \
  --vllm-base-url http://127.0.0.1:8000/v1
```

Use Vertex Gemini instead (e.g., to regenerate DSLs with Gemini):

```bash
python decomposition_equivalence/generate_dsls.py \
  --dataset musique \
  --model-name google/gemini-2.5-pro \
  --use-vertex
```

Infer the model name from existing generation outputs:

```bash
python decomposition_equivalence/generate_dsls.py \
  --dataset musique \
  --model-slug llama-3.1-8B-Instruct \
  --from-generation
```

## Evaluate Decomposition Equivalence

Run the judge + compute metrics:

```bash
python decomposition_equivalence/evaluate_decomposition.py \
  --dataset musique \
  --model-slug llama-3.1-8B-Instruct
```

### Batch judging (Gemini)

Use Vertex batch prediction to reduce costs. This writes a JSONL input file, optionally uploads it to GCS, and submits a batch job.

```bash
# Qwen3-8B
python decomposition_equivalence/evaluate_decomposition.py \
  --dataset hotpotqa \
  --model-slug qwen-qwen3-8b \
  --batch \
  --batch-input-uri gs://YOUR_BUCKET/decomp_equiv/hotpotqa/qwen-qwen3-8b.jsonl \
  --batch-output-uri gs://YOUR_BUCKET/decomp_equiv/output/ \
  --batch-upload

# Qwen3-32B
python decomposition_equivalence/evaluate_decomposition.py \
  --dataset hotpotqa \
  --model-slug qwen-qwen3-32b \
  --batch \
  --batch-input-uri gs://YOUR_BUCKET/decomp_equiv/hotpotqa/qwen-qwen3-32b.jsonl \
  --batch-output-uri gs://YOUR_BUCKET/decomp_equiv/output/ \
  --batch-upload
```

After the batch job finishes, fetch and parse outputs:

```bash
# Qwen3-8B
python decomposition_equivalence/evaluate_decomposition.py \
  --dataset hotpotqa \
  --model-slug qwen-qwen3-8b \
  --batch-fetch \
  --batch-output-uri gs://YOUR_BUCKET/decomp_equiv/output/ \
  --resume

# Qwen3-32B
python decomposition_equivalence/evaluate_decomposition.py \
  --dataset hotpotqa \
  --model-slug qwen-qwen3-32b \
  --batch-fetch \
  --batch-output-uri gs://YOUR_BUCKET/decomp_equiv/output/ \
  --resume
```

Outputs:

- Per-example results: `decomposition_equivalence/results/eval/<dataset>/<model_slug>.json`
- Summary metrics: `decomposition_equivalence/results/summary/<dataset>/<model_slug>.json`

## Metrics

- **Hop count**: average ratio/difference between model and gold hops.
- **Semantic equivalence**: `equivalent_final` from the LLM judge.
- **Hop coverage**: judge returns `matches` = # gold hops covered by any model hop.
- **Precision / Recall / F1**: computed from `matches`, gold hop count, and model hop count.

## Notes

- Generation prompts match the gold DSL construction prompt (few-shot examples included).
- Precision/recall/F1 are computed locally; coverage data comes from the LLM judge.
- Use `--resume` to continue partial runs.
