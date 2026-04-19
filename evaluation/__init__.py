"""LLM-judge evaluation for multi-hop QA consistency.

Two judges, both pointed at Gemini 2.5 Flash via Vertex AI:

  - `evaluation.correctness` — score each (regime, model) generation against
    the gold answer in `data/processed/<dataset>/<dataset>_dsl.json`.
  - `evaluation.consistency` — pairwise compare two regimes' answers for the
    same question (DBA-A: disagreement-based abstention).

Both judges share the Vertex client and JSON parsing in `evaluation.judge`,
and dispatch on the per-dataset prompt module via `evaluation.specs`.
"""
