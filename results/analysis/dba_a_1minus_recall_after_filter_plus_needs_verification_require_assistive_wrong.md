# DBA-A consistent-but-wrong (1-Recall) after filtering dataset issues

Filtered IDs (from hand review): exclude_needs_verification=True; require_assistive_wrong=True; total=271 bamboogle=16, crag=46, frames=41, hotpotqa=56, mintaka=31, musique=81

## After filtering

| Model | BAMBOOGLE | CRAG | FRAMES | HotpotQA | MINTAKA | MUSIQUE | Avg |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| GPT-5.1 | 0.0% | 11.8% | 24.4% | 4.8% | 7.1% | 12.7% | 10.1% |
| Gemini-2.5-Pro | 33.3% | 0.0% | 29.2% | 4.9% | 18.8% | 10.7% | 16.1% |
| Gemini-2.5-Flash | 16.7% | 5.0% | 24.7% | 16.1% | 8.7% | 14.4% | 14.3% |
| Llama-3.3-70B | 7.5% | 14.0% | 10.6% | 10.4% | 14.9% | 9.4% | 11.1% |
| Qwen2.5-72B | 16.7% | 16.4% | 12.9% | 13.5% | 15.0% | 12.8% | 14.6% |
| Qwen3-32B | 10.7% | 10.8% | -- | 13.9% | 13.0% | -- | 12.1% |
| Llama-3.1-8B | 12.5% | 13.7% | 8.7% | 19.6% | 16.8% | 5.2% | 12.7% |
| Qwen3-8B | 7.8% | 10.0% | 7.2% | 13.8% | 14.2% | 5.9% | 9.8% |
| Mistral-7B | 47.1% | 12.5% | 11.9% | 38.4% | 46.8% | 8.8% | 27.6% |


## Frontier-only (after filtering)

| Model | BAMBOOGLE | CRAG | FRAMES | HotpotQA | MINTAKA | MUSIQUE | Avg |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Gemini-2.5-Flash | 16.7% | 5.0% | 24.7% | 16.1% | 8.7% | 14.4% | 14.3% |
| GPT-5.1 | 0.0% | 11.8% | 24.4% | 4.8% | 7.1% | 12.7% | 10.1% |
| Gemini-2.5-Pro | 33.3% | 0.0% | 29.2% | 4.9% | 18.8% | 10.7% | 16.1% |
