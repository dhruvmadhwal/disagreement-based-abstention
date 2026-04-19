# DBA-A consistent-but-wrong (1-Recall) after filtering dataset issues

Filtered IDs (from hand review): exclude_needs_verification=True; total=271 bamboogle=16, crag=46, frames=41, hotpotqa=56, mintaka=31, musique=81

## After filtering

| Model | BAMBOOGLE | CRAG | FRAMES | HotpotQA | MINTAKA | MUSIQUE | Avg |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| GPT-5.1 | 0.0% | 11.8% | 26.2% | 4.8% | 7.1% | 15.8% | 10.9% |
| Gemini-2.5-Pro | 33.3% | 0.0% | 29.2% | 7.1% | 31.6% | 11.8% | 18.8% |
| Gemini-2.5-Flash | 16.7% | 5.0% | 24.7% | 16.1% | 22.2% | 15.2% | 16.7% |
| Llama-3.3-70B | 10.9% | 18.9% | 11.1% | 12.8% | 18.7% | 9.4% | 13.6% |
| Qwen2.5-72B | 20.3% | 17.6% | 13.7% | 16.2% | 17.2% | 12.8% | 16.3% |
| Qwen3-32B | 10.7% | 13.2% | -- | 15.1% | 14.1% | -- | 13.3% |
| Llama-3.1-8B | 14.6% | 14.9% | 9.1% | 22.7% | 21.6% | 5.8% | 14.8% |
| Qwen3-8B | 9.8% | 11.0% | 7.2% | 16.5% | 15.1% | 5.9% | 10.9% |
| Mistral-7B | 58.4% | 18.2% | 12.7% | 41.6% | 61.6% | 9.8% | 33.7% |


## Frontier-only (after filtering)

| Model | BAMBOOGLE | CRAG | FRAMES | HotpotQA | MINTAKA | MUSIQUE | Avg |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Gemini-2.5-Pro | 33.3% | 0.0% | 29.2% | 7.1% | 31.6% | 11.8% | 18.8% |
| Gemini-2.5-Flash | 16.7% | 5.0% | 24.7% | 16.1% | 22.2% | 15.2% | 16.7% |
| GPT-5.1 | 0.0% | 11.8% | 26.2% | 4.8% | 7.1% | 15.8% | 10.9% |
