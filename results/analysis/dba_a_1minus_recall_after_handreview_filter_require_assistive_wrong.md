# DBA-A consistent-but-wrong (1-Recall) after filtering dataset issues

Filtered IDs (from hand review): exclude_needs_verification=False; require_assistive_wrong=True; total=155 bamboogle=12, crag=40, frames=21, hotpotqa=25, mintaka=29, musique=28

## After filtering

| Model | BAMBOOGLE | CRAG | FRAMES | HotpotQA | MINTAKA | MUSIQUE | Avg |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| GPT-5.1 | 0.0% | 23.8% | 35.6% | 23.7% | 13.3% | 35.2% | 21.9% |
| Gemini-2.5-Pro | 44.4% | 5.9% | 41.8% | 24.2% | 23.5% | 24.6% | 27.4% |
| Gemini-2.5-Flash | 11.1% | 9.1% | 24.7% | 24.7% | 12.0% | 21.9% | 17.3% |
| Llama-3.3-70B | 7.3% | 16.1% | 10.6% | 13.1% | 14.8% | 11.9% | 12.3% |
| Qwen2.5-72B | 15.7% | 15.1% | 13.5% | 13.6% | 14.9% | 15.0% | 14.6% |
| Qwen3-32B | 10.2% | 12.5% | -- | 14.5% | 12.9% | -- | 12.5% |
| Llama-3.1-8B | 13.3% | 13.9% | 8.8% | 19.3% | 16.7% | 6.8% | 13.1% |
| Qwen3-8B | 7.4% | 9.5% | 7.1% | 13.4% | 14.1% | 9.8% | 10.2% |
| Mistral-7B | 47.3% | 14.1% | 11.4% | 38.1% | 46.4% | 8.2% | 27.6% |


## Frontier-only (after filtering)

| Model | BAMBOOGLE | CRAG | FRAMES | HotpotQA | MINTAKA | MUSIQUE | Avg |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| GPT-5.1 | 0.0% | 23.8% | 35.6% | 23.7% | 13.3% | 35.2% | 21.9% |
| Gemini-2.5-Pro | 44.4% | 5.9% | 41.8% | 24.2% | 23.5% | 24.6% | 27.4% |
| Gemini-2.5-Flash | 11.1% | 9.1% | 24.7% | 24.7% | 12.0% | 21.9% | 17.3% |
