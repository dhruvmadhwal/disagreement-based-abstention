# DBA-A consistent-but-wrong (1-Recall) after filtering dataset issues

Filtered IDs (from hand review): total=155 bamboogle=12, crag=40, frames=21, hotpotqa=25, mintaka=29, musique=28

## After filtering

| Model | BAMBOOGLE | CRAG | FRAMES | HotpotQA | MINTAKA | MUSIQUE | Avg |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| GPT-5.1 | 0.0% | 27.3% | 36.7% | 23.7% | 13.3% | 36.4% | 22.9% |
| Gemini-2.5-Pro | 50.0% | 11.1% | 42.6% | 26.5% | 38.1% | 25.8% | 32.4% |
| Gemini-2.5-Flash | 11.1% | 9.1% | 24.7% | 24.7% | 24.1% | 23.4% | 19.5% |
| Llama-3.3-70B | 10.5% | 20.3% | 11.1% | 15.0% | 18.5% | 11.9% | 14.5% |
| Qwen2.5-72B | 19.2% | 16.2% | 14.3% | 15.8% | 17.1% | 15.0% | 16.3% |
| Qwen3-32B | 10.2% | 14.6% | -- | 15.6% | 14.0% | -- | 13.6% |
| Llama-3.1-8B | 15.3% | 15.0% | 9.2% | 22.0% | 21.5% | 7.3% | 15.0% |
| Qwen3-8B | 9.4% | 10.4% | 7.1% | 16.1% | 15.0% | 9.8% | 11.3% |
| Mistral-7B | 58.1% | 19.3% | 12.2% | 40.8% | 61.4% | 9.0% | 33.5% |


## Frontier-only (after filtering)

| Model | BAMBOOGLE | CRAG | FRAMES | HotpotQA | MINTAKA | MUSIQUE | Avg |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Gemini-2.5-Pro | 50.0% | 11.1% | 42.6% | 26.5% | 38.1% | 25.8% | 32.4% |
| GPT-5.1 | 0.0% | 27.3% | 36.7% | 23.7% | 13.3% | 36.4% | 22.9% |
| Gemini-2.5-Flash | 11.1% | 9.1% | 24.7% | 24.7% | 24.1% | 23.4% | 19.5% |
