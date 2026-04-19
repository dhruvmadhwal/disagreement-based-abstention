# Baseline Evaluation Results

Source: analysis/derived/baseline_summary.json

## bamboogle — google-gemini-2-5-flash

| baseline | total | tp | fp | tn | fn | coverage | acc_at_cov | overall_acc | precision | recall | f1 | path |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ays | 492 | 32 | 16 | 396 | 48 | 0.902 | 0.892 | 0.805 | 0.667 | 0.400 | 0.500 | results/baselines/bamboogle/google-gemini-2-5-flash/bamboogle_ays.json |
| idk | 492 | 4 | 0 | 412 | 76 | 0.992 | 0.844 | 0.837 | 1.000 | 0.050 | 0.095 | results/baselines/bamboogle/google-gemini-2-5-flash/bamboogle_idk.json |
| pairwise_assistive | 123 | 12 | 16 | 87 | 8 | 0.772 | 0.916 | 0.707 | 0.429 | 0.600 | 0.500 | results/baselines/bamboogle/google-gemini-2-5-flash/bamboogle_pairwise_assistive.json |
| pairwise_incremental | 123 | 12 | 9 | 94 | 8 | 0.829 | 0.922 | 0.764 | 0.571 | 0.600 | 0.585 | results/baselines/bamboogle/google-gemini-2-5-flash/bamboogle_pairwise_incremental.json |
| pairwise_model_generated | 123 | 13 | 11 | 92 | 7 | 0.805 | 0.929 | 0.748 | 0.542 | 0.650 | 0.591 | results/baselines/bamboogle/google-gemini-2-5-flash/bamboogle_pairwise_model_generated.json |
| self_consistency | 123 | 7 | 7 | 96 | 13 | 0.886 | 0.881 | 0.780 | 0.500 | 0.350 | 0.412 | results/baselines/bamboogle/google-gemini-2-5-flash/bamboogle_self_consistency.json |

## bamboogle — google-gemini-2-5-pro

| baseline | total | tp | fp | tn | fn | coverage | acc_at_cov | overall_acc | precision | recall | f1 | path |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ays | 246 | 14 | 4 | 206 | 22 | 0.927 | 0.903 | 0.837 | 0.778 | 0.389 | 0.518 | results/baselines/bamboogle/google-gemini-2-5-pro/bamboogle_ays.json |
| idk | 246 | 2 | 0 | 210 | 34 | 0.992 | 0.861 | 0.854 | 1.000 | 0.056 | 0.105 | results/baselines/bamboogle/google-gemini-2-5-pro/bamboogle_idk.json |
| pairwise_assistive | 123 | 8 | 6 | 99 | 10 | 0.886 | 0.908 | 0.805 | 0.571 | 0.444 | 0.500 | results/baselines/bamboogle/google-gemini-2-5-pro/bamboogle_pairwise_assistive.json |
| pairwise_incremental | 123 | 8 | 4 | 101 | 10 | 0.902 | 0.910 | 0.821 | 0.667 | 0.444 | 0.533 | results/baselines/bamboogle/google-gemini-2-5-pro/bamboogle_pairwise_incremental.json |
| pairwise_model_generated | 123 | 9 | 8 | 97 | 9 | 0.862 | 0.915 | 0.789 | 0.529 | 0.500 | 0.514 | results/baselines/bamboogle/google-gemini-2-5-pro/bamboogle_pairwise_model_generated.json |
| self_consistency | 123 | 3 | 3 | 102 | 15 | 0.951 | 0.872 | 0.829 | 0.500 | 0.167 | 0.250 | results/baselines/bamboogle/google-gemini-2-5-pro/bamboogle_self_consistency.json |

## bamboogle — google-gemma-3-4b-it

| baseline | total | tp | fp | tn | fn | coverage | acc_at_cov | overall_acc | precision | recall | f1 | path |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ays | 246 | 98 | 6 | 16 | 126 | 0.577 | 0.113 | 0.065 | 0.942 | 0.438 | 0.598 | results/baselines/bamboogle/google-gemma-3-4b-it/bamboogle_ays.json |
| idk | 246 | 26 | 0 | 22 | 198 | 0.894 | 0.100 | 0.089 | 1.000 | 0.116 | 0.208 | results/baselines/bamboogle/google-gemma-3-4b-it/bamboogle_idk.json |
| pairwise_assistive | 123 | 20 | 3 | 8 | 92 | 0.813 | 0.080 | 0.065 | 0.870 | 0.179 | 0.296 | results/baselines/bamboogle/google-gemma-3-4b-it/bamboogle_pairwise_assistive.json |
| pairwise_incremental | 123 | 36 | 2 | 9 | 76 | 0.691 | 0.106 | 0.073 | 0.947 | 0.321 | 0.480 | results/baselines/bamboogle/google-gemma-3-4b-it/bamboogle_pairwise_incremental.json |
| pairwise_model_generated | 123 | 28 | 2 | 9 | 84 | 0.756 | 0.097 | 0.073 | 0.933 | 0.250 | 0.394 | results/baselines/bamboogle/google-gemma-3-4b-it/bamboogle_pairwise_model_generated.json |
| self_consistency | 123 | 1 | 0 | 11 | 111 | 0.992 | 0.090 | 0.089 | 1.000 | 0.009 | 0.018 | results/baselines/bamboogle/google-gemma-3-4b-it/bamboogle_self_consistency.json |

## bamboogle — gpt-5-1

| baseline | total | tp | fp | tn | fn | coverage | acc_at_cov | overall_acc | precision | recall | f1 | path |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ays | 123 | 7 | 4 | 100 | 12 | 0.911 | 0.893 | 0.813 | 0.636 | 0.368 | 0.467 | results/baselines/bamboogle/gpt-5-1/bamboogle_ays.json |
| idk | 123 | 6 | 3 | 101 | 13 | 0.927 | 0.886 | 0.821 | 0.667 | 0.316 | 0.429 | results/baselines/bamboogle/gpt-5-1/bamboogle_idk.json |
| pairwise_assistive | 123 | 12 | 2 | 102 | 7 | 0.886 | 0.936 | 0.829 | 0.857 | 0.632 | 0.727 | results/baselines/bamboogle/gpt-5-1/bamboogle_pairwise_assistive.json |
| pairwise_incremental | 123 | 9 | 4 | 100 | 10 | 0.894 | 0.909 | 0.813 | 0.692 | 0.474 | 0.562 | results/baselines/bamboogle/gpt-5-1/bamboogle_pairwise_incremental.json |
| pairwise_model_generated | 123 | 10 | 10 | 94 | 9 | 0.837 | 0.913 | 0.764 | 0.500 | 0.526 | 0.513 | results/baselines/bamboogle/gpt-5-1/bamboogle_pairwise_model_generated.json |

## bamboogle — meta-llama-llama-3-1-8b-instruct

| baseline | total | tp | fp | tn | fn | coverage | acc_at_cov | overall_acc | precision | recall | f1 | path |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ays | 246 | 96 | 8 | 46 | 96 | 0.577 | 0.324 | 0.187 | 0.923 | 0.500 | 0.649 | results/baselines/bamboogle/meta-llama-llama-3-1-8b-instruct/bamboogle_ays.json |
| idk | 246 | 10 | 2 | 50 | 184 | 0.951 | 0.214 | 0.203 | 0.833 | 0.051 | 0.097 | results/baselines/bamboogle/meta-llama-llama-3-1-8b-instruct/bamboogle_idk.json |
| pairwise_assistive | 123 | 81 | 6 | 20 | 16 | 0.293 | 0.556 | 0.163 | 0.931 | 0.835 | 0.880 | results/baselines/bamboogle/meta-llama-llama-3-1-8b-instruct/bamboogle_pairwise_assistive.json |
| pairwise_incremental | 123 | 67 | 5 | 21 | 30 | 0.415 | 0.412 | 0.171 | 0.931 | 0.691 | 0.793 | results/baselines/bamboogle/meta-llama-llama-3-1-8b-instruct/bamboogle_pairwise_incremental.json |
| pairwise_model_generated | 123 | 72 | 9 | 17 | 25 | 0.342 | 0.405 | 0.138 | 0.889 | 0.742 | 0.809 | results/baselines/bamboogle/meta-llama-llama-3-1-8b-instruct/bamboogle_pairwise_model_generated.json |
| self_consistency | 123 | 7 | 0 | 26 | 90 | 0.943 | 0.224 | 0.211 | 1.000 | 0.072 | 0.135 | results/baselines/bamboogle/meta-llama-llama-3-1-8b-instruct/bamboogle_self_consistency.json |

## bamboogle — meta-llama-llama-3-3-70b-instruct

| baseline | total | tp | fp | tn | fn | coverage | acc_at_cov | overall_acc | precision | recall | f1 | path |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ays | 246 | 50 | 12 | 100 | 84 | 0.748 | 0.543 | 0.406 | 0.806 | 0.373 | 0.510 | results/baselines/bamboogle/meta-llama-llama-3-3-70b-instruct/bamboogle_ays.json |
| idk | 246 | 48 | 18 | 94 | 86 | 0.732 | 0.522 | 0.382 | 0.727 | 0.358 | 0.480 | results/baselines/bamboogle/meta-llama-llama-3-3-70b-instruct/bamboogle_idk.json |
| pairwise_assistive | 123 | 60 | 4 | 52 | 7 | 0.480 | 0.881 | 0.423 | 0.938 | 0.895 | 0.916 | results/baselines/bamboogle/meta-llama-llama-3-3-70b-instruct/bamboogle_pairwise_assistive.json |
| pairwise_incremental | 123 | 54 | 7 | 49 | 13 | 0.504 | 0.790 | 0.398 | 0.885 | 0.806 | 0.844 | results/baselines/bamboogle/meta-llama-llama-3-3-70b-instruct/bamboogle_pairwise_incremental.json |
| pairwise_model_generated | 123 | 61 | 12 | 44 | 6 | 0.406 | 0.880 | 0.358 | 0.836 | 0.910 | 0.871 | results/baselines/bamboogle/meta-llama-llama-3-3-70b-instruct/bamboogle_pairwise_model_generated.json |
| self_consistency | 123 | 1 | 0 | 56 | 66 | 0.992 | 0.459 | 0.455 | 1.000 | 0.015 | 0.029 | results/baselines/bamboogle/meta-llama-llama-3-3-70b-instruct/bamboogle_self_consistency.json |

## bamboogle — mistralai-mistral-7b-instruct-v0-3

| baseline | total | tp | fp | tn | fn | coverage | acc_at_cov | overall_acc | precision | recall | f1 | path |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ays | 246 | 78 | 4 | 34 | 130 | 0.667 | 0.207 | 0.138 | 0.951 | 0.375 | 0.538 | results/baselines/bamboogle/mistralai-mistral-7b-instruct-v0-3/bamboogle_ays.json |
| idk | 246 | 46 | 4 | 34 | 162 | 0.797 | 0.173 | 0.138 | 0.920 | 0.221 | 0.357 | results/baselines/bamboogle/mistralai-mistral-7b-instruct-v0-3/bamboogle_idk.json |
| pairwise_assistive | 123 | 45 | 3 | 16 | 59 | 0.610 | 0.213 | 0.130 | 0.938 | 0.433 | 0.592 | results/baselines/bamboogle/mistralai-mistral-7b-instruct-v0-3/bamboogle_pairwise_assistive.json |
| pairwise_incremental | 123 | 42 | 4 | 15 | 62 | 0.626 | 0.195 | 0.122 | 0.913 | 0.404 | 0.560 | results/baselines/bamboogle/mistralai-mistral-7b-instruct-v0-3/bamboogle_pairwise_incremental.json |
| pairwise_model_generated | 123 | 33 | 4 | 15 | 71 | 0.699 | 0.174 | 0.122 | 0.892 | 0.317 | 0.468 | results/baselines/bamboogle/mistralai-mistral-7b-instruct-v0-3/bamboogle_pairwise_model_generated.json |
| self_consistency | 123 | 15 | 1 | 18 | 89 | 0.870 | 0.168 | 0.146 | 0.938 | 0.144 | 0.250 | results/baselines/bamboogle/mistralai-mistral-7b-instruct-v0-3/bamboogle_self_consistency.json |

## bamboogle — qwen-qwen2-5-72b-instruct

| baseline | total | tp | fp | tn | fn | coverage | acc_at_cov | overall_acc | precision | recall | f1 | path |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ays | 246 | 106 | 16 | 62 | 62 | 0.504 | 0.500 | 0.252 | 0.869 | 0.631 | 0.731 | results/baselines/bamboogle/qwen-qwen2-5-72b-instruct/bamboogle_ays.json |
| idk | 246 | 146 | 40 | 38 | 22 | 0.244 | 0.633 | 0.154 | 0.785 | 0.869 | 0.825 | results/baselines/bamboogle/qwen-qwen2-5-72b-instruct/bamboogle_idk.json |
| pairwise_assistive | 123 | 69 | 4 | 35 | 15 | 0.406 | 0.700 | 0.285 | 0.945 | 0.821 | 0.879 | results/baselines/bamboogle/qwen-qwen2-5-72b-instruct/bamboogle_pairwise_assistive.json |
| pairwise_incremental | 123 | 61 | 2 | 37 | 23 | 0.488 | 0.617 | 0.301 | 0.968 | 0.726 | 0.830 | results/baselines/bamboogle/qwen-qwen2-5-72b-instruct/bamboogle_pairwise_incremental.json |
| pairwise_model_generated | 123 | 70 | 8 | 31 | 14 | 0.366 | 0.689 | 0.252 | 0.897 | 0.833 | 0.864 | results/baselines/bamboogle/qwen-qwen2-5-72b-instruct/bamboogle_pairwise_model_generated.json |
| self_consistency | 123 | 15 | 2 | 37 | 69 | 0.862 | 0.349 | 0.301 | 0.882 | 0.179 | 0.297 | results/baselines/bamboogle/qwen-qwen2-5-72b-instruct/bamboogle_self_consistency.json |

## bamboogle — qwen-qwen3-32b

| baseline | total | tp | fp | tn | fn | coverage | acc_at_cov | overall_acc | precision | recall | f1 | path |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ays | 123 | 56 | 6 | 17 | 44 | 0.496 | 0.279 | 0.138 | 0.903 | 0.560 | 0.691 | results/baselines/bamboogle/qwen-qwen3-32b/bamboogle_ays.json |
| idk | 123 | 15 | 1 | 22 | 85 | 0.870 | 0.206 | 0.179 | 0.938 | 0.150 | 0.259 | results/baselines/bamboogle/qwen-qwen3-32b/bamboogle_idk.json |
| pairwise_assistive | 123 | 89 | 5 | 18 | 11 | 0.236 | 0.621 | 0.146 | 0.947 | 0.890 | 0.917 | results/baselines/bamboogle/qwen-qwen3-32b/bamboogle_pairwise_assistive.json |
| pairwise_incremental | 123 | 89 | 6 | 17 | 11 | 0.228 | 0.607 | 0.138 | 0.937 | 0.890 | 0.913 | results/baselines/bamboogle/qwen-qwen3-32b/bamboogle_pairwise_incremental.json |
| pairwise_model_generated | 123 | 92 | 11 | 12 | 8 | 0.163 | 0.600 | 0.098 | 0.893 | 0.920 | 0.906 | results/baselines/bamboogle/qwen-qwen3-32b/bamboogle_pairwise_model_generated.json |

## bamboogle — qwen-qwen3-8b

| baseline | total | tp | fp | tn | fn | coverage | acc_at_cov | overall_acc | precision | recall | f1 | path |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ays | 246 | 128 | 8 | 22 | 88 | 0.447 | 0.200 | 0.089 | 0.941 | 0.593 | 0.727 | results/baselines/bamboogle/qwen-qwen3-8b/bamboogle_ays.json |
| idk | 246 | 94 | 6 | 24 | 122 | 0.594 | 0.164 | 0.098 | 0.940 | 0.435 | 0.595 | results/baselines/bamboogle/qwen-qwen3-8b/bamboogle_idk.json |
| pairwise_assistive | 123 | 97 | 4 | 11 | 11 | 0.179 | 0.500 | 0.089 | 0.960 | 0.898 | 0.928 | results/baselines/bamboogle/qwen-qwen3-8b/bamboogle_pairwise_assistive.json |
| pairwise_incremental | 123 | 72 | 1 | 14 | 36 | 0.406 | 0.280 | 0.114 | 0.986 | 0.667 | 0.796 | results/baselines/bamboogle/qwen-qwen3-8b/bamboogle_pairwise_incremental.json |
| pairwise_model_generated | 123 | 96 | 2 | 13 | 12 | 0.203 | 0.520 | 0.106 | 0.980 | 0.889 | 0.932 | results/baselines/bamboogle/qwen-qwen3-8b/bamboogle_pairwise_model_generated.json |
| self_consistency | 123 | 11 | 1 | 14 | 97 | 0.902 | 0.126 | 0.114 | 0.917 | 0.102 | 0.183 | results/baselines/bamboogle/qwen-qwen3-8b/bamboogle_self_consistency.json |

## crag — google-gemini-2-5-flash

| baseline | total | tp | fp | tn | fn | coverage | acc_at_cov | overall_acc | precision | recall | f1 | path |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ays | 160 | 39 | 7 | 95 | 19 | 0.713 | 0.833 | 0.594 | 0.848 | 0.672 | 0.750 | results/baselines/crag/google-gemini-2-5-flash/crag_ays.json |
| ic_idk | 160 | 16 | 9 | 93 | 42 | 0.844 | 0.689 | 0.581 | 0.640 | 0.276 | 0.386 | results/baselines/crag/google-gemini-2-5-flash/crag_ic_idk.json |
| pairwise_assistive | 163 | 33 | 10 | 95 | 25 | 0.736 | 0.792 | 0.583 | 0.767 | 0.569 | 0.653 | results/baselines/crag/google-gemini-2-5-flash/crag_pairwise_assistive.json |
| pairwise_incremental | 163 | 33 | 13 | 92 | 25 | 0.718 | 0.786 | 0.564 | 0.717 | 0.569 | 0.635 | results/baselines/crag/google-gemini-2-5-flash/crag_pairwise_incremental.json |
| pairwise_model_generated | 163 | 29 | 17 | 88 | 29 | 0.718 | 0.752 | 0.540 | 0.630 | 0.500 | 0.558 | results/baselines/crag/google-gemini-2-5-flash/crag_pairwise_model_generated.json |
| self_consistency | 21 | 2 | 2 | 15 | 2 | 0.809 | 0.882 | 0.714 | 0.500 | 0.500 | 0.500 | results/baselines/crag/google-gemini-2-5-flash/crag_self_consistency.json |

## crag — google-gemini-2-5-pro

| baseline | total | tp | fp | tn | fn | coverage | acc_at_cov | overall_acc | precision | recall | f1 | path |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ays | 160 | 32 | 8 | 94 | 26 | 0.750 | 0.783 | 0.588 | 0.800 | 0.552 | 0.653 | results/baselines/crag/google-gemini-2-5-pro/crag_ays.json |
| ic_idk | 160 | 17 | 5 | 97 | 41 | 0.863 | 0.703 | 0.606 | 0.773 | 0.293 | 0.425 | results/baselines/crag/google-gemini-2-5-pro/crag_ic_idk.json |
| pairwise_assistive | 163 | 19 | 8 | 97 | 39 | 0.834 | 0.713 | 0.595 | 0.704 | 0.328 | 0.447 | results/baselines/crag/google-gemini-2-5-pro/crag_pairwise_assistive.json |
| pairwise_incremental | 163 | 17 | 9 | 96 | 41 | 0.841 | 0.701 | 0.589 | 0.654 | 0.293 | 0.405 | results/baselines/crag/google-gemini-2-5-pro/crag_pairwise_incremental.json |
| pairwise_model_generated | 163 | 16 | 17 | 88 | 42 | 0.797 | 0.677 | 0.540 | 0.485 | 0.276 | 0.352 | results/baselines/crag/google-gemini-2-5-pro/crag_pairwise_model_generated.json |
| self_consistency | 21 | 1 | 0 | 14 | 6 | 0.952 | 0.700 | 0.667 | 1.000 | 0.143 | 0.250 | results/baselines/crag/google-gemini-2-5-pro/crag_self_consistency.json |

## crag — google-gemma-3-4b-it

| baseline | total | tp | fp | tn | fn | coverage | acc_at_cov | overall_acc | precision | recall | f1 | path |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ays | 160 | 50 | 3 | 25 | 82 | 0.669 | 0.234 | 0.156 | 0.943 | 0.379 | 0.540 | results/baselines/crag/google-gemma-3-4b-it/crag_ays.json |
| ic_idk | 160 | 31 | 3 | 25 | 101 | 0.787 | 0.198 | 0.156 | 0.912 | 0.235 | 0.373 | results/baselines/crag/google-gemma-3-4b-it/crag_ic_idk.json |
| pairwise_assistive | 163 | 100 | 14 | 15 | 34 | 0.301 | 0.306 | 0.092 | 0.877 | 0.746 | 0.806 | results/baselines/crag/google-gemma-3-4b-it/crag_pairwise_assistive.json |
| pairwise_incremental | 163 | 107 | 16 | 13 | 27 | 0.245 | 0.325 | 0.080 | 0.870 | 0.798 | 0.833 | results/baselines/crag/google-gemma-3-4b-it/crag_pairwise_incremental.json |
| pairwise_model_generated | 163 | 109 | 19 | 10 | 25 | 0.215 | 0.286 | 0.061 | 0.852 | 0.813 | 0.832 | results/baselines/crag/google-gemma-3-4b-it/crag_pairwise_model_generated.json |
| self_consistency | 21 | 0 | 0 | 2 | 19 | 1.000 | 0.095 | 0.095 | 0.000 | 0.000 | 0.000 | results/baselines/crag/google-gemma-3-4b-it/crag_self_consistency.json |

## crag — gpt-5-1

| baseline | total | tp | fp | tn | fn | coverage | acc_at_cov | overall_acc | precision | recall | f1 | path |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ays | 160 | 33 | 11 | 93 | 23 | 0.725 | 0.802 | 0.581 | 0.750 | 0.589 | 0.660 | results/baselines/crag/gpt-5-1/crag_ays.json |
| ic_idk | 160 | 29 | 13 | 91 | 27 | 0.738 | 0.771 | 0.569 | 0.691 | 0.518 | 0.592 | results/baselines/crag/gpt-5-1/crag_ic_idk.json |
| idk | 21 | 2 | 0 | 15 | 4 | 0.905 | 0.789 | 0.714 | 1.000 | 0.333 | 0.500 | results/baselines/crag/gpt-5-1/crag_idk.json |
| pairwise_assistive | 21 | 2 | 0 | 15 | 4 | 0.905 | 0.789 | 0.714 | 1.000 | 0.333 | 0.500 | results/baselines/crag/gpt-5-1/crag_pairwise_assistive.json |
| pairwise_incremental | 21 | 2 | 0 | 15 | 4 | 0.905 | 0.789 | 0.714 | 1.000 | 0.333 | 0.500 | results/baselines/crag/gpt-5-1/crag_pairwise_incremental.json |
| pairwise_model_generated | 21 | 3 | 3 | 12 | 3 | 0.714 | 0.800 | 0.571 | 0.500 | 0.500 | 0.500 | results/baselines/crag/gpt-5-1/crag_pairwise_model_generated.json |

## crag — meta-llama-llama-3-1-8b-instruct

| baseline | total | tp | fp | tn | fn | coverage | acc_at_cov | overall_acc | precision | recall | f1 | path |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ays | 160 | 63 | 10 | 34 | 53 | 0.544 | 0.391 | 0.212 | 0.863 | 0.543 | 0.667 | results/baselines/crag/meta-llama-llama-3-1-8b-instruct/crag_ays.json |
| ic_idk | 160 | 29 | 3 | 41 | 87 | 0.800 | 0.320 | 0.256 | 0.906 | 0.250 | 0.392 | results/baselines/crag/meta-llama-llama-3-1-8b-instruct/crag_ic_idk.json |
| pairwise_assistive | 163 | 99 | 14 | 31 | 19 | 0.307 | 0.620 | 0.190 | 0.876 | 0.839 | 0.857 | results/baselines/crag/meta-llama-llama-3-1-8b-instruct/crag_pairwise_assistive.json |
| pairwise_incremental | 163 | 103 | 13 | 32 | 15 | 0.288 | 0.681 | 0.196 | 0.888 | 0.873 | 0.880 | results/baselines/crag/meta-llama-llama-3-1-8b-instruct/crag_pairwise_incremental.json |
| pairwise_model_generated | 163 | 99 | 21 | 24 | 19 | 0.264 | 0.558 | 0.147 | 0.825 | 0.839 | 0.832 | results/baselines/crag/meta-llama-llama-3-1-8b-instruct/crag_pairwise_model_generated.json |
| self_consistency | 21 | 1 | 0 | 7 | 13 | 0.952 | 0.350 | 0.333 | 1.000 | 0.071 | 0.133 | results/baselines/crag/meta-llama-llama-3-1-8b-instruct/crag_self_consistency.json |

## crag — meta-llama-llama-3-3-70b-instruct

| baseline | total | tp | fp | tn | fn | coverage | acc_at_cov | overall_acc | precision | recall | f1 | path |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ays | 160 | 37 | 8 | 59 | 56 | 0.719 | 0.513 | 0.369 | 0.822 | 0.398 | 0.536 | results/baselines/crag/meta-llama-llama-3-3-70b-instruct/crag_ays.json |
| ic_idk | 160 | 76 | 32 | 35 | 17 | 0.325 | 0.673 | 0.219 | 0.704 | 0.817 | 0.756 | results/baselines/crag/meta-llama-llama-3-3-70b-instruct/crag_ic_idk.json |
| pairwise_assistive | 163 | 62 | 16 | 52 | 33 | 0.521 | 0.612 | 0.319 | 0.795 | 0.653 | 0.717 | results/baselines/crag/meta-llama-llama-3-3-70b-instruct/crag_pairwise_assistive.json |
| pairwise_incremental | 163 | 62 | 9 | 59 | 33 | 0.564 | 0.641 | 0.362 | 0.873 | 0.653 | 0.747 | results/baselines/crag/meta-llama-llama-3-3-70b-instruct/crag_pairwise_incremental.json |
| pairwise_model_generated | 163 | 72 | 13 | 55 | 23 | 0.478 | 0.705 | 0.337 | 0.847 | 0.758 | 0.800 | results/baselines/crag/meta-llama-llama-3-3-70b-instruct/crag_pairwise_model_generated.json |
| self_consistency | 21 | 1 | 0 | 8 | 12 | 0.952 | 0.400 | 0.381 | 1.000 | 0.077 | 0.143 | results/baselines/crag/meta-llama-llama-3-3-70b-instruct/crag_self_consistency.json |

## crag — mistralai-mistral-7b-instruct-v0-3

| baseline | total | tp | fp | tn | fn | coverage | acc_at_cov | overall_acc | precision | recall | f1 | path |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ays | 160 | 25 | 2 | 38 | 95 | 0.831 | 0.286 | 0.237 | 0.926 | 0.208 | 0.340 | results/baselines/crag/mistralai-mistral-7b-instruct-v0-3/crag_ays.json |
| ic_idk | 160 | 62 | 7 | 33 | 58 | 0.569 | 0.363 | 0.206 | 0.899 | 0.517 | 0.656 | results/baselines/crag/mistralai-mistral-7b-instruct-v0-3/crag_ic_idk.json |
| pairwise_assistive | 163 | 98 | 13 | 28 | 24 | 0.319 | 0.538 | 0.172 | 0.883 | 0.803 | 0.841 | results/baselines/crag/mistralai-mistral-7b-instruct-v0-3/crag_pairwise_assistive.json |
| pairwise_incremental | 163 | 104 | 16 | 25 | 18 | 0.264 | 0.581 | 0.153 | 0.867 | 0.853 | 0.860 | results/baselines/crag/mistralai-mistral-7b-instruct-v0-3/crag_pairwise_incremental.json |
| pairwise_model_generated | 163 | 97 | 23 | 18 | 25 | 0.264 | 0.419 | 0.110 | 0.808 | 0.795 | 0.802 | results/baselines/crag/mistralai-mistral-7b-instruct-v0-3/crag_pairwise_model_generated.json |
| self_consistency | 21 | 3 | 0 | 6 | 12 | 0.857 | 0.333 | 0.286 | 1.000 | 0.200 | 0.333 | results/baselines/crag/mistralai-mistral-7b-instruct-v0-3/crag_self_consistency.json |

## crag — qwen-qwen2-5-72b-instruct

| baseline | total | tp | fp | tn | fn | coverage | acc_at_cov | overall_acc | precision | recall | f1 | path |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ays | 160 | 79 | 8 | 43 | 30 | 0.456 | 0.589 | 0.269 | 0.908 | 0.725 | 0.806 | results/baselines/crag/qwen-qwen2-5-72b-instruct/crag_ays.json |
| ic_idk | 160 | 51 | 10 | 41 | 58 | 0.619 | 0.414 | 0.256 | 0.836 | 0.468 | 0.600 | results/baselines/crag/qwen-qwen2-5-72b-instruct/crag_ic_idk.json |
| pairwise_assistive | 163 | 92 | 9 | 43 | 19 | 0.380 | 0.694 | 0.264 | 0.911 | 0.829 | 0.868 | results/baselines/crag/qwen-qwen2-5-72b-instruct/crag_pairwise_assistive.json |
| pairwise_incremental | 163 | 90 | 10 | 42 | 21 | 0.387 | 0.667 | 0.258 | 0.900 | 0.811 | 0.853 | results/baselines/crag/qwen-qwen2-5-72b-instruct/crag_pairwise_incremental.json |
| pairwise_model_generated | 163 | 94 | 9 | 43 | 17 | 0.368 | 0.717 | 0.264 | 0.913 | 0.847 | 0.878 | results/baselines/crag/qwen-qwen2-5-72b-instruct/crag_pairwise_model_generated.json |
| self_consistency | 21 | 5 | 0 | 6 | 10 | 0.762 | 0.375 | 0.286 | 1.000 | 0.333 | 0.500 | results/baselines/crag/qwen-qwen2-5-72b-instruct/crag_self_consistency.json |

## crag — qwen-qwen3-32b

| baseline | total | tp | fp | tn | fn | coverage | acc_at_cov | overall_acc | precision | recall | f1 | path |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ays | 161 | 71 | 8 | 35 | 47 | 0.509 | 0.427 | 0.217 | 0.899 | 0.602 | 0.721 | results/baselines/crag/qwen-qwen3-32b/crag_ays.json |
| ic_idk | 161 | 38 | 6 | 34 | 83 | 0.727 | 0.291 | 0.211 | 0.864 | 0.314 | 0.461 | results/baselines/crag/qwen-qwen3-32b/crag_ic_idk.json |
| pairwise_assistive | 163 | 104 | 13 | 31 | 15 | 0.282 | 0.674 | 0.190 | 0.889 | 0.874 | 0.881 | results/baselines/crag/qwen-qwen3-32b/crag_pairwise_assistive.json |
| pairwise_incremental | 163 | 103 | 12 | 32 | 16 | 0.294 | 0.667 | 0.196 | 0.896 | 0.866 | 0.880 | results/baselines/crag/qwen-qwen3-32b/crag_pairwise_incremental.json |
| pairwise_model_generated | 163 | 102 | 18 | 26 | 17 | 0.264 | 0.605 | 0.160 | 0.850 | 0.857 | 0.854 | results/baselines/crag/qwen-qwen3-32b/crag_pairwise_model_generated.json |
| self_consistency | 19 | 2 | 0 | 8 | 9 | 0.895 | 0.471 | 0.421 | 1.000 | 0.182 | 0.308 | results/baselines/crag/qwen-qwen3-32b/crag_self_consistency.json |

## crag — qwen-qwen3-8b

| baseline | total | tp | fp | tn | fn | coverage | acc_at_cov | overall_acc | precision | recall | f1 | path |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ays | 160 | 83 | 9 | 22 | 46 | 0.425 | 0.324 | 0.138 | 0.902 | 0.643 | 0.751 | results/baselines/crag/qwen-qwen3-8b/crag_ays.json |
| ic_idk | 160 | 88 | 14 | 17 | 41 | 0.362 | 0.293 | 0.106 | 0.863 | 0.682 | 0.762 | results/baselines/crag/qwen-qwen3-8b/crag_ic_idk.json |
| pairwise_assistive | 163 | 117 | 14 | 18 | 14 | 0.196 | 0.562 | 0.110 | 0.893 | 0.893 | 0.893 | results/baselines/crag/qwen-qwen3-8b/crag_pairwise_assistive.json |
| pairwise_incremental | 163 | 117 | 14 | 18 | 14 | 0.196 | 0.562 | 0.110 | 0.893 | 0.893 | 0.893 | results/baselines/crag/qwen-qwen3-8b/crag_pairwise_incremental.json |
| pairwise_model_generated | 163 | 112 | 24 | 8 | 19 | 0.166 | 0.296 | 0.049 | 0.824 | 0.855 | 0.839 | results/baselines/crag/qwen-qwen3-8b/crag_pairwise_model_generated.json |
| self_consistency | 21 | 6 | 0 | 3 | 12 | 0.714 | 0.200 | 0.143 | 1.000 | 0.333 | 0.500 | results/baselines/crag/qwen-qwen3-8b/crag_self_consistency.json |

## hotpotqa — google-gemini-2-5-flash

| baseline | total | tp | fp | tn | fn | coverage | acc_at_cov | overall_acc | precision | recall | f1 | path |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ays | 276 | 33 | 12 | 151 | 80 | 0.837 | 0.654 | 0.547 | 0.733 | 0.292 | 0.418 | results/baselines/hotpotqa/google-gemini-2-5-flash/hotpotqa_ays.json |
| ic_idk | 276 | 26 | 4 | 159 | 87 | 0.891 | 0.646 | 0.576 | 0.867 | 0.230 | 0.364 | results/baselines/hotpotqa/google-gemini-2-5-flash/hotpotqa_ic_idk.json |
| pairwise_assistive | 275 | 73 | 25 | 137 | 40 | 0.644 | 0.774 | 0.498 | 0.745 | 0.646 | 0.692 | results/baselines/hotpotqa/google-gemini-2-5-flash/hotpotqa_pairwise_assistive.json |
| pairwise_incremental | 275 | 79 | 40 | 122 | 34 | 0.567 | 0.782 | 0.444 | 0.664 | 0.699 | 0.681 | results/baselines/hotpotqa/google-gemini-2-5-flash/hotpotqa_pairwise_incremental.json |
| pairwise_model_generated | 275 | 71 | 48 | 114 | 42 | 0.567 | 0.731 | 0.414 | 0.597 | 0.628 | 0.612 | results/baselines/hotpotqa/google-gemini-2-5-flash/hotpotqa_pairwise_model_generated.json |
| self_consistency | 186 | 35 | 8 | 104 | 39 | 0.769 | 0.727 | 0.559 | 0.814 | 0.473 | 0.598 | results/baselines/hotpotqa/google-gemini-2-5-flash/hotpotqa_self_consistency.json |

## hotpotqa — google-gemini-2-5-pro

| baseline | total | tp | fp | tn | fn | coverage | acc_at_cov | overall_acc | precision | recall | f1 | path |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ays | 273 | 20 | 7 | 173 | 73 | 0.901 | 0.703 | 0.634 | 0.741 | 0.215 | 0.333 | results/baselines/hotpotqa/google-gemini-2-5-pro/hotpotqa_ays.json |
| ic_idk | 273 | 18 | 3 | 177 | 75 | 0.923 | 0.702 | 0.648 | 0.857 | 0.194 | 0.316 | results/baselines/hotpotqa/google-gemini-2-5-pro/hotpotqa_ic_idk.json |
| pairwise_assistive | 275 | 61 | 14 | 168 | 32 | 0.727 | 0.840 | 0.611 | 0.813 | 0.656 | 0.726 | results/baselines/hotpotqa/google-gemini-2-5-pro/hotpotqa_pairwise_assistive.json |
| pairwise_incremental | 275 | 54 | 21 | 161 | 39 | 0.727 | 0.805 | 0.586 | 0.720 | 0.581 | 0.643 | results/baselines/hotpotqa/google-gemini-2-5-pro/hotpotqa_pairwise_incremental.json |
| pairwise_model_generated | 275 | 65 | 62 | 120 | 28 | 0.538 | 0.811 | 0.436 | 0.512 | 0.699 | 0.591 | results/baselines/hotpotqa/google-gemini-2-5-pro/hotpotqa_pairwise_model_generated.json |
| self_consistency | 186 | 20 | 4 | 118 | 44 | 0.871 | 0.728 | 0.634 | 0.833 | 0.312 | 0.455 | results/baselines/hotpotqa/google-gemini-2-5-pro/hotpotqa_self_consistency.json |

## hotpotqa — google-gemma-3-4b-it

| baseline | total | tp | fp | tn | fn | coverage | acc_at_cov | overall_acc | precision | recall | f1 | path |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ays | 275 | 64 | 5 | 38 | 168 | 0.749 | 0.184 | 0.138 | 0.927 | 0.276 | 0.425 | results/baselines/hotpotqa/google-gemma-3-4b-it/hotpotqa_ays.json |
| ic_idk | 275 | 32 | 1 | 43 | 199 | 0.880 | 0.178 | 0.156 | 0.970 | 0.139 | 0.242 | results/baselines/hotpotqa/google-gemma-3-4b-it/hotpotqa_ic_idk.json |
| pairwise_assistive | 275 | 111 | 10 | 33 | 121 | 0.560 | 0.214 | 0.120 | 0.917 | 0.478 | 0.629 | results/baselines/hotpotqa/google-gemma-3-4b-it/hotpotqa_pairwise_assistive.json |
| pairwise_incremental | 275 | 102 | 9 | 34 | 130 | 0.596 | 0.207 | 0.124 | 0.919 | 0.440 | 0.595 | results/baselines/hotpotqa/google-gemma-3-4b-it/hotpotqa_pairwise_incremental.json |
| pairwise_model_generated | 275 | 128 | 14 | 29 | 104 | 0.484 | 0.218 | 0.105 | 0.901 | 0.552 | 0.684 | results/baselines/hotpotqa/google-gemma-3-4b-it/hotpotqa_pairwise_model_generated.json |
| self_consistency | 186 | 1 | 0 | 30 | 155 | 0.995 | 0.162 | 0.161 | 1.000 | 0.006 | 0.013 | results/baselines/hotpotqa/google-gemma-3-4b-it/hotpotqa_self_consistency.json |

## hotpotqa — gpt-5-1

| baseline | total | tp | fp | tn | fn | coverage | acc_at_cov | overall_acc | precision | recall | f1 | path |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ays | 275 | 26 | 4 | 199 | 46 | 0.891 | 0.812 | 0.724 | 0.867 | 0.361 | 0.510 | results/baselines/hotpotqa/gpt-5-1/hotpotqa_ays.json |
| ic_idk | 275 | 34 | 12 | 191 | 38 | 0.833 | 0.834 | 0.695 | 0.739 | 0.472 | 0.576 | results/baselines/hotpotqa/gpt-5-1/hotpotqa_ic_idk.json |
| pairwise_assistive | 186 | 33 | 11 | 123 | 19 | 0.763 | 0.866 | 0.661 | 0.750 | 0.635 | 0.688 | results/baselines/hotpotqa/gpt-5-1/hotpotqa_pairwise_assistive.json |
| pairwise_incremental | 186 | 31 | 8 | 126 | 21 | 0.790 | 0.857 | 0.677 | 0.795 | 0.596 | 0.681 | results/baselines/hotpotqa/gpt-5-1/hotpotqa_pairwise_incremental.json |
| pairwise_model_generated | 186 | 33 | 18 | 116 | 19 | 0.726 | 0.859 | 0.624 | 0.647 | 0.635 | 0.641 | results/baselines/hotpotqa/gpt-5-1/hotpotqa_pairwise_model_generated.json |

## hotpotqa — meta-llama-llama-3-1-8b-instruct

| baseline | total | tp | fp | tn | fn | coverage | acc_at_cov | overall_acc | precision | recall | f1 | path |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ays | 275 | 109 | 14 | 54 | 98 | 0.553 | 0.355 | 0.196 | 0.886 | 0.527 | 0.661 | results/baselines/hotpotqa/meta-llama-llama-3-1-8b-instruct/hotpotqa_ays.json |
| ic_idk | 275 | 4 | 0 | 70 | 201 | 0.986 | 0.258 | 0.255 | 1.000 | 0.019 | 0.038 | results/baselines/hotpotqa/meta-llama-llama-3-1-8b-instruct/hotpotqa_ic_idk.json |
| pairwise_assistive | 275 | 159 | 16 | 52 | 48 | 0.364 | 0.520 | 0.189 | 0.909 | 0.768 | 0.833 | results/baselines/hotpotqa/meta-llama-llama-3-1-8b-instruct/hotpotqa_pairwise_assistive.json |
| pairwise_incremental | 275 | 160 | 21 | 47 | 47 | 0.342 | 0.500 | 0.171 | 0.884 | 0.773 | 0.825 | results/baselines/hotpotqa/meta-llama-llama-3-1-8b-instruct/hotpotqa_pairwise_incremental.json |
| pairwise_model_generated | 275 | 171 | 36 | 32 | 36 | 0.247 | 0.471 | 0.116 | 0.826 | 0.826 | 0.826 | results/baselines/hotpotqa/meta-llama-llama-3-1-8b-instruct/hotpotqa_pairwise_model_generated.json |
| self_consistency | 186 | 9 | 0 | 49 | 128 | 0.952 | 0.277 | 0.263 | 1.000 | 0.066 | 0.123 | results/baselines/hotpotqa/meta-llama-llama-3-1-8b-instruct/hotpotqa_self_consistency.json |

## hotpotqa — meta-llama-llama-3-3-70b-instruct

| baseline | total | tp | fp | tn | fn | coverage | acc_at_cov | overall_acc | precision | recall | f1 | path |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ays | 274 | 64 | 17 | 94 | 99 | 0.704 | 0.487 | 0.343 | 0.790 | 0.393 | 0.525 | results/baselines/hotpotqa/meta-llama-llama-3-3-70b-instruct/hotpotqa_ays.json |
| ic_idk | 274 | 116 | 32 | 79 | 47 | 0.460 | 0.627 | 0.288 | 0.784 | 0.712 | 0.746 | results/baselines/hotpotqa/meta-llama-llama-3-3-70b-instruct/hotpotqa_ic_idk.json |
| pairwise_assistive | 275 | 138 | 25 | 86 | 26 | 0.407 | 0.768 | 0.313 | 0.847 | 0.842 | 0.844 | results/baselines/hotpotqa/meta-llama-llama-3-3-70b-instruct/hotpotqa_pairwise_assistive.json |
| pairwise_incremental | 275 | 131 | 26 | 85 | 33 | 0.429 | 0.720 | 0.309 | 0.834 | 0.799 | 0.816 | results/baselines/hotpotqa/meta-llama-llama-3-3-70b-instruct/hotpotqa_pairwise_incremental.json |
| pairwise_model_generated | 275 | 140 | 55 | 56 | 24 | 0.291 | 0.700 | 0.204 | 0.718 | 0.854 | 0.780 | results/baselines/hotpotqa/meta-llama-llama-3-3-70b-instruct/hotpotqa_pairwise_model_generated.json |
| self_consistency | 186 | 7 | 1 | 76 | 102 | 0.957 | 0.427 | 0.409 | 0.875 | 0.064 | 0.120 | results/baselines/hotpotqa/meta-llama-llama-3-3-70b-instruct/hotpotqa_self_consistency.json |

## hotpotqa — mistralai-mistral-7b-instruct-v0-3

| baseline | total | tp | fp | tn | fn | coverage | acc_at_cov | overall_acc | precision | recall | f1 | path |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ays | 275 | 95 | 11 | 54 | 115 | 0.615 | 0.320 | 0.196 | 0.896 | 0.452 | 0.601 | results/baselines/hotpotqa/mistralai-mistral-7b-instruct-v0-3/hotpotqa_ays.json |
| ic_idk | 275 | 21 | 1 | 64 | 189 | 0.920 | 0.253 | 0.233 | 0.955 | 0.100 | 0.181 | results/baselines/hotpotqa/mistralai-mistral-7b-instruct-v0-3/hotpotqa_ic_idk.json |
| pairwise_assistive | 275 | 127 | 15 | 50 | 83 | 0.484 | 0.376 | 0.182 | 0.894 | 0.605 | 0.722 | results/baselines/hotpotqa/mistralai-mistral-7b-instruct-v0-3/hotpotqa_pairwise_assistive.json |
| pairwise_incremental | 275 | 114 | 20 | 45 | 96 | 0.513 | 0.319 | 0.164 | 0.851 | 0.543 | 0.663 | results/baselines/hotpotqa/mistralai-mistral-7b-instruct-v0-3/hotpotqa_pairwise_incremental.json |
| pairwise_model_generated | 275 | 109 | 19 | 46 | 101 | 0.534 | 0.313 | 0.167 | 0.852 | 0.519 | 0.645 | results/baselines/hotpotqa/mistralai-mistral-7b-instruct-v0-3/hotpotqa_pairwise_model_generated.json |
| self_consistency | 186 | 23 | 0 | 43 | 120 | 0.876 | 0.264 | 0.231 | 1.000 | 0.161 | 0.277 | results/baselines/hotpotqa/mistralai-mistral-7b-instruct-v0-3/hotpotqa_self_consistency.json |

## hotpotqa — qwen-qwen2-5-72b-instruct

| baseline | total | tp | fp | tn | fn | coverage | acc_at_cov | overall_acc | precision | recall | f1 | path |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ays | 274 | 122 | 23 | 72 | 57 | 0.471 | 0.558 | 0.263 | 0.841 | 0.682 | 0.753 | results/baselines/hotpotqa/qwen-qwen2-5-72b-instruct/hotpotqa_ays.json |
| ic_idk | 274 | 73 | 8 | 88 | 105 | 0.704 | 0.456 | 0.321 | 0.901 | 0.410 | 0.564 | results/baselines/hotpotqa/qwen-qwen2-5-72b-instruct/hotpotqa_ic_idk.json |
| pairwise_assistive | 275 | 150 | 28 | 67 | 30 | 0.353 | 0.691 | 0.244 | 0.843 | 0.833 | 0.838 | results/baselines/hotpotqa/qwen-qwen2-5-72b-instruct/hotpotqa_pairwise_assistive.json |
| pairwise_incremental | 275 | 151 | 29 | 66 | 29 | 0.345 | 0.695 | 0.240 | 0.839 | 0.839 | 0.839 | results/baselines/hotpotqa/qwen-qwen2-5-72b-instruct/hotpotqa_pairwise_incremental.json |
| pairwise_model_generated | 275 | 161 | 53 | 42 | 19 | 0.222 | 0.689 | 0.153 | 0.752 | 0.894 | 0.817 | results/baselines/hotpotqa/qwen-qwen2-5-72b-instruct/hotpotqa_pairwise_model_generated.json |
| self_consistency | 186 | 29 | 2 | 61 | 94 | 0.833 | 0.394 | 0.328 | 0.935 | 0.236 | 0.377 | results/baselines/hotpotqa/qwen-qwen2-5-72b-instruct/hotpotqa_self_consistency.json |

## hotpotqa — qwen-qwen3-32b

| baseline | total | tp | fp | tn | fn | coverage | acc_at_cov | overall_acc | precision | recall | f1 | path |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ays | 275 | 91 | 6 | 78 | 100 | 0.647 | 0.438 | 0.284 | 0.938 | 0.476 | 0.632 | results/baselines/hotpotqa/qwen-qwen3-32b/hotpotqa_ays.json |
| ic_idk | 275 | 12 | 1 | 77 | 185 | 0.953 | 0.294 | 0.280 | 0.923 | 0.061 | 0.114 | results/baselines/hotpotqa/qwen-qwen3-32b/hotpotqa_ic_idk.json |
| pairwise_assistive | 275 | 158 | 25 | 61 | 31 | 0.335 | 0.663 | 0.222 | 0.863 | 0.836 | 0.850 | results/baselines/hotpotqa/qwen-qwen3-32b/hotpotqa_pairwise_assistive.json |
| pairwise_incremental | 275 | 159 | 27 | 59 | 30 | 0.324 | 0.663 | 0.214 | 0.855 | 0.841 | 0.848 | results/baselines/hotpotqa/qwen-qwen3-32b/hotpotqa_pairwise_incremental.json |
| pairwise_model_generated | 275 | 160 | 39 | 47 | 29 | 0.276 | 0.618 | 0.171 | 0.804 | 0.847 | 0.825 | results/baselines/hotpotqa/qwen-qwen3-32b/hotpotqa_pairwise_model_generated.json |
| self_consistency | 186 | 8 | 0 | 56 | 122 | 0.957 | 0.315 | 0.301 | 1.000 | 0.061 | 0.116 | results/baselines/hotpotqa/qwen-qwen3-32b/hotpotqa_self_consistency.json |

## hotpotqa — qwen-qwen3-8b

| baseline | total | tp | fp | tn | fn | coverage | acc_at_cov | overall_acc | precision | recall | f1 | path |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ays | 275 | 139 | 13 | 46 | 77 | 0.447 | 0.374 | 0.167 | 0.914 | 0.643 | 0.755 | results/baselines/hotpotqa/qwen-qwen3-8b/hotpotqa_ays.json |
| ic_idk | 275 | 101 | 10 | 50 | 114 | 0.596 | 0.305 | 0.182 | 0.910 | 0.470 | 0.620 | results/baselines/hotpotqa/qwen-qwen3-8b/hotpotqa_ic_idk.json |
| pairwise_assistive | 275 | 179 | 18 | 41 | 37 | 0.284 | 0.526 | 0.149 | 0.909 | 0.829 | 0.867 | results/baselines/hotpotqa/qwen-qwen3-8b/hotpotqa_pairwise_assistive.json |
| pairwise_incremental | 275 | 171 | 23 | 36 | 45 | 0.294 | 0.444 | 0.131 | 0.881 | 0.792 | 0.834 | results/baselines/hotpotqa/qwen-qwen3-8b/hotpotqa_pairwise_incremental.json |
| pairwise_model_generated | 275 | 188 | 25 | 34 | 28 | 0.226 | 0.548 | 0.124 | 0.883 | 0.870 | 0.876 | results/baselines/hotpotqa/qwen-qwen3-8b/hotpotqa_pairwise_model_generated.json |
| self_consistency | 186 | 17 | 0 | 40 | 129 | 0.909 | 0.237 | 0.215 | 1.000 | 0.116 | 0.209 | results/baselines/hotpotqa/qwen-qwen3-8b/hotpotqa_self_consistency.json |

## mintaka — google-gemini-2-5-flash

| baseline | total | tp | fp | tn | fn | coverage | acc_at_cov | overall_acc | precision | recall | f1 | path |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ays | 298 | 23 | 18 | 227 | 30 | 0.862 | 0.883 | 0.762 | 0.561 | 0.434 | 0.489 | results/baselines/mintaka/google-gemini-2-5-flash/mintaka_ays.json |
| ic_idk | 298 | 6 | 8 | 240 | 44 | 0.953 | 0.845 | 0.805 | 0.429 | 0.120 | 0.188 | results/baselines/mintaka/google-gemini-2-5-flash/mintaka_ic_idk.json |
| pairwise_assistive | 298 | 24 | 14 | 230 | 30 | 0.873 | 0.885 | 0.772 | 0.632 | 0.444 | 0.522 | results/baselines/mintaka/google-gemini-2-5-flash/mintaka_pairwise_assistive.json |
| pairwise_incremental | 298 | 19 | 14 | 230 | 35 | 0.889 | 0.868 | 0.772 | 0.576 | 0.352 | 0.437 | results/baselines/mintaka/google-gemini-2-5-flash/mintaka_pairwise_incremental.json |
| pairwise_model_generated | 298 | 17 | 24 | 220 | 37 | 0.862 | 0.856 | 0.738 | 0.415 | 0.315 | 0.358 | results/baselines/mintaka/google-gemini-2-5-flash/mintaka_pairwise_model_generated.json |
| self_consistency | 293 | 5 | 13 | 229 | 46 | 0.939 | 0.833 | 0.782 | 0.278 | 0.098 | 0.145 | results/baselines/mintaka/google-gemini-2-5-flash/mintaka_self_consistency.json |

## mintaka — google-gemini-2-5-pro

| baseline | total | tp | fp | tn | fn | coverage | acc_at_cov | overall_acc | precision | recall | f1 | path |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ays | 298 | 15 | 19 | 232 | 32 | 0.886 | 0.879 | 0.778 | 0.441 | 0.319 | 0.370 | results/baselines/mintaka/google-gemini-2-5-pro/mintaka_ays.json |
| ic_idk | 298 | 12 | 5 | 248 | 33 | 0.943 | 0.883 | 0.832 | 0.706 | 0.267 | 0.387 | results/baselines/mintaka/google-gemini-2-5-pro/mintaka_ic_idk.json |
| pairwise_assistive | 298 | 14 | 13 | 238 | 33 | 0.909 | 0.878 | 0.799 | 0.518 | 0.298 | 0.378 | results/baselines/mintaka/google-gemini-2-5-pro/mintaka_pairwise_assistive.json |
| pairwise_incremental | 298 | 15 | 8 | 243 | 32 | 0.923 | 0.884 | 0.815 | 0.652 | 0.319 | 0.429 | results/baselines/mintaka/google-gemini-2-5-pro/mintaka_pairwise_incremental.json |
| pairwise_model_generated | 298 | 26 | 100 | 151 | 21 | 0.577 | 0.878 | 0.507 | 0.206 | 0.553 | 0.301 | results/baselines/mintaka/google-gemini-2-5-pro/mintaka_pairwise_model_generated.json |
| self_consistency | 293 | 4 | 1 | 248 | 40 | 0.983 | 0.861 | 0.846 | 0.800 | 0.091 | 0.163 | results/baselines/mintaka/google-gemini-2-5-pro/mintaka_self_consistency.json |

## mintaka — google-gemma-3-4b-it

| baseline | total | tp | fp | tn | fn | coverage | acc_at_cov | overall_acc | precision | recall | f1 | path |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ays | 298 | 83 | 9 | 49 | 157 | 0.691 | 0.238 | 0.164 | 0.902 | 0.346 | 0.500 | results/baselines/mintaka/google-gemma-3-4b-it/mintaka_ays.json |
| ic_idk | 298 | 44 | 6 | 53 | 195 | 0.832 | 0.214 | 0.178 | 0.880 | 0.184 | 0.304 | results/baselines/mintaka/google-gemma-3-4b-it/mintaka_ic_idk.json |
| pairwise_assistive | 298 | 49 | 2 | 56 | 191 | 0.829 | 0.227 | 0.188 | 0.961 | 0.204 | 0.337 | results/baselines/mintaka/google-gemma-3-4b-it/mintaka_pairwise_assistive.json |
| pairwise_incremental | 298 | 55 | 5 | 53 | 185 | 0.799 | 0.223 | 0.178 | 0.917 | 0.229 | 0.367 | results/baselines/mintaka/google-gemma-3-4b-it/mintaka_pairwise_incremental.json |
| pairwise_model_generated | 298 | 64 | 7 | 51 | 176 | 0.762 | 0.225 | 0.171 | 0.901 | 0.267 | 0.412 | results/baselines/mintaka/google-gemma-3-4b-it/mintaka_pairwise_model_generated.json |
| self_consistency | 293 | 4 | 0 | 57 | 232 | 0.986 | 0.197 | 0.195 | 1.000 | 0.017 | 0.033 | results/baselines/mintaka/google-gemma-3-4b-it/mintaka_self_consistency.json |

## mintaka — gpt-5-1

| baseline | total | tp | fp | tn | fn | coverage | acc_at_cov | overall_acc | precision | recall | f1 | path |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ays | 298 | 17 | 23 | 234 | 24 | 0.866 | 0.907 | 0.785 | 0.425 | 0.415 | 0.420 | results/baselines/mintaka/gpt-5-1/mintaka_ays.json |
| ic_idk | 298 | 11 | 19 | 238 | 30 | 0.899 | 0.888 | 0.799 | 0.367 | 0.268 | 0.310 | results/baselines/mintaka/gpt-5-1/mintaka_ic_idk.json |
| pairwise_assistive | 293 | 13 | 6 | 246 | 28 | 0.935 | 0.898 | 0.840 | 0.684 | 0.317 | 0.433 | results/baselines/mintaka/gpt-5-1/mintaka_pairwise_assistive.json |
| pairwise_incremental | 293 | 16 | 9 | 243 | 25 | 0.915 | 0.907 | 0.829 | 0.640 | 0.390 | 0.485 | results/baselines/mintaka/gpt-5-1/mintaka_pairwise_incremental.json |
| pairwise_model_generated | 293 | 17 | 24 | 228 | 24 | 0.860 | 0.905 | 0.778 | 0.415 | 0.415 | 0.415 | results/baselines/mintaka/gpt-5-1/mintaka_pairwise_model_generated.json |
| self_consistency | 293 | 6 | 8 | 244 | 35 | 0.952 | 0.875 | 0.833 | 0.429 | 0.146 | 0.218 | results/baselines/mintaka/gpt-5-1/mintaka_self_consistency.json |

## mintaka — meta-llama-llama-3-1-8b-instruct

| baseline | total | tp | fp | tn | fn | coverage | acc_at_cov | overall_acc | precision | recall | f1 | path |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ays | 298 | 109 | 18 | 80 | 91 | 0.574 | 0.468 | 0.269 | 0.858 | 0.545 | 0.667 | results/baselines/mintaka/meta-llama-llama-3-1-8b-instruct/mintaka_ays.json |
| ic_idk | 298 | 14 | 3 | 98 | 183 | 0.943 | 0.349 | 0.329 | 0.824 | 0.071 | 0.131 | results/baselines/mintaka/meta-llama-llama-3-1-8b-instruct/mintaka_ic_idk.json |
| pairwise_assistive | 298 | 152 | 16 | 82 | 48 | 0.436 | 0.631 | 0.275 | 0.905 | 0.760 | 0.826 | results/baselines/mintaka/meta-llama-llama-3-1-8b-instruct/mintaka_pairwise_assistive.json |
| pairwise_incremental | 298 | 147 | 15 | 83 | 53 | 0.456 | 0.610 | 0.279 | 0.907 | 0.735 | 0.812 | results/baselines/mintaka/meta-llama-llama-3-1-8b-instruct/mintaka_pairwise_incremental.json |
| pairwise_model_generated | 298 | 166 | 42 | 56 | 34 | 0.302 | 0.622 | 0.188 | 0.798 | 0.830 | 0.814 | results/baselines/mintaka/meta-llama-llama-3-1-8b-instruct/mintaka_pairwise_model_generated.json |
| self_consistency | 293 | 16 | 0 | 97 | 180 | 0.945 | 0.350 | 0.331 | 1.000 | 0.082 | 0.151 | results/baselines/mintaka/meta-llama-llama-3-1-8b-instruct/mintaka_self_consistency.json |

## mintaka — meta-llama-llama-3-3-70b-instruct

| baseline | total | tp | fp | tn | fn | coverage | acc_at_cov | overall_acc | precision | recall | f1 | path |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ays | 298 | 43 | 19 | 166 | 70 | 0.792 | 0.703 | 0.557 | 0.694 | 0.381 | 0.491 | results/baselines/mintaka/meta-llama-llama-3-3-70b-instruct/mintaka_ays.json |
| ic_idk | 298 | 75 | 52 | 130 | 41 | 0.574 | 0.760 | 0.436 | 0.591 | 0.647 | 0.617 | results/baselines/mintaka/meta-llama-llama-3-3-70b-instruct/mintaka_ic_idk.json |
| pairwise_assistive | 298 | 83 | 17 | 165 | 33 | 0.664 | 0.833 | 0.554 | 0.830 | 0.716 | 0.768 | results/baselines/mintaka/meta-llama-llama-3-3-70b-instruct/mintaka_pairwise_assistive.json |
| pairwise_incremental | 298 | 89 | 20 | 162 | 27 | 0.634 | 0.857 | 0.544 | 0.817 | 0.767 | 0.791 | results/baselines/mintaka/meta-llama-llama-3-3-70b-instruct/mintaka_pairwise_incremental.json |
| pairwise_model_generated | 298 | 83 | 18 | 164 | 33 | 0.661 | 0.833 | 0.550 | 0.822 | 0.716 | 0.765 | results/baselines/mintaka/meta-llama-llama-3-3-70b-instruct/mintaka_pairwise_model_generated.json |
| self_consistency | 293 | 5 | 1 | 178 | 109 | 0.980 | 0.620 | 0.608 | 0.833 | 0.044 | 0.083 | results/baselines/mintaka/meta-llama-llama-3-3-70b-instruct/mintaka_self_consistency.json |

## mintaka — mistralai-mistral-7b-instruct-v0-3

| baseline | total | tp | fp | tn | fn | coverage | acc_at_cov | overall_acc | precision | recall | f1 | path |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ays | 298 | 52 | 12 | 111 | 123 | 0.785 | 0.474 | 0.372 | 0.812 | 0.297 | 0.435 | results/baselines/mintaka/mistralai-mistral-7b-instruct-v0-3/mintaka_ays.json |
| ic_idk | 298 | 37 | 5 | 117 | 139 | 0.859 | 0.457 | 0.393 | 0.881 | 0.210 | 0.339 | results/baselines/mintaka/mistralai-mistral-7b-instruct-v0-3/mintaka_ic_idk.json |
| pairwise_assistive | 298 | 67 | 12 | 108 | 111 | 0.735 | 0.493 | 0.362 | 0.848 | 0.376 | 0.521 | results/baselines/mintaka/mistralai-mistral-7b-instruct-v0-3/mintaka_pairwise_assistive.json |
| pairwise_incremental | 298 | 64 | 22 | 98 | 114 | 0.711 | 0.462 | 0.329 | 0.744 | 0.360 | 0.485 | results/baselines/mintaka/mistralai-mistral-7b-instruct-v0-3/mintaka_pairwise_incremental.json |
| pairwise_model_generated | 298 | 57 | 13 | 107 | 121 | 0.765 | 0.469 | 0.359 | 0.814 | 0.320 | 0.460 | results/baselines/mintaka/mistralai-mistral-7b-instruct-v0-3/mintaka_pairwise_model_generated.json |
| self_consistency | 293 | 25 | 1 | 118 | 149 | 0.911 | 0.442 | 0.403 | 0.962 | 0.144 | 0.250 | results/baselines/mintaka/mistralai-mistral-7b-instruct-v0-3/mintaka_self_consistency.json |

## mintaka — qwen-qwen2-5-72b-instruct

| baseline | total | tp | fp | tn | fn | coverage | acc_at_cov | overall_acc | precision | recall | f1 | path |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ays | 298 | 93 | 39 | 117 | 49 | 0.557 | 0.705 | 0.393 | 0.705 | 0.655 | 0.679 | results/baselines/mintaka/qwen-qwen2-5-72b-instruct/mintaka_ays.json |
| ic_idk | 298 | 37 | 13 | 144 | 104 | 0.832 | 0.581 | 0.483 | 0.740 | 0.262 | 0.387 | results/baselines/mintaka/qwen-qwen2-5-72b-instruct/mintaka_ic_idk.json |
| pairwise_assistive | 298 | 109 | 17 | 139 | 33 | 0.577 | 0.808 | 0.466 | 0.865 | 0.768 | 0.813 | results/baselines/mintaka/qwen-qwen2-5-72b-instruct/mintaka_pairwise_assistive.json |
| pairwise_incremental | 298 | 110 | 16 | 141 | 31 | 0.577 | 0.820 | 0.473 | 0.873 | 0.780 | 0.824 | results/baselines/mintaka/qwen-qwen2-5-72b-instruct/mintaka_pairwise_incremental.json |
| pairwise_model_generated | 298 | 114 | 17 | 140 | 27 | 0.560 | 0.838 | 0.470 | 0.870 | 0.808 | 0.838 | results/baselines/mintaka/qwen-qwen2-5-72b-instruct/mintaka_pairwise_model_generated.json |
| self_consistency | 293 | 19 | 9 | 145 | 120 | 0.904 | 0.547 | 0.495 | 0.679 | 0.137 | 0.228 | results/baselines/mintaka/qwen-qwen2-5-72b-instruct/mintaka_self_consistency.json |

## mintaka — qwen-qwen3-32b

| baseline | total | tp | fp | tn | fn | coverage | acc_at_cov | overall_acc | precision | recall | f1 | path |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ays | 298 | 97 | 18 | 94 | 89 | 0.614 | 0.514 | 0.315 | 0.844 | 0.521 | 0.644 | results/baselines/mintaka/qwen-qwen3-32b/mintaka_ays.json |
| ic_idk | 298 | 19 | 0 | 99 | 180 | 0.936 | 0.355 | 0.332 | 1.000 | 0.096 | 0.174 | results/baselines/mintaka/qwen-qwen3-32b/mintaka_ic_idk.json |
| pairwise_assistive | 298 | 152 | 20 | 95 | 31 | 0.423 | 0.754 | 0.319 | 0.884 | 0.831 | 0.856 | results/baselines/mintaka/qwen-qwen3-32b/mintaka_pairwise_assistive.json |
| pairwise_incremental | 298 | 152 | 23 | 92 | 31 | 0.413 | 0.748 | 0.309 | 0.869 | 0.831 | 0.849 | results/baselines/mintaka/qwen-qwen3-32b/mintaka_pairwise_incremental.json |
| pairwise_model_generated | 298 | 158 | 29 | 86 | 25 | 0.372 | 0.775 | 0.289 | 0.845 | 0.863 | 0.854 | results/baselines/mintaka/qwen-qwen3-32b/mintaka_pairwise_model_generated.json |
| self_consistency | 293 | 24 | 0 | 98 | 171 | 0.918 | 0.364 | 0.335 | 1.000 | 0.123 | 0.219 | results/baselines/mintaka/qwen-qwen3-32b/mintaka_self_consistency.json |

## mintaka — qwen-qwen3-8b

| baseline | total | tp | fp | tn | fn | coverage | acc_at_cov | overall_acc | precision | recall | f1 | path |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ays | 298 | 129 | 15 | 57 | 97 | 0.517 | 0.370 | 0.191 | 0.896 | 0.571 | 0.697 | results/baselines/mintaka/qwen-qwen3-8b/mintaka_ays.json |
| ic_idk | 298 | 68 | 17 | 55 | 158 | 0.715 | 0.258 | 0.185 | 0.800 | 0.301 | 0.437 | results/baselines/mintaka/qwen-qwen3-8b/mintaka_ic_idk.json |
| pairwise_assistive | 298 | 186 | 17 | 55 | 40 | 0.319 | 0.579 | 0.185 | 0.916 | 0.823 | 0.867 | results/baselines/mintaka/qwen-qwen3-8b/mintaka_pairwise_assistive.json |
| pairwise_incremental | 298 | 167 | 19 | 53 | 59 | 0.376 | 0.473 | 0.178 | 0.898 | 0.739 | 0.811 | results/baselines/mintaka/qwen-qwen3-8b/mintaka_pairwise_incremental.json |
| pairwise_model_generated | 298 | 188 | 28 | 44 | 38 | 0.275 | 0.537 | 0.148 | 0.870 | 0.832 | 0.851 | results/baselines/mintaka/qwen-qwen3-8b/mintaka_pairwise_model_generated.json |
| self_consistency | 293 | 21 | 1 | 70 | 201 | 0.925 | 0.258 | 0.239 | 0.955 | 0.095 | 0.172 | results/baselines/mintaka/qwen-qwen3-8b/mintaka_self_consistency.json |

## Averages by baseline

| dataset | baseline | models | coverage | acc_at_cov | overall_acc | precision | recall | f1 |
|---|---|---|---|---|---|---|---|---|
| bamboogle | ays | 10 | 0.676 | 0.485 | 0.373 | 0.842 | 0.463 | 0.593 |
| bamboogle | idk | 10 | 0.799 | 0.460 | 0.376 | 0.881 | 0.262 | 0.345 |
| bamboogle | pairwise_assistive | 10 | 0.556 | 0.631 | 0.364 | 0.839 | 0.663 | 0.714 |
| bamboogle | pairwise_incremental | 10 | 0.598 | 0.575 | 0.372 | 0.850 | 0.602 | 0.690 |
| bamboogle | pairwise_model_generated | 10 | 0.544 | 0.612 | 0.345 | 0.789 | 0.654 | 0.686 |
| bamboogle | self_consistency | 8 | 0.925 | 0.396 | 0.366 | 0.842 | 0.130 | 0.197 |
| crag | ays | 10 | 0.634 | 0.518 | 0.336 | 0.866 | 0.531 | 0.642 |
| crag | ic_idk | 10 | 0.663 | 0.472 | 0.317 | 0.809 | 0.437 | 0.540 |
| crag | idk | 1 | 0.905 | 0.789 | 0.714 | 1.000 | 0.333 | 0.500 |
| crag | pairwise_assistive | 10 | 0.478 | 0.630 | 0.323 | 0.860 | 0.687 | 0.746 |
| crag | pairwise_incremental | 10 | 0.470 | 0.640 | 0.322 | 0.856 | 0.694 | 0.749 |
| crag | pairwise_model_generated | 10 | 0.425 | 0.581 | 0.278 | 0.753 | 0.704 | 0.725 |
| crag | self_consistency | 9 | 0.877 | 0.423 | 0.370 | 0.833 | 0.204 | 0.296 |
| hotpotqa | ays | 10 | 0.682 | 0.489 | 0.349 | 0.853 | 0.432 | 0.561 |
| hotpotqa | ic_idk | 10 | 0.815 | 0.455 | 0.363 | 0.891 | 0.281 | 0.376 |
| hotpotqa | pairwise_assistive | 10 | 0.492 | 0.624 | 0.319 | 0.849 | 0.713 | 0.769 |
| hotpotqa | pairwise_incremental | 10 | 0.493 | 0.599 | 0.306 | 0.824 | 0.690 | 0.742 |
| hotpotqa | pairwise_model_generated | 10 | 0.411 | 0.596 | 0.251 | 0.749 | 0.732 | 0.730 |
| hotpotqa | self_consistency | 9 | 0.902 | 0.392 | 0.345 | 0.940 | 0.166 | 0.254 |
| mintaka | ays | 10 | 0.714 | 0.614 | 0.459 | 0.714 | 0.448 | 0.539 |
| mintaka | ic_idk | 10 | 0.849 | 0.559 | 0.477 | 0.722 | 0.243 | 0.328 |
| mintaka | pairwise_assistive | 10 | 0.670 | 0.699 | 0.476 | 0.804 | 0.554 | 0.632 |
| mintaka | pairwise_incremental | 10 | 0.669 | 0.685 | 0.471 | 0.789 | 0.550 | 0.629 |
| mintaka | pairwise_model_generated | 10 | 0.600 | 0.694 | 0.420 | 0.696 | 0.592 | 0.607 |
| mintaka | self_consistency | 10 | 0.944 | 0.535 | 0.506 | 0.793 | 0.098 | 0.166 |

## Averages by model

| dataset | model | baselines | coverage | acc_at_cov | overall_acc | precision | recall | f1 |
|---|---|---|---|---|---|---|---|---|
| bamboogle | google-gemini-2-5-flash | 6 | 0.865 | 0.897 | 0.774 | 0.618 | 0.442 | 0.447 |
| bamboogle | google-gemini-2-5-pro | 6 | 0.920 | 0.895 | 0.822 | 0.674 | 0.333 | 0.404 |
| bamboogle | google-gemma-3-4b-it | 6 | 0.787 | 0.098 | 0.076 | 0.949 | 0.219 | 0.332 |
| bamboogle | gpt-5-1 | 5 | 0.891 | 0.907 | 0.808 | 0.670 | 0.463 | 0.540 |
| bamboogle | meta-llama-llama-3-1-8b-instruct | 6 | 0.587 | 0.356 | 0.179 | 0.918 | 0.482 | 0.560 |
| bamboogle | meta-llama-llama-3-3-70b-instruct | 6 | 0.644 | 0.679 | 0.404 | 0.865 | 0.560 | 0.608 |
| bamboogle | mistralai-mistral-7b-instruct-v0-3 | 6 | 0.711 | 0.189 | 0.133 | 0.925 | 0.316 | 0.461 |
| bamboogle | qwen-qwen2-5-72b-instruct | 6 | 0.478 | 0.581 | 0.257 | 0.891 | 0.677 | 0.738 |
| bamboogle | qwen-qwen3-32b | 5 | 0.398 | 0.462 | 0.140 | 0.923 | 0.682 | 0.737 |
| bamboogle | qwen-qwen3-8b | 6 | 0.455 | 0.298 | 0.102 | 0.954 | 0.597 | 0.694 |
| crag | google-gemini-2-5-flash | 6 | 0.756 | 0.789 | 0.596 | 0.684 | 0.514 | 0.580 |
| crag | google-gemini-2-5-pro | 6 | 0.840 | 0.713 | 0.597 | 0.736 | 0.314 | 0.422 |
| crag | google-gemma-3-4b-it | 6 | 0.536 | 0.241 | 0.107 | 0.742 | 0.495 | 0.564 |
| crag | gpt-5-1 | 6 | 0.815 | 0.790 | 0.644 | 0.823 | 0.435 | 0.542 |
| crag | meta-llama-llama-3-1-8b-instruct | 6 | 0.526 | 0.487 | 0.223 | 0.893 | 0.569 | 0.627 |
| crag | meta-llama-llama-3-3-70b-instruct | 6 | 0.593 | 0.591 | 0.331 | 0.840 | 0.559 | 0.617 |
| crag | mistralai-mistral-7b-instruct-v0-3 | 6 | 0.517 | 0.420 | 0.194 | 0.897 | 0.563 | 0.639 |
| crag | qwen-qwen2-5-72b-instruct | 6 | 0.495 | 0.576 | 0.266 | 0.911 | 0.669 | 0.751 |
| crag | qwen-qwen3-32b | 6 | 0.495 | 0.522 | 0.233 | 0.899 | 0.616 | 0.684 |
| crag | qwen-qwen3-8b | 6 | 0.343 | 0.373 | 0.109 | 0.896 | 0.717 | 0.773 |
| hotpotqa | google-gemini-2-5-flash | 6 | 0.713 | 0.719 | 0.506 | 0.737 | 0.495 | 0.561 |
| hotpotqa | google-gemini-2-5-pro | 6 | 0.781 | 0.765 | 0.592 | 0.746 | 0.443 | 0.511 |
| hotpotqa | google-gemma-3-4b-it | 6 | 0.711 | 0.194 | 0.134 | 0.939 | 0.315 | 0.431 |
| hotpotqa | gpt-5-1 | 5 | 0.801 | 0.846 | 0.676 | 0.760 | 0.540 | 0.619 |
| hotpotqa | meta-llama-llama-3-1-8b-instruct | 6 | 0.574 | 0.397 | 0.198 | 0.917 | 0.496 | 0.551 |
| hotpotqa | meta-llama-llama-3-3-70b-instruct | 6 | 0.541 | 0.622 | 0.311 | 0.808 | 0.610 | 0.638 |
| hotpotqa | mistralai-mistral-7b-instruct-v0-3 | 6 | 0.657 | 0.307 | 0.196 | 0.908 | 0.397 | 0.515 |
| hotpotqa | qwen-qwen2-5-72b-instruct | 6 | 0.488 | 0.580 | 0.258 | 0.852 | 0.649 | 0.698 |
| hotpotqa | qwen-qwen3-32b | 6 | 0.582 | 0.498 | 0.245 | 0.897 | 0.520 | 0.564 |
| hotpotqa | qwen-qwen3-8b | 6 | 0.459 | 0.406 | 0.161 | 0.916 | 0.620 | 0.693 |
| mintaka | google-gemini-2-5-flash | 6 | 0.896 | 0.862 | 0.772 | 0.482 | 0.294 | 0.356 |
| mintaka | google-gemini-2-5-pro | 6 | 0.870 | 0.877 | 0.763 | 0.554 | 0.308 | 0.338 |
| mintaka | google-gemma-3-4b-it | 6 | 0.817 | 0.220 | 0.179 | 0.927 | 0.208 | 0.325 |
| mintaka | gpt-5-1 | 6 | 0.905 | 0.897 | 0.811 | 0.493 | 0.325 | 0.380 |
| mintaka | meta-llama-llama-3-1-8b-instruct | 6 | 0.609 | 0.505 | 0.278 | 0.882 | 0.504 | 0.567 |
| mintaka | meta-llama-llama-3-3-70b-instruct | 6 | 0.717 | 0.768 | 0.541 | 0.764 | 0.545 | 0.586 |
| mintaka | mistralai-mistral-7b-instruct-v0-3 | 6 | 0.795 | 0.466 | 0.370 | 0.844 | 0.285 | 0.415 |
| mintaka | qwen-qwen2-5-72b-instruct | 6 | 0.668 | 0.716 | 0.463 | 0.789 | 0.568 | 0.628 |
| mintaka | qwen-qwen3-32b | 6 | 0.613 | 0.585 | 0.316 | 0.907 | 0.544 | 0.600 |
| mintaka | qwen-qwen3-8b | 6 | 0.521 | 0.413 | 0.188 | 0.889 | 0.560 | 0.639 |
