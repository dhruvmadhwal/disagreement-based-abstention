# Decomposition Equivalence Results

Source: decomposition_equivalence/results/summary/

This analysis compares model-generated question decompositions against gold (human-annotated) decompositions.

## Key Metrics

- **equivalence_rate**: % of questions where model decomposition is semantically equivalent to gold
- **precision**: How many model hops match gold hops (from model's perspective)
- **recall**: How many gold hops are covered by model hops (from gold's perspective)
- **f1**: Harmonic mean of precision and recall
- **hop_ratio**: avg(model_hops / gold_hops) - 1.0 means same number of hops
- **avg_gold_hops / avg_model_hops**: Average number of reasoning steps

## bamboogle

| model | total | equiv_rate | precision | recall | f1 | gold_hops | model_hops | hop_ratio |
|---|---|---|---|---|---|---|---|---|
| google-gemma-3-4b-it | 125 | 0.776 | 0.853 | 0.827 | 0.831 | 2.02 | 2.00 | 0.989 |
| meta-llama-llama-3-1-8b-instruct | 125 | 0.608 | 0.689 | 0.801 | 0.727 | 2.02 | 2.79 | 1.383 |
| meta-llama-llama-3-3-70b-instruct | 125 | 0.952 | 0.931 | 0.969 | 0.946 | 2.02 | 2.16 | 1.069 |
| mistralai-mistral-7b-instruct-v0-3 | 125 | 0.768 | 0.807 | 0.873 | 0.827 | 2.02 | 2.50 | 1.237 |
| qwen-qwen2-5-72b-instruct | 125 | 0.968 | 0.976 | 0.977 | 0.975 | 2.02 | 2.04 | 1.009 |
| qwen-qwen3-32b | 125 | 0.960 | 0.961 | 0.951 | 0.951 | 2.02 | 2.00 | 0.991 |
| qwen-qwen3-8b | 125 | 0.928 | 0.943 | 0.925 | 0.928 | 2.02 | 1.98 | 0.984 |

## crag

| model | total | equiv_rate | precision | recall | f1 | gold_hops | model_hops | hop_ratio |
|---|---|---|---|---|---|---|---|---|
| google-gemma-3-4b-it | 300 | 0.387 | 0.425 | 0.545 | 0.462 | 1.37 | 2.08 | 1.640 |
| meta-llama-llama-3-1-8b-instruct | 300 | 0.263 | 0.256 | 0.543 | 0.327 | 1.37 | 3.97 | 3.281 |
| meta-llama-llama-3-3-70b-instruct | 300 | 0.643 | 0.507 | 0.776 | 0.589 | 1.37 | 2.35 | 1.886 |
| mistralai-mistral-7b-instruct-v0-3 | 300 | 0.403 | 0.315 | 0.576 | 0.387 | 1.37 | 2.92 | 2.317 |
| qwen-qwen2-5-72b-instruct | 300 | 0.740 | 0.729 | 0.831 | 0.760 | 1.37 | 1.68 | 1.288 |
| qwen-qwen3-32b | 300 | 0.830 | 0.791 | 0.820 | 0.783 | 1.37 | 1.46 | 1.139 |
| qwen-qwen3-8b | 300 | 0.677 | 0.658 | 0.783 | 0.691 | 1.37 | 1.79 | 1.404 |

## fanout

| model | total | equiv_rate | precision | recall | f1 | gold_hops | model_hops | hop_ratio |
|---|---|---|---|---|---|---|---|---|
| google-gemma-3-4b-it | 310 | 0.542 | 0.642 | 0.802 | 0.677 | 2.31 | 3.41 | 1.600 |
| meta-llama-llama-3-1-8b-instruct | 310 | 0.281 | 0.426 | 0.691 | 0.482 | 2.31 | 6.30 | 3.005 |
| meta-llama-llama-3-3-70b-instruct | 310 | 0.655 | 0.513 | 0.892 | 0.607 | 2.31 | 5.11 | 2.454 |
| mistralai-mistral-7b-instruct-v0-3 | 310 | 0.406 | 0.702 | 0.722 | 0.662 | 2.31 | 2.76 | 1.296 |
| qwen-qwen2-5-72b-instruct | 310 | 0.716 | 0.481 | 0.878 | 0.577 | 2.31 | 4.86 | 2.327 |
| qwen-qwen3-32b | 310 | 0.668 | 0.461 | 0.834 | 0.535 | 2.31 | 5.03 | 2.430 |
| qwen-qwen3-8b | 310 | 0.632 | 0.687 | 0.834 | 0.705 | 2.31 | 3.54 | 1.687 |

## frames

| model | total | equiv_rate | precision | recall | f1 | gold_hops | model_hops | hop_ratio |
|---|---|---|---|---|---|---|---|---|
| google-gemma-3-4b-it | 299 | 0.261 | 0.543 | 0.538 | 0.523 | 3.36 | 3.43 | 1.089 |
| meta-llama-llama-3-1-8b-instruct | 299 | 0.234 | 0.426 | 0.542 | 0.452 | 3.36 | 5.73 | 1.899 |
| meta-llama-llama-3-3-70b-instruct | 299 | 0.672 | 0.754 | 0.844 | 0.782 | 3.36 | 3.93 | 1.219 |
| mistralai-mistral-7b-instruct-v0-3 | 299 | 0.258 | 0.523 | 0.589 | 0.530 | 3.36 | 4.03 | 1.303 |
| qwen-qwen2-5-72b-instruct | 299 | 0.662 | 0.788 | 0.817 | 0.790 | 3.36 | 3.53 | 1.082 |
| qwen-qwen3-32b | 299 | 0.629 | 0.843 | 0.723 | 0.743 | 3.36 | 2.93 | 0.898 |
| qwen-qwen3-8b | 299 | 0.532 | 0.798 | 0.646 | 0.675 | 3.36 | 2.81 | 0.874 |

## hotpotqa

| model | total | equiv_rate | precision | recall | f1 | gold_hops | model_hops | hop_ratio |
|---|---|---|---|---|---|---|---|---|
| google-gemma-3-4b-it | 300 | 0.367 | 0.458 | 0.553 | 0.479 | 2.00 | 2.61 | 1.404 |
| meta-llama-llama-3-1-8b-instruct | 300 | 0.323 | 0.416 | 0.588 | 0.462 | 2.00 | 4.36 | 2.506 |
| meta-llama-llama-3-3-70b-instruct | 300 | 0.667 | 0.662 | 0.823 | 0.712 | 2.00 | 2.58 | 1.411 |
| mistralai-mistral-7b-instruct-v0-3 | 300 | 0.380 | 0.453 | 0.622 | 0.499 | 2.00 | 3.22 | 1.763 |
| qwen-qwen2-5-72b-instruct | 300 | 0.633 | 0.736 | 0.806 | 0.754 | 2.00 | 2.27 | 1.203 |
| qwen-qwen3-32b | 300 | 0.710 | 0.804 | 0.757 | 0.756 | 2.00 | 1.91 | 0.997 |
| qwen-qwen3-8b | 300 | 0.607 | 0.732 | 0.745 | 0.712 | 2.00 | 2.07 | 1.094 |

## mintaka

| model | total | equiv_rate | precision | recall | f1 | gold_hops | model_hops | hop_ratio |
|---|---|---|---|---|---|---|---|---|
| google-gemma-3-4b-it | 300 | 0.780 | 0.840 | 0.851 | 0.841 | 2.04 | 2.15 | 1.061 |
| meta-llama-llama-3-1-8b-instruct | 300 | 0.600 | 0.625 | 0.753 | 0.665 | 2.04 | 3.14 | 1.558 |
| meta-llama-llama-3-3-70b-instruct | 300 | 0.930 | 0.899 | 0.948 | 0.917 | 2.04 | 2.23 | 1.105 |
| mistralai-mistral-7b-instruct-v0-3 | 300 | 0.743 | 0.745 | 0.845 | 0.779 | 2.04 | 2.63 | 1.314 |
| qwen-qwen2-5-72b-instruct | 300 | 0.930 | 0.918 | 0.955 | 0.928 | 2.04 | 2.19 | 1.076 |
| qwen-qwen3-32b | 300 | 0.920 | 0.941 | 0.929 | 0.929 | 2.04 | 2.03 | 1.003 |
| qwen-qwen3-8b | 300 | 0.853 | 0.893 | 0.894 | 0.885 | 2.04 | 2.10 | 1.035 |

## musique

| model | total | equiv_rate | precision | recall | f1 | gold_hops | model_hops | hop_ratio |
|---|---|---|---|---|---|---|---|---|
| google-gemma-3-4b-it | 300 | 0.337 | 0.589 | 0.571 | 0.569 | 2.69 | 2.65 | 1.017 |
| meta-llama-llama-3-1-8b-instruct | 300 | 0.247 | 0.431 | 0.526 | 0.460 | 2.69 | 4.59 | 1.709 |
| meta-llama-llama-3-3-70b-instruct | 300 | 0.637 | 0.771 | 0.816 | 0.784 | 2.69 | 2.95 | 1.112 |
| mistralai-mistral-7b-instruct-v0-3 | 300 | 0.320 | 0.512 | 0.591 | 0.537 | 2.69 | 3.36 | 1.282 |
| qwen-qwen2-5-72b-instruct | 300 | 0.630 | 0.785 | 0.788 | 0.780 | 2.69 | 2.75 | 1.034 |
| qwen-qwen3-32b | 300 | 0.567 | 0.812 | 0.703 | 0.725 | 2.69 | 2.34 | 0.900 |
| qwen-qwen3-8b | 300 | 0.577 | 0.796 | 0.677 | 0.704 | 2.69 | 2.30 | 0.893 |

## Averages by Model (across all datasets)

| model | datasets | avg_equiv_rate | avg_precision | avg_recall | avg_f1 | avg_hop_ratio |
|---|---|---|---|---|---|---|
| google-gemma-3-4b-it | 7 | 0.493 | 0.621 | 0.670 | 0.626 | 1.257 |
| meta-llama-llama-3-1-8b-instruct | 7 | 0.365 | 0.467 | 0.635 | 0.511 | 2.192 |
| meta-llama-llama-3-3-70b-instruct | 7 | 0.737 | 0.720 | 0.867 | 0.762 | 1.465 |
| mistralai-mistral-7b-instruct-v0-3 | 7 | 0.468 | 0.579 | 0.688 | 0.603 | 1.502 |
| qwen-qwen2-5-72b-instruct | 7 | 0.754 | 0.773 | 0.865 | 0.795 | 1.289 |
| qwen-qwen3-32b | 7 | 0.755 | 0.802 | 0.817 | 0.775 | 1.194 |
| qwen-qwen3-8b | 7 | 0.686 | 0.787 | 0.786 | 0.757 | 1.139 |

## Averages by Dataset (across all models)

| dataset | models | avg_equiv_rate | avg_precision | avg_recall | avg_f1 | avg_gold_hops | avg_model_hops |
|---|---|---|---|---|---|---|---|
| bamboogle | 7 | 0.851 | 0.880 | 0.903 | 0.884 | 2.02 | 2.21 |
| crag | 7 | 0.563 | 0.526 | 0.696 | 0.571 | 1.37 | 2.32 |
| fanout | 7 | 0.557 | 0.559 | 0.808 | 0.606 | 2.31 | 4.43 |
| frames | 7 | 0.464 | 0.668 | 0.671 | 0.642 | 3.36 | 3.77 |
| hotpotqa | 7 | 0.527 | 0.609 | 0.699 | 0.625 | 2.00 | 2.72 |
| mintaka | 7 | 0.822 | 0.837 | 0.882 | 0.849 | 2.04 | 2.35 |
| musique | 7 | 0.473 | 0.671 | 0.667 | 0.651 | 2.69 | 2.99 |
