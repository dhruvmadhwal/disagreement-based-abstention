# Evaluation Summary (GPT-5.1 & Gemini-2.5 Pro)

## Consistency (raw, per dataset/model/regime vs open-ended)
| dataset | model | focused_regime | comparisons | equivalent | non_equivalent | not_meaningful | equivalence_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bamboogle | google-gemini-2-5-pro | assistive | 122 | 112 | 10 | 1.0 | 0.918 |
| bamboogle | google-gemini-2-5-pro | incremental | 123 | 109 | 14 | 0.0 | 0.886 |
| bamboogle | google-gemini-2-5-pro | model_generated | 122 | 104 | 18 | 1.0 | 0.852 |
| bamboogle | gpt-5-1 | assistive | 121 | 109 | 12 | 2.0 | 0.901 |
| bamboogle | gpt-5-1 | incremental | 121 | 108 | 13 | 2.0 | 0.893 |
| bamboogle | gpt-5-1 | model_generated | 120 | 102 | 18 | 3.0 | 0.850 |
| crag | google-gemini-2-5-pro | assistive | 162 | 136 | 26 | 1.0 | 0.840 |
| crag | google-gemini-2-5-pro | incremental | 163 | 136 | 27 | 0.0 | 0.834 |
| crag | google-gemini-2-5-pro | model_generated | 160 | 129 | 31 | 3.0 | 0.806 |
| crag | gpt-5-1 | assistive | 161 | 133 | 28 | 2.0 | 0.826 |
| crag | gpt-5-1 | incremental | 163 | 134 | 29 | 0.0 | 0.822 |
| crag | gpt-5-1 | model_generated | 152 | 116 | 36 | 11.0 | 0.763 |
| fanout | google-gemini-2-5-pro | assistive | 310 | 160 | 150 | nan | 0.516 |
| fanout | google-gemini-2-5-pro | model_generated | 310 | 43 | 267 | nan | 0.139 |
| fanout | google-gemini-2-5-pro | incremental | 310 | 162 | 148 | nan | 0.523 |
| frames | google-gemini-2-5-pro | assistive | 300 | 235 | 65 | nan | 0.783 |
| frames | google-gemini-2-5-pro | model_generated | 300 | 196 | 104 | nan | 0.653 |
| frames | google-gemini-2-5-pro | incremental | 299 | 211 | 88 | nan | 0.706 |
| hotpotqa | google-gemini-2-5-pro | assistive | 275 | 207 | 68 | 0.0 | 0.753 |
| hotpotqa | google-gemini-2-5-pro | incremental | 273 | 191 | 82 | 2.0 | 0.700 |
| hotpotqa | google-gemini-2-5-pro | model_generated | 267 | 177 | 90 | 8.0 | 0.663 |
| hotpotqa | gpt-5-1 | assistive | 262 | 217 | 45 | 15.0 | 0.828 |
| hotpotqa | gpt-5-1 | incremental | 269 | 209 | 60 | 8.0 | 0.777 |
| hotpotqa | gpt-5-1 | model_generated | 265 | 204 | 61 | 12.0 | 0.770 |
| mintaka | google-gemini-2-5-pro | assistive | 298 | 278 | 20 | 0.0 | 0.933 |
| mintaka | google-gemini-2-5-pro | incremental | 295 | 276 | 19 | 3.0 | 0.936 |
| mintaka | google-gemini-2-5-pro | model_generated | 290 | 240 | 50 | 8.0 | 0.828 |
| mintaka | gpt-5-1 | assistive | 297 | 279 | 18 | 1.0 | 0.939 |
| mintaka | gpt-5-1 | incremental | 297 | 274 | 23 | 1.0 | 0.923 |
| mintaka | gpt-5-1 | model_generated | 291 | 255 | 36 | 7.0 | 0.876 |
| musique | google-gemini-2-5-pro | assistive | 300 | 171 | 129 | nan | 0.570 |
| musique | google-gemini-2-5-pro | model_generated | 300 | 87 | 213 | nan | 0.290 |
| musique | google-gemini-2-5-pro | incremental | 299 | 162 | 137 | nan | 0.542 |

## Correctness (raw, per dataset/model/regime)
| dataset | model | regime | evaluated | correct | incorrect | not_meaningful | accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bamboogle | google-gemini-2-5-pro | assistive | 123 | 107 | 15 | 1 | 0.870 |
| bamboogle | google-gemini-2-5-pro | incremental | 122 | 106 | 16 | 0 | 0.869 |
| bamboogle | google-gemini-2-5-pro | model_generated | 123 | 102 | 20 | 1 | 0.829 |
| bamboogle | google-gemini-2-5-pro | open_ended | 123 | 105 | 18 | 0 | 0.854 |
| bamboogle | gpt-5-1 | assistive | 123 | 108 | 15 | 0 | 0.878 |
| bamboogle | gpt-5-1 | incremental | 123 | 107 | 16 | 0 | 0.870 |
| bamboogle | gpt-5-1 | model_generated | 123 | 100 | 22 | 1 | 0.813 |
| bamboogle | gpt-5-1 | open_ended | 123 | 104 | 17 | 2 | 0.846 |
| crag | google-gemini-2-5-pro | assistive | 163 | 100 | 61 | 2 | 0.613 |
| crag | google-gemini-2-5-pro | incremental | 163 | 97 | 63 | 3 | 0.595 |
| crag | google-gemini-2-5-pro | model_generated | 163 | 88 | 68 | 7 | 0.540 |
| crag | google-gemini-2-5-pro | open_ended | 163 | 105 | 57 | 1 | 0.644 |
| crag | gpt-5-1 | assistive | 163 | 104 | 57 | 2 | 0.638 |
| crag | gpt-5-1 | incremental | 163 | 101 | 60 | 2 | 0.620 |
| crag | gpt-5-1 | model_generated | 163 | 92 | 59 | 12 | 0.564 |
| crag | gpt-5-1 | open_ended | 163 | 106 | 56 | 1 | 0.650 |
| frames | google-gemini-2-5-pro | assistive | 300 | 212 | 88 | 0 | 0.707 |
| frames | google-gemini-2-5-pro | model_generated | 300 | 182 | 118 | 0 | 0.607 |
| frames | google-gemini-2-5-pro | open_ended | 300 | 217 | 83 | 0 | 0.723 |
| frames | google-gemini-2-5-pro | incremental | 300 | 191 | 109 | 0 | 0.637 |
| hotpotqa | google-gemini-2-5-pro | assistive | 270 | 184 | 85 | 1 | 0.681 |
| hotpotqa | google-gemini-2-5-pro | incremental | 272 | 175 | 94 | 3 | 0.643 |
| hotpotqa | google-gemini-2-5-pro | model_generated | 272 | 158 | 108 | 6 | 0.581 |
| hotpotqa | google-gemini-2-5-pro | open_ended | 271 | 186 | 85 | 0 | 0.686 |
| hotpotqa | gpt-5-1 | assistive | 273 | 198 | 65 | 10 | 0.725 |
| hotpotqa | gpt-5-1 | incremental | 273 | 193 | 72 | 8 | 0.707 |
| hotpotqa | gpt-5-1 | model_generated | 273 | 188 | 73 | 12 | 0.689 |
| hotpotqa | gpt-5-1 | open_ended | 274 | 202 | 65 | 7 | 0.737 |
| mintaka | google-gemini-2-5-pro | assistive | 298 | 254 | 42 | 2 | 0.852 |
| mintaka | google-gemini-2-5-pro | incremental | 297 | 247 | 49 | 1 | 0.832 |
| mintaka | google-gemini-2-5-pro | model_generated | 298 | 214 | 76 | 8 | 0.718 |
| mintaka | google-gemini-2-5-pro | open_ended | 298 | 255 | 42 | 1 | 0.856 |
| mintaka | gpt-5-1 | assistive | 298 | 259 | 39 | 0 | 0.869 |
| mintaka | gpt-5-1 | incremental | 298 | 245 | 52 | 1 | 0.822 |
| mintaka | gpt-5-1 | model_generated | 298 | 240 | 49 | 9 | 0.805 |
| mintaka | gpt-5-1 | open_ended | 298 | 257 | 40 | 1 | 0.862 |
| musique | google-gemini-2-5-pro | assistive | 300 | 150 | 150 | 0 | 0.500 |
| musique | google-gemini-2-5-pro | model_generated | 300 | 76 | 224 | 0 | 0.253 |
| musique | google-gemini-2-5-pro | open_ended | 300 | 147 | 153 | 0 | 0.490 |
| musique | google-gemini-2-5-pro | incremental | 299 | 134 | 165 | 0 | 0.448 |

## Consistency (average by model)
| model | assistive | incremental | model_generated |
| --- | --- | --- | --- |
| google-gemini-2-5-pro | 0.735 | 0.708 | 0.558 |
| gpt-5-1 | 0.878 | 0.853 | 0.818 |

## Consistency (average by dataset)
| dataset | assistive | incremental | model_generated |
| --- | --- | --- | --- |
| bamboogle | 0.909 | 0.889 | 0.851 |
| crag | 0.833 | 0.828 | 0.785 |
| fanout | 0.516 | 0.523 | 0.139 |
| frames | 0.783 | 0.706 | 0.653 |
| hotpotqa | 0.790 | 0.738 | 0.716 |
| mintaka | 0.936 | 0.929 | 0.852 |
| musique | 0.570 | 0.542 | 0.290 |

## Correctness (average by model)
| model | open_ended | assistive | incremental | model_generated |
| --- | --- | --- | --- | --- |
| google-gemini-2-5-pro | 0.698 | 0.693 | 0.654 | 0.563 |
| gpt-5-1 | 0.780 | 0.781 | 0.754 | 0.723 |

## Correctness (average by dataset)
| dataset | open_ended | assistive | incremental | model_generated |
| --- | --- | --- | --- | --- |
| bamboogle | 0.850 | 0.874 | 0.869 | 0.821 |
| crag | 0.647 | 0.626 | 0.607 | 0.552 |
| frames | 0.723 | 0.707 | 0.637 | 0.607 |
| hotpotqa | 0.712 | 0.703 | 0.675 | 0.635 |
| mintaka | 0.859 | 0.861 | 0.827 | 0.762 |
| musique | 0.490 | 0.500 | 0.448 | 0.253 |
