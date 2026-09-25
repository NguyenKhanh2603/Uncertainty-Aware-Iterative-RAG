# Four-dataset pure-cosine conformal comparison

## Unified protocol

Every query-level row in this table is **pure query-level cosine conformal selection**: per-query z-scored Top-30 cosine scores, calibrated on a disjoint 100-query split at α=0.10, with a deterministic Top-1 fallback only when no candidate passes. No internal hidden-state, LM-head, attention, reranker, or fusion feature is used in this report.

The HotpotQA query-level result was rerun on 25 September 2026 over all 1,000 test queries to make this definition match MMQA, TAT-QA, and WebQA. Its remaining selector rows are reused from the prior HotpotQA run only after asserting that the calibration thresholds and score-bank sizes exactly match the pure-cosine rerun.

All selectors use the same frozen Top-30 cosine candidates within each dataset. CCE, CONFLARE, TRAQ, BY, and BH are cosine-only. BY accommodates arbitrary p-value dependence; BH ordinarily needs independence or suitable positive dependence. The rank cap after BH is experimental; this report makes no capped-procedure FDR guarantee.

## Run inventory

| Dataset | Calibration | Test queries | Pure query-level downstream output |
|---|---:|---:|---|
| HotpotQA | 100 | 1,000 | [`hotpotqa_pure_query_level_cosine_1000_2026_09_25`](../hotpotqa_pure_query_level_cosine_1000_2026_09_25/REPORT.md) |
| MMQA | 100 | 1,000 | [`mmqa_downstream_predictions.jsonl`](mmqa_downstream_predictions.jsonl) |
| TAT-QA | 100 | 1,000 | [`tatqa_downstream_predictions.jsonl`](tatqa_downstream_predictions.jsonl) |
| WebQA | 100 | 250 | [`webqa_downstream_predictions.jsonl`](webqa_downstream_predictions.jsonl) |

## HOTPOTQA (n=1000)

| Method | Chunks | Precision | Support recall | Empty | Any support | All support | EM | F1 | Numeric |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed Top-10 | 10.00 | 16.2% | 93.0% | 0.0% | 98.7% | 88.3% | 0.383 | 0.500 | 0.405 |
| Fixed Top-20 | 20.00 | 8.5% | 97.8% | 0.0% | 99.7% | 96.1% | 0.364 | 0.480 | 0.386 |
| ECIR CCE (α=0.10) | 7.62 | 20.0% | 87.6% | 0.7% | 97.4% | 80.0% | 0.364 | 0.482 | 0.385 |
| CONFLARE (α=0.10) | 7.46 | 20.4% | 87.4% | 0.7% | 97.3% | 79.7% | 0.366 | 0.484 | 0.387 |
| TRAQ retrieval (α=0.10) | 14.18 | 11.6% | 94.5% | 0.0% | 99.0% | 90.8% | 0.374 | 0.495 | 0.397 |
| Query-level cosine (α=0.10) | 9.31 | 17.4% | 93.2% | 0.0% | 98.8% | 88.4% | 0.370 | 0.491 | 0.392 |
| BY cosine (α=0.10) | 0.15 | 73.3% | 6.3% | 90.9% | 9.7% | 4.2% | 0.167 | 0.256 | 0.182 |
| BY cosine (α=0.30) | 0.75 | 57.1% | 24.7% | 63.1% | 36.7% | 16.3% | 0.211 | 0.308 | 0.223 |
| BY cosine (α=0.50) | 1.46 | 43.9% | 36.8% | 52.2% | 47.3% | 28.7% | 0.245 | 0.339 | 0.254 |
| BH cosine (α=0.10, ctx=10) | 0.99 | 52.5% | 30.0% | 57.8% | 41.7% | 21.7% | 0.220 | 0.314 | 0.229 |
| BH cosine (α=0.30, ctx=10) | 2.86 | 37.3% | 61.4% | 23.1% | 75.6% | 50.7% | 0.299 | 0.415 | 0.314 |
| BH cosine (α=0.50, ctx=10) | 4.57 | 28.2% | 74.2% | 10.1% | 88.9% | 63.2% | 0.337 | 0.457 | 0.356 |
| BH cosine (α=0.90, ctx=10) | 8.31 | 18.3% | 87.3% | 2.5% | 96.3% | 80.5% | 0.373 | 0.492 | 0.395 |
| BH cosine (α=0.99, ctx=10) | 9.88 | 16.3% | 92.5% | 0.3% | 98.4% | 87.7% | 0.382 | 0.498 | 0.404 |
| BH cosine (α=0.99, ctx=20) | 19.73 | 8.6% | 97.2% | 0.3% | 99.4% | 95.5% | 0.362 | 0.479 | 0.384 |

## MMQA (n=1000)

| Method | Chunks | Precision | Support recall | Empty | Any support | All support | EM | F1 | Numeric |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed Top-10 | 10.00 | 11.0% | 93.0% | 0.0% | 98.3% | 91.9% | 0.471 | 0.525 | 0.492 |
| Fixed Top-20 | 20.00 | 5.8% | 97.9% | 0.0% | 99.5% | 97.5% | 0.470 | 0.520 | 0.487 |
| ECIR CCE (α=0.10) | 13.54 | 8.1% | 92.2% | 2.0% | 96.9% | 91.3% | 0.470 | 0.521 | 0.488 |
| CONFLARE (α=0.10) | 12.42 | 8.7% | 91.1% | 2.4% | 96.3% | 90.0% | 0.468 | 0.519 | 0.488 |
| TRAQ retrieval (α=0.10) | 19.99 | 5.7% | 96.3% | 1.0% | 98.6% | 95.7% | 0.466 | 0.518 | 0.485 |
| Query-level cosine (α=0.10) | 11.94 | 9.4% | 94.4% | 0.0% | 98.7% | 93.5% | 0.481 | 0.537 | 0.502 |
| BY cosine (α=0.10) | 0.07 | 52.3% | 2.9% | 96.3% | 8.4% | 7.3% | 0.145 | 0.192 | 0.158 |
| BY cosine (α=0.30) | 0.39 | 27.3% | 8.8% | 88.8% | 15.2% | 12.6% | 0.176 | 0.222 | 0.191 |
| BY cosine (α=0.50) | 0.66 | 22.1% | 12.3% | 85.3% | 18.5% | 15.9% | 0.182 | 0.231 | 0.198 |
| BH cosine (α=0.10, ctx=10) | 0.33 | 35.8% | 9.9% | 87.7% | 16.3% | 13.5% | 0.178 | 0.226 | 0.194 |
| BH cosine (α=0.30, ctx=10) | 1.23 | 25.5% | 26.5% | 68.9% | 33.9% | 29.2% | 0.244 | 0.292 | 0.260 |
| BH cosine (α=0.50, ctx=10) | 2.39 | 19.1% | 38.5% | 56.0% | 46.0% | 41.2% | 0.290 | 0.337 | 0.307 |
| BH cosine (α=0.90, ctx=10) | 7.84 | 11.8% | 78.0% | 15.7% | 83.5% | 77.6% | 0.425 | 0.478 | 0.445 |
| BH cosine (α=0.99, ctx=10) | 9.76 | 11.1% | 91.3% | 1.9% | 96.5% | 90.2% | 0.468 | 0.522 | 0.489 |
| BH cosine (α=0.99, ctx=20) | 19.51 | 5.8% | 96.1% | 1.9% | 97.7% | 95.7% | 0.465 | 0.515 | 0.482 |

## TATQA (n=1000)

| Method | Chunks | Precision | Support recall | Empty | Any support | All support | EM | F1 | Numeric |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed Top-10 | 10.00 | 8.8% | 85.8% | 0.0% | 88.7% | 86.1% | 0.241 | 0.350 | 0.305 |
| Fixed Top-20 | 20.00 | 4.9% | 95.3% | 0.0% | 96.3% | 95.2% | 0.220 | 0.335 | 0.293 |
| ECIR CCE (α=0.10) | 18.26 | 5.0% | 90.1% | 3.9% | 92.6% | 90.5% | 0.221 | 0.333 | 0.293 |
| CONFLARE (α=0.10) | 18.14 | 5.1% | 89.9% | 4.0% | 92.5% | 90.3% | 0.222 | 0.335 | 0.294 |
| TRAQ retrieval (α=0.10) | 23.06 | 4.2% | 95.6% | 1.4% | 96.5% | 95.6% | 0.224 | 0.343 | 0.297 |
| Query-level cosine (α=0.10) | 8.84 | 10.0% | 86.4% | 0.0% | 89.9% | 86.5% | 0.238 | 0.349 | 0.304 |
| BY cosine (α=0.10) | 0.88 | 14.0% | 12.0% | 85.7% | 19.9% | 18.6% | 0.049 | 0.112 | 0.102 |
| BY cosine (α=0.30) | 1.82 | 9.7% | 17.2% | 79.2% | 25.0% | 23.5% | 0.064 | 0.130 | 0.115 |
| BY cosine (α=0.50) | 2.84 | 8.2% | 22.8% | 74.0% | 30.0% | 28.2% | 0.079 | 0.148 | 0.127 |
| BH cosine (α=0.10, ctx=10) | 1.07 | 16.5% | 17.2% | 77.6% | 25.2% | 23.4% | 0.069 | 0.136 | 0.118 |
| BH cosine (α=0.30, ctx=10) | 2.64 | 12.1% | 31.3% | 60.0% | 38.5% | 35.2% | 0.096 | 0.177 | 0.155 |
| BH cosine (α=0.50, ctx=10) | 4.14 | 10.3% | 41.5% | 47.8% | 48.2% | 45.0% | 0.115 | 0.204 | 0.178 |
| BH cosine (α=0.90, ctx=10) | 8.45 | 8.8% | 72.4% | 12.9% | 76.7% | 73.6% | 0.194 | 0.297 | 0.261 |
| BH cosine (α=0.99, ctx=10) | 9.81 | 8.8% | 84.1% | 1.5% | 87.3% | 84.8% | 0.240 | 0.346 | 0.301 |
| BH cosine (α=0.99, ctx=20) | 19.62 | 4.9% | 93.5% | 1.5% | 94.9% | 93.8% | 0.219 | 0.332 | 0.290 |

## WEBQA (n=250)

| Method | Chunks | Precision | Support recall | Empty | Any support | All support | EM | F1 | Numeric |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed Top-10 | 10.00 | 6.2% | 68.1% | 0.0% | 82.0% | 73.6% | 0.000 | 0.124 | 0.028 |
| Fixed Top-20 | 20.00 | 4.1% | 89.1% | 0.0% | 93.6% | 90.0% | 0.000 | 0.127 | 0.028 |
| ECIR CCE (α=0.10) | 21.88 | 3.6% | 85.6% | 5.2% | 91.6% | 88.8% | 0.000 | 0.157 | 0.028 |
| CONFLARE (α=0.10) | 21.63 | 3.6% | 85.2% | 5.6% | 91.2% | 88.4% | 0.000 | 0.158 | 0.028 |
| TRAQ retrieval (α=0.10) | 26.03 | 3.3% | 93.4% | 1.6% | 96.4% | 94.8% | 0.000 | 0.147 | 0.028 |
| Query-level cosine (α=0.10) | 19.24 | 4.2% | 88.6% | 0.0% | 93.2% | 89.6% | 0.000 | 0.132 | 0.028 |
| BY cosine (α=0.10) | 0.02 | 20.0% | 0.4% | 99.6% | 27.6% | 27.2% | 0.060 | 0.487 | 0.152 |
| BY cosine (α=0.30) | 0.55 | 2.2% | 1.3% | 96.0% | 28.4% | 28.0% | 0.056 | 0.475 | 0.144 |
| BY cosine (α=0.50) | 1.20 | 2.7% | 3.5% | 92.0% | 30.0% | 30.0% | 0.052 | 0.465 | 0.132 |
| BH cosine (α=0.10, ctx=10) | 0.40 | 4.0% | 1.7% | 93.2% | 28.8% | 28.4% | 0.052 | 0.459 | 0.136 |
| BH cosine (α=0.30, ctx=10) | 1.86 | 6.2% | 12.7% | 78.0% | 38.0% | 36.4% | 0.044 | 0.403 | 0.108 |
| BH cosine (α=0.50, ctx=10) | 3.78 | 5.7% | 23.6% | 58.4% | 46.8% | 44.4% | 0.036 | 0.318 | 0.084 |
| BH cosine (α=0.90, ctx=10) | 7.46 | 5.6% | 45.4% | 24.8% | 65.6% | 58.8% | 0.008 | 0.211 | 0.040 |
| BH cosine (α=0.99, ctx=10) | 9.28 | 6.2% | 62.9% | 7.2% | 78.4% | 70.0% | 0.000 | 0.149 | 0.036 |
| BH cosine (α=0.99, ctx=20) | 18.56 | 4.1% | 82.5% | 7.2% | 89.2% | 85.6% | 0.000 | 0.152 | 0.036 |

## Metrics

- **Precision / support recall** are chunk-level evidence metrics within the frozen Top-30 candidate pool.
- **Any support / all support** are query-level evidence coverage.
- **EM, F1, Numeric** are Qwen2-VL-7B-Instruct greedy downstream QA measurements, generated with at most 24 new tokens. They must not be averaged across datasets.
- The old [HotpotQA fusion report](../official_seed42_hotpot_six_methods/REPORT.md) is intentionally separate; it is an internal-model ablation, not a row in this pure-cosine comparison.

## Reproducibility artifacts

- Current three-dataset aggregate: [`summary.json`](summary.json)
- Pure HotpotQA aggregate: [`../hotpotqa_pure_query_level_cosine_1000_2026_09_25/summary.json`](../hotpotqa_pure_query_level_cosine_1000_2026_09_25/summary.json)
- Runner: [`../../run_all_datasets_cosine_six_methods.py`](../../run_all_datasets_cosine_six_methods.py)
