# Four-dataset cosine conformal selector comparison

Every row within a dataset uses the same frozen Top-30 cosine candidates and its disjoint 100-query calibration plan. Query-level cosine is the pure per-query z-scored cosine threshold with a deterministic Top-1 empty-context fallback. BH context caps are experimental rank-ordered policies and have no claimed capped-procedure FDR guarantee.

## mmqa (complete; n=1000)

| Method | Chunks | Precision | Recall | Empty | Any support | All support | EM | F1 | Numeric |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_top10 | 10.00 | 11.0% | 93.0% | 0.0% | 98.3% | 91.9% | 0.471 | 0.525 | 0.492 |
| fixed_top20 | 20.00 | 5.8% | 97.9% | 0.0% | 99.5% | 97.5% | 0.470 | 0.520 | 0.487 |
| cce_alpha_0.10 | 13.54 | 8.1% | 92.2% | 2.0% | 96.9% | 91.3% | 0.470 | 0.521 | 0.488 |
| conflare_alpha_0.10 | 12.42 | 8.7% | 91.1% | 2.4% | 96.3% | 90.0% | 0.468 | 0.519 | 0.488 |
| traq_alpha_0.10 | 19.99 | 5.7% | 96.3% | 1.0% | 98.6% | 95.7% | 0.466 | 0.518 | 0.485 |
| query_level_cosine_alpha_0.10 | 11.94 | 9.4% | 94.4% | 0.0% | 98.7% | 93.5% | 0.481 | 0.537 | 0.502 |
| by_cosine_alpha_0.10 | 0.07 | 52.3% | 2.9% | 96.3% | 8.4% | 7.3% | 0.145 | 0.192 | 0.158 |
| by_cosine_alpha_0.30 | 0.39 | 27.3% | 8.8% | 88.8% | 15.2% | 12.6% | 0.176 | 0.222 | 0.191 |
| by_cosine_alpha_0.50 | 0.66 | 22.1% | 12.3% | 85.3% | 18.5% | 15.9% | 0.182 | 0.231 | 0.198 |
| bh_cosine_alpha_0.10_ctx10 | 0.33 | 35.8% | 9.9% | 87.7% | 16.3% | 13.5% | 0.178 | 0.226 | 0.194 |
| bh_cosine_alpha_0.30_ctx10 | 1.23 | 25.5% | 26.5% | 68.9% | 33.9% | 29.2% | 0.244 | 0.292 | 0.260 |
| bh_cosine_alpha_0.50_ctx10 | 2.39 | 19.1% | 38.5% | 56.0% | 46.0% | 41.2% | 0.290 | 0.337 | 0.307 |
| bh_cosine_alpha_0.90_ctx10 | 7.84 | 11.8% | 78.0% | 15.7% | 83.5% | 77.6% | 0.425 | 0.478 | 0.445 |
| bh_cosine_alpha_0.99_ctx10 | 9.76 | 11.1% | 91.3% | 1.9% | 96.5% | 90.2% | 0.468 | 0.522 | 0.489 |
| bh_cosine_alpha_0.99_ctx20 | 19.51 | 5.8% | 96.1% | 1.9% | 97.7% | 95.7% | 0.465 | 0.515 | 0.482 |

## tatqa (complete; n=1000)

| Method | Chunks | Precision | Recall | Empty | Any support | All support | EM | F1 | Numeric |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_top10 | 10.00 | 8.8% | 85.8% | 0.0% | 88.7% | 86.1% | 0.241 | 0.350 | 0.305 |
| fixed_top20 | 20.00 | 4.9% | 95.3% | 0.0% | 96.3% | 95.2% | 0.220 | 0.335 | 0.293 |
| cce_alpha_0.10 | 18.26 | 5.0% | 90.1% | 3.9% | 92.6% | 90.5% | 0.221 | 0.333 | 0.293 |
| conflare_alpha_0.10 | 18.14 | 5.1% | 89.9% | 4.0% | 92.5% | 90.3% | 0.222 | 0.335 | 0.294 |
| traq_alpha_0.10 | 23.06 | 4.2% | 95.6% | 1.4% | 96.5% | 95.6% | 0.224 | 0.343 | 0.297 |
| query_level_cosine_alpha_0.10 | 8.84 | 10.0% | 86.4% | 0.0% | 89.9% | 86.5% | 0.238 | 0.349 | 0.304 |
| by_cosine_alpha_0.10 | 0.88 | 14.0% | 12.0% | 85.7% | 19.9% | 18.6% | 0.049 | 0.112 | 0.102 |
| by_cosine_alpha_0.30 | 1.82 | 9.7% | 17.2% | 79.2% | 25.0% | 23.5% | 0.064 | 0.130 | 0.115 |
| by_cosine_alpha_0.50 | 2.84 | 8.2% | 22.8% | 74.0% | 30.0% | 28.2% | 0.079 | 0.148 | 0.127 |
| bh_cosine_alpha_0.10_ctx10 | 1.07 | 16.5% | 17.2% | 77.6% | 25.2% | 23.4% | 0.069 | 0.136 | 0.118 |
| bh_cosine_alpha_0.30_ctx10 | 2.64 | 12.1% | 31.3% | 60.0% | 38.5% | 35.2% | 0.096 | 0.177 | 0.155 |
| bh_cosine_alpha_0.50_ctx10 | 4.14 | 10.3% | 41.5% | 47.8% | 48.2% | 45.0% | 0.115 | 0.204 | 0.178 |
| bh_cosine_alpha_0.90_ctx10 | 8.45 | 8.8% | 72.4% | 12.9% | 76.7% | 73.6% | 0.194 | 0.297 | 0.261 |
| bh_cosine_alpha_0.99_ctx10 | 9.81 | 8.8% | 84.1% | 1.5% | 87.3% | 84.8% | 0.240 | 0.346 | 0.301 |
| bh_cosine_alpha_0.99_ctx20 | 19.62 | 4.9% | 93.5% | 1.5% | 94.9% | 93.8% | 0.219 | 0.332 | 0.290 |

## webqa (complete; n=250)

| Method | Chunks | Precision | Recall | Empty | Any support | All support | EM | F1 | Numeric |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_top10 | 10.00 | 6.2% | 68.1% | 0.0% | 82.0% | 73.6% | 0.000 | 0.124 | 0.028 |
| fixed_top20 | 20.00 | 4.1% | 89.1% | 0.0% | 93.6% | 90.0% | 0.000 | 0.127 | 0.028 |
| cce_alpha_0.10 | 21.88 | 3.6% | 85.6% | 5.2% | 91.6% | 88.8% | 0.000 | 0.157 | 0.028 |
| conflare_alpha_0.10 | 21.63 | 3.6% | 85.2% | 5.6% | 91.2% | 88.4% | 0.000 | 0.158 | 0.028 |
| traq_alpha_0.10 | 26.03 | 3.3% | 93.4% | 1.6% | 96.4% | 94.8% | 0.000 | 0.147 | 0.028 |
| query_level_cosine_alpha_0.10 | 19.24 | 4.2% | 88.6% | 0.0% | 93.2% | 89.6% | 0.000 | 0.132 | 0.028 |
| by_cosine_alpha_0.10 | 0.02 | 20.0% | 0.4% | 99.6% | 27.6% | 27.2% | 0.060 | 0.487 | 0.152 |
| by_cosine_alpha_0.30 | 0.55 | 2.2% | 1.3% | 96.0% | 28.4% | 28.0% | 0.056 | 0.475 | 0.144 |
| by_cosine_alpha_0.50 | 1.20 | 2.7% | 3.5% | 92.0% | 30.0% | 30.0% | 0.052 | 0.465 | 0.132 |
| bh_cosine_alpha_0.10_ctx10 | 0.40 | 4.0% | 1.7% | 93.2% | 28.8% | 28.4% | 0.052 | 0.459 | 0.136 |
| bh_cosine_alpha_0.30_ctx10 | 1.86 | 6.2% | 12.7% | 78.0% | 38.0% | 36.4% | 0.044 | 0.403 | 0.108 |
| bh_cosine_alpha_0.50_ctx10 | 3.78 | 5.7% | 23.6% | 58.4% | 46.8% | 44.4% | 0.036 | 0.318 | 0.084 |
| bh_cosine_alpha_0.90_ctx10 | 7.46 | 5.6% | 45.4% | 24.8% | 65.6% | 58.8% | 0.008 | 0.211 | 0.040 |
| bh_cosine_alpha_0.99_ctx10 | 9.28 | 6.2% | 62.9% | 7.2% | 78.4% | 70.0% | 0.000 | 0.149 | 0.036 |
| bh_cosine_alpha_0.99_ctx20 | 18.56 | 4.1% | 82.5% | 7.2% | 89.2% | 85.6% | 0.000 | 0.152 | 0.036 |

## Protocol

```json
{
  "selector_families": [
    "CCE",
    "CONFLARE",
    "TRAQ retrieval",
    "BY cosine",
    "query-level cosine",
    "BH cosine"
  ],
  "top_l": 30,
  "calibration_queries_per_dataset": 100,
  "BH_note": "BH normally needs independence or suitable positive dependence; its rank cap is experimental."
}
```
