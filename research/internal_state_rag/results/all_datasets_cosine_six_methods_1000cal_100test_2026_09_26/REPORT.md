# Cosine conformal selector comparison

Every row within a dataset uses the same frozen Top-30 cosine candidates and its disjoint calibration plan. Query-level cosine is the pure per-query z-scored cosine threshold with a deterministic Top-1 empty-context fallback. BH context caps are experimental rank-ordered policies and have no claimed capped-procedure FDR guarantee.

## hotpotqa (selection_complete; n=100)

| Method | Chunks | Precision | Recall | Empty | Any support | All support | EM | F1 | Numeric |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_top10 | 10.00 | 16.0% | 93.0% | 0.0% | 99.0% | 88.0% | pending | pending | pending |
| fixed_top20 | 20.00 | 8.4% | 97.7% | 0.0% | 100.0% | 96.0% | pending | pending | pending |
| cce_alpha_0.10 | 8.75 | 17.8% | 90.7% | 0.0% | 97.0% | 86.0% | pending | pending | pending |
| conflare_alpha_0.10 | 8.74 | 17.8% | 90.7% | 0.0% | 97.0% | 86.0% | pending | pending | pending |
| traq_alpha_0.10 | 14.53 | 11.2% | 94.8% | 0.0% | 98.0% | 92.0% | pending | pending | pending |
| query_level_cosine_alpha_0.10 | 10.60 | 15.1% | 93.0% | 0.0% | 99.0% | 88.0% | pending | pending | pending |
| by_cosine_alpha_0.10 | 0.17 | 70.6% | 7.0% | 88.0% | 14.0% | 5.0% | pending | pending | pending |
| by_cosine_alpha_0.30 | 0.42 | 71.4% | 17.4% | 73.0% | 29.0% | 12.0% | pending | pending | pending |
| by_cosine_alpha_0.50 | 0.74 | 58.1% | 25.0% | 65.0% | 37.0% | 19.0% | pending | pending | pending |
| bh_cosine_alpha_0.10_ctx10 | 0.52 | 69.2% | 20.9% | 70.0% | 32.0% | 15.0% | pending | pending | pending |
| bh_cosine_alpha_0.30_ctx10 | 2.42 | 34.3% | 48.3% | 36.0% | 64.0% | 40.0% | pending | pending | pending |
| bh_cosine_alpha_0.50_ctx10 | 4.11 | 28.7% | 68.6% | 18.0% | 83.0% | 59.0% | pending | pending | pending |
| bh_cosine_alpha_0.90_ctx10 | 6.92 | 20.8% | 83.7% | 5.0% | 94.0% | 77.0% | pending | pending | pending |
| bh_cosine_alpha_0.99_ctx10 | 9.76 | 16.1% | 91.3% | 1.0% | 98.0% | 86.0% | pending | pending | pending |
| bh_cosine_alpha_0.99_ctx20 | 19.46 | 8.5% | 95.9% | 1.0% | 99.0% | 94.0% | pending | pending | pending |

## mmqa (selection_complete; n=100)

| Method | Chunks | Precision | Recall | Empty | Any support | All support | EM | F1 | Numeric |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_top10 | 10.00 | 10.9% | 96.5% | 0.0% | 100.0% | 96.0% | pending | pending | pending |
| fixed_top20 | 20.00 | 5.5% | 97.3% | 0.0% | 100.0% | 97.0% | pending | pending | pending |
| cce_alpha_0.10 | 10.00 | 10.0% | 88.5% | 5.0% | 96.0% | 88.0% | pending | pending | pending |
| conflare_alpha_0.10 | 10.00 | 10.0% | 88.5% | 5.0% | 96.0% | 88.0% | pending | pending | pending |
| traq_alpha_0.10 | 18.34 | 5.9% | 95.6% | 2.0% | 98.0% | 95.0% | pending | pending | pending |
| query_level_cosine_alpha_0.10 | 9.05 | 12.0% | 96.5% | 0.0% | 100.0% | 96.0% | pending | pending | pending |
| by_cosine_alpha_0.10 | 0.05 | 80.0% | 3.5% | 95.0% | 11.0% | 9.0% | pending | pending | pending |
| by_cosine_alpha_0.30 | 0.24 | 45.8% | 9.7% | 88.0% | 18.0% | 16.0% | pending | pending | pending |
| by_cosine_alpha_0.50 | 0.40 | 37.5% | 13.3% | 82.0% | 22.0% | 20.0% | pending | pending | pending |
| bh_cosine_alpha_0.10_ctx10 | 0.35 | 42.9% | 13.3% | 83.0% | 22.0% | 20.0% | pending | pending | pending |
| bh_cosine_alpha_0.30_ctx10 | 0.96 | 31.2% | 26.5% | 67.0% | 36.0% | 31.0% | pending | pending | pending |
| bh_cosine_alpha_0.50_ctx10 | 2.58 | 19.4% | 44.2% | 52.0% | 53.0% | 48.0% | pending | pending | pending |
| bh_cosine_alpha_0.90_ctx10 | 8.11 | 11.7% | 84.1% | 14.0% | 88.0% | 85.0% | pending | pending | pending |
| bh_cosine_alpha_0.99_ctx10 | 9.90 | 10.9% | 95.6% | 1.0% | 99.0% | 95.0% | pending | pending | pending |
| bh_cosine_alpha_0.99_ctx20 | 19.80 | 5.5% | 96.5% | 1.0% | 99.0% | 96.0% | pending | pending | pending |

## tatqa (selection_complete; n=100)

| Method | Chunks | Precision | Recall | Empty | Any support | All support | EM | F1 | Numeric |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_top10 | 10.00 | 8.9% | 84.8% | 0.0% | 89.0% | 84.0% | pending | pending | pending |
| fixed_top20 | 20.00 | 5.1% | 96.2% | 0.0% | 97.0% | 96.0% | pending | pending | pending |
| cce_alpha_0.10 | 17.80 | 5.2% | 87.6% | 6.0% | 90.0% | 88.0% | pending | pending | pending |
| conflare_alpha_0.10 | 17.80 | 5.2% | 87.6% | 6.0% | 90.0% | 88.0% | pending | pending | pending |
| traq_alpha_0.10 | 23.00 | 4.3% | 95.2% | 4.0% | 95.0% | 95.0% | pending | pending | pending |
| query_level_cosine_alpha_0.10 | 11.30 | 8.1% | 87.6% | 0.0% | 91.0% | 87.0% | pending | pending | pending |
| by_cosine_alpha_0.10 | 0.40 | 10.0% | 3.8% | 94.0% | 13.0% | 12.0% | pending | pending | pending |
| by_cosine_alpha_0.30 | 1.04 | 6.7% | 6.7% | 90.0% | 16.0% | 15.0% | pending | pending | pending |
| by_cosine_alpha_0.50 | 2.85 | 6.0% | 16.2% | 80.0% | 25.0% | 22.0% | pending | pending | pending |
| bh_cosine_alpha_0.10_ctx10 | 0.69 | 11.6% | 7.6% | 87.0% | 17.0% | 15.0% | pending | pending | pending |
| bh_cosine_alpha_0.30_ctx10 | 2.25 | 8.0% | 17.1% | 68.0% | 27.0% | 22.0% | pending | pending | pending |
| bh_cosine_alpha_0.50_ctx10 | 3.89 | 8.2% | 30.5% | 51.0% | 40.0% | 35.0% | pending | pending | pending |
| bh_cosine_alpha_0.90_ctx10 | 8.35 | 8.7% | 69.5% | 12.0% | 77.0% | 71.0% | pending | pending | pending |
| bh_cosine_alpha_0.99_ctx10 | 9.80 | 8.9% | 82.9% | 2.0% | 87.0% | 82.0% | pending | pending | pending |
| bh_cosine_alpha_0.99_ctx20 | 19.60 | 5.1% | 94.3% | 2.0% | 95.0% | 94.0% | pending | pending | pending |

## webqa (selection_complete; n=100)

| Method | Chunks | Precision | Recall | Empty | Any support | All support | EM | F1 | Numeric |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_top10 | 10.00 | 5.8% | 68.2% | 0.0% | 81.0% | 74.0% | pending | pending | pending |
| fixed_top20 | 20.00 | 4.0% | 92.9% | 0.0% | 97.0% | 94.0% | pending | pending | pending |
| cce_alpha_0.10 | 19.82 | 3.6% | 83.5% | 9.0% | 90.0% | 86.0% | pending | pending | pending |
| conflare_alpha_0.10 | 19.80 | 3.6% | 83.5% | 9.0% | 90.0% | 86.0% | pending | pending | pending |
| traq_alpha_0.10 | 23.72 | 3.4% | 94.1% | 5.0% | 96.0% | 95.0% | pending | pending | pending |
| query_level_cosine_alpha_0.10 | 17.63 | 4.4% | 90.6% | 0.0% | 96.0% | 92.0% | pending | pending | pending |
| by_cosine_alpha_0.10 | 0.00 | 0.0% | 0.0% | 100.0% | 30.0% | 30.0% | pending | pending | pending |
| by_cosine_alpha_0.30 | 0.93 | 1.1% | 1.2% | 95.0% | 31.0% | 31.0% | pending | pending | pending |
| by_cosine_alpha_0.50 | 1.06 | 1.9% | 2.4% | 94.0% | 32.0% | 32.0% | pending | pending | pending |
| bh_cosine_alpha_0.10_ctx10 | 0.34 | 2.9% | 1.2% | 95.0% | 31.0% | 31.0% | pending | pending | pending |
| bh_cosine_alpha_0.30_ctx10 | 2.07 | 5.8% | 14.1% | 74.0% | 41.0% | 38.0% | pending | pending | pending |
| bh_cosine_alpha_0.50_ctx10 | 3.89 | 6.2% | 28.2% | 58.0% | 51.0% | 48.0% | pending | pending | pending |
| bh_cosine_alpha_0.90_ctx10 | 8.70 | 6.0% | 61.2% | 13.0% | 75.0% | 68.0% | pending | pending | pending |
| bh_cosine_alpha_0.99_ctx10 | 10.00 | 5.8% | 68.2% | 0.0% | 81.0% | 74.0% | pending | pending | pending |
| bh_cosine_alpha_0.99_ctx20 | 20.00 | 4.0% | 92.9% | 0.0% | 97.0% | 94.0% | pending | pending | pending |

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
  "calibration_queries_per_dataset": {
    "hotpotqa": 1000,
    "mmqa": 1000,
    "tatqa": 1000,
    "webqa": 1000
  },
  "test_queries_per_dataset": {
    "hotpotqa": 100,
    "mmqa": 100,
    "tatqa": 100,
    "webqa": 100
  },
  "plan_root": "research/internal_state_rag/results/all_datasets_cosine_six_methods_1000cal_100test_2026_09_26/splits",
  "BH_note": "BH normally needs independence or suitable positive dependence; its rank cap is experimental."
}
```
