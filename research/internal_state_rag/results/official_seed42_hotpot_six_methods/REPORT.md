# HotpotQA cosine conformal selector comparison

Frozen cosine Top-30 candidates; 100 disjoint calibration queries; 1,000 official HotpotQA dev queries selected by seed 42. BH caps are experimental rank-ordered backfill policies, so no capped-procedure FDR guarantee is claimed.

| Method | Chunks | Precision | Recall | Empty | Any support | All support | EM | F1 | Numeric |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_top10 | 10.00 | 16.2% | 93.0% | 0.0% | 98.7% | 88.3% | 0.383 | 0.500 | 0.405 |
| fixed_top20 | 20.00 | 8.5% | 97.8% | 0.0% | 99.7% | 96.1% | 0.364 | 0.480 | 0.386 |
| cce_alpha_0.10 | 7.62 | 20.0% | 87.6% | 0.7% | 97.4% | 80.0% | 0.364 | 0.482 | 0.385 |
| conflare_alpha_0.10 | 7.46 | 20.4% | 87.4% | 0.7% | 97.3% | 79.7% | 0.366 | 0.484 | 0.387 |
| traq_alpha_0.10 | 14.18 | 11.6% | 94.5% | 0.0% | 99.0% | 90.8% | 0.374 | 0.495 | 0.397 |
| query_level_cosine_internal_alpha_0.10 | 2.80 | 56.9% | 91.7% | 0.0% | 98.5% | 86.1% | 0.395 | 0.522 | 0.416 |
| by_cosine_alpha_0.10 | 0.15 | 73.3% | 6.3% | 90.9% | 9.7% | 4.2% | 0.167 | 0.256 | 0.182 |
| by_cosine_alpha_0.30 | 0.75 | 57.1% | 24.7% | 63.1% | 36.7% | 16.3% | 0.211 | 0.308 | 0.223 |
| by_cosine_alpha_0.50 | 1.46 | 43.9% | 36.8% | 52.2% | 47.3% | 28.7% | 0.245 | 0.339 | 0.254 |
| bh_cosine_alpha_0.10_ctx10 | 0.99 | 52.5% | 30.0% | 57.8% | 41.7% | 21.7% | 0.220 | 0.314 | 0.229 |
| bh_cosine_alpha_0.30_ctx10 | 2.86 | 37.3% | 61.4% | 23.1% | 75.6% | 50.7% | 0.299 | 0.415 | 0.314 |
| bh_cosine_alpha_0.50_ctx10 | 4.57 | 28.2% | 74.2% | 10.1% | 88.9% | 63.2% | 0.337 | 0.457 | 0.356 |
| bh_cosine_alpha_0.90_ctx10 | 8.31 | 18.3% | 87.3% | 2.5% | 96.3% | 80.5% | 0.373 | 0.492 | 0.395 |
| bh_cosine_alpha_0.99_ctx10 | 9.88 | 16.3% | 92.5% | 0.3% | 98.4% | 87.7% | 0.382 | 0.498 | 0.404 |
| bh_cosine_alpha_0.99_ctx20 | 19.73 | 8.6% | 97.2% | 0.3% | 99.4% | 95.5% | 0.362 | 0.479 | 0.384 |

## Calibration

```json
{
  "calibration_queries": 100,
  "test_queries": 1000,
  "thresholds": {
    "cce_alpha_0.10": 0.4592403471469879,
    "conflare_alpha_0.10": 0.4604645162820816,
    "traq_alpha_0.10": 0.4241151511669159,
    "false_score_bank_size": 2826,
    "support_score_bank_size": 174
  },
  "methods": [
    "fixed_top10",
    "fixed_top20",
    "cce_alpha_0.10",
    "conflare_alpha_0.10",
    "traq_alpha_0.10",
    "query_level_cosine_internal_alpha_0.10",
    "by_cosine_alpha_0.10",
    "by_cosine_alpha_0.30",
    "by_cosine_alpha_0.50",
    "bh_cosine_alpha_0.10_ctx10",
    "bh_cosine_alpha_0.30_ctx10",
    "bh_cosine_alpha_0.50_ctx10",
    "bh_cosine_alpha_0.90_ctx10",
    "bh_cosine_alpha_0.99_ctx10",
    "bh_cosine_alpha_0.99_ctx20"
  ],
  "bh_note": "BH is calculated from the same candidate false-score p-values as BY. BH context caps are experimental."
}
```
