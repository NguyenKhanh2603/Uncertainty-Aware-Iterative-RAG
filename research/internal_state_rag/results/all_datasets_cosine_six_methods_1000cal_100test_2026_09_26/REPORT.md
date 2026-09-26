# Cosine conformal selector comparison

Every row within a dataset uses the same frozen Top-30 cosine candidates and its disjoint calibration plan. Query-level cosine is the pure per-query z-scored cosine threshold with a deterministic Top-1 empty-context fallback. BH context caps are experimental rank-ordered policies and have no claimed capped-procedure FDR guarantee.

> **Baseline scope:** `CCE`, `CONFLARE`, and `TRAQ` labels below are
> shared-Jina-cosine threshold proxies, rather than faithful executions of the
> original papers. In particular, CCE and CONFLARE use nearly the same
> positive-score quantile here, so their rows are expected to be nearly
> identical. See [BASELINE_SCOPE.md](BASELINE_SCOPE.md) before interpreting
> them as a literature comparison.

## Exact 1,000-calibration / 100-test splits

The qid lists used for every row are versioned here. Each calibration manifest
has exactly 1,000 qids and each test manifest has exactly 100 distinct held-out
qids:

| Dataset | Calibration qids | Held-out test qids |
|---|---|---|
| HotpotQA | [calibration manifest](splits/hotpotqa/calibration_manifest.json) | [test manifest](splits/hotpotqa/test_manifest.json) |
| MMQA | [calibration manifest](splits/mmqa/calibration_manifest.json) | [test manifest](splits/mmqa/test_manifest.json) |
| TAT-QA | [calibration manifest](splits/tatqa/calibration_manifest.json) | [test manifest](splits/tatqa/test_manifest.json) |
| WebQA | [calibration manifest](splits/webqa/calibration_manifest.json) | [test manifest](splits/webqa/test_manifest.json) |

The shared split-level provenance and no-overlap audit is in
[SPLIT_INTEGRITY.json](splits/SPLIT_INTEGRITY.json).

<!-- metric-definitions-start -->
## Metric definitions

All selection metrics use the frozen Jina Top-30 candidate pool. A `support` is a dataset-labelled evidence chunk inside that pool.

- **Chunks:** mean number of retained candidates per test query.
- **Precision:** micro precision, `retained support chunks / all retained chunks`.
- **Recall:** micro support recall, `retained support chunks / all labelled support chunks in Top-30`. It cannot recover evidence absent from Top-30.
- **Empty:** fraction of queries for which a selector retains no chunk. Fixed Top-K and query-level cosine use their documented non-empty policies; pointwise selectors may be empty.
- **Any support:** legacy retrieval-pool coverage: at least one support is retained for a query with labelled Top-30 support. A query with no labelled support in Top-30 is counted as covered because no available support can be lost. The retrieval ceiling for each dataset is available in its summary JSON.
- **All support:** fraction of all test queries whose labelled Top-30 support set is a subset of the retained context; a query with no labelled Top-30 support is likewise counted as covered. This is why all-support should be read alongside Recall and the retrieval ceiling.
- **EM / F1 / Numeric:** mean Exact Match, token-level F1, and numerical accuracy of the same deterministic Qwen direct-answer evaluation over the 100 held-out qids. These are downstream diagnostics, not conformal coverage guarantees.

Rows marked **†** are the separately run literature-protocol records: they retain the same qids and Top-30 candidates but use CCE's positive-pair unit, CONFLARE's one-relevant-record unit, and TRAQ's Bonferroni retrieval unit. TRAQ's row is retrieval-only, not its full semantic answer-prediction-set system.
<!-- metric-definitions-end -->

## hotpotqa (complete; n=100)

| Method | Chunks | Precision | Recall | Empty | Any support | All support | EM | F1 | Numeric |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_top10 | 10.00 | 16.0% | 93.0% | 0.0% | 99.0% | 88.0% | 0.430 | 0.538 | 0.450 |
| fixed_top20 | 20.00 | 8.4% | 97.7% | 0.0% | 100.0% | 96.0% | 0.390 | 0.504 | 0.410 |
| CCE-style positive-cosine proxy (α=0.10) | 8.75 | 17.8% | 90.7% | 0.0% | 97.0% | 86.0% | 0.460 | 0.559 | 0.480 |
| CONFLARE-style positive-cosine proxy (α=0.10) | 8.74 | 17.8% | 90.7% | 0.0% | 97.0% | 86.0% | 0.460 | 0.559 | 0.480 |
| TRAQ-style loose-positive-cosine proxy (α=0.10) | 14.53 | 11.2% | 94.8% | 0.0% | 98.0% | 92.0% | 0.430 | 0.539 | 0.450 |
| query_level_cosine_alpha_0.10 | 10.60 | 15.1% | 93.0% | 0.0% | 99.0% | 88.0% | 0.420 | 0.537 | 0.440 |
| Fixed Top-10† | 10.00 | 16.0% | 93.0% | 0.0% | 99.0% | 88.0% | 0.430 | 0.538 | 0.450 |
| CCE Conformal-Embedding (Jina adaptation)† | 8.75 | 17.8% | 90.7% | 0.0% | 97.0% | 86.0% | 0.460 | 0.559 | 0.480 |
| CONFLARE source-question (Jina adaptation)† | 2.14 | 57.0% | 70.9% | 8.0% | 89.0% | 56.0% | 0.430 | 0.527 | 0.440 |
| TRAQ retrieval, Bonferroni (Jina adaptation)† | 3.39 | 41.3% | 81.4% | 3.0% | 93.0% | 72.0% | 0.440 | 0.553 | 0.450 |
| Query-level all-support cosine† | 10.60 | 15.1% | 93.0% | 0.0% | 99.0% | 88.0% | 0.420 | 0.537 | 0.440 |
| by_cosine_alpha_0.10 | 0.17 | 70.6% | 7.0% | 88.0% | 14.0% | 5.0% | 0.130 | 0.233 | 0.130 |
| by_cosine_alpha_0.30 | 0.42 | 71.4% | 17.4% | 73.0% | 29.0% | 12.0% | 0.170 | 0.266 | 0.170 |
| by_cosine_alpha_0.50 | 0.74 | 58.1% | 25.0% | 65.0% | 37.0% | 19.0% | 0.210 | 0.309 | 0.210 |
| bh_cosine_alpha_0.10_ctx10 | 0.52 | 69.2% | 20.9% | 70.0% | 32.0% | 15.0% | 0.180 | 0.276 | 0.180 |
| bh_cosine_alpha_0.30_ctx10 | 2.42 | 34.3% | 48.3% | 36.0% | 64.0% | 40.0% | 0.290 | 0.398 | 0.290 |
| bh_cosine_alpha_0.50_ctx10 | 4.11 | 28.7% | 68.6% | 18.0% | 83.0% | 59.0% | 0.360 | 0.463 | 0.360 |
| bh_cosine_alpha_0.90_ctx10 | 6.92 | 20.8% | 83.7% | 5.0% | 94.0% | 77.0% | 0.430 | 0.530 | 0.440 |
| bh_cosine_alpha_0.99_ctx10 | 9.76 | 16.1% | 91.3% | 1.0% | 98.0% | 86.0% | 0.420 | 0.529 | 0.440 |
| bh_cosine_alpha_0.99_ctx20 | 19.46 | 8.5% | 95.9% | 1.0% | 99.0% | 94.0% | 0.390 | 0.506 | 0.410 |

## mmqa (complete; n=100)

| Method | Chunks | Precision | Recall | Empty | Any support | All support | EM | F1 | Numeric |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_top10 | 10.00 | 10.9% | 96.5% | 0.0% | 100.0% | 96.0% | 0.470 | 0.512 | 0.490 |
| fixed_top20 | 20.00 | 5.5% | 97.3% | 0.0% | 100.0% | 97.0% | 0.460 | 0.497 | 0.480 |
| CCE-style positive-cosine proxy (α=0.10) | 10.00 | 10.0% | 88.5% | 5.0% | 96.0% | 88.0% | 0.480 | 0.519 | 0.500 |
| CONFLARE-style positive-cosine proxy (α=0.10) | 10.00 | 10.0% | 88.5% | 5.0% | 96.0% | 88.0% | 0.480 | 0.519 | 0.500 |
| TRAQ-style loose-positive-cosine proxy (α=0.10) | 18.34 | 5.9% | 95.6% | 2.0% | 98.0% | 95.0% | 0.480 | 0.521 | 0.500 |
| query_level_cosine_alpha_0.10 | 9.05 | 12.0% | 96.5% | 0.0% | 100.0% | 96.0% | 0.510 | 0.552 | 0.530 |
| Fixed Top-10† | 10.00 | 10.9% | 96.5% | 0.0% | 100.0% | 96.0% | 0.470 | 0.512 | 0.490 |
| CCE Conformal-Embedding (Jina adaptation)† | 10.00 | 10.0% | 88.5% | 5.0% | 96.0% | 88.0% | 0.480 | 0.519 | 0.500 |
| CONFLARE source-question (Jina adaptation)† | 5.70 | 16.0% | 80.5% | 8.0% | 90.0% | 80.0% | 0.480 | 0.520 | 0.500 |
| TRAQ retrieval, Bonferroni (Jina adaptation)† | 9.97 | 10.0% | 88.5% | 5.0% | 96.0% | 88.0% | 0.480 | 0.519 | 0.500 |
| Query-level all-support cosine† | 9.05 | 12.0% | 96.5% | 0.0% | 100.0% | 96.0% | 0.510 | 0.552 | 0.530 |
| by_cosine_alpha_0.10 | 0.05 | 80.0% | 3.5% | 95.0% | 11.0% | 9.0% | 0.160 | 0.196 | 0.170 |
| by_cosine_alpha_0.30 | 0.24 | 45.8% | 9.7% | 88.0% | 18.0% | 16.0% | 0.190 | 0.225 | 0.200 |
| by_cosine_alpha_0.50 | 0.40 | 37.5% | 13.3% | 82.0% | 22.0% | 20.0% | 0.190 | 0.234 | 0.200 |
| bh_cosine_alpha_0.10_ctx10 | 0.35 | 42.9% | 13.3% | 83.0% | 22.0% | 20.0% | 0.200 | 0.244 | 0.210 |
| bh_cosine_alpha_0.30_ctx10 | 0.96 | 31.2% | 26.5% | 67.0% | 36.0% | 31.0% | 0.240 | 0.283 | 0.250 |
| bh_cosine_alpha_0.50_ctx10 | 2.58 | 19.4% | 44.2% | 52.0% | 53.0% | 48.0% | 0.340 | 0.381 | 0.350 |
| bh_cosine_alpha_0.90_ctx10 | 8.11 | 11.7% | 84.1% | 14.0% | 88.0% | 85.0% | 0.440 | 0.488 | 0.460 |
| bh_cosine_alpha_0.99_ctx10 | 9.90 | 10.9% | 95.6% | 1.0% | 99.0% | 95.0% | 0.480 | 0.522 | 0.500 |
| bh_cosine_alpha_0.99_ctx20 | 19.80 | 5.5% | 96.5% | 1.0% | 99.0% | 96.0% | 0.470 | 0.507 | 0.490 |

## tatqa (complete; n=100)

| Method | Chunks | Precision | Recall | Empty | Any support | All support | EM | F1 | Numeric |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_top10 | 10.00 | 8.9% | 84.8% | 0.0% | 89.0% | 84.0% | 0.190 | 0.292 | 0.270 |
| fixed_top20 | 20.00 | 5.1% | 96.2% | 0.0% | 97.0% | 96.0% | 0.150 | 0.253 | 0.230 |
| CCE-style positive-cosine proxy (α=0.10) | 17.80 | 5.2% | 87.6% | 6.0% | 90.0% | 88.0% | 0.150 | 0.253 | 0.230 |
| CONFLARE-style positive-cosine proxy (α=0.10) | 17.80 | 5.2% | 87.6% | 6.0% | 90.0% | 88.0% | 0.150 | 0.253 | 0.230 |
| TRAQ-style loose-positive-cosine proxy (α=0.10) | 23.00 | 4.3% | 95.2% | 4.0% | 95.0% | 95.0% | 0.140 | 0.243 | 0.220 |
| query_level_cosine_alpha_0.10 | 11.30 | 8.1% | 87.6% | 0.0% | 91.0% | 87.0% | 0.180 | 0.292 | 0.260 |
| Fixed Top-10† | 10.00 | 8.9% | 84.8% | 0.0% | 89.0% | 84.0% | 0.190 | 0.292 | 0.270 |
| CCE Conformal-Embedding (Jina adaptation)† | 17.80 | 5.2% | 87.6% | 6.0% | 90.0% | 88.0% | 0.150 | 0.253 | 0.230 |
| CONFLARE source-question (Jina adaptation)† | 16.52 | 5.6% | 87.6% | 7.0% | 90.0% | 88.0% | 0.150 | 0.248 | 0.220 |
| TRAQ retrieval, Bonferroni (Jina adaptation)† | 21.57 | 4.5% | 92.4% | 4.0% | 94.0% | 92.0% | 0.130 | 0.234 | 0.210 |
| Query-level all-support cosine† | 11.30 | 8.1% | 87.6% | 0.0% | 91.0% | 87.0% | 0.180 | 0.292 | 0.260 |
| by_cosine_alpha_0.10 | 0.40 | 10.0% | 3.8% | 94.0% | 13.0% | 12.0% | 0.030 | 0.100 | 0.100 |
| by_cosine_alpha_0.30 | 1.04 | 6.7% | 6.7% | 90.0% | 16.0% | 15.0% | 0.040 | 0.113 | 0.110 |
| by_cosine_alpha_0.50 | 2.85 | 6.0% | 16.2% | 80.0% | 25.0% | 22.0% | 0.040 | 0.102 | 0.090 |
| bh_cosine_alpha_0.10_ctx10 | 0.69 | 11.6% | 7.6% | 87.0% | 17.0% | 15.0% | 0.040 | 0.113 | 0.110 |
| bh_cosine_alpha_0.30_ctx10 | 2.25 | 8.0% | 17.1% | 68.0% | 27.0% | 22.0% | 0.060 | 0.129 | 0.120 |
| bh_cosine_alpha_0.50_ctx10 | 3.89 | 8.2% | 30.5% | 51.0% | 40.0% | 35.0% | 0.090 | 0.167 | 0.150 |
| bh_cosine_alpha_0.90_ctx10 | 8.35 | 8.7% | 69.5% | 12.0% | 77.0% | 71.0% | 0.170 | 0.260 | 0.250 |
| bh_cosine_alpha_0.99_ctx10 | 9.80 | 8.9% | 82.9% | 2.0% | 87.0% | 82.0% | 0.190 | 0.290 | 0.270 |
| bh_cosine_alpha_0.99_ctx20 | 19.60 | 5.1% | 94.3% | 2.0% | 95.0% | 94.0% | 0.150 | 0.250 | 0.230 |

## webqa (complete; n=100)

| Method | Chunks | Precision | Recall | Empty | Any support | All support | EM | F1 | Numeric |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_top10 | 10.00 | 5.8% | 68.2% | 0.0% | 81.0% | 74.0% | 0.000 | 0.125 | 0.050 |
| fixed_top20 | 20.00 | 4.0% | 92.9% | 0.0% | 97.0% | 94.0% | 0.000 | 0.128 | 0.050 |
| CCE-style positive-cosine proxy (α=0.10) | 19.82 | 3.6% | 83.5% | 9.0% | 90.0% | 86.0% | 0.000 | 0.179 | 0.060 |
| CONFLARE-style positive-cosine proxy (α=0.10) | 19.80 | 3.6% | 83.5% | 9.0% | 90.0% | 86.0% | 0.000 | 0.179 | 0.060 |
| TRAQ-style loose-positive-cosine proxy (α=0.10) | 23.72 | 3.4% | 94.1% | 5.0% | 96.0% | 95.0% | 0.000 | 0.154 | 0.060 |
| query_level_cosine_alpha_0.10 | 17.63 | 4.4% | 90.6% | 0.0% | 96.0% | 92.0% | 0.000 | 0.125 | 0.050 |
| Fixed Top-10† | 10.00 | 5.8% | 68.2% | 0.0% | 81.0% | 74.0% | 0.000 | 0.125 | 0.050 |
| CCE Conformal-Embedding (Jina adaptation)† | 19.82 | 3.6% | 83.5% | 9.0% | 90.0% | 86.0% | 0.000 | 0.179 | 0.060 |
| CONFLARE source-question (Jina adaptation)† | 18.27 | 3.8% | 82.4% | 10.0% | 89.0% | 85.0% | 0.000 | 0.189 | 0.060 |
| TRAQ retrieval, Bonferroni (Jina adaptation)† | 22.17 | 3.5% | 91.8% | 8.0% | 95.0% | 93.0% | 0.000 | 0.174 | 0.050 |
| Query-level all-support cosine† | 17.63 | 4.4% | 90.6% | 0.0% | 96.0% | 92.0% | 0.000 | 0.125 | 0.050 |
| by_cosine_alpha_0.10 | 0.00 | 0.0% | 0.0% | 100.0% | 30.0% | 30.0% | 0.020 | 0.462 | 0.110 |
| by_cosine_alpha_0.30 | 0.93 | 1.1% | 1.2% | 95.0% | 31.0% | 31.0% | 0.020 | 0.449 | 0.110 |
| by_cosine_alpha_0.50 | 1.06 | 1.9% | 2.4% | 94.0% | 32.0% | 32.0% | 0.020 | 0.443 | 0.110 |
| bh_cosine_alpha_0.10_ctx10 | 0.34 | 2.9% | 1.2% | 95.0% | 31.0% | 31.0% | 0.020 | 0.445 | 0.110 |
| bh_cosine_alpha_0.30_ctx10 | 2.07 | 5.8% | 14.1% | 74.0% | 41.0% | 38.0% | 0.010 | 0.376 | 0.070 |
| bh_cosine_alpha_0.50_ctx10 | 3.89 | 6.2% | 28.2% | 58.0% | 51.0% | 48.0% | 0.010 | 0.295 | 0.070 |
| bh_cosine_alpha_0.90_ctx10 | 8.70 | 6.0% | 61.2% | 13.0% | 75.0% | 68.0% | 0.000 | 0.187 | 0.080 |
| bh_cosine_alpha_0.99_ctx10 | 10.00 | 5.8% | 68.2% | 0.0% | 81.0% | 74.0% | 0.000 | 0.125 | 0.050 |
| bh_cosine_alpha_0.99_ctx20 | 20.00 | 4.0% | 92.9% | 0.0% | 97.0% | 94.0% | 0.000 | 0.128 | 0.050 |

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
