# Full context-selection comparison: original cosine, literature baselines, and internal signals

Every row evaluates the same frozen Jina Top-30 candidates on the same 100 held-out test qids per dataset. This consolidates completed artifacts; it does not rerun, average, or overwrite any experiment.

> **Fidelity status:** CCE, CONFLARE, and TRAQ rows are **original-formula / calibration-unit adaptations**, not end-to-end executions of their released pipelines. The exact boundary is documented in [FIDELITY_AUDIT.md](FIDELITY_AUDIT.md). Do not claim these values reproduce or outperform the published systems.

## Protocol distinctions that must remain visible

- **Original cosine / CCE / CONFLARE / TRAQ / alpha-free:** 1,000 calibration qids and 100 held-out test qids. CCE, CONFLARE, and TRAQ are Jina adaptations with their distinct retrieval calibration units. TRAQ is its retrieval component only, not the full semantic answer prediction-set procedure.
- **Internal Qwen-7B rows:** the same 1,000 parent calibration qids are split into disjoint probe-train and conformal-calibration roles (about 500 / 500, exact counts shown in the internal report), followed by the same 100 test qids. Their use of 500 calibration qids means they are an ablation, not a strictly matched 1,000-calibration head-to-head result.
- **Alpha-free rows:** threshold is chosen only on the 1,000 calibration qids for the stated empirical utility. It is frozen for test inference, but it has no conformal risk guarantee.
- **BY/BH:** `alpha` is the multiple-testing level. BH rows use the reported experimental context cap; no capped-procedure FDR guarantee is claimed.

## Metrics

Chunks is mean retained chunks/query; Precision and Recall are micro support metrics inside frozen Top-30; Empty is the fraction retaining no chunk. Any/All are query support-retention rates recorded by the source artifact. EM, token F1, and Numeric are shared deterministic Qwen direct-answer diagnostics on the held-out 100 qids, not conformal guarantees.

## hotpotqa

| Family | Method | Chunks | Precision | Recall | Empty | Any support | All support | EM | F1 | Numeric |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed | Fixed Top-10 | 10.00 | 16.0% | 93.0% | 0.0% | 99.0% | 88.0% | 0.430 | 0.538 | 0.450 |
| Fixed | Fixed Top-20 | 20.00 | 8.4% | 97.7% | 0.0% | 100.0% | 96.0% | 0.390 | 0.504 | 0.410 |
| Literature | CCE Conformal-Embedding, Jina adaptation (α=.10) | 8.75 | 17.8% | 90.7% | 0.0% | 96.9% | 85.7% | 0.460 | 0.559 | 0.480 |
| Literature | CONFLARE source-question, Jina adaptation (α=.10) | 2.14 | 57.0% | 70.9% | 8.0% | 88.8% | 55.1% | 0.430 | 0.527 | 0.440 |
| Literature | TRAQ retrieval Bonferroni, Jina adaptation (α=.10; α_R=.05) | 3.39 | 41.3% | 81.4% | 3.0% | 92.9% | 71.4% | 0.440 | 0.553 | 0.450 |
| Original cosine | Query-level cosine all-support (α=.10; 1,000 cal) | 10.60 | 15.1% | 93.0% | 0.0% | 99.0% | 88.0% | 0.420 | 0.537 | 0.440 |
| Original cosine | BY cosine (α=.10) | 0.17 | 70.6% | 7.0% | 88.0% | 14.0% | 5.0% | 0.130 | 0.233 | 0.130 |
| Original cosine | BY cosine (α=.30) | 0.42 | 71.4% | 17.4% | 73.0% | 29.0% | 12.0% | 0.170 | 0.266 | 0.170 |
| Original cosine | BY cosine (α=.50) | 0.74 | 58.1% | 25.0% | 65.0% | 37.0% | 19.0% | 0.210 | 0.309 | 0.210 |
| Original cosine | BH cosine, ctx=10 (α=.10) | 0.52 | 69.2% | 20.9% | 70.0% | 32.0% | 15.0% | 0.180 | 0.276 | 0.180 |
| Original cosine | BH cosine, ctx=10 (α=.30) | 2.42 | 34.3% | 48.3% | 36.0% | 64.0% | 40.0% | 0.290 | 0.398 | 0.290 |
| Original cosine | BH cosine, ctx=10 (α=.50) | 4.11 | 28.7% | 68.6% | 18.0% | 83.0% | 59.0% | 0.360 | 0.463 | 0.360 |
| Original cosine | BH cosine, ctx=10 (α=.90) | 6.92 | 20.8% | 83.7% | 5.0% | 94.0% | 77.0% | 0.430 | 0.530 | 0.440 |
| Original cosine | BH cosine, ctx=10 (α=.99) | 9.76 | 16.1% | 91.3% | 1.0% | 98.0% | 86.0% | 0.420 | 0.529 | 0.440 |
| Original cosine | BH cosine, ctx=20 (α=.99) | 19.46 | 8.5% | 95.9% | 1.0% | 99.0% | 94.0% | 0.390 | 0.506 | 0.410 |
| Internal Qwen-7B | Cosine baseline (500 probe / 500 conformal cal; α=.10) | 9.89 | 16.2% | 93.0% | 0.0% | 99.0% | 88.0% | 0.400 | 0.508 | 0.420 |
| Internal Qwen-7B | LM-head only (500 / 500; α=.10) | 8.20 | 19.9% | 94.8% | 0.0% | 96.0% | 91.0% | 0.430 | 0.551 | 0.440 |
| Internal Qwen-7B | Hidden-state relevance probe only (500 / 500; α=.10) | 2.67 | 60.7% | 94.2% | 0.0% | 96.0% | 91.0% | 0.440 | 0.576 | 0.460 |
| Internal Qwen-7B | LM-head + hidden-state probe (500 / 500; α=.10) | 2.67 | 60.7% | 94.2% | 0.0% | 96.0% | 91.0% | 0.440 | 0.576 | 0.460 |
| Internal Qwen-7B | Cosine + LM-head + hidden-state probe (500 / 500; α=.10) | 2.50 | 65.2% | 94.8% | 0.0% | 98.0% | 92.0% | 0.440 | 0.576 | 0.460 |
| Alpha-free | Query-level cosine F1-selected (1,000 cal; no α at test) | 1.62 | 75.3% | 70.9% | 0.0% | 90.8% | 53.1% | 0.430 | 0.534 | 0.440 |
| Alpha-free | Query-level cosine budget-10 all-support (1,000 cal; no α at test) | 9.89 | 16.2% | 93.0% | 0.0% | 99.0% | 87.8% | 0.400 | 0.508 | 0.420 |

## mmqa

| Family | Method | Chunks | Precision | Recall | Empty | Any support | All support | EM | F1 | Numeric |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed | Fixed Top-10 | 10.00 | 10.9% | 96.5% | 0.0% | 100.0% | 96.0% | 0.470 | 0.512 | 0.490 |
| Fixed | Fixed Top-20 | 20.00 | 5.5% | 97.3% | 0.0% | 100.0% | 97.0% | 0.460 | 0.497 | 0.480 |
| Literature | CCE Conformal-Embedding, Jina adaptation (α=.10) | 10.00 | 10.0% | 88.5% | 5.0% | 95.7% | 87.1% | 0.480 | 0.519 | 0.500 |
| Literature | CONFLARE source-question, Jina adaptation (α=.10) | 5.70 | 16.0% | 80.5% | 8.0% | 89.2% | 78.5% | 0.480 | 0.520 | 0.500 |
| Literature | TRAQ retrieval Bonferroni, Jina adaptation (α=.10; α_R=.05) | 9.97 | 10.0% | 88.5% | 5.0% | 95.7% | 87.1% | 0.480 | 0.519 | 0.500 |
| Original cosine | Query-level cosine all-support (α=.10; 1,000 cal) | 9.05 | 12.0% | 96.5% | 0.0% | 100.0% | 96.0% | 0.510 | 0.552 | 0.530 |
| Original cosine | BY cosine (α=.10) | 0.05 | 80.0% | 3.5% | 95.0% | 11.0% | 9.0% | 0.160 | 0.196 | 0.170 |
| Original cosine | BY cosine (α=.30) | 0.24 | 45.8% | 9.7% | 88.0% | 18.0% | 16.0% | 0.190 | 0.225 | 0.200 |
| Original cosine | BY cosine (α=.50) | 0.40 | 37.5% | 13.3% | 82.0% | 22.0% | 20.0% | 0.190 | 0.234 | 0.200 |
| Original cosine | BH cosine, ctx=10 (α=.10) | 0.35 | 42.9% | 13.3% | 83.0% | 22.0% | 20.0% | 0.200 | 0.244 | 0.210 |
| Original cosine | BH cosine, ctx=10 (α=.30) | 0.96 | 31.2% | 26.5% | 67.0% | 36.0% | 31.0% | 0.240 | 0.283 | 0.250 |
| Original cosine | BH cosine, ctx=10 (α=.50) | 2.58 | 19.4% | 44.2% | 52.0% | 53.0% | 48.0% | 0.340 | 0.381 | 0.350 |
| Original cosine | BH cosine, ctx=10 (α=.90) | 8.11 | 11.7% | 84.1% | 14.0% | 88.0% | 85.0% | 0.440 | 0.488 | 0.460 |
| Original cosine | BH cosine, ctx=10 (α=.99) | 9.90 | 10.9% | 95.6% | 1.0% | 99.0% | 95.0% | 0.480 | 0.522 | 0.500 |
| Original cosine | BH cosine, ctx=20 (α=.99) | 19.80 | 5.5% | 96.5% | 1.0% | 99.0% | 96.0% | 0.470 | 0.507 | 0.490 |
| Internal Qwen-7B | Cosine baseline (500 probe / 500 conformal cal; α=.10) | 8.07 | 13.5% | 96.5% | 0.0% | 100.0% | 96.0% | 0.500 | 0.547 | 0.520 |
| Internal Qwen-7B | LM-head only (500 / 500; α=.10) | 8.55 | 12.7% | 96.5% | 0.0% | 98.0% | 96.0% | 0.500 | 0.537 | 0.520 |
| Internal Qwen-7B | Hidden-state relevance probe only (500 / 500; α=.10) | 4.17 | 25.9% | 95.6% | 0.0% | 98.0% | 95.0% | 0.500 | 0.545 | 0.520 |
| Internal Qwen-7B | LM-head + hidden-state probe (500 / 500; α=.10) | 4.12 | 26.2% | 95.6% | 0.0% | 98.0% | 95.0% | 0.500 | 0.545 | 0.520 |
| Internal Qwen-7B | Cosine + LM-head + hidden-state probe (500 / 500; α=.10) | 3.70 | 29.7% | 97.3% | 0.0% | 99.0% | 97.0% | 0.500 | 0.539 | 0.520 |
| Alpha-free | Query-level cosine F1-selected (1,000 cal; no α at test) | 1.04 | 71.2% | 65.5% | 0.0% | 79.6% | 62.4% | 0.490 | 0.523 | 0.510 |
| Alpha-free | Query-level cosine budget-10 all-support (1,000 cal; no α at test) | 9.63 | 11.3% | 96.5% | 0.0% | 100.0% | 95.7% | 0.510 | 0.551 | 0.530 |

## tatqa

| Family | Method | Chunks | Precision | Recall | Empty | Any support | All support | EM | F1 | Numeric |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed | Fixed Top-10 | 10.00 | 8.9% | 84.8% | 0.0% | 89.0% | 84.0% | 0.190 | 0.292 | 0.270 |
| Fixed | Fixed Top-20 | 20.00 | 5.1% | 96.2% | 0.0% | 97.0% | 96.0% | 0.150 | 0.253 | 0.230 |
| Literature | CCE Conformal-Embedding, Jina adaptation (α=.10) | 17.80 | 5.2% | 87.6% | 6.0% | 89.0% | 86.8% | 0.150 | 0.253 | 0.230 |
| Literature | CONFLARE source-question, Jina adaptation (α=.10) | 16.52 | 5.6% | 87.6% | 7.0% | 89.0% | 86.8% | 0.150 | 0.248 | 0.220 |
| Literature | TRAQ retrieval Bonferroni, Jina adaptation (α=.10; α_R=.05) | 21.57 | 4.5% | 92.4% | 4.0% | 93.4% | 91.2% | 0.130 | 0.234 | 0.210 |
| Original cosine | Query-level cosine all-support (α=.10; 1,000 cal) | 11.30 | 8.1% | 87.6% | 0.0% | 91.0% | 87.0% | 0.180 | 0.292 | 0.260 |
| Original cosine | BY cosine (α=.10) | 0.40 | 10.0% | 3.8% | 94.0% | 13.0% | 12.0% | 0.030 | 0.100 | 0.100 |
| Original cosine | BY cosine (α=.30) | 1.04 | 6.7% | 6.7% | 90.0% | 16.0% | 15.0% | 0.040 | 0.113 | 0.110 |
| Original cosine | BY cosine (α=.50) | 2.85 | 6.0% | 16.2% | 80.0% | 25.0% | 22.0% | 0.040 | 0.102 | 0.090 |
| Original cosine | BH cosine, ctx=10 (α=.10) | 0.69 | 11.6% | 7.6% | 87.0% | 17.0% | 15.0% | 0.040 | 0.113 | 0.110 |
| Original cosine | BH cosine, ctx=10 (α=.30) | 2.25 | 8.0% | 17.1% | 68.0% | 27.0% | 22.0% | 0.060 | 0.129 | 0.120 |
| Original cosine | BH cosine, ctx=10 (α=.50) | 3.89 | 8.2% | 30.5% | 51.0% | 40.0% | 35.0% | 0.090 | 0.167 | 0.150 |
| Original cosine | BH cosine, ctx=10 (α=.90) | 8.35 | 8.7% | 69.5% | 12.0% | 77.0% | 71.0% | 0.170 | 0.260 | 0.250 |
| Original cosine | BH cosine, ctx=10 (α=.99) | 9.80 | 8.9% | 82.9% | 2.0% | 87.0% | 82.0% | 0.190 | 0.290 | 0.270 |
| Original cosine | BH cosine, ctx=20 (α=.99) | 19.60 | 5.1% | 94.3% | 2.0% | 95.0% | 94.0% | 0.150 | 0.250 | 0.230 |
| Internal Qwen-7B | Cosine baseline (500 probe / 500 conformal cal; α=.10) | 10.95 | 8.4% | 87.6% | 0.0% | 91.0% | 87.0% | 0.190 | 0.293 | 0.270 |
| Internal Qwen-7B | LM-head only (500 / 500; α=.10) | 10.11 | 9.9% | 95.2% | 0.0% | 99.0% | 95.0% | 0.200 | 0.315 | 0.270 |
| Internal Qwen-7B | Hidden-state relevance probe only (500 / 500; α=.10) | 4.26 | 22.1% | 89.5% | 0.0% | 94.0% | 89.0% | 0.150 | 0.249 | 0.240 |
| Internal Qwen-7B | LM-head + hidden-state probe (500 / 500; α=.10) | 4.14 | 23.2% | 91.4% | 0.0% | 96.0% | 91.0% | 0.170 | 0.280 | 0.280 |
| Internal Qwen-7B | Cosine + LM-head + hidden-state probe (500 / 500; α=.10) | 3.96 | 24.5% | 92.4% | 0.0% | 97.0% | 92.0% | 0.150 | 0.253 | 0.260 |
| Alpha-free | Query-level cosine F1-selected (1,000 cal; no α at test) | 1.08 | 46.3% | 47.6% | 0.0% | 54.9% | 46.2% | 0.130 | 0.234 | 0.220 |
| Alpha-free | Query-level cosine budget-10 all-support (1,000 cal; no α at test) | 10.05 | 9.0% | 85.7% | 0.0% | 89.0% | 83.5% | 0.160 | 0.275 | 0.250 |

## webqa

| Family | Method | Chunks | Precision | Recall | Empty | Any support | All support | EM | F1 | Numeric |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed | Fixed Top-10 | 10.00 | 5.8% | 68.2% | 0.0% | 81.0% | 74.0% | 0.000 | 0.125 | 0.050 |
| Fixed | Fixed Top-20 | 20.00 | 4.0% | 92.9% | 0.0% | 97.0% | 94.0% | 0.000 | 0.128 | 0.050 |
| Literature | CCE Conformal-Embedding, Jina adaptation (α=.10) | 19.82 | 3.6% | 83.5% | 9.0% | 85.7% | 80.0% | 0.000 | 0.179 | 0.060 |
| Literature | CONFLARE source-question, Jina adaptation (α=.10) | 18.27 | 3.8% | 82.4% | 10.0% | 84.3% | 78.6% | 0.000 | 0.189 | 0.060 |
| Literature | TRAQ retrieval Bonferroni, Jina adaptation (α=.10; α_R=.05) | 22.17 | 3.5% | 91.8% | 8.0% | 92.9% | 90.0% | 0.000 | 0.174 | 0.050 |
| Original cosine | Query-level cosine all-support (α=.10; 1,000 cal) | 17.63 | 4.4% | 90.6% | 0.0% | 96.0% | 92.0% | 0.000 | 0.125 | 0.050 |
| Original cosine | BY cosine (α=.10) | 0.00 | 0.0% | 0.0% | 100.0% | 30.0% | 30.0% | 0.020 | 0.462 | 0.110 |
| Original cosine | BY cosine (α=.30) | 0.93 | 1.1% | 1.2% | 95.0% | 31.0% | 31.0% | 0.020 | 0.449 | 0.110 |
| Original cosine | BY cosine (α=.50) | 1.06 | 1.9% | 2.4% | 94.0% | 32.0% | 32.0% | 0.020 | 0.443 | 0.110 |
| Original cosine | BH cosine, ctx=10 (α=.10) | 0.34 | 2.9% | 1.2% | 95.0% | 31.0% | 31.0% | 0.020 | 0.445 | 0.110 |
| Original cosine | BH cosine, ctx=10 (α=.30) | 2.07 | 5.8% | 14.1% | 74.0% | 41.0% | 38.0% | 0.010 | 0.376 | 0.070 |
| Original cosine | BH cosine, ctx=10 (α=.50) | 3.89 | 6.2% | 28.2% | 58.0% | 51.0% | 48.0% | 0.010 | 0.295 | 0.070 |
| Original cosine | BH cosine, ctx=10 (α=.90) | 8.70 | 6.0% | 61.2% | 13.0% | 75.0% | 68.0% | 0.000 | 0.187 | 0.080 |
| Original cosine | BH cosine, ctx=10 (α=.99) | 10.00 | 5.8% | 68.2% | 0.0% | 81.0% | 74.0% | 0.000 | 0.125 | 0.050 |
| Original cosine | BH cosine, ctx=20 (α=.99) | 20.00 | 4.0% | 92.9% | 0.0% | 97.0% | 94.0% | 0.000 | 0.128 | 0.050 |
| Internal Qwen-7B | Cosine baseline (500 probe / 500 conformal cal; α=.10) | 18.89 | 4.2% | 92.9% | 0.0% | 97.0% | 94.0% | 0.000 | 0.122 | 0.050 |
| Internal Qwen-7B | LM-head only (500 / 500; α=.10) | 16.24 | 4.9% | 94.1% | 0.0% | 97.0% | 96.0% | 0.000 | 0.151 | 0.040 |
| Internal Qwen-7B | Hidden-state relevance probe only (500 / 500; α=.10) | 6.91 | 9.4% | 76.5% | 0.0% | 89.0% | 83.0% | 0.000 | 0.120 | 0.040 |
| Internal Qwen-7B | LM-head + hidden-state probe (500 / 500; α=.10) | 6.90 | 9.9% | 80.0% | 0.0% | 92.0% | 85.0% | 0.000 | 0.123 | 0.040 |
| Internal Qwen-7B | Cosine + LM-head + hidden-state probe (500 / 500; α=.10) | 5.77 | 11.6% | 78.8% | 0.0% | 90.0% | 84.0% | 0.000 | 0.122 | 0.040 |
| Alpha-free | Query-level cosine F1-selected (1,000 cal; no α at test) | 1.59 | 11.3% | 21.2% | 0.0% | 25.7% | 20.0% | 0.000 | 0.115 | 0.030 |
| Alpha-free | Query-level cosine budget-10 all-support (1,000 cal; no α at test) | 10.18 | 6.0% | 71.8% | 0.0% | 77.1% | 67.1% | 0.000 | 0.120 | 0.050 |

## Exact source artifacts

- [Original cosine, BY/BH α sweep](../all_datasets_cosine_six_methods_1000cal_100test_2026_09_26/REPORT.md)
- [CCE / CONFLARE / TRAQ adapted protocol](../literature_protocol_1000cal_100test_2026_09_26/REPORT.md)
- [Qwen-7B internal-signal ablation](../all_datasets_cosine_six_methods_1000cal_100test_2026_09_26/internal_signal_ablation_7b/REPORT.md)
- [Alpha-free query-level operating points](../alpha_free_query_level_1000cal_100test_2026_09_26/DOWNSTREAM_REPORT.md)
- [Fidelity audit for CCE / CONFLARE / TRAQ](FIDELITY_AUDIT.md)
