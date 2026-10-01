# Three post-retrieval baselines — official eligible full test (2026-09-30)

The experiment uses the complete official splits with public answer and support labels: HotpotQA distractor/validation, MMQA dev, TAT-QA dev, and WebQA validation. The three selectors share the same Jina-v4 ranked candidate pool (dataset-provided candidates, Top-L≤30), the same 1,000-query calibration thresholds from `splits_khanh_27_09`, and greedy Qwen2-VL-7B generation with at most 24 new tokens.

- Split definition: [`SPLIT_SUMMARY.json`](../../splits/official_evaluable_test_2026_09_30/SPLIT_SUMMARY.json)
- Runner: [`run_eligible_full_three_baselines.py`](../../run_eligible_full_three_baselines.py)
- Retrieval launcher: [`run_eligible_full_three_baselines_2026_09_30.sh`](../../../../scripts/run_eligible_full_three_baselines_2026_09_30.sh)

## hotpotqa (complete; calibration=1,000, test=7,405)

| Family | Method | Chunks | Precision | Recall | Empty | Any support | All support | EM | F1 | Numeric |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CCE | CCE Conformal-Embedding (α=.10) | 8.25 | 21.3% | 88.0% | 0.8% | 96.0% | 80.0% | 0.443 | 0.564 | 0.471 |
| CONFLARE | CONFLARE source-question (α=.10) | 6.37 | 22.1% | 70.5% | 3.9% | 85.4% | 55.6% | 0.402 | 0.516 | 0.428 |
| TRAQ | TRAQ retrieval Bonferroni (α=.10; α_R=.05) | 7.72 | 21.6% | 83.5% | 1.5% | 93.7% | 73.2% | 0.434 | 0.555 | 0.463 |

Retrieval ceiling: 100.0% (7,405/7,405 queries contain at least one labelled support in the frozen candidate pool). Any-support and all-support are conditional on those retrievable queries.

## mmqa (complete; calibration=1,000, test=2,441)

| Family | Method | Chunks | Precision | Recall | Empty | Any support | All support | EM | F1 | Numeric |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CCE | CCE Conformal-Embedding (α=.10) | 17.35 | 8.5% | 89.1% | 2.6% | 91.0% | 83.2% | 0.413 | 0.467 | 0.433 |
| CONFLARE | CONFLARE source-question (α=.10) | 16.97 | 8.5% | 87.2% | 3.0% | 89.7% | 80.7% | 0.408 | 0.461 | 0.428 |
| TRAQ | TRAQ retrieval Bonferroni (α=.10; α_R=.05) | 18.39 | 8.4% | 93.0% | 1.6% | 93.9% | 88.9% | 0.422 | 0.477 | 0.442 |

Retrieval ceiling: 100.0% (2,440/2,441 queries contain at least one labelled support in the frozen candidate pool). Any-support and all-support are conditional on those retrievable queries.

## tatqa (complete; calibration=1,000, test=1,668)

| Family | Method | Chunks | Precision | Recall | Empty | Any support | All support | EM | F1 | Numeric |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CCE | CCE Conformal-Embedding (α=.10) | 4.78 | 25.2% | 91.1% | 1.0% | 93.9% | 88.7% | 0.328 | 0.464 | 0.411 |
| CONFLARE | CONFLARE source-question (α=.10) | 4.41 | 26.1% | 87.2% | 1.7% | 90.6% | 84.1% | 0.321 | 0.452 | 0.401 |
| TRAQ | TRAQ retrieval Bonferroni (α=.10; α_R=.05) | 4.84 | 25.1% | 92.1% | 1.0% | 94.7% | 90.0% | 0.333 | 0.470 | 0.415 |

Retrieval ceiling: 100.0% (1,668/1,668 queries contain at least one labelled support in the frozen candidate pool). Any-support and all-support are conditional on those retrievable queries.

## webqa (complete; calibration=1,000, test=4,966)

| Family | Method | Chunks | Precision | Recall | Empty | Any support | All support | EM | F1 | Numeric |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CCE | CCE Conformal-Embedding (α=.10) | 24.41 | 6.4% | 92.3% | 0.0% | 93.5% | 88.5% | 0.140 | 0.289 | 0.169 |
| CONFLARE | CONFLARE source-question (α=.10) | 22.47 | 6.7% | 88.6% | 0.1% | 90.7% | 83.3% | 0.140 | 0.288 | 0.168 |
| TRAQ | TRAQ retrieval Bonferroni (α=.10; α_R=.05) | 25.83 | 6.2% | 95.1% | 0.0% | 96.0% | 92.8% | 0.140 | 0.289 | 0.169 |

Retrieval ceiling: 99.1% (4,923/4,966 queries contain at least one labelled support in the frozen candidate pool). Any-support and all-support are conditional on those retrievable queries.

## Metric definitions

- **Chunks:** mean selected chunks per query.
- **Precision:** selected labelled-support chunks divided by all selected chunks.
- **Recall:** selected labelled-support chunks divided by labelled supports present in the candidate pool.
- **Empty:** fraction of queries for which the selector keeps no chunk.
- **Any support / All support:** fraction of retrievable queries retaining at least one / every labelled support.
- **EM / F1 / Numeric:** exact match, token F1, and numerical answer accuracy from the shared deterministic downstream generator.

## Selection rules

CCE pools every labelled support chunk from the 1,000 calibration queries, takes the finite-sample 90th percentile of nonconformity `1 − cosine`, and keeps test chunks whose cosine reaches the resulting cutoff.

CONFLARE contributes one calibration value per query: the highest-scoring labelled support. It takes the 90th percentile of cosine distance and keeps test chunks with distance strictly below that cutoff.

TRAQ contributes the best labelled-support cosine from each calibration query, assigns half of α=.10 to retrieval, and keeps test chunks at or above the lower 5% retrieval quantile.
