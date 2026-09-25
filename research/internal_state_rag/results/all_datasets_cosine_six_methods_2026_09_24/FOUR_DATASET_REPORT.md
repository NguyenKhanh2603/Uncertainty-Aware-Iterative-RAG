# Four-dataset pure-cosine conformal comparison

## Unified protocol

Rows labelled **Query-level cosine** are pure query-level cosine conformal selection: per-query z-scored Top-30 cosine scores, calibrated on a disjoint 100-query split at α=0.10, with a deterministic Top-1 fallback only when no candidate passes. Rows labelled **Query-level cosine + internal fusion** use the separately reported Qwen LM-head and hidden-probe signals; their score calibration remains pooled within each dataset. All other selector rows are cosine-only.

The HotpotQA query-level result was rerun on 25 September 2026 over all 1,000 test queries to make this definition match MMQA, TAT-QA, and WebQA. Its remaining selector rows are reused from the prior HotpotQA run only after asserting that the calibration thresholds and score-bank sizes exactly match the pure-cosine rerun.

All selectors use the same frozen Top-30 cosine candidates within each dataset. CCE, CONFLARE, TRAQ, BY, and BH are cosine-only. BY accommodates arbitrary p-value dependence; BH ordinarily needs independence or suitable positive dependence. The rank cap after BH is experimental; this report makes no capped-procedure FDR guarantee.

## Calibration and test qid files

| Dataset | Calibration qids | Test qids |
|---|---|---|
| HotpotQA | [`calibration_manifest.json`](splits/hotpotqa/calibration_manifest.json) — 100 | [`test_manifest.json`](splits/hotpotqa/test_manifest.json) — 1,000 |
| MMQA | [`calibration_manifest.json`](splits/mmqa/calibration_manifest.json) — 100 | [`test_manifest.json`](splits/mmqa/test_manifest.json) — 1,000 |
| TAT-QA | [`calibration_manifest.json`](splits/tatqa/calibration_manifest.json) — 100 | [`test_manifest.json`](splits/tatqa/test_manifest.json) — 1,000 |
| WebQA | [`calibration_manifest.json`](splits/webqa/calibration_manifest.json) — 100 | [`test_manifest.json`](splits/webqa/test_manifest.json) — 250 |

Each file is the ordered qid list actually used; calibration and test are disjoint for every dataset.

## Run inventory

| Dataset | Calibration qids | Test qids | Pure query-level downstream output | Pooled internal-fusion artifacts |
|---|---:|---:|---|---|
| HotpotQA | [100](splits/hotpotqa/calibration_manifest.json) | [1,000](splits/hotpotqa/test_manifest.json) | [`hotpotqa_pure_query_level_cosine_1000_2026_09_25`](../hotpotqa_pure_query_level_cosine_1000_2026_09_25/REPORT.md) | [`analysis`](../qwen2vl_7b_jina4/full_test_internal_fusion_2026_09_25_v2/analysis/hotpotqa/fusion_analysis.json), [`QA`](../qwen2vl_7b_jina4/full_test_internal_fusion_2026_09_25_v2/downstream/hotpotqa/summary.json) |
| MMQA | [100](splits/mmqa/calibration_manifest.json) | [1,000](splits/mmqa/test_manifest.json) | [`mmqa_downstream_predictions.jsonl`](mmqa_downstream_predictions.jsonl) | [`analysis`](../qwen2vl_7b_jina4/full_test_internal_fusion_2026_09_25_v2/analysis/mmqa/fusion_analysis.json), [`QA`](../qwen2vl_7b_jina4/full_test_internal_fusion_2026_09_25_v2/downstream/mmqa/summary.json) |
| TAT-QA | [100](splits/tatqa/calibration_manifest.json) | [1,000](splits/tatqa/test_manifest.json) | [`tatqa_downstream_predictions.jsonl`](tatqa_downstream_predictions.jsonl) | [`analysis`](../qwen2vl_7b_jina4/full_test_internal_fusion_2026_09_25_v2/analysis/tatqa/fusion_analysis.json), [`QA`](../qwen2vl_7b_jina4/full_test_internal_fusion_2026_09_25_v2/downstream/tatqa/summary.json) |
| WebQA | [100](splits/webqa/calibration_manifest.json) | [250](splits/webqa/test_manifest.json) | [`webqa_downstream_predictions.jsonl`](webqa_downstream_predictions.jsonl) | [`analysis`](../qwen2vl_7b_jina4/full_test_internal_fusion_2026_09_25_v2/analysis/webqa/fusion_analysis.json), [`QA`](../qwen2vl_7b_jina4/full_test_internal_fusion_2026_09_25_v2/downstream/webqa/summary.json) |

Each linked manifest contains the exact ordered qid list, source manifest, Top-L, and role. The eight materialized manifests were checked for duplicate qids and for calibration/test overlap; every dataset has zero overlap.

## HOTPOTQA (n=1000)

| Method | Chunks | Precision | Support recall | Empty | Any support | All support | EM | F1 | Numeric |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed Top-10 | 10.00 | 16.2% | 93.0% | 0.0% | 98.7% | 88.3% | 0.383 | 0.500 | 0.405 |
| Fixed Top-20 | 20.00 | 8.5% | 97.8% | 0.0% | 99.7% | 96.1% | 0.364 | 0.480 | 0.386 |
| ECIR CCE (α=0.10) | 7.62 | 20.0% | 87.6% | 0.7% | 97.4% | 80.0% | 0.364 | 0.482 | 0.385 |
| CONFLARE (α=0.10) | 7.46 | 20.4% | 87.4% | 0.7% | 97.3% | 79.7% | 0.366 | 0.484 | 0.387 |
| TRAQ retrieval (α=0.10) | 14.18 | 11.6% | 94.5% | 0.0% | 99.0% | 90.8% | 0.374 | 0.495 | 0.397 |
| Query-level cosine (α=0.10) | 9.31 | 17.4% | 93.2% | 0.0% | 98.8% | 88.4% | 0.370 | 0.491 | 0.392 |
| Query-level cosine + internal fusion (α=0.10; pooled) | 3.40 | 48.8% | 95.5% | 0.0% | 99.1% | 92.3% | 0.399 | 0.526 | 0.422 |
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
| Query-level cosine + internal fusion (α=0.10; pooled) | 4.70 | 24.1% | 95.5% | 0.0% | 98.3% | 94.6% | 0.480 | 0.534 | 0.499 |
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
| Query-level cosine + internal fusion (α=0.10; pooled) | 4.06 | 22.4% | 88.7% | 0.0% | 94.1% | 89.1% | 0.231 | 0.345 | 0.311 |
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
| Query-level cosine + internal fusion (α=0.10; pooled) | 7.62 | 11.1% | 92.1% | 0.0% | 96.8% | 93.2% | 0.004 | 0.138 | 0.036 |
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
- Pooled full-test cosine + internal fusion: [`full_test_internal_fusion_2026_09_25_v2`](../qwen2vl_7b_jina4/full_test_internal_fusion_2026_09_25_v2/)
- Pure HotpotQA aggregate: [`../hotpotqa_pure_query_level_cosine_1000_2026_09_25/summary.json`](../hotpotqa_pure_query_level_cosine_1000_2026_09_25/summary.json)
- Runner: [`../../run_all_datasets_cosine_six_methods.py`](../../run_all_datasets_cosine_six_methods.py)

## Code for the matched internal-fusion column

The completed `Query-level cosine + internal fusion` rows use a separate selector, `z(cosine) + w_lm z(LM-head relevance) + w_hidden z(hidden-probe relevance)`. Its layer, logistic-probe regularization, and two fusion weights are selected only on the disjoint probe-training split; its all-support conformal threshold is calibrated only on the disjoint calibration split. The full-test runner verifies every qid and Top-30 chunk order before it generates an answer.

- Frozen full-test plans: [`../../prepare_full_internal_fusion_plans.py`](../../prepare_full_internal_fusion_plans.py)
- Qwen hidden-state and LM-head feature extraction: [`../../run_qwen2vl_pairwise_features.py`](../../run_qwen2vl_pairwise_features.py)
- Disjoint probe fitting, fusion selection, and calibration: [`../../analyze_qwen2vl_jina_ablation.py`](../../analyze_qwen2vl_jina_ablation.py)
- Cosine and internal-fusion selector/downstream runner: [`../../run_all_datasets_cosine_six_methods.py`](../../run_all_datasets_cosine_six_methods.py)
- Resumable four-dataset orchestration: [`../../run_full_query_level_internal_fusion.sh`](../../run_full_query_level_internal_fusion.sh)

## Executed code, method mapping, and configuration

The pure-cosine table above was generated with the first runner below. The pooled internal-fusion row was generated with the remaining files over every frozen test qid. Every row uses Qwen2-VL-7B-Instruct at revision `eed13092ef92e448dd6875b2a00151bd3f7db0ac`, greedy decoding with at most 24 new tokens, Top-L=30, and the A100 GPU. Calibration, probe training, and test qids are disjoint.

| Table row / method | Selection implementation | Executed code | Frozen configuration |
|---|---|---|---|
| Fixed Top-10 / Top-20 | Retrieval-rank context | [`run_all_datasets_cosine_six_methods.py`](../../run_all_datasets_cosine_six_methods.py) | 10 or 20 highest-ranked candidates from the shared Top-30 log. |
| ECIR CCE | Positive-support split-conformal cosine threshold | [`run_all_datasets_cosine_six_methods.py`](../../run_all_datasets_cosine_six_methods.py) | α=0.10; 100 disjoint calibration queries; the report calls this the retrieval component/adapter, not the complete CCE system. |
| CONFLARE | Positive-support percentile cosine adapter | [`run_all_datasets_cosine_six_methods.py`](../../run_all_datasets_cosine_six_methods.py) | α=0.10; same 100-query calibration split and cosine candidates. |
| TRAQ retrieval | Retrieval threshold with α/2 allocation | [`run_all_datasets_cosine_six_methods.py`](../../run_all_datasets_cosine_six_methods.py) | α=0.10; retrieval component/adapter, not TRAQ's complete answer-set procedure. |
| BY cosine | Candidate p-values plus Benjamini--Yekutieli | [`run_all_datasets_cosine_six_methods.py`](../../run_all_datasets_cosine_six_methods.py), [`conformal_selection.py`](../../../../src/uncertainty_rag/core/conformal_selection.py) | Same false-score calibration bank; α∈{0.10, 0.30, 0.50}. |
| BH cosine | Candidate p-values plus BH, then experimental rank cap | [`run_hotpotqa_cosine_six_methods.py`](../../run_hotpotqa_cosine_six_methods.py), [`conformal_selection.py`](../../../../src/uncertainty_rag/core/conformal_selection.py) | α∈{0.10,0.30,0.50,0.90,0.99}; ctx=10 except α=0.99 also ctx=20. |
| Query-level cosine | Query-wise z-scored cosine and an all-support finite-sample threshold | [`run_all_datasets_cosine_six_methods.py`](../../run_all_datasets_cosine_six_methods.py) | α=0.10; 100 calibration queries; deterministic Top-1 fallback if no candidate passes. |
| Query-level cosine + internal fusion *(running)* | `z(cosine)+w_lm z(LM-head)+w_hidden z(hidden probe)` | [`run_qwen2vl_pairwise_features.py`](../../run_qwen2vl_pairwise_features.py), [`analyze_qwen2vl_jina_ablation.py`](../../analyze_qwen2vl_jina_ablation.py), [`run_all_datasets_cosine_six_methods.py`](../../run_all_datasets_cosine_six_methods.py) | 100 disjoint probe-train queries choose layer, L2 regularization, and weights using grouped OOF AP; 100 different calibration queries set α=0.10 all-support threshold; test is 1,000/1,000/1,000/250 queries. |

The exact dataset/log mapping and test-count guard are in [`prepare_full_internal_fusion_plans.py`](../../prepare_full_internal_fusion_plans.py). The runnable configuration, paths, pixel limits, batch sizes, model revision, output directories, and resume conditions are in [`run_full_query_level_internal_fusion.sh`](../../run_full_query_level_internal_fusion.sh). For WebQA, feature extraction uses `max_pixels=262144` to match its precomputed probe/calibration features; downstream QA still uses the common `max_pixels=200704` setting.

### Modality status

The 19–20 September fusion experiments and the matched full-test internal-fusion column use **modality-aware model inputs but pooled score calibration**. Text/table candidates and image candidates are encoded through their respective Qwen2-VL input paths, but the query-wise z-scores, probe, fusion weights, and conformal threshold pool all Top-30 candidates within a dataset. Earlier MMQA-only modality-conditioned pilots are separate ablations: [`analyze_modality_conditional_conformal.py`](../../analyze_modality_conditional_conformal.py) and [`analyze_modality_score_calibration.py`](../../analyze_modality_score_calibration.py). They are not table rows in this report.

### Queued modality-conditioned internal-fusion ablation

The pooled `Query-level cosine + internal fusion` run remains unchanged. A second four-dataset ablation is queued behind it so the two runs never contend for the A100. It reuses the exact same frozen test feature tensors, 100 probe-train qids, 100 calibration qids, Qwen revision, and Top-L=30 candidates.

Its primary selector is `Query-level cosine + internal fusion, Mondrian-Bonferroni (α=0.10)`. The score remains `z(cosine) + w_lm z(LM-head relevance) + w_hidden z(hidden-probe relevance)`: layer, regularization, and weights are inherited from the pooled run's probe-only OOF selection. Five-fold probe OOF scores choose how to allocate the total α=0.10 across the support modalities; the independent calibration role fits one all-support threshold per modality. The allocation sums to α=0.10, so the reported rule has a union-bound joint target. It will use `mask_cosine_internal_mondrian_bonferroni_probe_allocated` and the same downstream QA configuration.

- Modality-conditioned analysis: [`analyze_modality_conditional_conformal.py`](../../analyze_modality_conditional_conformal.py)
- Selector/downstream runner with exact qid and candidate-order checks: [`run_all_datasets_cosine_six_methods.py`](../../run_all_datasets_cosine_six_methods.py)
- Resumable orchestration, including the pooled-run completion gate: [`run_full_query_level_internal_fusion_modality.sh`](../../run_full_query_level_internal_fusion_modality.sh)

The modality set, allocations, and thresholds are determined from probe/calibration data only; the implementation does not derive them from test support labels.
