# MMQA conformal extensions: modality calibration and adaptive internal state

Date: 2026-09-14

## Question

The main MMQA run found a strong modality imbalance at alpha = 0.1: Jina-m0 retrieved 98.65% of image support in a fixed Top-10, but the single conformal threshold retained only 75.68% of image support. This report tests whether modality-aware calibration or query-level Qwen2-VL uncertainty can repair that loss without simply retaining many more chunks.

All methods use the same 504-query probe-training role, disjoint 496-query conformal-calibration role, and 1,000-query official test role. Metrics below are conditional on the 948 test queries with support in Jina-v4 Top-30. The Qwen probe remains decoder layer 19 with logistic `C=0.1`.

## Modality-conditional conformal

Three rules were tested:

- Same-alpha Mondrian applies alpha = 0.1 separately to image, table, and text. This is only a marginal diagnostic; it does not provide 90% simultaneous query-level coverage.
- Equal Bonferroni divides the total failure budget equally across three modalities.
- Probe-allocated Bonferroni searches allocations whose sum is 0.1 using five-fold OOF probe predictions, then freezes the allocation before final calibration. Both Jina methods selected image/table/text alpha = `0.030/0.005/0.065`.

### Alpha = 0.1 results

| Score and calibration rule | Chunks | Precision | Micro recall | Query any | Query all | Image recall | Table recall | Text recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Jina-m0, global | 2.93 | 40.13% | 93.85% | 98.42% | 92.62% | 75.68% | 99.55% | 90.71% |
| Jina-m0, same alpha per modality | 3.60 | 32.22% | 92.59% | 96.62% | 90.93% | 87.16% | 93.61% | 92.90% |
| Jina-m0, equal Bonferroni | 7.71 | 16.08% | 98.99% | 99.58% | 98.73% | 100.00% | 98.96% | 98.63% |
| Jina-m0, probe-allocated Bonferroni | 5.48 | 22.44% | 98.23% | 99.79% | 97.78% | 100.00% | 100.00% | 94.26% |
| Jina-m0 + internal, global | 2.94 | 40.07% | 94.02% | 98.63% | 92.83% | 76.35% | 99.70% | 90.71% |
| Jina-m0 + internal, same alpha per modality | 3.58 | 32.48% | 92.84% | 96.94% | 91.24% | 87.84% | 93.91% | 92.90% |
| Jina-m0 + internal, equal Bonferroni | 7.70 | 16.10% | 98.99% | 99.58% | 98.73% | 100.00% | 98.96% | 98.63% |
| Jina-m0 + internal, probe-allocated Bonferroni | **5.40** | **22.80%** | **98.23%** | **99.79%** | **97.78%** | **100.00%** | **100.00%** | **94.26%** |

Probe allocation is substantially more efficient than equal allocation. Internal fusion saves 0.09 chunks per query relative to Jina alone under the selected modality rule, with identical support metrics.

The large recall increase over global alpha = 0.1 is not a free gain: Jina+internal retains 2.46 additional chunks per query. A global alpha = 0.03 comparison retains 5.62 chunks, reaches 97.89% micro recall and 97.36% query-all coverage, and retains 91.89% of image support. The modality rule at a slightly smaller 5.40 chunks reaches 98.23%, 97.78%, and 100%, respectively. This is a favorable observed operating point, but its paired recall interval includes zero; it is evidence for further testing rather than a confirmed broad improvement.

## Modality-specific score maps

Platt scaling, isotonic regression, and a negative-candidate empirical CDF were fitted separately for each modality on the probe role. A single query-level conformal threshold was still fitted on the disjoint calibration role.

| Score map | Chunks | Precision | Micro recall | Query any | Query all | Image recall |
|---|---:|---:|---:|---:|---:|---:|
| Jina-m0 raw | 2.93 | 40.13% | 93.85% | 98.42% | 92.62% | 75.68% |
| Modality Platt | 2.96 | 39.67% | 93.85% | 98.84% | 92.51% | 79.05% |
| Modality isotonic | 3.23 | 36.74% | 94.69% | 99.05% | 93.57% | 79.73% |
| Modality negative CDF | 2.98 | 39.46% | 93.93% | 98.63% | 92.62% | 74.32% |
| Jina-m0 + internal raw | 2.94 | 40.07% | 94.02% | 98.63% | 92.83% | 76.35% |
| Internal + modality isotonic | 2.89 | 40.75% | 93.93% | 98.84% | 92.62% | 79.73% |

Jina isotonic raises recall reliably relative to raw alpha = 0.1, but global alpha = 0.09 reaches the same 94.69% micro recall and 93.57% all-support coverage with only 3.13 chunks. Score rescaling therefore does not beat ordinary threshold relaxation. It also does not improve the internal fusion.

## Adaptive query difficulty from internal state

The adaptive experiment predicts each query's worst support score from candidate-score distributions. Subtracting that predicted difficulty from every candidate score changes the number of chunks retained per query without changing the within-query Jina ranking. Regressors are selected with five-fold OOF probe predictions; final conformal calibration remains disjoint.

| Base score | Query features | Selected regressor | Chunks | Precision | Micro recall | Query any | Query all |
|---|---|---|---:|---:|---:|---:|---:|
| Jina-m0 | none | none | 2.93 | 40.13% | 93.85% | 98.42% | 92.62% |
| Jina-m0 | external summaries | Ridge | 3.21 | 36.35% | 93.26% | 99.05% | 91.77% |
| Jina-m0 | external + Qwen hidden/LM | HistGradientBoosting | 3.19 | 36.54% | 93.18% | 99.37% | 91.56% |
| Jina-m0 + internal | none | none | 2.94 | 40.07% | 94.02% | 98.63% | 92.83% |
| Jina-m0 + internal | external summaries | Ridge | 3.29 | 35.48% | 93.09% | 99.05% | 91.56% |
| Jina-m0 + internal | external + Qwen hidden/LM | HistGradientBoosting | 3.11 | 37.49% | 93.01% | 99.26% | 91.35% |

The worst-support predictor has OOF correlation around 0.72–0.74, but that predictive fit does not improve test pruning. It increases any-support coverage while losing complete support sets on multi-support queries. Aggregate hidden/LM distributions are therefore not a reliable query-level stopping signal in this experiment.

## Research conclusion

The experiments separate three claims that should not be conflated:

1. The MMQA image problem is real and is partly a cross-modality score/calibration problem. It is not caused mainly by the Top-30 retrieval ceiling.
2. Modality-aware Bonferroni calibration can recover image support and reach a useful high-recall operating point, but most of its gain comes from retaining more chunks. Its advantage over a similarly loose global threshold is small on this test set.
3. The Qwen layer-19 probe contains chunk-level relevance information, but simple linear fusion, modality rescaling, and aggregate query-difficulty regression do not turn it into a large independent gain over Jina-m0.

The next internal-state experiment should change the supervision rather than add another score-combination rule. Train a small probe to predict a chunk's **causal answer value**: the change in gold-answer log likelihood when that chunk is masked, or direct-logit attribution from chunk-token states/attention-head outputs to answer tokens. Use attention only to nominate candidate heads, then validate each head by individual causal masking. Freeze this scorer before conformal calibration. This tests whether internal state identifies evidence that changes generation, rather than merely reproducing support annotations or reranker relevance.

Raw artifacts:

- `mmqa_modality_conformal_report.json` and `mmqa_modality_conformal_predictions.npz`
- `mmqa_modality_score_calibration_report.json` and `mmqa_modality_score_calibration_predictions.npz`
- `mmqa_adaptive_query_conformal_report.json` and `mmqa_adaptive_query_conformal_predictions.npz`

