# Qwen2-VL-7B candidate-level fusion: 100-query evaluation

Date: 2026-09-15

## Main answer

Candidate-level fusion with Qwen2-VL-7B is now evaluated correctly on all four
datasets. It improves the raw cosine selector substantially, but it does not
give a consistent recall improvement over the already strong Jina-m0 reranker.
At alpha = 0.1, the fusion is better than Jina on WebQA, slightly more
efficient on TAT-QA and HotpotQA, and indistinguishable from Jina on MMQA. The
recall changes are small enough that the 100-query test does not establish a
general recall gain.

## Protocol

- Model: local `Qwen/Qwen2-VL-7B-Instruct`, unquantized BF16 on an A100 40 GB.
- Candidate pool: Jina embeddings-v4 Top-30, then the stored Jina-m0 reranker
  scores. The aligned reranker files are
  `*_jina_v4_top30_bge_jina_m0.jsonl.gz` for TAT-QA/HotpotQA and
  `*_jina_v4_top30_jina_m0.jsonl.gz` for MMQA/WebQA.
- Candidate internal features: Qwen Yes-vs-No LM-head score plus a 256-dimensional
  random projection of normalized hidden states from layers
  `{3, 7, 11, 15, 19, 23, 27}`.
- Each dataset uses 100 disjoint probe-training, 100 calibration, and 100 test
  queries (1,200 queries and 36,000 candidate states total).
- Hidden-probe layer/C and fusion weights are selected by query-grouped OOF on
  the probe role. The conformal threshold is fitted only on calibration.
- `Jina + internal` is the primary fusion: standardized Jina score plus the
  OOF-selected LM-head and hidden-probe residual. `Cosine + internal` is shown
  as the comparison to the original embedding selector.
- All pruning rows use query-level all-support conformal calibration at alpha
  = 0.1. Conditional metrics include only test queries with at least one
  support chunk in Top-30; E2E includes every test query.
- WebQA images use `max_pixels=262144` to keep single-candidate prompts within
  the model context limit.

## Pruning at alpha = 0.1

| Dataset | Method | Mean chunks | Chunk precision | Support recall | Conditional all-support | E2E all-support |
|---|---|---:|---:|---:|---:|---:|
| TAT-QA (91% ceiling) | Jina-m0 | 3.36 | 31.05% | **90.48%** | **91.21%** | **83%** |
|  | Jina-m0 + Qwen-7B internal | **3.20** | **32.30%** | 89.52% | 90.11% | 82% |
|  | Cosine | 9.11 | 10.86% | 85.71% | 83.52% | 76% |
|  | Cosine + Qwen-7B internal | 3.49 | 29.87% | 90.48% | 89.01% | 81% |
| HotpotQA (100% ceiling) | Jina-m0 | 3.13 | 55.59% | **94.57%** | **90%** | **90%** |
|  | Jina-m0 + Qwen-7B internal | **2.79** | **61.29%** | 92.93% | 88% | 88% |
|  | Cosine | 9.98 | 17.94% | 97.28% | 95% | 95% |
|  | Cosine + Qwen-7B internal | 4.08 | 44.36% | **98.37%** | **97%** | **97%** |
| MMQA (93% ceiling) | Jina-m0 | 2.85 | 41.51% | 97.35% | 96.77% | 90% |
|  | Jina-m0 + Qwen-7B internal | 2.85 | 41.51% | 97.35% | 96.77% | 90% |
|  | Cosine | 11.95 | 9.81% | 96.46% | 95.70% | 89% |
|  | Cosine + Qwen-7B internal | 4.74 | 25.17% | **98.23%** | **97.85%** | **91%** |
| WebQA (70% ceiling) | Jina-m0 | 8.07 | 13.63% | 90.59% | 88.57% | 62% |
|  | Jina-m0 + Qwen-7B internal | **7.61** | **14.63%** | **91.76%** | **90%** | **63%** |
|  | Cosine | 19.17 | 5.89% | **92.94%** | **91.43%** | **64%** |
|  | Cosine + Qwen-7B internal | 7.97 | 13.44% | 88.24% | 87.14% | 61% |

The fusion residual changes the Jina operating point rather than creating a
new high-recall regime. Relative to Jina at alpha = 0.1, its recall changes
are -0.95 points on TAT-QA, -1.63 on HotpotQA, 0.00 on MMQA, and +1.18 on
WebQA. The corresponding chunk changes are -0.165, -0.340, 0.000, and
-0.457. The WebQA recall gain has a paired bootstrap 95% interval of
[-3.57, +5.71] points, so it is promising but not statistically established.

The comparison with cosine is stronger. On TAT-QA, fusion reduces the set from
9.11 to 3.49 chunks while raising recall from 85.71% to 90.48%. On HotpotQA it
reduces 9.98 to 4.08 chunks and raises recall from 97.28% to 98.37%. On MMQA
it reduces 11.95 to 4.74 chunks and raises recall from 96.46% to 98.23%. WebQA
is the exception: the set shrinks from 19.17 to 7.97 chunks but recall falls
4.71 points. This shows that the internal residual is useful for correcting a
weak embedding score, while a strong reranker leaves less headroom.

## Ranking quality on retrievable test queries

| Dataset | Jina AP | Jina + Qwen-7B AP | AP change | Jina Top-1 support | Fusion Top-1 support |
|---|---:|---:|---:|---:|---:|
| TAT-QA (91) | 0.863 | 0.862 | -0.000 | 81.3% | 81.3% |
| HotpotQA (100) | 0.919 | **0.931** | **+0.011** | 94.0% | 93.0% |
| MMQA (93) | 0.948 | **0.953** | **+0.004** | 93.5% | 94.6% |
| WebQA (70) | 0.660 | **0.696** | **+0.035** | 57.1% | 58.6% |

Ranking AP gains do not automatically translate to the alpha = 0.1 conformal
coverage point because each score has a different calibration tail. This is why
WebQA can improve ranking and pruning together, while HotpotQA improves ranking
but loses some all-support coverage at the chosen threshold.

## Learned fusion parameters

All four datasets select hidden layer 19. The selected regularization C values
are 0.03 (TAT-QA), 0.10 (HotpotQA), 0.003 (MMQA), and 0.03 (WebQA). The
reranker residual weights are:

| Dataset | LM-head weight | Hidden-probe weight |
|---|---:|---:|
| TAT-QA | 0.00 | 0.20 |
| HotpotQA | 0.00 | 0.35 |
| MMQA | 0.025 | 0.20 |
| WebQA | 0.05 | 1.50 |

The nonzero hidden weights confirm that Qwen's intermediate representation has
complementary ranking information. The effect is dataset-specific, however;
the validation does not support a fixed universal weight.

## Conclusion

Fusion is useful, but the accurate claim is narrower than “internal states solve
low recall.” Qwen-7B internal features substantially improve cosine-based
selection and improve WebQA over Jina at this operating point. Against Jina on
the other datasets, they mostly trade a small amount of recall for fewer chunks
and higher precision. MMQA is effectively unchanged.

For the current system, keep Jina-m0 as the default selector and use the
internal residual only when an OOF validation gate supports it. The research
opportunity is to learn a candidate-level residual with a larger calibration
bank or causal chunk-value labels, then conformal-calibrate that residual's
final set; the current 100-query run does not justify a universal fusion rule.

## Artifacts

- Aligned features: `fusion_features_reranker_aligned/{tatqa,hotpotqa,mmqa,webqa}/{probe_train,calibration,test}/features.npz`.
- Analyzer outputs: `{tatqa,hotpotqa,mmqa,webqa}_fusion_ablation_100_aligned.json`.
- Per-test scores/masks: `{tatqa,hotpotqa,mmqa,webqa}_fusion_ablation_100_aligned_predictions.npz`.
- The earlier `fusion_features/` run used a retrieval file without Jina scores
  and is invalid for Jina fusion; it is intentionally excluded from the result
  set.
