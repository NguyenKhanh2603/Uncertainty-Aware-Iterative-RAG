# MMQA: Jina + Qwen2-VL multimodal internal-state conformal ablation

Date: 2026-09-14

## Protocol

- Candidate pool: global-corpus Top-30 from `jinaai/jina-embeddings-v4`, using images directly with a 200,704-pixel cap.
- Probe training: 504 train-side queries; 464 are retrievable.
- Conformal calibration: a disjoint 496 train-side queries; 463 are retrievable.
- Evaluation: 1,000 official dev queries; 948 have at least one support chunk in Top-30, so end-to-end retrieval ceiling is 94.8%.
- Test support rows in Top-30: 148 image, 673 table, and 366 text.
- Internal model: `Qwen/Qwen2-VL-2B-Instruct`. Image candidates are passed as pixels, while table and text candidates are passed as text.
- The probe layer and regularization are selected by five-fold query-grouped out-of-fold mean AP on the probe role. Fusion weights are selected on the same out-of-fold predictions. Test labels are not used for selection.
- Split-conformal thresholds target retention of all support chunks and are calibrated only on retrievable calibration queries.

## Retrieval and reranking quality

Before Qwen probing, the multimodal Jina reranker improves candidate discrimination on every modality:

| Candidate subset | Jina-v4 cosine AUC / AP | Jina-m0 AUC / AP | Cosine support recall@10 | Jina-m0 support recall@10 |
|---|---:|---:|---:|---:|
| All | 0.889 / 0.413 | 0.964 / 0.705 | 93.01% | 99.83% |
| Image | 0.718 / 0.145 | 0.849 / 0.381 | 70.27% | 98.65% |
| Table | 0.957 / 0.848 | 0.995 / 0.976 | 99.41% | 100.00% |
| Text | 0.856 / 0.239 | 0.952 / 0.470 | 90.44% | 100.00% |

The largest gain is on image candidates. This validates the use of a true multimodal reranker instead of converting images to captions for a text-only BGE model.

## Full comparison at alpha = 0.1

All metrics except the last two columns are conditional on the 948 retrievable test queries.

| Method | Mean chunks kept | Chunk precision | Micro support recall | Query any-support | Query all-support | End-to-end any | End-to-end all |
|---|---:|---:|---:|---:|---:|---:|---:|
| Jina-v4 cosine | 8.14 | 14.30% | 92.92% | 98.42% | 91.35% | 93.3% | 86.6% |
| Qwen LM-head only | 16.49 | 7.13% | 93.93% | 95.89% | 92.41% | 90.9% | 87.6% |
| Qwen hidden probe only | 8.71 | 13.34% | 92.84% | 96.52% | 91.24% | 91.5% | 86.5% |
| Qwen internal only | 8.43 | 13.77% | 92.67% | 96.52% | 91.14% | 91.5% | 86.4% |
| Jina-m0 reranker | 2.93 | 40.13% | 93.85% | 98.42% | 92.62% | 93.3% | 87.8% |
| Jina-v4 cosine + internal | 4.84 | 24.16% | 93.43% | 98.73% | 91.98% | 93.6% | 87.2% |
| Jina-m0 reranker + internal | 2.94 | 40.07% | **94.02%** | **98.63%** | **92.83%** | **93.5%** | **88.0%** |

Internal-only does not beat cosine on the aggregate MMQA operating point. It keeps 0.29 more chunks, lowers mean support recall by 1.03 percentage points, and lowers query-any coverage by 1.90 points. The query-any loss is statistically reliable; the mean-recall interval includes zero.

Jina-m0 + internal rescues two test support chunks without losing any support: one image and one table. It adds 15 chunks and removes 6 relative to Jina-m0, a net increase of only 9 chunks across 948 retrievable queries. Mean support recall and both query-level coverage metrics increase by 0.21 points. Their paired bootstrap 95% interval is `[0.00, +0.53]` points; the lower endpoint is exactly zero because many bootstrap samples do not contain either rescued query. This is a real observed rescue, but its magnitude is small and should not be described as a broad gain.

## Support recall by modality at alpha = 0.1

| Method | Image support recall | Table support recall | Text support recall |
|---|---:|---:|---:|
| Jina-v4 cosine | 70.95% | 99.41% | 89.89% |
| Qwen LM-head only | **97.97%** | 95.69% | 89.07% |
| Qwen hidden probe only | 84.46% | 96.43% | 89.62% |
| Qwen internal only | 85.14% | 96.43% | 88.80% |
| Jina-m0 reranker | 75.68% | 99.55% | **90.71%** |
| Jina-v4 cosine + internal | 77.70% | 99.41% | 88.80% |
| Jina-m0 reranker + internal | 76.35% | **99.70%** | **90.71%** |

The global conformal threshold exposes a modality-calibration problem. Jina-m0 ranks image support very well—98.65% image support recall by fixed Top-10—but its single alpha-0.1 threshold retains only 75.68% of image support while retaining 99.55% of table support. Score distributions differ by modality, so one threshold allocates the error budget unevenly.

The LM-head threshold retains nearly every image support but keeps 16.49 chunks per query and has only 7.13% chunk precision. It is therefore not a useful standalone pruner. The hidden/internal probe improves image support recall over cosine, but loses table/text coverage and does not improve the aggregate selector.

The strongest next change supported by these results is modality-conditional conformal calibration: calibrate separate thresholds for image, table, and text, with an explicit allocation of alpha across modalities. This targets the observed failure directly while preserving the Jina-m0 ordering.

That follow-up is now complete. Probe-allocated Bonferroni calibration raises Jina+internal micro recall from 94.02% to 98.23% and query-all coverage from 92.83% to 97.78%, while chunks rise from 2.94 to 5.40. Against a similarly loose global alpha = 0.03 rule (5.62 chunks, 97.89% recall, 97.36% query-all), the observed gain is small and its paired recall interval includes zero. Modality-specific score maps and an internal-state query-difficulty regressor also fail to beat ordinary global threshold relaxation. See `MMQA_CONFORMAL_EXTENSIONS_REPORT_2026-09-14.md` for the full controlled comparison.

## Ranking quality before conformal calibration

Metrics are conditional on retrievable queries.

| Method | Mean query AP | MRR | Top-1 support rate | Top-10 mean support recall |
|---|---:|---:|---:|---:|
| Jina-v4 cosine | 0.824 | 0.875 | 80.59% | 94.83% |
| Qwen LM-head only | 0.622 | 0.662 | 52.00% | 87.80% |
| Qwen hidden probe only | 0.773 | 0.810 | 72.15% | 94.28% |
| Qwen internal only | 0.773 | 0.810 | 71.94% | 94.59% |
| Jina-m0 reranker | 0.927 | 0.952 | 91.77% | **99.92%** |
| Jina-v4 cosine + internal | 0.866 | 0.906 | 85.34% | 98.22% |
| Jina-m0 reranker + internal | **0.930** | **0.956** | **92.51%** | **99.92%** |

Fusion improves Jina-m0 ranking slightly on MMQA. The effect is larger on candidate-level image AP than on the final conformal metrics: image AP is 0.494 for Jina-m0 and 0.496 after internal fusion.

## Alpha sensitivity

Conditional mean chunks / micro support recall:

| Method | alpha 0.2 | alpha 0.1 | alpha 0.05 |
|---|---:|---:|---:|
| Jina-v4 cosine | 2.77 / 82.90% | 8.14 / 92.92% | 15.89 / 96.46% |
| Qwen internal only | 4.26 / 84.75% | 8.43 / 92.67% | 17.82 / 97.14% |
| Jina-m0 reranker | 2.00 / 90.48% | 2.93 / 93.85% | 4.09 / 96.12% |
| Jina-v4 cosine + internal | 1.97 / 83.32% | 4.84 / 93.43% | 7.96 / 96.63% |
| Jina-m0 reranker + internal | 2.00 / 90.65% | 2.94 / 94.02% | 4.06 / 96.12% |

At alpha 0.05, internal-only has slightly higher recall than cosine but keeps 1.93 more chunks; it does not dominate cosine. Jina+internal keeps 0.04 fewer chunks than Jina at identical recall. The useful operating behavior is therefore tied to the external reranker, not an internal-only replacement.

## Internal-state finding

MMQA independently selects decoder layer 19 and logistic-regression `C=0.1`, matching TAT-QA and HotpotQA. Development OOF mean query AP by layer is:

| Layer | 3 | 7 | 11 | 15 | 19 | 23 | 27 |
|---|---:|---:|---:|---:|---:|---:|---:|
| OOF AP | 0.427 | 0.443 | 0.502 | 0.675 | **0.725** | 0.705 | 0.656 |

The repeated rise to layer 19 and decline toward layer 27 now appears on three independently processed datasets, including a multimodal one. This is the strongest internal-state result in the current experiments. It supports a trained intermediate-layer relevance probe, while the final LM-head margin remains a weak standalone pruning score.

Raw artifacts:

- `mmqa_ablation_report.json`: thresholds, metrics, layer/C audit, selected weights, and bootstrap intervals.
- `mmqa_ablation_predictions.npz`: per-query scores, labels, modalities, masks, and chunk IDs.
- `features/mmqa/*/features.npz`: Qwen projected hidden states and external scores for probe, calibration, and test.
- `mmqa_jina_v4_top30_jina_m0.jsonl.gz`: complete multimodal reranking of 60,000 candidates.
