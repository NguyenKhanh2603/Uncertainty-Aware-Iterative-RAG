# TAT-QA: Jina + Qwen2-VL internal-state conformal ablation

Date: 2026-09-14

## Protocol

- Candidate pool: global-corpus Top-30 from `jinaai/jina-embeddings-v4`, 512 dimensions, visual-only image mode.
- Probe training: 516 train-side queries (494 retrievable).
- Conformal calibration: a disjoint 484 train-side queries (447 retrievable).
- Evaluation: 1,000 official dev/test queries; 919 have at least one support chunk in Top-30, so end-to-end retrieval ceiling is 91.9%.
- Internal model: `Qwen/Qwen2-VL-2B-Instruct`. The probe uses the last-token hidden state from one selected decoder layer; LM-head score is the Yes-versus-No logit margin.
- Probe layer and regularization are selected using five-fold query-grouped out-of-fold mean AP on the probe-training role. Fusion weights are also selected on those out-of-fold predictions. Test labels are never used for selection.
- Split-conformal thresholds target retention of all support chunks and are calibrated only on retrievable calibration queries. Conditional and end-to-end results are reported separately.

## Main result at alpha = 0.1

All metrics except the two end-to-end columns are conditional on the 919 retrievable test queries.

| Method | Mean chunks kept | Chunk precision | Micro support recall | Query any-support | Query all-support | End-to-end any | End-to-end all |
|---|---:|---:|---:|---:|---:|---:|---:|
| Jina-v4 cosine | 10.84 | 9.08% | 88.37% | 90.64% | 87.27% | 83.3% | 80.2% |
| Qwen LM-head only | 18.32 | 5.50% | 90.52% | 92.17% | 89.88% | 84.7% | 82.6% |
| Qwen hidden probe only | 9.08 | 11.04% | 90.03% | 93.04% | 89.23% | 85.5% | 82.0% |
| Qwen internal only: LM-head + hidden | 9.53 | 10.65% | 91.20% | 93.91% | 90.64% | 86.3% | 83.3% |
| BGE-v2-m3 reranker | 9.00 | 11.50% | 92.96% | 96.52% | 92.49% | 88.7% | 85.0% |
| Jina-m0 reranker | 3.92 | 26.24% | 92.38% | 95.10% | 92.27% | 87.4% | 84.8% |
| Jina-v4 cosine + internal | 6.27 | 16.06% | 90.52% | 93.04% | 89.88% | 85.5% | 82.6% |
| Jina-m0 reranker + internal | 3.93 | 26.30% | 92.77% | 95.43% | 92.71% | 87.7% | 85.2% |

The internal-only score dominates Jina-v4 cosine at this operating point: it keeps 1.31 fewer chunks, increases mean support recall by 3.30 percentage points, and increases query-any coverage by 3.26 points. Paired bootstrap 95% intervals are `[-1.51, -1.10]` chunks, `[+1.03, +5.77]` points mean recall, and `[+1.09, +5.66]` points query-any coverage.

Adding internal features to Jina-m0 gives only a small change at alpha 0.1: +0.38 points mean support recall and +0.33 points query-any coverage at essentially the same chunk count. Both recall intervals include zero, so this run does not establish a reliable improvement over Jina-m0 alone.

## Ranking quality before conformal thresholding

Metrics below are conditional on retrievable queries.

| Method | Mean query AP | MRR | Top-1 support rate | Top-5 query coverage | Top-10 mean support recall |
|---|---:|---:|---:|---:|---:|
| Jina-v4 cosine | 0.613 | 0.630 | 49.73% | 79.76% | 86.31% |
| Qwen LM-head only | 0.393 | 0.404 | 24.16% | 57.89% | 73.65% |
| Qwen hidden probe only | 0.599 | 0.615 | 47.33% | 80.20% | 91.73% |
| Qwen internal only | 0.595 | 0.610 | 46.03% | 80.30% | 92.27% |
| BGE-v2-m3 reranker | 0.712 | 0.731 | 61.70% | 87.27% | 93.33% |
| Jina-m0 reranker | **0.856** | **0.871** | **79.98%** | **96.84%** | 98.60% |
| Jina-v4 cosine + internal | 0.700 | 0.716 | 58.11% | 89.12% | 95.05% |
| Jina-m0 reranker + internal | 0.852 | 0.867 | 79.11% | 96.74% | **98.82%** |

## Alpha sensitivity

The compact table shows conditional mean chunks / micro support recall.

| Method | alpha 0.2 | alpha 0.1 | alpha 0.05 |
|---|---:|---:|---:|
| Jina-v4 cosine | 5.15 / 78.59% | 10.84 / 88.37% | 19.53 / 94.62% |
| Qwen internal only | 5.22 / 78.98% | 9.53 / 91.20% | 13.08 / 94.92% |
| BGE-v2-m3 reranker | 4.39 / 83.68% | 9.00 / 92.96% | 22.72 / 97.56% |
| Jina-m0 reranker | 2.03 / 84.07% | 3.92 / 92.38% | 6.68 / 97.36% |
| Jina-v4 cosine + internal | 2.66 / 77.52% | 6.27 / 90.52% | 10.79 / 95.70% |
| Jina-m0 reranker + internal | 1.90 / 82.99% | 3.93 / 92.77% | 6.38 / 97.65% |

At alpha 0.05, Jina-m0 + internal keeps 0.30 fewer chunks than Jina-m0 and raises query-all coverage by 0.54 points. The query-all bootstrap interval is `[+0.11, +1.09]` points, but the mean-recall interval still includes zero. This is a narrow efficiency result rather than evidence of a broad recall gain.

## Internal-state finding

The best probe is decoder layer 19 with logistic-regression `C=0.1`. Development OOF mean query AP rises from about 0.30 at layer 3 to 0.63 at layer 19, then falls to 0.57 at layer 27. This non-monotonic layer profile supports the hypothesis that intermediate hidden states encode chunk relevance more cleanly than the final LM decision alone.

The LM-head margin alone is weak. The trained layer-19 probe is the useful internal component. Internal-only improves over embedding cosine and can rescue Top-10 support recall, but the strong Jina-m0 multimodal reranker remains the best standalone selector on TAT-QA. The current fusion does not yet provide a statistically reliable improvement over that reranker at alpha 0.1.

Raw artifacts:

- `tatqa_ablation_report.json`: all thresholds, metrics, selected weights, layer/C audit, and bootstrap intervals.
- `tatqa_ablation_predictions.npz`: per-query labels, scores, and masks for every method and alpha.
- `features/tatqa/*/features.npz`: projected hidden states, LM-head margins, external scores, labels, modalities, and chunk IDs.
