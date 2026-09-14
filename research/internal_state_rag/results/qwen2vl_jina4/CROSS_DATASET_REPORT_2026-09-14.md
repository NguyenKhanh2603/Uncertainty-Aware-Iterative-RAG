# Qwen2-VL internal-state pruning: TAT-QA and HotpotQA

Date: 2026-09-14

## Question

Can a decoder's internal state identify which retrieved chunks should be retained, and does it improve the high-precision/low-recall behavior of cosine conformal pruning?

This run uses `Qwen/Qwen2-VL-2B-Instruct` as the internal model and `jinaai/jina-embeddings-v4` for global-corpus Top-30 retrieval. It compares embedding cosine, an LM-head score, a trained hidden-state probe, `jinaai/jina-reranker-m0`, `BAAI/bge-reranker-v2-m3`, and score-level fusion. The probe-training, conformal-calibration, and test query roles are disjoint.

## Protocol and retrieval ceiling

| Dataset | Probe-training queries | Calibration queries | Official test queries | Test queries retrievable in Top-30 | End-to-end ceiling |
|---|---:|---:|---:|---:|---:|
| TAT-QA | 516 | 484 | 1,000 | 919 | 91.9% |
| HotpotQA | 509 | 491 | 1,000 | 996 | 99.6% |

The hidden probe is logistic regression over a fixed 256-dimensional projection of one Qwen2-VL last-token decoder state. Decoder layer and regularization are selected by five-fold, query-grouped out-of-fold mean AP on the probe-training role. Fusion weights are selected on the same out-of-fold predictions. Split-conformal thresholds are then fitted on the separate calibration role and evaluated once on test.

All conditional metrics below exclude queries whose support is absent from the Jina-v4 Top-30. The end-to-end columns include all 1,000 test queries and therefore include retrieval failures.

## Full comparison at alpha = 0.1

### TAT-QA

| Method | Chunks kept | Chunk precision | Micro support recall | Query any-support | Query all-support | End-to-end any | End-to-end all |
|---|---:|---:|---:|---:|---:|---:|---:|
| Jina-v4 cosine | 10.84 | 9.08% | 88.37% | 90.64% | 87.27% | 83.3% | 80.2% |
| Qwen LM-head only | 18.32 | 5.50% | 90.52% | 92.17% | 89.88% | 84.7% | 82.6% |
| Qwen hidden probe only | 9.08 | 11.04% | 90.03% | 93.04% | 89.23% | 85.5% | 82.0% |
| Qwen internal only | 9.53 | 10.65% | 91.20% | 93.91% | 90.64% | 86.3% | 83.3% |
| BGE-v2-m3 reranker | 9.00 | 11.50% | 92.96% | 96.52% | 92.49% | 88.7% | 85.0% |
| Jina-m0 reranker | 3.92 | 26.24% | 92.38% | 95.10% | 92.27% | 87.4% | 84.8% |
| Jina-v4 cosine + internal | 6.27 | 16.06% | 90.52% | 93.04% | 89.88% | 85.5% | 82.6% |
| Jina-m0 reranker + internal | 3.93 | 26.30% | 92.77% | 95.43% | 92.71% | 87.7% | 85.2% |

### HotpotQA

| Method | Chunks kept | Chunk precision | Micro support recall | Query any-support | Query all-support | End-to-end any | End-to-end all |
|---|---:|---:|---:|---:|---:|---:|---:|
| Jina-v4 cosine | 9.66 | 17.90% | 94.41% | 99.40% | 89.76% | 99.0% | 89.4% |
| Qwen LM-head only | 20.12 | 8.73% | 95.84% | 98.49% | 92.37% | 98.1% | 92.0% |
| Qwen hidden probe only | 6.54 | 26.49% | 94.52% | 99.40% | 90.06% | 99.0% | 89.7% |
| Qwen internal only | 6.54 | 26.49% | 94.52% | 99.40% | 90.06% | 99.0% | 89.7% |
| BGE-v2-m3 reranker | 2.48 | 68.40% | 92.49% | 99.50% | 86.55% | 99.1% | 86.2% |
| Jina-m0 reranker | 2.91 | 59.24% | 94.19% | 99.90% | 89.46% | 99.5% | 89.1% |
| Jina-v4 cosine + internal | 3.89 | 44.24% | 93.92% | 99.30% | 88.96% | 98.9% | 88.6% |
| Jina-m0 reranker + internal | 2.76 | 62.59% | 94.25% | 99.90% | 89.56% | 99.5% | 89.2% |

## Paired findings at alpha = 0.1

| Dataset and comparison | Change in chunks | 95% CI | Change in mean support recall | 95% CI | Change in query-any | 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| TAT-QA: internal minus cosine | -1.31 | [-1.51, -1.10] | +3.30 pp | [+1.03, +5.77] | +3.26 pp | [+1.09, +5.66] |
| HotpotQA: internal minus cosine | -3.13 | [-3.32, -2.93] | +0.15 pp | [-1.15, +1.51] | 0.00 pp | [-0.70, +0.70] |
| TAT-QA: Jina+internal minus Jina | +0.01 | [-0.03, +0.04] | +0.38 pp | [-0.16, +0.92] | +0.33 pp | [-0.11, +0.87] |
| HotpotQA: Jina+internal minus Jina | -0.15 | [-0.19, -0.12] | +0.05 pp | [-0.30, +0.45] | 0.00 pp | [0.00, 0.00] |

The internal score clearly improves on raw embedding cosine, but in different ways. On TAT-QA it raises recall while keeping fewer chunks. On HotpotQA it preserves essentially the same recall while pruning 32% more aggressively. The paired confidence intervals support TAT-QA's recall gain and the chunk reductions on both datasets.

Adding internal state to a strong Jina reranker does not establish a recall gain at alpha 0.1. On HotpotQA it saves 0.15 chunks per query with statistically stable direction, but the recall change is indistinguishable from zero. On TAT-QA even the chunk-count change is negligible.

## Ranking quality before conformal calibration

Mean query AP / Top-10 mean support recall:

| Method | TAT-QA | HotpotQA |
|---|---:|---:|
| Jina-v4 cosine | 0.613 / 86.31% | 0.819 / 93.88% |
| Qwen LM-head only | 0.393 / 73.65% | 0.546 / 81.78% |
| Qwen hidden probe only | 0.599 / 91.73% | 0.787 / 96.79% |
| Qwen internal only | 0.595 / 92.27% | 0.787 / 96.79% |
| BGE-v2-m3 reranker | 0.712 / 93.33% | **0.943** / 99.20% |
| Jina-m0 reranker | **0.856** / 98.60% | 0.934 / **99.75%** |
| Jina-v4 cosine + internal | 0.700 / 95.05% | 0.882 / 98.59% |
| Jina-m0 reranker + internal | 0.852 / **98.82%** | 0.935 / 99.70% |

The best external reranker is dataset-dependent: Jina-m0 wins clearly on TAT-QA, while BGE has the best mean AP on HotpotQA. A better ranking score does not automatically produce the best conformal operating point because each score has its own calibration distribution. At Hotpot alpha 0.1, BGE keeps the fewest chunks but also has lower all-support recall than Jina-m0.

## Alpha sensitivity

Conditional mean chunks / micro support recall:

| Dataset | Method | alpha 0.2 | alpha 0.1 | alpha 0.05 |
|---|---|---:|---:|---:|
| TAT-QA | Jina-v4 cosine | 5.15 / 78.59% | 10.84 / 88.37% | 19.53 / 94.62% |
| TAT-QA | Qwen internal | 5.22 / 78.98% | 9.53 / 91.20% | 13.08 / 94.92% |
| TAT-QA | Jina-m0 | 2.03 / 84.07% | 3.92 / 92.38% | 6.68 / 97.36% |
| TAT-QA | Jina-m0 + internal | 1.90 / 82.99% | 3.93 / 92.77% | 6.38 / 97.65% |
| HotpotQA | Jina-v4 cosine | 4.71 / 88.44% | 9.66 / 94.41% | 18.74 / 98.08% |
| HotpotQA | Qwen internal | 4.42 / 89.21% | 6.54 / 94.52% | 8.85 / 96.93% |
| HotpotQA | Jina-m0 | 2.22 / 89.04% | 2.91 / 94.19% | 4.05 / 97.32% |
| HotpotQA | Jina-m0 + internal | 2.15 / 88.88% | 2.76 / 94.25% | 3.77 / 97.04% |

Lowering alpha does increase recall, but cosine pays much more in retained chunks. This is the concrete way the internal probe helps the original high-precision/low-recall issue: it gives a better pruning score before conformal calibration, so a looser threshold does not require retaining as much of Top-30.

## Internal-state result

Both datasets independently select decoder layer 19 and logistic-regression `C=0.1`. Best development OOF AP by layer is:

| Layer | 3 | 7 | 11 | 15 | 19 | 23 | 27 |
|---|---:|---:|---:|---:|---:|---:|---:|
| TAT-QA | 0.318 | 0.388 | 0.401 | 0.562 | **0.632** | 0.619 | 0.574 |
| HotpotQA | 0.405 | 0.504 | 0.577 | 0.779 | **0.795** | 0.759 | 0.680 |

This repeated rise-then-fall profile is evidence that intermediate Qwen2-VL hidden states encode chunk relevance more cleanly than early states or the final decision. Attention magnitude alone was not used as a relevance label. The current signal is a trained relevance probe over the hidden representation.

The LM-head margin is consistently weak. On HotpotQA, validation assigns it exactly zero weight in the internal-only score, so the useful signal is entirely the layer-19 hidden probe. On TAT-QA, validation retains a small LM-head contribution, but the hidden probe remains the main component.

## What this establishes

1. Internal model state is useful for pruning. It beats Jina-v4 embedding cosine on both evaluated datasets at alpha 0.1, with a recall gain on TAT-QA and a large efficiency gain on HotpotQA.
2. Internal state does not yet beat a strong cross-encoder reranker. Jina-m0 remains the strongest practical selector on TAT-QA, while Jina-m0 and BGE define different precision/recall trade-offs on HotpotQA.
3. Simple linear score fusion is not a general answer. Its incremental gain over a reranker is small and recall confidence intervals include zero.
4. Raising Top-L can raise the retrieval ceiling only when support is missing from the candidate set. It cannot repair pruning errors within an already retrievable Top-30. TAT-QA has more room for retrieval improvement (8.1% of queries missing support) than HotpotQA (0.4%).
5. The strongest research direction from these results is adaptive use of the internal probe: use the external reranker for ordering, then use internal uncertainty/evidence state to choose a query-specific stopping depth or abstain/escalate to a larger candidate pool. That should be evaluated against fixed score fusion and fixed Top-K with a separately calibrated stopping rule.

## Reproducibility

- GPU: NVIDIA A100-PCIE-40GB.
- Working environment: PyTorch 2.7.1 with CUDA 11.8 runtime (`torch 2.7.1+cu118`), compatible with the installed NVIDIA driver. The environment was not upgraded to CUDA 12.8 or 13.
- Probe/calibration data: 1,000 train-side queries per dataset, split approximately 50/50 into disjoint probe and calibration roles.
- Test data: 1,000 official held-out queries per dataset.
- Raw reports: `tatqa_ablation_report.json`, `hotpotqa_ablation_report.json`.
- Per-query predictions and masks: `tatqa_ablation_predictions.npz`, `hotpotqa_ablation_predictions.npz`.
- Qwen features: `features/{tatqa,hotpotqa}/{probe_train,calibration,test}/features.npz`.

