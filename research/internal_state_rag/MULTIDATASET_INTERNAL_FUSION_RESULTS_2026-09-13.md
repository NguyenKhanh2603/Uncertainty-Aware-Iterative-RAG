# Multi-dataset internal-fusion validation — 2026-09-13

## Executive finding

The original dense-cosine baseline is included below. This changes the practical
interpretation substantially: with the same query-level all-support conformal
rule, cosine can obtain high recall only by retaining a large fraction of Top-30.
At alpha 0.10 it retains 21.46 chunks on TAT-QA, 12.97 on HotpotQA, and 9.16 on
the MMQA pilot. BGE and the selected internal fusion reach similar recall with far
fewer chunks.

This must be kept separate from the proposal's original **cosine + candidate-wise
BY-FDR** result. BY is allowed to return an empty set and, on the original TAT-QA
diagnostic, it leaves 93.3% of queries empty at alpha 0.10. Thus the old
“high-precision, low-recall” failure comes mainly from the BY selection rule;
weak cosine separation then makes the alternative all-support rule inefficient.

The experiments do **not** support a universal hidden-state relevance score. They
support a narrower and more useful result: a supervised internal-state probe can
provide complementary ranking information when the external reranker has
headroom, while a validation-selected residual weight correctly collapses to
zero when the reranker is already near saturation.

- TAT-QA: internal fusion gives a large improvement over BGE.
- HotpotQA: BGE is already very strong; OOF selects zero internal weight.
- MMQA text/table pilot: BGE + hidden probe gives a small, statistically positive
  ranking improvement and a modest conformal efficiency improvement.
- A TAT-QA-trained hidden probe transfers poorly to both new domains. The useful
  signal is therefore not yet a dataset-invariant notion of relevance.

This is a research finding, but “hidden states contain relevance information” is
not a new claim by itself. The strongest potential contribution is an
**uncertainty-gated internal residual reranker with an all-support conformal risk
target**: use the internal score only on queries where the external reranker is
uncertain, and let held-out OOF selection assign zero weight when it adds no value.

## Locked protocol

All methods see the same frozen dense Top-30 candidate reserve. BGE reranks only
within that reserve. Qwen2.5-3B-Instruct is frozen; for each question--chunk pair
we extract the Yes-vs-No LM-head logit and a fixed random projection of hidden
states. Only a class-balanced logistic relevance probe is trained.

Hyperparameters and fusion weights are chosen using five-fold query-grouped OOF
predictions on the probe-training queries. The conformal threshold uses a separate
calibration set and targets the worst support score for query-level all-support
coverage. Test queries are not used for layer, regularization, weight, or threshold
selection.

| Dataset | Probe train | Conformal calibration | Test | Retrievable test | Notes |
|---|---:|---:|---:|---:|---|
| TAT-QA | 452 (351 retrievable) | 452 (363) | 1,000 | 768 | Official train/dev role split |
| HotpotQA | 500 (494) | 500 (497) | 1,000 | 990 | Official distractor/train to official validation |
| MMQA text/table | 477 (473) | 523 (521) | 300 | 295 | Exploratory sample from official dev; no image-support queries |

The MMQA corpus contains 20,227 official text/table documents: the union of the
official candidates attached to 1,000 selected train and 1,000 selected dev
queries. Retrieval is global within this frozen benchmark corpus. This is not a
claim about full multimodal MMQA retrieval or image understanding.

## Test ranking

The primary ranking metric is mean per-query average precision over retrievable
queries.

| Dataset | Method | AP | MRR | Top-1 support | AP change vs BGE |
|---|---|---:|---:|---:|---:|
| TAT-QA | Dense cosine | 0.5285 | 0.5504 | 0.4284 | -0.2161 |
| TAT-QA | BGE | 0.7445 | 0.7677 | 0.6641 | — |
| TAT-QA | BGE + LM-head + hidden | **0.7983** | **0.8171** | **0.7214** | **+0.0538** |
| HotpotQA | Dense cosine | 0.7919 | 0.9057 | 0.8556 | -0.1603 |
| HotpotQA | BGE | **0.9522** | **0.9814** | **0.9657** | — |
| HotpotQA | Hidden probe alone | 0.8820 | 0.9294 | 0.8838 | -0.0702 |
| HotpotQA | OOF-selected fusion | **0.9522** | **0.9814** | **0.9657** | 0.0000 |
| HotpotQA | TAT-QA probe, zero-shot | 0.5669 | 0.6931 | 0.5778 | -0.3853 |
| MMQA text/table | Dense cosine | 0.7823 | 0.8303 | 0.7458 | -0.1273 |
| MMQA text/table | BGE | 0.9096 | 0.9253 | 0.8678 | — |
| MMQA text/table | LM-head alone | 0.8045 | 0.8545 | 0.7729 | -0.1051 |
| MMQA text/table | Hidden probe alone | 0.9028 | **0.9339** | **0.8949** | -0.0068 |
| MMQA text/table | BGE + LM-head | 0.9239 | 0.9392 | 0.8915 | +0.0143 |
| MMQA text/table | BGE + hidden | **0.9250** | 0.9375 | 0.8847 | **+0.0154** |
| MMQA text/table | BGE + LM-head + hidden | 0.9249 | 0.9392 | 0.8881 | +0.0153 |
| MMQA text/table | TAT-QA probe, zero-shot | 0.7292 | 0.7996 | 0.7153 | -0.1804 |

For the MMQA pilot, the paired query bootstrap AP change is:

- BGE + LM-head: +0.0143, 95% CI [+0.0035, +0.0266].
- BGE + hidden: +0.0154, 95% CI [+0.0013, +0.0307].
- BGE + LM-head + hidden: +0.0153, 95% CI [+0.0023, +0.0289].

The selected Hotpot weights are LM-head 0 and hidden 0. The selected MMQA weights
are 0.10 for LM-head in BGE+LM, 0.35 for hidden in BGE+hidden, and 0.10/0.20 in
the three-signal fusion. MMQA selected layer 24 with C=0.03 from the bounded
precommitted grid `{layer 24, layer 30} x {0.03, 0.10}`. Hotpot selected layer 24
with C=0.03 from the full six-layer grid. TAT-QA previously selected layer 30.

## Conformal pruning at alpha = 0.10

All values below are conditional on support being present in Top-30. Precision is
support chunks divided by retained chunks. Recall is micro support recall.

| Dataset | Method | Chunks kept | Precision | Recall | All-support coverage |
|---|---|---:|---:|---:|---:|
| TAT-QA | Dense cosine | 21.462 | 4.86% | **92.71%** | **92.19%** |
| TAT-QA | BGE | 5.391 | 18.33% | 87.85% | 86.72% |
| TAT-QA | BGE + LM-head + hidden | **3.233** | **30.85%** | 88.66% | 88.41% |
| HotpotQA | Dense cosine | 12.970 | 12.64% | **92.90%** | **87.68%** |
| HotpotQA | BGE | **2.257** | **72.38%** | 92.56% | 87.47% |
| HotpotQA | Hidden probe alone | 3.543 | 46.89% | **94.16%** | **89.80%** |
| HotpotQA | OOF-selected fusion | **2.257** | **72.38%** | 92.56% | 87.47% |
| MMQA text/table | Dense cosine | 9.159 | 12.55% | 88.98% | 87.46% |
| MMQA text/table | BGE | 1.841 | 64.46% | **91.86%** | **90.17%** |
| MMQA text/table | Hidden probe alone | 2.451 | 46.61% | 88.45% | 86.44% |
| MMQA text/table | BGE + hidden | **1.763** | **67.31%** | **91.86%** | **90.17%** |
| MMQA text/table | BGE + LM-head + hidden | 1.766 | 66.99% | 91.60% | 89.83% |

On MMQA, BGE + hidden reduces mean set size by 0.078 chunks/query, paired bootstrap
95% CI [-0.119, -0.041], while the changes in mean support recall and all-support
coverage are exactly 0 in the pilot point estimate. This is a real efficiency
gain, but it is only about 4.2% relative and should not be described as a large
improvement.

## Full alpha sweep

| Dataset | alpha | Method | Chunks | Precision | Recall | All-support |
|---|---:|---|---:|---:|---:|---:|
| TAT-QA | .20 | Dense cosine | 16.422 | 5.80% | 84.72% | 83.33% |
| TAT-QA | .20 | BGE | 3.236 | 28.05% | 80.67% | 80.08% |
| TAT-QA | .20 | Fusion | 1.936 | 45.46% | 78.24% | 77.73% |
| TAT-QA | .10 | Dense cosine | 21.462 | 4.86% | 92.71% | 92.19% |
| TAT-QA | .10 | BGE | 5.391 | 18.33% | 87.85% | 86.72% |
| TAT-QA | .10 | Fusion | 3.233 | 30.85% | 88.66% | 88.41% |
| TAT-QA | .05 | Dense cosine | 24.484 | 4.45% | 96.76% | 96.48% |
| TAT-QA | .05 | BGE | 15.990 | 6.64% | 94.33% | 93.75% |
| TAT-QA | .05 | Fusion | 4.974 | 20.89% | 92.36% | 92.06% |
| HotpotQA | .20 | Dense cosine | 6.837 | 22.25% | 86.20% | 76.46% |
| HotpotQA | .20 | BGE / selected fusion | 1.793 | 85.86% | 87.24% | 78.38% |
| HotpotQA | .10 | Dense cosine | 12.970 | 12.64% | 92.90% | 87.68% |
| HotpotQA | .10 | BGE / selected fusion | 2.257 | 72.38% | 92.56% | 87.47% |
| HotpotQA | .05 | Dense cosine | 19.007 | 8.98% | 96.74% | 94.34% |
| HotpotQA | .05 | BGE / selected fusion | 2.933 | 57.92% | 96.28% | 93.54% |
| MMQA text/table | .20 | Dense cosine | 4.078 | 24.94% | 78.74% | 75.93% |
| MMQA text/table | .20 | BGE | 1.529 | 72.95% | 86.35% | 84.07% |
| MMQA text/table | .20 | BGE + hidden | 1.403 | 76.81% | 83.46% | 80.68% |
| MMQA text/table | .10 | Dense cosine | 9.159 | 12.55% | 88.98% | 87.46% |
| MMQA text/table | .10 | BGE | 1.841 | 64.46% | 91.86% | 90.17% |
| MMQA text/table | .10 | BGE + hidden | 1.763 | 67.31% | 91.86% | 90.17% |
| MMQA text/table | .05 | Dense cosine | 15.651 | 7.88% | 95.54% | 94.58% |
| MMQA text/table | .05 | BGE | 2.153 | 56.54% | 94.23% | 92.88% |
| MMQA text/table | .05 | BGE + hidden | 2.075 | 59.15% | 95.01% | 93.90% |

At alpha .20, MMQA fusion prunes more aggressively but loses all-support coverage;
at alpha .10 it lands on the useful matched-recall point; at alpha .05 it keeps
fewer chunks and has a slightly higher point-estimate recall. The result is not a
uniform Pareto improvement at every alpha.

## Original proposal baseline: cosine + BY-FDR

These are the original TAT-QA diagnostic numbers on all 1,000 test queries. They
are shown because this is the proposal's baseline, but they are **not an
apples-to-apples comparison** with the table above: the unit of error control and
fallback behavior differ, and the old sweep reused development-smoke p-values.

| alpha | BY calibration | Chunks | Precision | Micro recall | Empty queries |
|---:|---|---:|---:|---:|---:|
| .20 | pooled | 0.33 | 25.3% | 9.6% | 89.5% |
| .20 | modality-aware | 0.28 | 29.0% | 9.4% | 90.5% |
| .10 | pooled | 0.15 | 38.5% | 6.6% | 93.3% |
| .10 | modality-aware | 0.12 | **52.2%** | 6.9% | 93.3% |
| .05 | pooled | 0.09 | 54.7% | 5.4% | 94.5% |
| .05 | modality-aware | 0.06 | 61.0% | 4.2% | 96.0% |

At alpha .10, the selection-rule contrast is clear. Raw cosine with query-level
all-support conformal retains 21.46 chunks and reaches
92.71% recall; cosine + modality-aware BY retains 0.12 chunks and reaches only
6.9% recall. The score is the same family, but the statistical target is not.
The fair signal comparison is therefore the matched query-level table; the BY
table diagnoses why the original system “leaves everything.”

## What is new and what is established

Intermediate hidden-state probing for adaptive retrieval is established. For
example, [Probing-RAG](https://arxiv.org/abs/2410.13339) trains probes on internal
representations to decide whether more retrieval is needed. LLM relevance and
passage reranking are also established, including
[PaRaDe](https://aclanthology.org/2023.findings-emnlp.950/) and later
[batched self-consistency relevance ranking](https://aclanthology.org/2025.emnlp-main.1661/).
Recent work also studies how retrieved context changes internal representations
([ACL 2026 Findings](https://aclanthology.org/2026.findings-acl.758/)) and treats
relevance as dependent on the surrounding candidate set and order
([ACL 2026](https://aclanthology.org/2026.acl-long.94/)).

Consequently, the following claims are not defensible as novelty:

1. Hidden states contain relevance information.
2. A linear probe can classify relevant passages.
3. LLM logits can be combined with an external reranker.

The evidence here motivates a more specific contribution:

1. Predict a **residual correction** to an external reranker from internal states,
   rather than replacing the reranker.
2. Gate that correction using query-level reranker uncertainty or disagreement.
   HotpotQA shows why the gate must be able to turn the internal path off.
3. Calibrate the final set for a query-level **all-support** event, then report the
   efficiency frontier at fixed alpha.
4. Evaluate leave-one-dataset-out transfer. The current zero-shot failure is a
   negative result and defines the next technical problem: learning a
   domain-invariant residual rather than a dataset-specific relevance classifier.

This combination is a potentially new method framing. A broader literature review
and direct baseline implementation are still needed before calling it a novel
algorithm in a paper.

## Practical conclusion for the mentor discussion

Raising Top-L only changes the retrieval ceiling; it does not solve poor separation
inside the reserve. Internal signals can improve that separation, but they do not
replace a good embedder or reranker. The best current system is therefore:

1. retrieve a sufficiently high-recall Top-L reserve;
2. rerank with BGE;
3. apply an OOF-selected hidden residual only when validation supports it;
4. calibrate an all-support conformal threshold on a separate bank;
5. keep the BGE-only path when the selected internal weight is zero.

The robust multi-dataset claim is conditional complementarity, not universal
internal relevance. TAT-QA shows a large benefit, MMQA shows a small reproducible
benefit, and HotpotQA is the necessary null result.

## Artifacts

- `results/scaled_hidden_probe_train452_cal452_test1000.json`
- `results/hotpotqa_internal_fusion_train500_cal500_test1000.json`
- `results/mmqa_text_table_internal_fusion_train477_cal523_test300.json`
- `results/multidataset_cosine_baselines.json`
- `results/mmqa_text_table_bundle_manifest.json`
- `results/mmqa_text_table_top30_global_manifest.json`
