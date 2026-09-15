# Qwen2-VL-7B internal-state gate: 100-query evaluation

Date: 2026-09-15
Branch: `results/qwen2vl-jina4-four-datasets-2026-09-14`

## Result

Scaling the internal-state gate from Qwen2-VL-2B to
`Qwen/Qwen2-VL-7B-Instruct` does not produce a reliable improvement over the
Jina conformal baseline. The learned gate generalizes weakly on all four test
sets. Mondrian calibration gives at most a 0.54 percentage-point support-recall
gain on HotpotQA, no gain on TAT-QA or MMQA, and loses 3.53 points on WebQA.

The Top-10 rescue heuristic raises recall on every dataset, but this is a
context-size tradeoff rather than a better conformal selector. For example,
TAT-QA recall rises from 90.48% to 94.29% while mean retained chunks rise from
3.36 to 5.97 and precision falls from 31.05% to 18.23%.

## Protocol

- Model: `Qwen/Qwen2-VL-7B-Instruct`, unquantized BF16 on one A100 40 GB.
- Retrieval pool: full-corpus `jinaai/jina-embeddings-v4` Top-30 candidates.
- Candidate score: `jinaai/jina-reranker-m0`.
- Data per dataset: 100 disjoint probe-training queries, 100 calibration
  queries, and 100 test queries. Across four datasets, 1,200 query states were
  extracted.
- Probe target: no labelled support in **Jina-reranker Top-1**. This alignment
  matters because the conformal selector operates on the Jina ordering.
- Probe input: selected-layer draft-token probabilities, margins, entropy,
  layer agreement, residual statistics, attention entropy, and projected hidden
  states from a short Qwen draft over the Top-30 context.
- Conformal setting: alpha = 0.1. `Baseline` uses one calibration threshold.
  `Mondrian` uses separate thresholds for queries classified easy and hard.
  `Rescue` unions the conformal output with Jina Top-10 on predicted-hard
  queries and has no standalone conformal guarantee.
- WebQA images were capped at 262,144 pixels each so that all Top-30 prompts fit
  Qwen's 32,768-token context. Prompt lengths were 2,429–9,644 for probe,
  2,202–9,579 for calibration, and 2,131–9,531 for test.

## Gate quality on held-out test queries

The positive rate is included because average precision must be interpreted
against this base rate. Training AP is approximately 1.0 on every dataset,
while test generalization is weak; the high-dimensional probe is overfitting
100 training examples.

| Dataset | Positive rate | ROC AUC | Average precision | Hard-query rate |
|---|---:|---:|---:|---:|
| TAT-QA | 26% | 0.578 | 0.349 | 41% |
| HotpotQA | 6% | 0.546 | 0.073 | 18% |
| MMQA | 13% | 0.679 | 0.252 | 33% |
| WebQA | 60% | 0.528 | 0.620 | 48% |

WebQA AP of 0.620 is only slightly above its 0.600 positive-rate baseline, and
its AUC is 0.528. It therefore does not indicate a useful gate.

## Pruning results at alpha = 0.1

`Recall`, `precision`, and `chunks` below are conditional on queries for which
the Top-30 retrieval pool contains at least one labelled evidence chunk. `E2E all-support`
uses all 100 test queries and exposes failures of Top-30 retrieval itself.

| Dataset | Retrieval ceiling | Method | Mean chunks | Chunk precision | Support recall | Conditional all-support | E2E all-support |
|---|---:|---|---:|---:|---:|---:|---:|
| TAT-QA | 91% | Jina conformal baseline | 3.36 | 31.05% | 90.48% | 91.21% | 83% |
|  |  | Internal-state Mondrian | 3.64 | 28.70% | 90.48% | 91.21% | 83% |
|  |  | Hard-query Top-10 rescue | 5.97 | 18.23% | **94.29%** | **95.60%** | **87%** |
| HotpotQA | 100% | Jina conformal baseline | 3.13 | 55.59% | 94.57% | 90% | 90% |
|  |  | Internal-state Mondrian | 3.43 | 51.02% | 95.11% | 91% | 91% |
|  |  | Hard-query Top-10 rescue | 4.44 | 39.86% | **96.20%** | **93%** | **93%** |
| MMQA | 93% | Jina conformal baseline | 2.85 | 41.51% | 97.35% | 96.77% | 90% |
|  |  | Internal-state Mondrian | 3.03 | 39.01% | 97.35% | 96.77% | 90% |
|  |  | Hard-query Top-10 rescue | 5.05 | 23.62% | **98.23%** | **97.85%** | **91%** |
| WebQA | 70% | Jina conformal baseline | 8.07 | 13.63% | 90.59% | 88.57% | 62% |
|  |  | Internal-state Mondrian | 8.00 | 13.21% | 87.06% | 85.71% | 60% |
|  |  | Hard-query Top-10 rescue | 9.03 | 12.34% | **91.76%** | **90%** | **63%** |

## What the experiment establishes

The 7B hidden trajectory contains some relevance information on MMQA, where
test AUC reaches 0.679, but the current query-level probe cannot turn it into a
materially better conformal pruning rule. The gap between near-perfect training
scores and weak calibration/test scores is evidence of probe overfitting, not
evidence that the model has no useful internal signal.

The main bottleneck differs by dataset. TAT-QA and MMQA lose 9% and 7% of test
queries before pruning because labelled evidence is absent from Top-30. WebQA's
retrieval ceiling is only 70%, so no pruning policy can exceed 70% end-to-end
all-support coverage with this candidate pool. Raising `L` can address that
ceiling, but it cannot repair errors made after a support chunk is retrieved;
it also increases reranking and model-context cost.

The deployable result remains the Jina conformal baseline. The rescue rule is
useful only when recall is worth a substantial precision/context penalty. A
stronger research direction is candidate-level causal probing: predict each
chunk's change in draft-answer log probability or hidden trajectory when that
chunk is masked, then calibrate those candidate scores. That directly matches
the pruning decision, unlike the current query-level easy/hard gate.

## Reproducibility note

The earlier Qwen2-VL-2B Probing-gate table used a target based on embedding
retrieval order while pruning used Jina-reranker order. Corrected 2B analyzer
artifacts use `no_support_in_jina_reranker_top1`; the old gate table is
superseded. The causal LOO and head-masking pilot in the earlier report is a
separate experiment and is unaffected by this correction.
