# Causal internal evidence and Probing-RAG gate report

Date: 2026-09-14  
Branch: `results/qwen2vl-jina4-four-datasets-2026-09-14`

## Scope and protocol

This report follows the three internal-state ideas requested before the
Probing-RAG experiment:

1. answer-conditioned leave-one-chunk-out (LOO) change in gold-answer
   log-probability;
2. hidden-state change after removing a candidate (residual L2 and cosine
   distance);
3. attention-based head screening followed by direct causal head masking.

The model is `Qwen/Qwen2-VL-2B-Instruct`, loaded locally in BF16 on the A100.
Retrieval is the frozen full-corpus `jinaai/jina-embeddings-v4` Top-30 pool and
the external comparison score is `jinaai/jina-reranker-m0`. BF16 is used because
FP16 eager attention produced overflow on long multimodal prompts. The existing
unit suite still passes (`22 passed`).

The causal pilot uses six discovery and six evaluation queries for TAT-QA,
HotpotQA, and MMQA. WebQA uses four plus four queries; its multimodal LOO and
hidden-state scores are computed for only the first eight candidates per query
to keep the pilot tractable. All causal ranking values below are mean query AP.
The gold-answer LOO and hidden-state traces are oracle diagnostics: they use the
first gold answer to define the teacher-forced answer trajectory and therefore
are not directly deployable without replacing the answer with a generated
draft.

## Causal ranking pilot

| Dataset (evaluation queries) | Cosine | Jina m0 | Attention fraction | Gold LOO log-prob drop | Hidden delta L2 | Hidden cosine distance |
|---|---:|---:|---:|---:|---:|---:|
| TAT-QA (6) | 0.5917 | 0.7292 | 0.2767 | **0.8417** | 0.7002 | 0.6078 |
| HotpotQA (6) | 0.8433 | **0.9222** | 0.4737 | 0.4767 | 0.6966 | 0.5611 |
| MMQA (6) | 0.8525 | **0.9028** | 0.5590 | 0.7751 | 0.7671 | 0.7659 |
| WebQA (4; 8 scored candidates) | 0.2000 | **0.6458** | 0.1190 | 0.3750 | 0.6000 | 0.3750 |

The LOO value is promising on TAT-QA and MMQA, and the hidden-state L2 value
beats cosine on TAT-QA, HotpotQA, and WebQA. Neither signal beats the frozen
Jina reranker consistently. The attention fraction is below cosine on every
evaluation split, so a large attention mass is not evidence that a chunk is a
good pruning candidate.

## Causal head masking

The screen ranks heads by support-vs-distractor attention on discovery queries,
then masks the selected heads on evaluation queries. The value is the change
in mean gold-answer log-probability, `masked - full`; more negative means the
masked heads were more necessary for the measured trajectory.

| Dataset | Selected heads | Random controls | Bottom-score controls |
|---|---:|---:|---:|
| TAT-QA (6 eval.) | -0.0304 | +0.0365 | -0.0692 |
| HotpotQA (6 eval.) | -0.0251 | -0.0586 | +0.0290 |
| MMQA (6 eval.) | -0.1758 | -0.1796 | -0.0124 |
| WebQA (4 eval.) | -0.0449 | -0.0065 | -0.1452 |

The selected heads do not reliably beat both controls. In particular, bottom
attention-score heads are more causal than the selected heads on TAT-QA and
WebQA. This is a useful negative result: attention is suitable for generating
a head shortlist, but the final causal test must be direct masking or direct
logit attribution, and the current shortlist is not a stable cross-dataset
signal.

## Probing-RAG query gate with conformal pruning

The gate follows the Probing-RAG idea of using intermediate hidden trajectories
to decide whether the current retrieval state is risky. Qwen first generates a
short draft from the full Top-30 context. The state vector pools selected-layer
target log-probability, margin, entropy, token-agreement, residual statistics,
attention entropy, and a fixed 32-dimensional random projection of normalized
residual means. Gold labels are used only as audit targets after state
extraction. A logistic gate is trained on a disjoint 64-query (32 for WebQA)
probe slice to predict `no_support_top1`, meaning that the first reranked
candidate contains no labelled support. The calibration slice sets the normal
Jina-reranker all-support conformal threshold and separate easy/hard Mondrian
thresholds. The test slice is untouched until evaluation.

`Cond.` columns are calculated only on test queries whose Top-30 pool contains a
support chunk. `E2E all-support` includes every test query and therefore also
shows the retrieval ceiling. Numbers are for alpha = 0.1.

| Dataset (test slice) | Gate AUC / AP | Method | Cond. chunks kept | Cond. precision | Cond. support recall | E2E all-support |
|---|---:|---|---:|---:|---:|---:|
| TAT-QA (64) | 0.539 / 0.573 | Jina baseline | 4.27 | 25.00% | 91.43% | 85.94% |
|  |  | Probing Mondrian | 4.45 | 23.97% | 91.43% | 85.94% |
|  |  | Hard-query Top-10 rescue* | 7.52 | 14.86% | 95.71% | 90.63% |
| HotpotQA (64) | 0.523 / 0.205 | Jina baseline | 2.84 | 61.54% | 93.33% | 87.50% |
|  |  | Probing Mondrian | 2.81 | 61.67% | 92.50% | 85.94% |
|  |  | Hard-query Top-10 rescue* | 3.88 | 46.37% | 95.83% | 92.19% |
| MMQA (64) | 0.677 / 0.474 | Jina baseline | 2.73 | 43.29% | 97.26% | 90.63% |
|  |  | Probing Mondrian | 2.95 | 40.11% | 97.26% | 90.63% |
|  |  | Hard-query Top-10 rescue* | 4.93 | 24.66% | 100.00% | 93.75% |
| WebQA (32) | 0.724 / 0.928 | Jina baseline | 9.76 | 11.48% | 93.33% | 71.88% |
|  |  | Probing Mondrian | 10.48 | 10.69% | 93.33% | 71.88% |
|  |  | Hard-query Top-10 rescue* | 10.56 | 10.61% | 93.33% | 71.88% |

\*The rescue row is a heuristic union of the conformal output and the Top-10
reranker candidates for gate-hard queries. It has no conformal guarantee and
is shown to quantify the explicit recall-versus-context trade-off.

The gate is most useful as a WebQA risk detector (test AP 0.928) and has a
modest MMQA signal. TAT-QA and HotpotQA are close to chance on this small test
slice. The gate-conditional conformal thresholds do not improve recall over the
ordinary Jina threshold in this pilot. Rescue increases support recall on
TAT-QA, HotpotQA, and MMQA only by retaining substantially more candidates;
WebQA is already limited by its retrieval ceiling (25/32 queries retrievable
in this slice).

The `no_support_top3` target was not used for the main table because the
HotpotQA probe slice has zero positive examples at Top-3. A larger probe slice
is required before claiming a stable risk gate at that operating point.

## Conclusion and next experiment

The current evidence does not support replacing Jina-m0 with attention or a
generic internal-state fusion score. The strongest research direction is a
deployable version of the causal LOO signal: generate a draft answer without
gold information, measure per-candidate leave-one-out change in the draft's
token log-probability (or direct-logit attribution), and then apply query-level
conformal calibration to that score. A practical next run should use 1,000
development queries for probe/calibration, score Top-8 or Top-16 candidates in
batches, and reserve the full official test role for the final audit. Head
attention can remain a cheap shortlist, while direct causal masking/DLA is used
only to validate the shortlist. This separates a new internal-evidence signal
from the current external reranker instead of assuming that high attention
alone means the model has enough evidence.

## Artifacts

- `causal_pilot/{tatqa,hotpotqa,mmqa}_bf16_dev12.json`
- `causal_pilot/webqa_bf16_dev8.json`
- `probing_gate/{tatqa,hotpotqa,mmqa}_gate_conformal.json`
- `probing_gate/webqa_gate_conformal.json`
- `run_qwen2vl_causal_evidence_pilot.py`
- `run_qwen2vl_probing_gate_states.py`
- `analyze_probing_gate_conformal.py`

