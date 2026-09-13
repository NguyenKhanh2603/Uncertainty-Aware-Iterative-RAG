# Internal model signals for chunk pruning

## Decision

The experiments now support using a generator-internal signal **with** BGE for
chunk pruning. The best validated score is:

```text
z(BGE) + 0.20 z(zero-shot Yes-vs-No LM-head score)
       + 0.75 z(trained layer-30 relevance probe)
```

The `z` transformations are computed within each query's Top-30. Qwen remains
frozen. The only trained component is an L2 logistic probe over the final token
state of an explicit question--chunk relevance prompt.

On 256 fresh test-role TAT-QA queries, disjoint from all previously used test
queries, this score improved mean query AP from 0.759 to 0.813 and Top-1 support
from 67.2% to 73.8%. Both gains were significant under paired query bootstrap.
At the conformal all-support threshold with `alpha=0.05`, it retained 9.92 of
30 chunks on average versus 26.76 for BGE while achieving 97.3% all-support
coverage versus 99.2% for BGE. A 64-query answer-generation check found no
detectable F1, EM, or numerical-accuracy difference between those two
`alpha=0.05` keep-sets.

The useful target still must be specified carefully:

- answer hidden states and layerwise LM-head trajectories measure whether the
  model appears to have enough evidence;
- chunk attention, gradients, and causal interventions measure which context
  influenced the model's current answer;
- labelled support relevance measures which context should be retained for the
  task.

These targets are correlated but not equivalent. In particular, a faithful
attribution method can identify a distractor that caused a wrong answer while
ranking annotated support poorly. Raw attention, raw LM-head score, and a
passive full-context hidden-state probe all failed as standalone replacements
for BGE. The gain comes from combining BGE with explicit pairwise computation.

## What was tested

All runs used Qwen2.5-3B-Instruct in BF16 on an A100 40 GB. The production RAG
code was not changed. New code is isolated under `research/internal_state_rag`.

### 1. Layerwise contrastive saliency

`contrastive_saliency.py` compares answer-token distributions with and without
context, selects context-sensitive tokens, and differentiates a contrastive
answer objective through:

- the residual stream at 12 layers;
- attention-block outputs at the same layers;
- MLP-block outputs at the same layers.

The LLM is frozen. Gradients are used only for per-example attribution.

On a deliberately small candidate task (support plus three high-ranked false
chunks), layer-33 residual saliency looked promising. A configuration selected
on 48 discovery queries and frozen before 64 disjoint validation queries gave:

| Score | Mean query AP | MRR | Top-1 support |
|---|---:|---:|---:|
| BGE | 0.561 | 0.559 | 35.9% |
| Frozen BGE + residual saliency | 0.666 | 0.677 | 48.4% |

The paired improvements were +0.105 AP, 95% CI `[+0.020, +0.191]`, and +0.118
MRR, 95% CI `[+0.027, +0.210]`.

This result did not survive complete Top-30 evaluation. On 61 fresh test-role
queries (three additional queries exceeded the 5,600-token backward resource
limit), the same frozen scorer produced:

| Score | Global AP | MRR | Top-1 support |
|---|---:|---:|---:|
| BGE | 0.344 | 0.748 | 60.7% |
| Residual saliency | 0.039 | 0.138 | 3.3% |
| Frozen fusion, internal weight 3 | 0.129 | 0.275 | 4.9% |

The small-candidate improvement was therefore a candidate-set artifact. The
saliency could often distinguish support from the first few BGE distractors,
but many other Top-30 chunks also influenced the generated draft. It did not
rank labelled support across the complete set.

### 2. Frozen hidden-state relevance probe

`run_hidden_chunk_features.py` performs one full-context prefill, mean-pools
each chunk's query-conditioned hidden state at layers 6, 12, 18, 24, 30, and
35, L2-normalizes it, and applies a fixed random projection from 2,048 to 256
dimensions. `analyze_hidden_chunk_probe.py` trains only an L2 logistic probe;
Qwen remains frozen.

- Training/model selection: 96 calibration-role, source-train queries.
- Evaluation: 64 fresh test-role, source-dev queries, disjoint from every prior
  locked-test and saliency query.
- Cross-validation grouped by query selected layer 30 and `C=0.001`.
- Fusion weight 2 was selected using only out-of-fold calibration predictions.

| Split and score | Mean query AP | MRR | Top-1 support |
|---|---:|---:|---:|
| Calibration BGE | 0.722 | 0.751 | 62.5% |
| Calibration probe OOF | 0.763 | 0.784 | 68.8% |
| Calibration fusion OOF | 0.767 | 0.788 | 67.7% |
| Test BGE | 0.688 | 0.716 | 60.9% |
| Test probe | 0.687 | 0.713 | 59.4% |
| Test fusion | 0.696 | 0.717 | 59.4% |

On test, fusion minus BGE was +0.008 AP, 95% CI `[-0.028, +0.047]`, and +0.001
MRR, 95% CI `[-0.033, +0.038]`. The apparent calibration gain did not
generalize. This is evidence that chunk states encode relevance, but not that a
small support-label probe is a better deployable ranker.

### 3. Gold-answer causal-value teacher

`run_causal_value_labels.py` measures each chunk with 30 full-context
leave-one-out interventions. Its target is the mean gold-token log-likelihood
drop when that chunk is removed. Positive values mean the model assigns the
gold answer lower probability without the chunk.

On 32 source-train queries, 82.1% of annotated supports had positive causal
value versus 54.3% of non-support chunks. Mean causal value was 2.006 for
support and 0.018 for non-support. It therefore measures a real difference,
but it is not identical to dataset relevance: four of 32 queries had no support
chunk with a positive leave-one-out effect. Causal-value ranking produced
68.8% Top-1 support versus 65.6% for BGE, while its mean query AP was lower,
0.724 versus 0.749.

A Ridge probe trained to predict within-query causal-value rank from the old
passive chunk states did not generalize. On eight held-out causal-labelled
queries, mean per-query Spearman was 0.036 for the internal-only model and
0.044 after adding BGE. On 64 separate test queries, its BGE/internal score did
not beat raw BGE for support ranking. The causal teacher is useful, but the
feature must represent an explicit question--chunk interaction.

### 4. Explicit pairwise LM-head and hidden-state probe

`run_pairwise_relevance_features.py` asks Qwen whether each passage contains
information needed for the question. It performs three batched forward passes
for 30 candidates and does not generate an answer. It extracts:

- the next-token logit contrast over `Yes/yes/YES` versus `No/no/NO`;
- the final prompt-token hidden state at layers 6, 12, 18, 24, 30, and 35;
- the existing BGE score.

The zero-shot LM-head contrast alone was weaker than BGE. A group-safe
support-label probe selected layer 30 and `C=0.1` on 96 source-train queries.
Fusion weights were also selected from those 96 out-of-fold predictions. No
configuration was selected on the 256-query evaluation set.

| Frozen score on 256 fresh test queries | Mean query AP | MRR | Top-1 support | Top-5 any-support coverage | Top-10 any-support coverage |
|---|---:|---:|---:|---:|---:|
| BGE | 0.759 | 0.774 | 67.2% | 92.2% | 96.1% |
| BGE + zero-shot LM head | 0.779 | 0.797 | 70.3% | 92.6% | 96.1% |
| BGE + trained hidden probe | 0.777 | 0.794 | 67.6% | 96.5% | 98.8% |
| Three-signal fusion | **0.813** | **0.830** | **73.8%** | **95.7%** | **98.4%** |

For three-signal fusion minus BGE, the paired improvements were +0.054 AP,
95% CI `[+0.023, +0.087]`; +0.056 MRR, CI `[+0.024, +0.089]`; and +6.6
percentage points Top-1 support, CI `[+2.0, +11.3]`. Top-5 any-support
coverage improved by 3.5 points, CI `[+1.2, +5.9]`, and Top-10 by 2.3 points,
CI `[+0.8, +4.3]`.

### 5. Conformal keep-sets and answer quality

`analyze_pairwise_conformal_pruning.py` calibrates the critical support score
under two events: retaining at least one support and retaining every annotated
support. This experiment uses group-OOF probe predictions on 96 calibration
queries, then evaluates fixed thresholds on the 256 fresh queries.

| Target and alpha | Score | Mean kept / 30 | Support precision | Mean support recall | Query coverage |
|---|---|---:|---:|---:|---:|
| Any support, 0.05 | BGE | 5.00 | 20.2% | 92.3% | 94.5% |
| Any support, 0.05 | Fusion | **3.49** | **28.6%** | **92.4%** | **95.3%** |
| All support, 0.10 | BGE | 7.46 | 13.9% | 94.6% | 92.6% |
| All support, 0.10 | Fusion | **5.05** | **20.6%** | **95.2%** | 92.6% |
| All support, 0.05 | BGE | 26.76 | 4.1% | 99.4% | 99.2% |
| All support, 0.05 | Fusion | **9.92** | **10.9%** | 97.9% | 97.3% |

At all-support `alpha=0.10`, fusion removed 2.41 additional chunks per query,
95% CI `[2.03, 2.80]`, at the same observed all-support coverage. At
`alpha=0.05`, it removed 16.84 additional chunks, CI `[15.92, 17.76]`; its
observed 97.3% coverage remained above the 95% nominal target.

Downstream generation on the first 64 queries supports the conservative
`alpha=0.05` setting. Fusion used 10.83 chunks versus 26.28 for BGE. Fusion
minus BGE was -0.008 F1, 95% CI `[-0.076, +0.055]`, with zero mean change in EM
and numerical accuracy. At `alpha=0.10`, fusion numerical accuracy was 6.25
points lower, CI `[-12.5, -1.6]`; that threshold is too aggressive for the
current TAT-QA generator.

### 6. Attention heads, LM heads, and MLPs

Previous experiments in this directory establish the following:

- Averaging attention across layers and heads loses information. On the
  120-query discovery set, a group-safe sparse probe over individual
  `(layer, head)` attention masses reached mean query AP 0.782 versus 0.685 for
  averaged attention. This was not a full-Top-30 locked result.
- Heads selected by support-attention concentration changed gold likelihood
  when jointly masked, but did not beat layer-matched causal controls
  significantly. Raw attention is a locator for candidate circuits, not a
  signed contribution score.
- LM-head/logit-lens trajectories react strongly when evidence is added, but
  they are answer-level signals. They cannot assign a score to chunk `j`
  without contrasting a run in which `j` is removed or intervened on.
- MLPs are token-local. An MLP activation at an answer token can reveal answer
  formation or confidence, but it has no source-chunk identity. Source
  provenance requires later attention paths, gradients, activation patching,
  or chunk removal. In the 48-query layerwise experiment, attention- and
  MLP-update features did not improve over residual features.

## Relation to prior work

[Probing-RAG](https://aclanthology.org/2025.findings-naacl.181/) trains a small
prober on intermediate answer/rationale hidden states to decide whether more
retrieval is needed. Its official implementation builds labels from correct
versus incorrect generations with and without retrieval. It is a query/answer
sufficiency gate, not a per-document relevance ranker.

[MIRAGE](https://aclanthology.org/2024.emnlp-main.347/) first detects generated
tokens whose distributions change with context and then uses contrastive input
gradients to attribute those tokens to source documents. This directly
supports the claim that internals can expose context usage. It attributes the
model's actual generation, which can differ from gold support relevance.

[Retrieval Head](https://openreview.net/forum?id=EytBpUGB1Z) identifies sparse
heads responsible for long-context copying and paraphrastic retrieval and
validates them causally. Such heads are useful circuit candidates. Their
attention weights alone do not say whether a chunk helps the correct task
answer.

[ContextCite](https://openreview.net/forum?id=9kJperA2a4) learns a sparse local
surrogate from random source ablations and explicitly evaluates whether
removing attributed sources changes response probability. It provides a good
causal teacher for the proposed probe, although its repeated inference passes
are too expensive to run for every production query.

## Recommended method

Use the three-signal fusion as the candidate conformity score and calibrate the
all-support event at `alpha=0.05`. This setting fixes the original failure mode
where a weak score forces the conformal threshold to retain almost every
candidate. It also leaves enough evidence for the downstream generator in the
current 64-query check.

The 96-query OOF calibration above demonstrates compatibility, but it is not a
formal deployment guarantee because the same 96 labels were used for probe and
fusion selection. The frozen layer, `C`, prompt, and fusion weights must next be
run on the separate 1,000-query development calibration bank for each dataset.
Only the conformal threshold is fitted on that bank. Test queries must remain
untouched. The probe can be trained on the dataset training split or transferred
from TAT-QA and evaluated cross-dataset.

Causal-value supervision remains the strongest follow-up for separating
"annotated support" from "evidence this generator needs." Random-subset
ablations should replace leave-one-out when redundant chunks or multi-chunk
interactions matter. A useful test is to add that target to the explicit
pairwise layer-30 representation, since the passive full-context representation
was the component that failed here.

Direct-logit attribution is still worth including as an input feature. For an
attention head it computes the signed contribution of each source token's
attention-weighted value vector, after the head's output projection, in the
target-versus-alternative logit direction. It needs no separate training, but
should be validated against chunk-removal effects before being trusted. An MLP
has no comparable source-token decomposition, so its useful score is the
change in MLP contribution under chunk removal or activation patching.

## Reproducibility

- Layerwise saliency implementation: `contrastive_saliency.py`
- Saliency runners: `run_contrastive_saliency.py`,
  `run_full_topl_saliency.py`
- Hidden-state extraction: `run_hidden_chunk_features.py`
- Probe fitting and locked evaluation: `analyze_hidden_chunk_probe.py`
- Hidden-probe result: `results/hidden_probe_n96_test64_analysis.json`
- Causal labels: `results/causal_value_train_n32.json`
- Causal probe result: `results/causal_value_probe_n32_test64.json`
- Pairwise extractor: `run_pairwise_relevance_features.py`
- Pairwise analysis: `analyze_pairwise_relevance_probe.py`
- Fresh 256-query result:
  `results/pairwise_three_signal_locked_n96_fresh_test256.json`
- Conformal result: `results/pairwise_conformal_n96_fresh_test256.json`
- Downstream answer result:
  `results/pairwise_conformal_answers_fresh_n64_summary.json`
- Full-Top-30 saliency result:
  `results/full_top30_saliency_test_n64_seed311.json`

The complete internal-state test suite passes: 17 tests.
