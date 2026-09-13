# Internal model signals for chunk pruning

## Decision

The experiments do **not** support replacing BGE/conformal pruning with a raw
internal-model score. They do show that internal states contain useful
information, but the useful target must be specified carefully:

- answer hidden states and layerwise LM-head trajectories measure whether the
  model appears to have enough evidence;
- chunk attention, gradients, and causal interventions measure which context
  influenced the model's current answer;
- labelled support relevance measures which context should be retained for the
  task.

These targets are correlated but not equivalent. In particular, a faithful
attribution method can faithfully identify the distractor that caused a wrong
answer.

The strongest next method is a **causal-value probe**: freeze the LLM, create
chunk-removal utility labels on development questions, and train a small probe
to predict unsafe pruning from query-conditioned chunk states and mechanistic
features. The conformal bank then calibrates that predicted unsafe-prune score.

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

### 3. Attention heads, LM heads, and MLPs

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

## Recommended method: conformal causal-value probe

The next experiment should train a probe, but should not fine-tune the LLM.

1. On development queries, retrieve the complete Top-30 and generate a fixed
   draft.
2. Sample chunk subsets or perform leave-one-chunk-out recomputation. Define a
   chunk's target as the change in gold-answer loss and generated-answer
   quality when it is removed. Use a binary label such as `unsafe_to_prune` or
   a continuous utility delta. Random subset ablations capture interactions
   better than one-at-a-time masking.
3. From one full forward pass, collect cheap features: BGE score, selected
   per-head attention value/logit contributions, chunk hidden states at
   intermediate layers, and answer-level context sensitivity.
4. Train a group-sparse linear probe or shallow MLP to approximate the causal
   utility labels. Split by query and dataset; do not split candidate rows.
5. On the separate 1,000-query development calibration bank for each dataset,
   calibrate the nonconformity score for the event that any required chunk is
   pruned. Test queries remain untouched until all feature, layer, and threshold
   choices are frozen.

This target resolves the main mismatch in all failed variants: the probe learns
whether pruning a chunk harms the task, rather than whether the model attended
to it, represented it, or used it while producing a possibly wrong draft.

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
- Full-Top-30 saliency result:
  `results/full_top30_saliency_test_n64_seed311.json`

The complete internal-state test suite passes: 17 tests.
