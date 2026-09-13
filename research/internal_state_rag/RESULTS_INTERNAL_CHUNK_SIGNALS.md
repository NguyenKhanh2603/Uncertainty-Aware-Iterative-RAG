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

## Why cosine plus BY has high precision and low recall

The original method constructs one conformal p-value per candidate and applies
Benjamini--Yekutieli (BY) across all `L` candidates. Here, `alpha=0.10` is the
target false-discovery level; it is not a cosine threshold of 0.10.

The benchmark metrics separate retrieval from statistical selection:

- **conditional reserve support recall** is selected support divided by support
  already present in Top-L. Low values therefore mean conformal selection
  discarded retrievable support;
- **empty-query rate** is the fraction of queries for which BY certified no
  candidate;
- **evidence precision** is support divided by labelled support plus false
  chunks among the selected candidates.

On 1,000 TAT-QA evaluation queries from the labelled official-dev source, with
a separate 1,000-query official-train calibration role, the frozen Top-30
diagnostic produced:

| Candidate score and bank | BY alpha | Precision | Conditional recall | Empty queries | AUROC | AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| Jina cosine | 0.10 | 0.522 | **0.069** | **0.933** | 0.762 | 0.186 |
| BGE reranker, modality bank | 0.10 | 0.491 | 0.097 | 0.912 | 0.906 | 0.378 |
| BGE reranker, pooled bank | 0.10 | 0.508 | 0.116 | 0.894 | 0.906 | 0.378 |
| BGE plus metadata/rank fusion, five-split mean | 0.10 | **0.702** | 0.145 | 0.866 | 0.929 | 0.455 |
| BGE reranker, pooled bank | 0.20 | 0.437 | 0.163 | 0.850 | 0.906 | 0.378 |

The cosine result is the clearest statement of the failure: among support rows
that retrieval already placed in Top-30, BY selected only 6.9%, and 93.3% of
queries received an empty certified context. BGE approximately doubled AUPRC
and raised recall, but 89.4% of queries were still empty. Raising alpha to 0.20
traded precision for recall without removing the failure.

The reported precision must be read together with the empty rate. A selector
can appear precise by returning evidence only for a small subset of easy
queries. Precision 0.522 alongside an empty rate of 0.933 does not describe a
usable RAG context builder: it means the metric ignores the absence of evidence
on most queries.

### Why increasing L does not fix it

BY rejects ordered p-values only when

```text
p_(i) <= i * alpha / (L * H_L),
H_L = 1 + 1/2 + ... + 1/L.
```

At `alpha=0.10`, the first BY cutoff is approximately 0.00341 for `L=10` and
0.000834 for `L=30`. Increasing L from 10 to 30 makes the first cutoff about
4.1 times stricter. A larger reserve helps only when support is missing from
the smaller reserve; it hurts power once candidates enter the simultaneous
test.

The measured retrieval headroom is small on this TAT-QA slice. Top-10 already
contains 793 of the 864 support rows present in Top-30 and covers 738 of the 768
queries for which Top-30 finds any support. Moving from Top-10 to Top-30 adds 71
support rows and 30 covered queries, while adding 20 hypotheses to every query.
`L=30` is a reasonable reserve; increasing it further is not a remedy for the
6.9--14.5% post-selection recall.

The Top-L/Top-K backfill order does not change this diagnosis. Backfill can move
a BY-accepted candidate from ranks `K+1..L` into an unused Top-K slot. It never
adds a BY-rejected candidate merely to fill the context. Increasing L helps
only when a new support enters the reserve and also survives the stricter BY
test; it cannot rescue the majority of support already rejected inside Top-10.

### Why full-corpus retrieval does not fix it

Full-corpus retrieval fixes a different problem. It guarantees a real reserve
of L distinct candidates, removes dependence on short per-query candidate
lists, and can recover support absent from those lists. It does not change the
fact that BY later rejects most candidates whose p-values are not extreme
enough. Once conditional recall is defined relative to support already in
Top-L, a value such as 0.069 directly identifies statistical selection as the
bottleneck.

The correct conclusion is therefore not that full-corpus retrieval is useless.
It repairs candidate construction and makes calibration/test retrieval
comparable. On this dataset it does not repair the high-precision/low-recall
behavior of cosine-based candidate-wise BY.

## Resolution options and their statistical claims

### A. Recommended: query-level conformal coverage

If the RAG objective is to keep sufficient evidence, calibrate a prediction set
for the event that at least one support, or every required support, remains in
the context. This changes the controlled event from candidate-wise false
discoveries to query-level evidence coverage.

The new three-signal score demonstrates this operating point on 256 fresh
queries:

| Coverage target | Alpha | Score | Mean kept | Support recall | Query coverage |
|---|---:|---|---:|---:|---:|
| Any support | 0.05 | BGE | 5.00 | 0.923 | 0.945 |
| Any support | 0.05 | Three-signal fusion | **3.49** | **0.924** | **0.953** |
| All support | 0.05 | BGE | 26.76 | 0.994 | 0.992 |
| All support | 0.05 | Three-signal fusion | **9.92** | 0.979 | 0.973 |

This is the cleanest answer to the recall problem. It does not preserve the old
candidate-wise BY-FDR claim; it makes a coverage claim aligned with the RAG
failure event. For a formal result, train the probe before calibration, freeze
the prompt/layer/weights, and fit only the coverage threshold on the separate
1,000-query development bank for each dataset.

### B. Keep candidate-wise BY and expose a fallback channel

If candidate-level FDR control must remain, return two fields:

- `certified_chunks`: candidates rejected by valid BY;
- `fallback_chunks`: highest-ranked uncertified candidates used only when the
  certified set is empty.

In one held-out fusion split, the certified set had precision 0.676, recall
0.147, and empty rate 0.863. Adding an explicitly unverified Top-1 fallback
raised recall to 0.653 with zero empty queries and 1.05 chunks per query;
precision became 0.537. A Top-3 fallback reached recall 0.802 but precision fell
to 0.250. The conformal guarantee applies only to `certified_chunks`, so the two
sets must not be merged under one certification label.

### C. Improve the score but retain BY

BGE, metadata/rank fusion, and the pairwise internal score improve candidate
separation. They can make p-values smaller and recover some power, but no score
removes BY's `L * H_L` multiplicity factor. The next valid candidate-wise test
is to train the pairwise layer-30 probe on a training split, construct its
false-score bank on a disjoint 1,000-query calibration split, and evaluate BY
once on the 1,000 test-role queries. Until that run is complete, the
three-signal result supports query-level coverage pruning, not a claim that the
old BY procedure has been solved.

### Decision for the current system

1. Keep full-corpus Top-30 as the reserve; do not increase L as the primary
   fix.
2. Use the three-signal fusion for candidate ordering.
3. Use all-support query-level calibration at `alpha=0.05` when evidence recall
   is the product requirement.
4. If candidate-wise certification is mandatory, use BY plus a separately
   labelled Top-1 fallback and report certified precision/recall separately
   from end-to-end RAG recall.
5. Validate the frozen choice on each dataset's separate 1,000-query bank and
   untouched test role. Do not train the probe and calibrate its bank on the
   same examples.

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

- Vietnamese mentor brief: `MENTOR_BRIEF_CONFORMAL_RECALL.md`
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
