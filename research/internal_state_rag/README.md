# Internal-state RAG research prototype

This directory isolates a research direction in which the generator decides
whether its evidence is sufficient. Embedding similarity remains a cheap way to
build a reserve list, but it is no longer treated as the uncertainty signal or
the stopping rule.

The first 39-query calibration smoke experiment is reported in
[`RESULTS_TATQA_SMOKE.md`](RESULTS_TATQA_SMOKE.md). It finds a significant
evidence-induced gold log-probability shift, while highly concentrated support
attention alone does not predict whether the final answer improves.

The held-out causal masking experiment is reported in
[`RESULTS_CAUSAL_HEADS.md`](RESULTS_CAUSAL_HEADS.md). Masking top
support-attention heads lowers gold likelihood, but the effect is not
significantly larger than layer-matched controls; raw attention is therefore a
candidate-head discovery statistic rather than a validated confidence signal.

## Research question

Can the model's internal computation distinguish four cases that look similar
to a retriever?

1. The current evidence is sufficient and causally supports the answer.
2. The model already knows the answer without retrieval.
3. A missing document would repair the answer, so retrieval should continue.
4. The evidence is present but the model integrates it badly, so retrieving more
   is the wrong action.

The current conformal selector cannot answer this question. It tests up to 30
candidate relevance hypotheses and applies BY correction. With a small positive
calibration bank, the corrected threshold becomes so strict that most selected
sets are empty. Increasing `L` creates more simultaneous tests and can make that
failure worse.

## What the cited methods establish

**Probing-RAG (NAACL Findings 2025).** It mean-pools generated answer/rationale
hidden states, trains binary probes at several residual layers, and retrieves
when the probes predict that the current answer is incorrect. This establishes
that the residual stream contains a useful self-assessment signal. Its label is
still correlational: an incorrect answer is labelled “needs retrieval” even when
gold evidence is already present or more evidence cannot repair the reasoning.

**Retrieval Heads (ICLR 2025) and Retrieval-Transition Heads (EMNLP Findings
2026).** A small subset of attention heads performs behavior-specific retrieval
or transfers retrieved concepts into output. Masking the selected heads causes a
much larger failure than masking random heads. This establishes a useful
methodological standard: score heads by a specific behavior, then verify them by
causal intervention. Averaging every head in the last layer, as the current repo
does, erases this sparse structure.

**ProbeRAG (ACL Findings 2026).** This separate work probes whether a knowledge
statement conflicts with model memory, first prunes statements with embedding
similarity, then fine-tunes attention toward conflict-marked tokens. It targets
context-versus-parametric conflict and needs an attention-guided SFT stage. It
does not learn whether another retrieval round has causal value, distinguish a
retrieval failure from a reasoning failure, or calibrate an adaptive stopping
policy.

## Proposed method: Causal Internal-State RAG

At retrieval round `t`, the model sees the question and current evidence `C_t`
and produces a short draft answer. We extract three signal families from the
answer-token computation:

### 1. Residual sufficiency

For selected middle and late layers, mean-pool the residual states at answer
positions. This is the direct Probing-RAG baseline. The prototype keeps the
vectors for a linear probe instead of immediately collapsing layer logits.

### 2. Evidence-induced logit trajectory

Apply the final normalization and LM head to each selected residual layer. For
each answer token, record its log probability, margin against the best competing
token, normalized entropy, and agreement with the final-layer prediction. Run
the same teacher-forced answer with and without context and take the difference:

`context gain = log p(answer | q, C_t) - log p(answer | q)`.

A model may be very confident from parametric memory while ignoring the supplied
evidence. Absolute confidence cannot distinguish that case; the context-induced
trajectory can.

### 3. Sparse evidence-use heads

For every selected layer and head, measure attention mass from answer positions
to each aligned chunk span. Candidate heads are ranked by how strongly this mass
changes when gold support is added or removed. A head is accepted as an
**Evidence-Use Head** only if masking or patching it changes answer likelihood or
correctness more than matched random and bottom-ranked heads.

Raw attention is therefore a discovery signal, not the claimed explanation.
The causal intervention is the evidence that a head participates in grounding.
For a first TATQA experiment, the crucial contrast is whether the same heads
route evidence from text and table-token spans.

## Labels come from interventions

For each development question, construct four versions:

- no context;
- current top-K context;
- current context with labelled support removed;
- current context with missing gold support added.

`labels.py` maps the four outcomes to five states:

| State | Evidence pattern | Action |
|---|---|---|
| `grounded_sufficient` | correct now; removing support breaks it | stop |
| `parametric_known` | remains correct without support | stop or verify |
| `retrieval_insufficient` | wrong now; oracle evidence repairs it | retrieve next |
| `context_misled` | context breaks an answer known without it | reweight/remove |
| `reasoning_failure` | oracle evidence still does not repair it | reason again or abstain |

This fixes the key ambiguity in the Probing-RAG binary target: “wrong” and “more
retrieval will help” are different events.

## Sequential conformal stopping

Train a small probe on the internal features to output an unsafe-to-stop score.
Calibration uses one row per complete query trajectory. For a threshold
`lambda`, a calibration loss is one if *any* unsafe round has score at or below
`lambda`. `calibration.py` selects the largest threshold whose corrected loss

`(number of failed trajectories + 1) / (number of trajectories + 1)`

is at most `alpha`. This controls marginal anytime false-stop risk under the
usual exchangeability assumption. It deliberately does not claim conditional
error among the subset of answered queries.

This changes the multiplicity structure. The system calibrates one stopping
policy per query trajectory instead of running BY across `L` candidate chunks.
`L` can stay at 30 as a reserve pool; the generator retrieves the next chunk
only when its internal state says the evidence is insufficient.

## Code in this directory

- `signals.py` performs a memory-bounded teacher-forced trace with
  `HuggingFaceLocalClient`. It records residual vectors, per-layer LM-head
  statistics, and per-layer/per-head chunk attention, including Qwen2-VL image
  spans.
- `labels.py` creates action-aware counterfactual labels.
- `calibration.py` calibrates an anytime stopping threshold on whole query
  trajectories.

Example import:

```python
from research.internal_state_rag import QwenInternalStateExtractor
from uncertainty_rag.models.llm_client import HuggingFaceLocalClient

client = HuggingFaceLocalClient("Qwen/Qwen2-VL-7B-Instruct")
extractor = QwenInternalStateExtractor(client)
trace = extractor.extract(question, current_chunks, draft_answer)
trace.save("trace.npz", metadata={"question_id": question_id, "round": 0})
```

The extractor reuses the repo's exact chunk-token alignment. It encodes the long
prompt once, then teacher-forces answer tokens one at a time. This avoids keeping
quadratic prompt attention tensors in memory on the A100.

## Falsifiable experiment

Start on TATQA; do not retrieve the full corpus again. Reuse the existing top-30
candidate logs and the 1,000 development questions used for calibration.

1. Generate nested contexts `K = 0, 2, 5, 10` plus support-removal and
   oracle-addition interventions.
2. Extract traces for the same short draft answer under each intervention.
3. Train on development folds only. Keep the existing test questions untouched
   until the model, head selection, and stopping threshold are frozen.
4. Identify Evidence-Use Heads on training folds; mask top, random, and bottom
   heads on held-out development folds before using the test set.
5. Compare embedding-only, token entropy, Probing-RAG residual mean, logit
   trajectory, all-head attention, and causally validated sparse heads.

Primary endpoints are grounded answer EM/F1, anytime false-stop risk, and the
risk-coverage curve. Support precision/recall, retrieved tokens, and retrieval
rounds are diagnostic endpoints. The proposed signal is useful only if it
improves action classification and grounded answers at matched risk; a higher
probe AUROC alone is insufficient.

## Novelty boundary

The defensible contribution is the combination of:

- counterfactual **retrieval value** labels instead of correctness or conflict;
- internal **context-induced layer trajectories** instead of absolute model
  confidence;
- sparse heads validated by causal evidence removal/patching;
- trajectory-level conformal calibration of an adaptive stop decision.

Each component has nearby prior work, so the novelty claim must be tested by a
broader literature review before submission. The current proposal is distinct
from the three papers above at the level of target variable, intervention, and
guarantee.

## Primary sources

- Baek et al., [Probing-RAG: Self-Probing to Guide Language Models in Selective
  Document Retrieval](https://aclanthology.org/2025.findings-naacl.181/), 2025.
- Wu et al., [Retrieval Head Mechanistically Explains Long-Context
  Factuality](https://arxiv.org/abs/2404.15574), ICLR 2025.
- Sui et al., [Identifying Crucial Attention Heads for Multilingual Language
  Models: Retrieval and Retrieval-Transition
  Heads](https://arxiv.org/abs/2602.22453), 2026.
- Gao et al., [Beyond Black-Box Interventions: Latent Probing for Faithful
  Retrieval-Augmented Generation](https://aclanthology.org/2026.findings-acl.1499/),
  2026.
- Belrose et al., [Eliciting Latent Predictions from Transformers with the Tuned
  Lens](https://arxiv.org/abs/2303.08112), 2023.
