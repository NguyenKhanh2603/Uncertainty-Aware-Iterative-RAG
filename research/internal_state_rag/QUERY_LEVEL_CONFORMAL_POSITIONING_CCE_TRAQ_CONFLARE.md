# Positioning query-level conformal pruning against CCE, TRAQ, and CONFLARE

## Claim boundary

The query-level selector is **not** a new general conformal-prediction
algorithm. It is ordinary split conformal prediction applied to a different
nonconformity target: the lowest-scoring required support for one query. That
target is useful for multi-hop and multi-fact retrieval, but the distinction
must be stated precisely. A paper should not claim to be the first conformal
method for RAG context filtering: CCE, TRAQ, and CONFLARE already establish
that broader category.

The defensible methodological contribution is a query-level **all-retrieved-
support** coverage objective, evaluated on a shared candidate set against the
pointwise and any-support objectives. The independent internal-signal component
asks a second question: can a frozen generator's LM-head and hidden states give
a score that attains that same target with a smaller context than cosine alone?

## Different conformal events

Let `C(q)` be frozen Top-L candidates for query `q`, let `S(q) ⊆ C(q)` be its
labelled supports, and let higher `g(q,c)` mean more relevant.

| Method | Calibration statistic | Intended retained-evidence event | Consequence for multi-support queries |
|---|---|---|---|
| CCE Conformal-Embedding / Conformal-LLM | Each positive `(q,c)` score independently | A randomly sampled relevant snippet is retained with marginal probability at least `1−α` | Does not directly control whether **every** support for the same query survives. |
| CONFLARE | Similarity of the answer-bearing source chunk for generated calibration questions | The source/answer-bearing context is retrieved above a global cosine-distance cutoff | Its released pipeline has one source chunk per generated question; it does not calibrate the minimum over multiple required supports. |
| TRAQ retrieval component | Score of the first retrieved true document, then a separate QA conformity score | At least one relevant retrieval and a semantically correct answer set | Optimizes end-to-end answer-set coverage, not retention of all evidence chunks. |
| This query-level selector | `T(q)=min_{c∈S(q)} g(q,c)` for one retrievable calibration query | `S(q) ⊆ K(q)` conditional on support being present in Top-L | Directly protects the weakest retrieved support; it is deliberately conservative when an answer needs several pieces of evidence. |

For the last row, an all-support split-conformal threshold is the finite-sample
quantile of `T(q)` from calibration queries. At inference, retain candidates
whose score is at least that threshold, with deterministic Top-1 fallback only
for a numerically empty retained set. The coverage claim is conditional on the
retrieval pool containing supports; retrieval itself can still fail.

CCE explicitly calibrates the positive snippet score distribution and states
the marginal target `P(c ∈ K(q) | c is relevant) ≥ 1−α` for its
Conformal-Embedding or Conformal-LLM scorer. Its released documentation uses
`1−cosine` or an LLM relevance rating, but the calibration unit remains a
relevant snippet rather than a full query. [CCE repository](https://github.com/hltcoe/conformal-context-engineering)

TRAQ decomposes the total error budget between retrieval and answering, then
uses conformal thresholds and Bayesian optimization to control its end-to-end
answer-set objective. It is therefore related, but its retrieval statistic is
an any-correct-retrieval event rather than an all-support context-pruning
criterion. [TRAQ paper](https://aclanthology.org/2024.naacl-long.210/)

CONFLARE constructs answerable calibration questions from documents and filters
retrieval by a global similarity-distance threshold learned from their
answer-bearing chunks. It is a pointwise/global retrieval uncertainty method,
not a query-minimum support construction. [CONFLARE paper](https://arxiv.org/abs/2404.04287)

## What is potentially new, and what is not

Potentially publishable if supported by matched experiments:

1. **Target:** formulate context pruning around an all-retrieved-support event
   appropriate for multi-hop/multi-fact QA, rather than marginal snippet or
   any-retrieval coverage.
2. **Score:** evaluate whether generator-internal evidence signals
   (`Yes`-versus-`No` LM-head logit and query-conditioned hidden-state probe)
   improve the precision/context-cost frontier at that fixed target.
3. **Protocol:** use three disjoint roles—internal-probe training, conformal
   calibration, and test—and report support recall, all-support coverage,
   retained chunks, empty rate, and downstream QA together.

Not sufficient by itself for novelty:

- replacing CCE's embedding with cosine z-scores;
- applying split conformal prediction to a new score without showing a
  different retained-evidence objective;
- tuning a probe or fusion weights on the calibration or test qids;
- claiming a candidate-level BY/BH false-discovery guarantee for a later
  rank-capped context.

## Current matched experiment

The inverse protocol has 1,000 parent calibration qids and 100 held-out qids
per dataset. To make the internal scorer valid, its parent calibration pool is
partitioned before looking at labels used for model selection:

| Dataset | Probe train | Conformal calibration | Held-out test |
|---|---:|---:|---:|
| HotpotQA | 500 | 500 | 100 |
| MMQA | 504 | 496 | 100 |
| TAT-QA | 516 | 484 | 100 |
| WebQA | 465 | 535 | 100 |

The resulting table compares matched query-level cosine, LM-head-only,
hidden-probe-only, internal-only fusion, and cosine+LM-head+hidden fusion at
`α=0.10`. The experiment directory contains the exact qid audit, masks, and
per-query downstream answers:
[`internal_signal_ablation_7b`](results/all_datasets_cosine_six_methods_1000cal_100test_2026_09_26/internal_signal_ablation_7b/).

## Required comparison language

Use “query-level all-support conformal pruning” for the method. Describe it as
a **different coverage objective** from CCE/CONFLARE/TRAQ, not as a replacement
for their guarantees. A positive result should be stated as: under the same
Top-L candidates and a disjoint split, the proposed score reaches comparable
all-support coverage with fewer retained chunks or improves downstream QA. If
the internal ablation does not improve that frontier, report it as a negative
result rather than attributing gains to the conformal target.
