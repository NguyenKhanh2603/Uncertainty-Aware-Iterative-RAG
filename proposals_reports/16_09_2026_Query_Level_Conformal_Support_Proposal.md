# Proposal: Query-Level Conformal Support for RAG Pruning

**Date:** 2026-09-16
**Status:** proposed protocol. Existing pilot results establish feasibility; this document does not claim a new formal theorem beyond split conformal prediction.

## Proposal

After retrieval returns a fixed Top-*L* pool, predict a **query-specific set of chunks containing every retrieved support chunk**. Calibrate one nonconformity score per query. This replaces candidate-wise testing plus Benjamini--Yekutieli (BY) correction with an all-support coverage objective that directly matches evidence pruning.

## Why it is not the existing conformal/BY method

The current Conformal Backfill procedure scores each candidate against a reference bank, tests up to *L* candidates for a query, then uses BY to control false discoveries under arbitrary dependence. Its useful guarantee is precision-oriented: selected candidates have controlled expected false-discovery proportion. It can validly choose no candidates and it does not guarantee that evidence required by an answer remains.

With `L=30`, BY is intentionally conservative, explaining the observed high-precision / low-recall and frequently-empty selections. The proposal changes the calibrated object, error event, and output.

| Aspect | Candidate-level conformal + BY | Query-level conformal support |
|---|---|---|
| Calibration unit | one candidate chunk | one complete query |
| Output | discovery list after *L* hypothesis tests | retained evidence set |
| Error event | false discovery among selected chunks | any retrieved gold support was pruned |
| Multiplicity | BY over correlated chunks | none: one scalar per query |
| Target | candidate precision / FDR | all-support recall and context efficiency |
| Empty set | valid and common | error unless a separate sufficiency test certifies no evidence is needed |

This is **not loosening BY**. It is split conformal prediction of an evidence-support set, conditional on the retriever including the evidence in Top-*L*.

## Formal task and guarantee

For query `q`, retrieve

```text
C(q) = {c_1, ..., c_L}.
```

Let `S(q) subseteq C(q)` be all annotated necessary-support chunks. A frozen chunk scorer `g(q, c_j, C(q))` is larger when a chunk should remain. Return

```text
R(q) = {c_j in C(q) : g(q, c_j, C(q)) >= tau}.
```

The primary target is:

```text
P[S(q) subseteq R(q) | S(q) subseteq C(q)] >= 1 - alpha.
```

This is conditional because pruning cannot rescue support that retrieval missed. Every evaluation must report both conditional coverage among retrievable queries and end-to-end coverage over every test query; their difference is the retrieval ceiling.

## Calibration algorithm

For each of `n` calibration queries with all support in Top-*L*, calculate its critical (weakest support) score:

```text
t_i = min_{c in S(q_i)} g(q_i, c, C(q_i))
a_i = -t_i
```

Use the finite-sample split-conformal quantile:

```text
k = ceil((n + 1) * (1 - alpha))
q_hat = k-th smallest of {a_1, ..., a_n}
tau = -q_hat
```

At inference retain every chunk whose score is `>= tau` (tie handling must be conservative or pre-fixed). Under exchangeability of calibration and test queries, `a_test <= q_hat` with probability at least `1-alpha`. That event is exactly that every retrieved support chunk is retained. A query creates one score, so there are no candidate `p`-values and no BY correction.

For a secondary *any-support* target, replace `min` by `max`. It only guarantees that one support survives and must never be presented as multi-hop all-support coverage.

```python
critical = [min(g(q, C)[j] for j in support) for q, C, support in calibration]
# ascending critical; exact order statistic corresponding to the score above
tau = sorted(critical)[len(critical) - ceil((len(critical) + 1) * alpha)]

def retain(q, candidates):
    scores = g(q, candidates)
    kept = [chunk for chunk, score in zip(candidates, scores) if score >= tau]
    return kept if kept else [candidates[argmax(scores)]]
```

The fallback Top-1 only enlarges the set and therefore cannot lower support coverage, but its rate must be reported.

## What supplies the chunk score

Conformal calibration determines a threshold; it cannot make a weak relevance score strong. Score families must be fitted and frozen **before** conformal calibration.

| Candidate scorer | Role |
|---|---|
| Jina-v4 cosine | original retrieval-score baseline |
| Jina-m0 reranker | main strong external-score baseline |
| BGE reranker | text-only ablation where modality supports it |
| Qwen hidden-state probe | trained internal relevance signal |
| Direct-logit attribution / causal chunk masking | proposed internal evidence score |
| score fusion | optional pre-registered ablation, with weights selected out-of-fold before calibration |

Raw attention is not a relevance label. It may find candidate heads, but a head must be accepted using direct-logit attribution or chunk-specific causal masking, with matched non-support controls. The final internal score should be either a low-capacity hidden-state probe or a causally validated DLA/masking feature.

## Data isolation

| Role | Data | Allowed use |
|---|---|---|
| Score development | official train or a disjoint train partition | head discovery, causal labels, probe/fusion fitting, score selection |
| Calibration | 1,000 official dev queries per dataset | calculate threshold only, with scorer frozen |
| Final evaluation | official test set | one locked evaluation |

Do not calibrate on test queries. The older 1,000-train-side pilot split (approximately half probe training, half calibration) should remain labelled as a pilot; the intended protocol is 1,000 dev queries for calibration. Calibrate separately by dataset initially. MMQA modality-conditional calibration is a secondary analysis and must allocate alpha across modalities before test evaluation.

## Context budget and empty outputs

Do not apply a hard Top-*K* after conformal selection: that may delete support and void the all-support claim. For a fixed generator budget, either calibrate a support-containing prefix, permit a larger context, or mark budget overflow/abstain. Report the overflow rate.

An empty retained set is a pruning failure whenever external evidence is required. A truly empty context needs a separate, conformally calibrated query-sufficiency decision; it is not justified merely because every candidate score fell below threshold.

## Evaluation plan

At alpha `{0.20, 0.10, 0.05}`, report per dataset:

| Metric | Meaning |
|---|---|
| Top-*L* retrieval ceiling | fraction with all support in candidate pool |
| mean and median chunks kept | context efficiency |
| chunk precision | selected annotated support fraction |
| micro support recall | support chunks retained |
| query all-support / any-support coverage | primary / secondary events |
| end-to-end all-support coverage | includes retrieval misses |
| budget overflow and empty/fallback rate | deployment behavior |
| answer F1/EM and token cost | downstream value |

Use paired bootstrap confidence intervals versus query-level cosine. Required comparisons are: matched-size Top-*K*; candidate-level BY using cosine and Jina; query-level cosine; query-level Jina-m0; internal-only; frozen fusion; and uncalibrated attention-only (explicitly labelled recall-oriented baseline).

## Existing evidence and claim boundary

The pilots already show that query-level calibration can return small, non-empty support sets with high conditional recall. Hidden-state features improve cosine pruning on TAT-QA and HotpotQA. Jina-m0 is still the strongest broad candidate score, and simple internal-plus-Jina fusion has small or inconsistent incremental benefit.

The defensible claim today is: **query-level conformal support fixes the objective mismatch of candidate-level BY for evidence retention.** It does not yet prove that model attention, LM-head margins, or generic fusion is a universally superior evidence detector. A causally validated internal scorer remains the central research question.

## Falsifiable hypotheses

1. At matched recall, query-level calibration retains fewer chunks than candidate-level BY; at matched size, it yields higher all-support coverage.
2. Hidden-state/DLA scores beat cosine in query-grouped AP, and DLA correlates with exact answer-logit loss under chunk masking better than attention mass.
3. After fully separate calibration, an internal score improves the chunk-count/coverage frontier over cosine on at least two datasets. Fusion is a gain only when its paired confidence interval beats its best constituent.
4. Increasing *L* raises end-to-end coverage only by recovering retrieval misses; it cannot fix ranking errors inside an existing Top-*L* pool.

## Execution sequence

1. Freeze Top-30 candidates and split manifests for TAT-QA, HotpotQA, MMQA, and WebQA.
2. Implement one shared query-level conformal module accepting score matrices and support masks; persist threshold, alpha, score version, candidate pool, calibration manifest, and per-query mask.
3. Use it identically for cosine, Jina, BGE where applicable, hidden probe, and frozen fusion.
4. Validate causal/DLA features on score-development data before fitting a low-capacity internal scorer.
5. Calibrate with 1,000 dev queries/dataset, then evaluate once on official tests.

## Repository links

- [Internal chunk-pruning plan](../research/internal_state_rag/PLAN_CONFORMAL_INTERNAL_CHUNK_PRUNING.md)
- [Adaptive query-conformal analysis](../research/internal_state_rag/analyze_adaptive_query_conformal.py)
- [Cross-dataset pilot report](../research/internal_state_rag/results/qwen2vl_jina4/CROSS_DATASET_REPORT_2026-09-14.md)
- [Experiment/result index](../16_09_2026_nhunggidachay.md)
