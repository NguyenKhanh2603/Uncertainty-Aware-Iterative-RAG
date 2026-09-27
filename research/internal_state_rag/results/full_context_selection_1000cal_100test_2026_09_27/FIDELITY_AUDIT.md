# Fidelity audit: CCE, CONFLARE, and TRAQ rows

## Finding

The literature rows in the consolidated report are **not full reproductions of
the released CCE, CONFLARE, or TRAQ systems**. The repositories were checked
out and used to derive the relevant scoring and calibration constructions, but
`research/internal_state_rag/run_literature_protocol_1000cal.py` is a local
adapter. It does not import or execute the end-to-end pipelines from those
repositories.

The rows remain useful as a controlled shared-candidate ablation: all methods
receive exactly the same frozen Jina-v4 Top-30 candidates, the same 1,000
calibration qids, and the same 100 held-out qids. They must not be described
as published-baseline reproduction results.

## Checked-out source revisions

| Method | Local source revision | What was retained |
|---|---|---|
| CCE | `baselines/conformal-context-engineering` at `91732d6058267f180ba9f47873d743288b2625af` | Conformal-Embedding score `1 - cosine` and the positive-pair `1 - alpha` calibration quantile. |
| CONFLARE | `baselines/conflare` at `ce081a45fb452704daa87f3b37f601b4accc7a82` | One calibration record per question and the strict cosine-distance percentile acceptance rule. |
| TRAQ | `baselines/TRAQ` at `e9b66b5b7fd8fcd0b3c3dbfebc5a3391ca79aa55` | Retrieval threshold's lower empirical quantile and the equal Bonferroni retrieval allocation `alpha_R = alpha / 2`. |

## What differs from each released pipeline

### CCE Conformal-Embedding

The checked-out CCE material provides the scoring and threshold specification,
but no released executable retrieval run or scored candidate artifact for
these four datasets. The adapter uses Jina-v4 cosine values and benchmark
labelled support pairs, whereas the reported CCE setup uses its own embedding
model, snippet construction, and dataset protocol. Its result is therefore a
**Conformal-Embedding construction adapted to Jina candidates**.

### CONFLARE

Released CONFLARE creates calibration records by generating questions from
documents, evaluating those questions against a vector database, and then
filters Chroma distances. The adapter instead uses each benchmark human
question plus its highest-scoring labelled support in the frozen Top-30. It
preserves the one-record-per-question and strict percentile filtering rule but
does not execute question generation, QA evaluation, Chroma retrieval, or the
released RAG pipeline.

### TRAQ

Released TRAQ calibrates a retrieval component together with an answer
prediction set: it needs generated answer samples, semantic clustering,
answer conformity scores, a second calibration/evaluation stage, and an
error-budget allocation. The adapter measures only the retrieval threshold
on frozen Jina candidate scores with `alpha_R=.05` under total `alpha=.10`.
The shared Qwen EM/F1 columns are a common downstream diagnostic; they are
not TRAQ answer-set coverage or a TRAQ end-to-end result.

## Correct claims supported by the current table

- It is a controlled comparison of **adapted retrieval calibration rules** on
  the same candidate pool and disjoint qids.
- It measures chunk precision, support recall, context size, and a shared
  direct-answer diagnostic for those adapted rules.
- It does not establish that query-level conformal pruning beats the released
  CCE, CONFLARE, or TRAQ systems on their published datasets or protocols.
