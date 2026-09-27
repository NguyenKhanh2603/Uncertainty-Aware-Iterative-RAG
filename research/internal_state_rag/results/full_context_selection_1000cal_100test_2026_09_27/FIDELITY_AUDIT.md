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

## What “fidelity” means here

There is no defensible single fidelity percentage: weighing a retrieval model,
chunking, corpus, calibration data, answer generation, and evaluation into one
number would be arbitrary. The auditable numbers are instead:

| Dimension | CCE adaptation | CONFLARE adaptation | TRAQ adaptation |
|---|---|---|---|
| Original repository code executed end-to-end | **0%** | **0%** | **0%** |
| Mathematical threshold/filtering kernel retained | Conformal-Embedding `1 - cosine` positive-pair quantile | Strict distance-percentile filter | Lower retrieval-score quantile with `alpha_R = alpha / 2` |
| Original calibration-record construction retained | No | No: generated questions replaced with benchmark question/support pairs | No: original calibration split and answer stage omitted |
| Original retriever, scorer, corpus, chunking retained | No: frozen Jina-v4 Top-30 is substituted | No: frozen Jina-v4 Top-30 replaces Chroma pipeline | No: frozen Jina-v4 Top-30 replaces released retriever artifacts |
| Full answer-set method retained | Not applicable to this retrieval table | No | No: semantic answer prediction set is omitted |

Thus the accurate wording is **"higher construction fidelity than the legacy
cosine proxy"** only for the CONFLARE and TRAQ adaptations, never
"high-fidelity reproduction." The percentage of end-to-end released pipeline
execution is zero for all three rows.

## Why use an adaptation at all?

The adaptation makes the *selection decision* comparable: every method sees
the same candidate chunks, labels, calibration qids, test qids, embedding
score, and downstream Qwen prompt. A direct execution of CONFLARE or TRAQ
would change question generation, corpus, retriever, chunking, score scale,
and sometimes the desired output itself. Any gain could then come from those
changed inputs rather than from the pruning/calibration rule.

This is a **matched comparison of retrieval rules**, not a head-to-head
published-system benchmark. A published-system benchmark requires running each
repository's full data, retrieval, generation, and evaluation protocol and
reporting its native metric separately.

## Legacy proxy rows versus later adaptation rows

The consolidated report contains both, so no old result is hidden:

| Row family | What it does | Fidelity position |
|---|---|---|
| `CCE-style`, `CONFLARE-style`, `TRAQ-style` legacy cosine proxy | One global threshold from pooled positive Jina cosine scores | Lowest; historical shared-score ablation only |
| CCE Conformal-Embedding Jina adaptation | Positive **question–chunk** calibration records with the CCE nonconformity construction | **Numerically identical** to the old CCE proxy here: both pool all positive Jina scores and use the same finite lower-alpha quantile |
| CONFLARE source-question Jina adaptation | One best labelled support per question, strict distance filtering | Higher construction fidelity than the old CONFLARE proxy |
| TRAQ retrieval Bonferroni Jina adaptation | One best true-retrieval score per question, `alpha_R=.05` | Higher construction fidelity than the old TRAQ proxy, still retrieval-only |

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

For the frozen data in this repository, the earlier `CCE-style global
positive-cosine proxy (alpha=.10)` performs this same calculation. Writing
the score as `A=1-cosine`, taking the CCE `1-alpha` upper nonconformity
quantile, and retaining `A <= tau` is algebraically the same as taking the
finite lower-alpha Jina cosine quantile and retaining `cosine >= threshold`.
The two thresholds, masks, retrieval metrics, and Qwen answers are exactly
equal on all four datasets. The later row is retained only to expose its CCE
provenance, not as an independent improvement.

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
