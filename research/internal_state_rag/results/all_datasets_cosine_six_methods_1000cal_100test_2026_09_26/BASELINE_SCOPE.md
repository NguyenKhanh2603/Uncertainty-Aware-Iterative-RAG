# Scope of the CCE, CONFLARE, and TRAQ rows

The rows in this directory are **shared-candidate cosine proxies**, not
faithful executions of the released CCE, CONFLARE, or TRAQ pipelines. They
answer a narrow, useful question: what happens when three pointwise/global
threshold styles are applied to exactly the same frozen Jina-v4 Top-30
candidates and the same 1,000 calibration qids?

## What this run actually does

For each dataset, the runner pools the raw Jina cosine scores of labelled
support chunks from the 1,000 calibration qids, then selects test candidates
whose raw cosine score clears one global threshold:

| Artifact key | Implemented threshold |
|---|---|
| `cce_alpha_0.10` | finite-sample lower 10% positive-score quantile |
| `conflare_alpha_0.10` | NumPy lower 10th-percentile positive-score quantile |
| `traq_alpha_0.10` | lower 5% positive-score quantile |

The first two are mathematically almost identical in a 1,000-query bank; that
is why their retained contexts and downstream values are nearly identical in
this report. TRAQ's proxy has a lower threshold, so it retains more chunks.
The exact implementation is in
[`run_all_datasets_cosine_six_methods.py`](../../run_all_datasets_cosine_six_methods.py).

## Why this can lose to Fixed Top-10

Fixed Top-10 applies a **per-query rank threshold** and always passes ten of
that query's strongest retrieved candidates. These proxies instead apply one
**global absolute cosine threshold** across queries. Score scales vary by
query and dataset, so the proxy can retain fewer than ten chunks on one query
or 18--24 noisy chunks on another. More support recall therefore does not
necessarily improve direct-answer QA: extra distractors can reduce answer
quality.

On the 100-query evaluation, they are not uniformly worse than Fixed Top-10:
CCE-style and CONFLARE-style proxies have higher Qwen F1 on HotpotQA (0.559 vs
0.538), MMQA (0.519 vs 0.512), and WebQA (0.179 vs 0.125), but lower F1 on
TAT-QA (0.253 vs 0.292). The TRAQ-style proxy is much looser and mainly trades
precision for recall.

## What an exact literature comparison requires

- **CCE:** its snippet construction, labels, and Conformal-Embedding or
  Conformal-LLM scores; the public materials do not provide the original
  retrieval runs and scored artifacts needed to reproduce its reported
  NeuCLIR/ARGUE results.
- **CONFLARE:** generated calibration questions paired with their
  answer-bearing source chunks and its global similarity construction, rather
  than labelled QA supports in this repository.
- **TRAQ:** the retrieval-plus-answer prediction-set procedure, including
  error-budget allocation and answer conformity calibration, rather than a
  retrieval-only cosine cutoff.

The report should therefore call these rows `CCE-style positive-cosine proxy`,
`CONFLARE-style positive-cosine proxy`, and `TRAQ-style loose-positive-cosine
proxy`. They are valid shared-score ablations, but cannot support a claim of
outperforming the original papers.
