# Downstream QA: query-level cosine + internal fusion vs cosine baselines

## Matched evaluation

Every result uses the same frozen Jina-v4 Top-30 candidates, 100 calibration
queries, 100 untouched test queries, Qwen2-VL-7B greedy generation, and 24
new answer tokens.  MMQA uses the same 3,136--200,704 pixel bounds as feature
extraction.

`Query-level cosine + internal fusion` is the all-support split-conformal
selector applied to

`z(cosine) + w_lm z(LM-head relevance) + w_hidden z(hidden-state probe)`.

The hidden probe and weights are selected on a separate 100-query probe-train
split; its threshold is calibrated on the frozen calibration split.  The
weights differ by dataset and are saved in each summary artifact.

CCE, CONFLARE, and TRAQ in this table all use **Jina-v4 cosine** as their
retrieval score.  CCE applies a positive-support split-conformal threshold;
the CONFLARE adapter applies its percentile threshold with gold supports in
place of synthetic questions; TRAQ is its retrieval `alpha/2` allocation, not
the complete answer-set TRAQ method.

## Results at alpha=0.10

| Dataset | Method | EM | F1 | Numerical accuracy | Mean chunks |
|---|---|---:|---:|---:|---:|
| HotpotQA | BY cosine | 0.150 | 0.263 | 0.170 | 0.13 |
|  | CCE cosine | **0.430** | **0.580** | **0.450** | 11.19 |
|  | CONFLARE cosine adapter | 0.410 | 0.558 | 0.430 | 10.80 |
|  | TRAQ cosine retrieval adapter | 0.390 | 0.542 | 0.420 | 14.97 |
|  | Query-level cosine + internal fusion | 0.400 | 0.564 | 0.410 | **4.08** |
| MMQA | BY cosine | 0.160 | 0.196 | 0.170 | 0.05 |
|  | CCE cosine | **0.500** | **0.548** | **0.520** | 14.19 |
|  | CONFLARE cosine adapter | 0.490 | 0.531 | 0.510 | 13.02 |
|  | TRAQ cosine retrieval adapter | 0.480 | 0.531 | 0.500 | 20.43 |
|  | Query-level cosine + internal fusion | **0.500** | 0.544 | **0.520** | **4.75** |
| TAT-QA | BY cosine | 0.040 | 0.116 | 0.120 | 0.85 |
|  | CCE cosine | **0.160** | 0.263 | 0.240 | 20.10 |
|  | CONFLARE cosine adapter | **0.160** | **0.268** | 0.250 | 20.02 |
|  | TRAQ cosine retrieval adapter | 0.150 | 0.256 | 0.230 | 24.08 |
|  | Query-level cosine + internal fusion | **0.160** | 0.262 | **0.260** | **3.49** |

## What this establishes

Fusion does not produce a large EM gain over CCE on these 100-query tests.
Its useful result is substantially smaller evidence sets while preserving
downstream answer quality: 64% fewer chunks than CCE on HotpotQA, 67% fewer
on MMQA, and 83% fewer on TAT-QA.  MMQA matches CCE's EM and numerical
accuracy with 4.75 rather than 14.19 chunks; TAT-QA matches EM with 3.49
rather than 20.10 chunks.

This is therefore evidence for *efficient pruning*, not a claim that internal
signals outperform cosine calibration in absolute QA accuracy.  BY cosine is
not competitive downstream because its candidate-wise correction yields empty
contexts for 86--97% of queries.

## Artifacts

- Query-level fusion summaries and predictions:
  `downstream_query_level_fusion_a10/`.
- Cosine baseline summaries and predictions:
  `downstream_conformal_baselines_a10_fixedcal/` (HotpotQA) and
  `downstream_conformal_baselines_a10_fixedcal_px200704/` (MMQA, TAT-QA).
- Fusion runner: `research/internal_state_rag/run_downstream_query_level_fusion.py`.
- Cosine baseline runner: `research/internal_state_rag/run_downstream_qa_conformal_baselines.py`.
