# Candidate-wise BY: cosine score versus Jina reranker score

Date: 2026-09-16

## Question

Can the original candidate-wise Benjamini--Yekutieli (BY) procedure keep the
same cosine-retrieved Top-30 candidates but use a reranker score for its
false-score bank, p-values, and Top-K ordering?

Yes. This run uses `jina_reranker_score` from Jina reranker-m0 in place of
`cosine_score`. The candidate pool is unchanged: Jina embeddings-v4 retrieves
the fixed Top-30. This isolates the effect of the score used by BY.

## Protocol

- Candidate pool: frozen Jina embeddings-v4 global Top-30.
- Cosine variant: `cosine_score`, ordered by the original dense rank.
- Reranker variant: `jina_reranker_score`, ordered by Jina m0 rank.
- Reranker: `jinaai/jina-reranker-m0`, revision `94bfe0a...`.
- False-score bank: only rows with `split_role=calibration` and
  `support_label=false`; the development role is not used.
- Test: the untouched `split_role=test` rows.
- BY level: `alpha=0.20, 0.10, 0.05`; context cap `K=10`.
- The pooled bank combines modalities. The modality bank is reported as an
  ablation. All metrics are empirical; the Top-K cap remains an experimental
  post-processing step rather than a new formal BY guarantee.

The calibration/test counts are 484/1,000 for TAT-QA, 491/1,000 for HotpotQA,
496/1,000 for MMQA, and 535/250 for WebQA. The Top-30 retrieval coverage
ceilings are 91.9%, 99.6%, 94.8%, and 72.8%, respectively.

## Pooled false-score bank at alpha = 0.10

`Recall` is micro support recall conditional on support being present in Top-30.
`Empty` is the fraction of all test queries for which BY selects no chunk.
`E2E all` is the fraction of all test queries for which every labelled support
chunk is retained.

| Dataset | Score | Mean chunks | Precision | Recall | Empty | E2E all |
|---|---|---:|---:|---:|---:|---:|
| TAT-QA | Cosine | 0.208 | 25.00% | 5.08% | 93.7% | 4.4% |
|  | Jina m0 | **0.113** | **75.22%** | **8.31%** | **91.3%** | **7.4%** |
| HotpotQA | Cosine | 0.238 | 69.33% | 9.04% | 85.1% | 2.6% |
|  | Jina m0 | 0.293 | **88.40%** | **14.19%** | **78.4%** | **6.7%** |
| MMQA | Cosine | 0.065 | 50.77% | 2.78% | 96.3% | 1.9% |
|  | Jina m0 | 0.309 | **89.64%** | **23.34%** | **72.5%** | **24.4%** |
| WebQA | Cosine | 0.000 | -- | 0.00% | 100.0% | 0.0% |
|  | Jina m0 | 0.044 | -- | 0.00% | 98.8% | 0.0% |

The Jina score improves BY substantially on the first three datasets. It raises
precision by 50.22, 19.07, and 38.87 percentage points and raises conditional
recall by 3.23, 5.15, and 20.56 points on TAT-QA, HotpotQA, and MMQA. WebQA has
too little separation at this strict BY level: the reranker selects 11 chunks
but none is labelled support in the pooled result.

## Modality-conditioned bank at alpha = 0.10

| Dataset | Score | Mean chunks | Precision | Recall | Empty | E2E all |
|---|---|---:|---:|---:|---:|---:|
| TAT-QA | Cosine | 0.191 | 21.99% | 4.11% | 94.4% | 3.3% |
|  | Jina m0 | **0.149** | **71.14%** | **10.36%** | **88.9%** | **8.8%** |
| HotpotQA | Cosine | 0.238 | 69.33% | 9.04% | 85.1% | 2.6% |
|  | Jina m0 | 0.293 | **88.40%** | **14.19%** | **78.4%** | **6.7%** |
| MMQA | Cosine | 0.102 | 42.16% | 3.62% | 95.3% | 3.0% |
|  | Jina m0 | 0.366 | **86.89%** | **26.79%** | **68.5%** | **28.2%** |
| WebQA | Cosine | 0.052 | 15.38% | 0.87% | 97.6% | 0.4% |
|  | Jina m0 | 0.088 | 9.09% | 0.87% | 98.4% | 0.8% |

## Alpha sweep for the Jina score, pooled bank

| Dataset | alpha | Mean chunks | Precision | Recall | Empty |
|---|---:|---:|---:|---:|---:|
| TAT-QA | 0.20 | 0.308 | 55.52% | 16.72% | 82.5% |
|  | 0.10 | 0.113 | 75.22% | 8.31% | 91.3% |
|  | 0.05 | 0.096 | 85.42% | 8.02% | 91.5% |
| HotpotQA | 0.20 | 0.505 | 79.41% | 21.97% | 66.1% |
|  | 0.10 | 0.293 | 88.40% | 14.19% | 78.4% |
|  | 0.05 | 0.178 | 94.38% | 9.21% | 85.2% |
| MMQA | 0.20 | 0.423 | 80.61% | 28.73% | 66.6% |
|  | 0.10 | 0.309 | 89.64% | 23.34% | 72.5% |
|  | 0.05 | 0.193 | 87.05% | 14.15% | 83.1% |
| WebQA | 0.20 | 0.168 | 9.52% | 1.75% | 96.4% |
|  | 0.10 | 0.044 | 0.00% | 0.00% | 98.8% |
|  | 0.05 | 0.036 | 0.00% | 0.00% | 99.2% |

## Interpretation

Replacing cosine with Jina m0 is possible and materially improves the BY
candidate test. The reranker makes the false-score p-values more separated,
so BY admits more true support while retaining high measured precision. It does
not, however, remove the basic candidate-wise BY power problem: at alpha 0.10
the selector is still empty on 72.5--91.3% of queries on three datasets, and
WebQA remains effectively empty. Increasing the reranker quality helps, but
the multiplicity correction and strict candidate-wise event still dominate
recall.

This is a score-swap ablation, not a fusion method and not a comparison of
different candidate pools. The next recall-oriented option remains query-level
conformal coverage or an explicitly unverified fallback channel.

## Artifacts

- `tatqa_by_cosine.json`, `hotpotqa_by_cosine.json`, `mmqa_by_cosine.json`,
  `webqa_by_cosine.json`.
- `tatqa_by_jina_m0.json`, `hotpotqa_by_jina_m0.json`, `mmqa_by_jina_m0.json`,
  `webqa_by_jina_m0.json`.
- The score-parametrized runner is
  `research/internal_state_rag/analyze_current_cosine_by.py`.
