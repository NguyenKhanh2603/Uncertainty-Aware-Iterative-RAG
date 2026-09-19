# BY, TRAQ, CONFLARE, and ECIR CCE — Four-Dataset Retrieval Benchmark

**Date:** 2026-09-19  
**Scope:** retrieval/chunk selection only. Query-level conformal methods and internal-model fusion are intentionally excluded.

## Shared protocol

- Frozen Jina v4 cosine candidate scores; each query has the same Top-30 chunks.
- Each dataset uses the project’s disjoint 100-query calibration and 100-query test roles.
- Precision and recall are conditional on labelled support present in Top-30.
- `Kept` is mean retained chunks per test query; `Recall` is conditional support recall; `Empty` is fraction of test queries retaining no chunk.

## What each method implements

| Method | Retrieval rule benchmarked | Important limitation |
|---|---|---|
| BY cosine | Candidate p-values from calibration false-score bank; Benjamini–Yekutieli across 30 candidates. | Candidate-wise FDR-style selection; often returns no context. |
| ECIR CCE Embedding | Finite-sample split-conformal threshold over calibration positive chunk cosine scores. | Public ECIR supplement has prompts only; this reproduces its stated Conformal-Embedding rule on our shared logs. |
| CONFLARE | Official percentile threshold on calibration relevant-chunk cosine distance, converted to cosine similarity. | Original code uses LLM-generated synthetic questions; this controlled adapter uses gold calibration support chunks. |
| TRAQ retrieval | Official retrieval threshold at `alpha/2`, matching TRAQ’s Bonferroni allocation between retrieval and answer generation. | Full TRAQ also requires answer candidate sets and semantic correctness scores, so its end-to-end guarantee is not evaluated here. |

## Results at α = .10

| Dataset | Method | Kept | Precision | Recall | Empty |
|---|---|---:|---:|---:|---:|
| HotpotQA | BY cosine | 0.40 | 92.5% | 20.1% | 65.0% |
| HotpotQA | ECIR CCE | 11.19 | 14.9% | 90.8% | 1.0% |
| HotpotQA | CONFLARE | 10.80 | 15.5% | 90.8% | 1.0% |
| HotpotQA | TRAQ retrieval | 14.97 | 11.4% | 92.9% | 0.0% |
| MMQA | BY cosine | 0.13 | 84.6% | 9.7% | 87.0% |
| MMQA | ECIR CCE | 14.19 | 7.4% | 92.9% | 3.0% |
| MMQA | CONFLARE | 13.02 | 8.0% | 92.0% | 3.0% |
| MMQA | TRAQ retrieval | 20.43 | 5.4% | 97.3% | 1.0% |
| TAT-QA | BY cosine | 0.12 | 83.3% | 9.5% | 88.0% |
| TAT-QA | ECIR CCE | 20.10 | 4.7% | 89.5% | 5.0% |
| TAT-QA | CONFLARE | 20.02 | 4.7% | 89.5% | 5.0% |
| TAT-QA | TRAQ retrieval | 24.08 | 4.2% | 95.2% | 3.0% |
| WebQA | BY cosine | 0.00 | 0.0% | 0.0% | 100.0% |
| WebQA | ECIR CCE | 21.98 | 3.5% | 91.8% | 8.0% |
| WebQA | CONFLARE | 21.71 | 3.5% | 90.6% | 8.0% |
| WebQA | TRAQ retrieval | 26.16 | 3.2% | 97.6% | 2.0% |

## Results at α = .20

| Dataset | Method | Kept | Precision | Recall | Empty |
|---|---|---:|---:|---:|---:|
| HotpotQA | BY cosine | 1.16 | 84.5% | 53.3% | 22.0% |
| HotpotQA | ECIR CCE | 6.04 | 25.2% | 82.6% | 4.0% |
| HotpotQA | CONFLARE | 6.05 | 25.1% | 82.6% | 4.0% |
| HotpotQA | TRAQ retrieval | 11.19 | 14.9% | 90.8% | 1.0% |
| MMQA | BY cosine | 0.41 | 87.8% | 31.9% | 59.0% |
| MMQA | ECIR CCE | 4.70 | 18.3% | 76.1% | 14.0% |
| MMQA | CONFLARE | 4.72 | 18.2% | 76.1% | 14.0% |
| MMQA | TRAQ retrieval | 14.19 | 7.4% | 92.9% | 3.0% |
| TAT-QA | BY cosine | 0.16 | 81.2% | 12.4% | 85.0% |
| TAT-QA | ECIR CCE | 14.66 | 6.0% | 83.8% | 8.0% |
| TAT-QA | CONFLARE | 14.66 | 6.0% | 83.8% | 8.0% |
| TAT-QA | TRAQ retrieval | 20.10 | 4.7% | 89.5% | 5.0% |
| WebQA | BY cosine | 0.01 | 100.0% | 1.2% | 99.0% |
| WebQA | ECIR CCE | 20.32 | 3.5% | 84.7% | 9.0% |
| WebQA | CONFLARE | 20.10 | 3.5% | 83.5% | 9.0% |
| WebQA | TRAQ retrieval | 21.98 | 3.5% | 91.8% | 8.0% |

## Interpretation

- **BY cosine** is the only method here that controls candidate-wise multiple testing. It achieves the highest precision but loses most support because its simultaneous correction is severe, especially on TAT-QA and WebQA.
- **ECIR CCE** and **CONFLARE** are mathematically nearly the same threshold rule in this controlled setting: retain chunks above a global calibration-derived cosine threshold. Their small differences come from finite-sample quantile choice versus `np.percentile`.
- **TRAQ retrieval** is deliberately more conservative because the original end-to-end method reserves half the error budget for answer-generation coverage. It therefore retains more chunks and has higher retrieval recall than CCE/CONFLARE at the same reported alpha.
- This is a retrieval-component benchmark, not a claim that TRAQ, CONFLARE, or CCE end-to-end answer quality has been reproduced.

## Downloaded official code

- TRAQ: `baselines/TRAQ` at `e9b66b5` (downloaded 2026-09-19).
- CONFLARE: `baselines/conflare` at `ce081a4` (downloaded 2026-09-19).
- ECIR CCE: `baselines/conformal-context-engineering` at `91732d6`.

## Artifacts

- `research/internal_state_rag/results/qwen2vl_7b_jina4/conformal_retrieval_baselines_four_datasets.json`
- `research/internal_state_rag/results/qwen2vl_7b_jina4/by_internal_ablation_four_datasets.json`
