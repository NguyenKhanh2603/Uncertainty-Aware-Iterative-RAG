# BY conformal with cosine only — four-dataset results

**Date:** 2026-09-18  
**Score:** Jina v4 query–chunk cosine only; no LM-head, hidden-state probe, reranker, or fusion.  
**Candidate reserve:** Top-L = 30.  
**Protocol:** the false-score bank is pooled from a disjoint 100-query calibration split. BY is applied independently across the 30 candidates of each untouched 100-query test split. Precision and recall are conditional on support already occurring in Top-30.

## Results

### BY α = 0.1

| Dataset | Mean chunks kept | Empty rate | Precision | Conditional support recall | Query any-support |
|---|---:|---:|---:|---:|---:|
| HotpotQA | 0.40 | 65.0% | 92.5% | 20.1% | 34.0% |
| MMQA | 0.13 | 87.0% | 84.6% | 9.7% | 11.8% |
| TAT-QA | 0.12 | 88.0% | 83.3% | 9.5% | 11.0% |
| WebQA | 0.00 | 100.0% | 0.0% | 0.0% | 0.0% |

### BY α = 0.2

| Dataset | Mean chunks kept | Empty rate | Precision | Conditional support recall | Query any-support |
|---|---:|---:|---:|---:|---:|
| HotpotQA | 1.16 | 22.0% | 84.5% | 53.3% | 74.0% |
| MMQA | 0.41 | 59.0% | 87.8% | 31.9% | 38.7% |
| TAT-QA | 0.16 | 85.0% | 81.2% | 12.4% | 14.3% |
| WebQA | 0.01 | 99.0% | 100.0% | 1.2% | 1.4% |

## Reading the result

BY cosine has high precision when it returns chunks, but it returns no context for most queries: empty rates range from 65% to 100% at α=.10. Raising α to .20 improves recall but leaves WebQA effectively empty and TAT-QA highly sparse. These are candidate-wise BY results; they are distinct from the query-level conformal coverage experiments.

## Source artifact

- `research/internal_state_rag/results/qwen2vl_7b_jina4/by_internal_ablation_four_datasets.json`
