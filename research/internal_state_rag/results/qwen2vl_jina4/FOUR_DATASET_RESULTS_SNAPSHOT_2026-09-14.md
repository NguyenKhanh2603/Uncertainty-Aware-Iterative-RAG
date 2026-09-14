# Four-dataset pruning results snapshot

Date: 2026-09-14

## Scope

This snapshot evaluates chunk pruning on TAT-QA, HotpotQA, MMQA, and WebQA. Retrieval uses `jinaai/jina-embeddings-v4` over each dataset's full corpus and retains Top-30 candidates. Internal features use `Qwen/Qwen2-VL-2B-Instruct`; the external reranker is `jinaai/jina-reranker-m0`.

Probe-training, conformal-calibration, and official test queries are disjoint. At most 1,000 train-side queries per dataset are split between probe training and calibration. Test metrics use only the official test role.

The main table reports alpha = 0.1. Chunks, precision, support recall, and query all-support coverage are conditional on support being retrievable in Top-30. `E2E all` includes every test query and therefore includes retrieval failures.

## Retrieval coverage

| Dataset | Probe train | Calibration | Test | Test retrievable | Top-30 ceiling |
|---|---:|---:|---:|---:|---:|
| TAT-QA | 516 | 484 | 1,000 | 919 | 91.9% |
| HotpotQA | 509 | 491 | 1,000 | 996 | 99.6% |
| MMQA | 504 | 496 | 1,000 | 948 | 94.8% |
| WebQA | 465 | 535 | 250 | 182 | 72.8% |

## Full comparison at alpha = 0.1

| Dataset | Method | Chunks kept /30 | Chunk precision | Support recall | Query all-support | E2E all |
|---|---|---:|---:|---:|---:|---:|
| TAT-QA | Original cosine-BY | 0.21 | 25.00% | 5.08% | 4.79% | 4.40% |
| TAT-QA | Jina-v4 cosine + query conformal | 10.84 | 9.08% | 88.37% | 87.27% | 80.20% |
| TAT-QA | Attention-only | 21.28 | 4.80% | 91.79% | 91.40% | 84.00% |
| TAT-QA | Qwen internal-only | 9.53 | 10.65% | 91.20% | 90.64% | 83.30% |
| TAT-QA | Jina-m0 reranker-only | 3.92 | 26.24% | 92.38% | 92.27% | 84.80% |
| TAT-QA | Jina-m0 + Qwen internal | 3.93 | 26.30% | 92.77% | 92.71% | 85.20% |
| HotpotQA | Original cosine-BY | 0.24 | 69.33% | 9.04% | 2.61% | 2.60% |
| HotpotQA | Jina-v4 cosine + query conformal | 9.66 | 17.90% | 94.41% | 89.76% | 89.40% |
| HotpotQA | Attention-only | 14.04 | 12.39% | 94.96% | 91.47% | 91.10% |
| HotpotQA | Qwen internal-only | 6.54 | 26.49% | 94.52% | 90.06% | 89.70% |
| HotpotQA | Jina-m0 reranker-only | 2.91 | 59.24% | 94.19% | 89.46% | 89.10% |
| HotpotQA | Jina-m0 + Qwen internal | 2.76 | 62.59% | 94.25% | 89.56% | 89.20% |
| MMQA | Original cosine-BY | 0.07 | 50.77% | 2.78% | 2.00% | 1.90% |
| MMQA | Jina-v4 cosine + query conformal | 8.14 | 14.30% | 92.92% | 91.35% | 86.60% |
| MMQA | Attention-only | 16.34 | 7.20% | 93.93% | 93.04% | 88.20% |
| MMQA | Qwen internal-only | 8.43 | 13.77% | 92.67% | 91.14% | 86.40% |
| MMQA | Jina-m0 reranker-only | 2.93 | 40.13% | 93.85% | 92.62% | 87.80% |
| MMQA | Jina-m0 + Qwen internal | 2.94 | 40.07% | 94.02% | 92.83% | 88.00% |
| WebQA | Original cosine-BY | 0.00 | 0.00% | 0.00% | 0.00% | 0.00% |
| WebQA | Jina-v4 cosine + query conformal | 19.07 | 5.85% | 88.65% | 85.71% | 62.40% |
| WebQA | Attention-only | 22.21 | 5.57% | 98.25% | 97.80% | 71.20% |
| WebQA | Qwen internal-only | 7.41 | 14.69% | 86.46% | 84.07% | 61.20% |
| WebQA | Jina-m0 reranker-only | 7.29 | 15.38% | 89.08% | 87.36% | 63.60% |
| WebQA | Jina-m0 + Qwen internal | 5.35 | 20.66% | 87.77% | 84.62% | 61.60% |

## Original cosine-BY failure

The original proposal computes empirical p-values from calibration false scores, applies candidate-wise Benjamini-Yekutieli, caps the result at Top-K = 10, and permits an empty context. Its alpha = 0.1 empty-context rates are:

| Dataset | Empty context | Mean chunks, all test | Conditional support recall |
|---|---:|---:|---:|
| TAT-QA | 93.7% | 0.208 | 5.08% |
| HotpotQA | 85.1% | 0.238 | 9.04% |
| MMQA | 96.3% | 0.065 | 2.78% |
| WebQA, pooled bank | 100.0% | 0.000 | 0.00% |
| WebQA, modality bank | 97.6% | 0.052 | 0.87% |

Full-corpus retrieval and Top-L = 30 give high retrieval ceilings on the first three datasets, but BY still discards nearly every candidate. Increasing L alone therefore does not repair this failure. The false-score bank and candidate-wise multiple-testing rule make the selection too conservative.

## Findings

1. Query-level conformal calibration restores recall, but raw cosine needs 8-19 chunks per retrievable query at alpha = 0.1.
2. Attention-only also restores recall by retaining 14-22 chunks. Its precision is only 4.8-12.4%, so attention magnitude is not an effective standalone chunk-pruning score. WebQA makes this especially clear: position-controlled attention reaches 98.25% support recall but retains 22.21 of 30 candidates at 5.57% precision.
3. Jina-m0 reranker-only is the strongest practical selector. It retains about three chunks on TAT-QA, HotpotQA, and MMQA with 92-94% support recall. WebQA is harder and needs 7.29 chunks for 89.08% recall.
4. Qwen hidden states contain a reproducible relevance signal. TAT-QA, HotpotQA, and MMQA select decoder layer 19; WebQA selects layer 15. Internal-only improves over cosine on TAT-QA and HotpotQA, but does not generalize as a replacement on MMQA or WebQA.
5. Adding internal state to Jina-m0 changes recall by +0.39 percentage points on TAT-QA, +0.05 on HotpotQA, +0.17 on MMQA, and -1.31 on WebQA. These results do not establish a broad recall improvement over the reranker.
6. WebQA LM-head-only reaches 97.38% conditional recall, but retains 19.90 chunks with 6.16% precision. This signal may help detect uncertainty or trigger escalation, but it is not a precise standalone ranker.

The current evidence supports replacing candidate-wise cosine-BY with a stronger learned ranking score followed by query-level conformal calibration. The internal state is scientifically interesting and useful for analysis, but the current linear fusion is not a substantially better general-purpose pruning method than Jina-m0.

## Protocol caveats

The original cosine-BY row and the query-level conformal rows implement different selection guarantees. The BY method can emit an empty set; query-level conformal uses a Top-1 fallback. They are shown together to expose the failure mode, not as identical conformal procedures.

Some Qwen attention traces contain non-finite values. The evaluator counts those queries as tied zero scores rather than dropping them. The invalid-trace counts are 547/1,000 test queries for TAT-QA, 205/1,000 for HotpotQA, 190/1,000 for MMQA, and 32/250 for WebQA.

## Result files

- `{tatqa,hotpotqa,mmqa,webqa}_cosine_by_report.json`: original candidate-wise cosine-BY.
- `{tatqa,hotpotqa,mmqa,webqa}_ablation_report.json`: cosine, LM-head, hidden probe, internal-only, Jina-m0, and fusion results.
- `{tatqa,hotpotqa,mmqa,webqa}_attention_only_report.json`: completed attention-only results.
- `{dataset}_ablation_predictions.npz`: per-query scores and retained masks.
- `features/{dataset}/{probe_train,calibration,test}/features.npz`: Qwen internal feature arrays.
- `attention/{dataset}/{role}/queries.jsonl`: raw attention extraction for completed roles.

Environment: NVIDIA A100-PCIE-40GB, PyTorch 2.7.1 with CUDA 11.8 runtime. The existing working environment was preserved.
