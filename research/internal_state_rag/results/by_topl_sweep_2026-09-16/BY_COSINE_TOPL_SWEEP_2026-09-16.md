# Candidate-wise BY cosine: candidate-pool-size sweep

**Date:** 2026-09-16
**Question:** Does increasing the retrieval reserve `L` improve the original candidate-wise cosine conformal + Benjamini--Yekutieli (BY) selector?

## Protocol

This is the **old candidate-level method**, not query-level conformal support:

1. Take the first `L` candidates from the same frozen Jina-v4 global-corpus Top-30 retrieval logs.
2. Rebuild the calibration false-score bank using only that same Top-`L` prefix.
3. Calculate candidate `p`-values from cosine scores, apply BY independently within each query, then retain selected candidates in cosine-rank order.

The sweep uses `L in {5, 10, 20, 30}`, alpha in `{0.05, 0.10, 0.20, 0.50}`, pooled false-score banks, and four datasets. `K=10` is the original context cap. `K=30` is a cap-removal diagnostic; it does not alter the BY correction.

The `L<=30` rows reuse the frozen Top-30 logs. To test a real expansion rather than only a prefix, TAT-QA, HotpotQA, and MMQA were also re-retrieved against their full corpus at Top-50 using the same cached Jina-v4 embeddings; those rows rebuild their calibration bank from the fresh Top-50 logs.

## Main result: alpha = 0.10, original K = 10

Values are on the official held-out test role. `Ceiling` is the percentage of test queries with at least one support in the candidate prefix. Recall is conditional micro support recall. `All E2E` requires every in-pool support to be selected and counts retrieval misses as failures.

| Dataset | L | Ceiling | Chunks kept | Precision | Recall | Empty | All E2E |
|---|---:|---:|---:|---:|---:|---:|---:|
| TAT-QA | 5 | 73.3% | 0.196 | 30.61% | 7.69% | 91.8% | 5.6% |
|  | 10 | 80.6% | 0.263 | 22.05% | 6.61% | 92.7% | 5.0% |
|  | 20 | 88.2% | 0.226 | 23.45% | 5.44% | 93.6% | 4.5% |
|  | 30 | 91.9% | 0.208 | 25.00% | 5.08% | 93.7% | 4.4% |
| HotpotQA | 5 | 98.3% | 0.262 | 69.08% | 11.32% | 83.8% | 7.3% |
|  | 10 | 98.9% | 0.277 | 64.26% | 10.41% | 84.1% | 5.1% |
|  | 20 | 99.4% | 0.257 | 65.76% | 9.42% | 84.7% | 3.0% |
|  | 30 | 99.6% | 0.238 | 69.33% | 9.04% | 85.1% | 2.6% |
| MMQA | 5 | 91.3% | 0.091 | 56.04% | 4.87% | 94.5% | 4.0% |
|  | 10 | 93.1% | 0.093 | 52.69% | 4.44% | 94.7% | 3.6% |
|  | 20 | 94.3% | 0.072 | 50.00% | 3.10% | 96.0% | 2.3% |
|  | 30 | 94.8% | 0.065 | 50.77% | 2.78% | 96.3% | 1.9% |
| WebQA | 5 | 40.8% | 0.020 | 20.00% | 0.86% | 99.6% | 0.4% |
|  | 10 | 54.8% | 0.016 | 0.00% | 0.00% | 99.6% | 0.0% |
|  | 20 | 66.4% | 0.016 | 0.00% | 0.00% | 99.6% | 0.0% |
|  | 30 | 72.8% | 0.000 | -- | 0.00% | 100.0% | 0.0% |

Increasing `L` raises the retrieval ceiling on every dataset, especially TAT-QA and WebQA. It nevertheless **lowers BY support recall** almost monotonically from `L=5` to `L=30`. The reason is structural: each extra candidate adds a correlated hypothesis to BY, while the additional lower-ranked candidates also populate the false-score bank. The very small gain in available support is outweighed by the stricter candidate-level discovery threshold.

## Fresh full-corpus Top-50 check: alpha = .10, K = 10

| Dataset | L=30: ceiling / kept / precision / recall / empty | L=50: ceiling / kept / precision / recall / empty |
|---|---|---|
| TAT-QA | 91.9% / 0.208 / 25.00% / 5.08% / 93.7% | 94.2% / 0.199 / 26.13% / 4.90% / 93.8% |
| HotpotQA | 99.6% / 0.238 / 69.33% / 9.04% / 85.1% | 99.7% / 0.205 / 69.27% / 7.63% / 86.9% |
| MMQA | 94.8% / 0.065 / 50.77% / 2.78% / 96.3% | 95.6% / 0.057 / 52.63% / 2.46% / 96.7% |

This confirms the prefix sweep with an actual larger candidate pool: Top-50 raises retrieval availability by `0.1--2.3` points, but the original BY selector keeps fewer chunks and loses recall on all three datasets. The expanded retrieval was ranked on CPU because the current `uv` environment's PyTorch build rejects this host's CUDA 12.0 driver; it reuses the exact frozen embeddings and does not change scores or the evaluation protocol.

## Alpha relaxation at L = 30, K = 10

Cell format: `chunks kept / precision / conditional recall / empty rate`.

| Dataset | alpha=.05 | alpha=.10 | alpha=.20 | alpha=.50 |
|---|---|---|---|---|
| TAT-QA | 0.132 / 24.2% / 3.1% / 95.9% | 0.208 / 25.0% / 5.1% / 93.7% | 0.354 / 20.3% / 7.0% / 90.6% | 0.774 / 15.6% / 11.8% / 84.4% |
| HotpotQA | 0.118 / 74.6% / 4.8% / 91.8% | 0.238 / 69.3% / 9.0% / 85.1% | 0.474 / 57.6% / 15.0% / 77.5% | 1.060 / 49.5% / 28.8% / 58.5% |
| MMQA | 0.046 / 60.9% / 2.4% / 97.1% | 0.065 / 50.8% / 2.8% / 96.3% | 0.150 / 46.0% / 5.8% / 92.7% | 0.420 / 37.1% / 13.1% / 83.9% |
| WebQA | 0.000 / -- / 0.0% / 100.0% | 0.000 / -- / 0.0% / 100.0% | 0.064 / 12.5% / 0.9% / 98.8% | 0.432 / 4.6% / 2.2% / 94.0% |

Loosening alpha does produce more selections, but even alpha `.50` leaves low recall and degrades precision substantially. It is not a credible route to high evidence recall under the original candidate-wise formulation.

## Is the K=10 cap the bottleneck?

At `L=30, alpha=.50`, removing the cap (`K=30`) changes `chunks / recall / precision` as follows:

| Dataset | K=10 | K=30 | Interpretation |
|---|---|---|---|
| TAT-QA | 0.774 / 11.83% / 15.63% | 1.922 / 13.78% / 7.34% | Extra selections mostly add false chunks; recall rises only 1.95 pp. |
| HotpotQA | 1.060 / 28.77% / 49.53% | 1.411 / 28.93% / 37.42% | Cap is nearly irrelevant for recall. |
| MMQA | 0.420 / 13.14% / 37.14% | 0.631 / 13.23% / 24.88% | Cap is nearly irrelevant for recall. |
| WebQA | 0.432 / 2.18% / 4.63% | 1.000 / 3.06% / 2.80% | No useful recovery. |

The cap is not the primary cause of low recall. BY produces too few discoveries before the cap applies.

## Conclusion

For the old BY cosine selector, increasing the pool from Top-5 to Top-30 improves retrieval availability but makes selection worse at a fixed alpha. Raising alpha or removing `K` recovers only modest recall while giving up precision. This supports moving to query-level conformal support for the pruning stage: it calibrates the error event “a query loses evidence” once per query instead of treating all `L` chunks as simultaneous discoveries.

## Reproducibility artifacts

- Evaluator: [`analyze_current_cosine_by.py`](../../analyze_current_cosine_by.py) (the `--top-l` option reconstructs the false bank and BY decision for each prefix).
- Raw output: one JSON for every `(dataset, L, K)` combination in this directory, e.g. [`tatqa_L30_K10.json`](tatqa_L30_K10.json), [`hotpotqa_L30_K10.json`](hotpotqa_L30_K10.json), [`mmqa_L30_K10.json`](mmqa_L30_K10.json), [`webqa_L30_K10.json`](webqa_L30_K10.json), and fresh Top-50 outputs [`tatqa_L50_K10.json`](tatqa_L50_K10.json), [`hotpotqa_L50_K10.json`](hotpotqa_L50_K10.json), [`mmqa_L50_K10.json`](mmqa_L50_K10.json).
- Fresh Top-50 retrieval logs and manifests: [`top50_logs/`](top50_logs/).
- Frozen input logs: [`TAT-QA`](../qwen2vl_jina4/tatqa_jina_v4_top30.jsonl.gz), [`HotpotQA`](../qwen2vl_jina4/hotpotqa_jina_v4_top30.jsonl.gz), [`MMQA`](../qwen2vl_jina4/mmqa_jina_v4_top30.jsonl.gz), [`WebQA`](../qwen2vl_jina4/webqa_jina_v4_top30.jsonl.gz).
