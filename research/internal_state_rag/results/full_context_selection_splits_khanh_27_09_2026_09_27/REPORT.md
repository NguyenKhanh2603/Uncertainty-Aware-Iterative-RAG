# Full context-selection comparison — `splits_khanh_27_09`

This is a new, split-specific rerun. It uses the qid manifests from `splits_khanh_27_09.zip` (SHA-256 `c9d4ea29fea7ddee5b8eabe7ddbf8d019ff990c371710d660fb088e178f34580`): 1,000 calibration qids and 100 held-out test qids per dataset, with zero qid overlap. Every evaluated selector receives the frozen Jina-v4 dataset-provided candidate pool, capped at Top-L=30, and is scored by the shared greedy Qwen2-VL-7B direct-answer diagnostic. Test-pool sizes are ragged: hotpotqa: 2–10 candidates/query (mean 9.85); mmqa: 12–26 candidates/query (mean 21.87); tatqa: 3–22 candidates/query (mean 6.13); webqa: 8–30 candidates/query (mean 28.11).

Only the three literature retrieval adaptations below were run on this split: CCE Conformal-Embedding, CONFLARE source-question, and TRAQ retrieval Bonferroni. Every other row is deliberately `pending`; no value from an earlier split is copied into this report.

## Metrics

Chunks is mean retained chunks/query. Precision and Recall are micro support metrics in each frozen candidate pool. Empty is the fraction of queries that retained no chunk. Any/All are conditional support-retention rates among queries with at least one labelled support in the candidate pool. EM, token F1, and Numeric are Qwen direct-answer measurements on the 100 held-out queries; they are not conformal guarantees.

## hotpotqa (calibration=1,000; test=100; split=`splits_khanh_27_09`)

| Family | Method | Chunks | Precision | Recall | Empty | Any support | All support | EM | F1 | Numeric |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed | Fixed Top-10 | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Fixed | Fixed Top-20 | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Legacy cosine proxy | CCE-style global positive-cosine proxy (α=.10) | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Legacy cosine proxy | CONFLARE-style global positive-cosine proxy (α=.10) | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Legacy cosine proxy | TRAQ-style loose positive-cosine proxy (α=.10) | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Literature | CCE Conformal-Embedding, Jina adaptation (α=.10; identical to CCE-style proxy) | 8.61 | 21.0% | 90.5% | 2.0% | 97.0% | 84.0% | 0.490 | 0.653 | 0.570 |
| Literature | CONFLARE source-question, Jina adaptation (α=.10) | 7.05 | 21.0% | 74.0% | 3.0% | 88.0% | 60.0% | 0.460 | 0.623 | 0.540 |
| Literature | TRAQ retrieval Bonferroni, Jina adaptation (α=.10; α_R=.05) | 8.17 | 21.4% | 87.5% | 2.0% | 96.0% | 79.0% | 0.490 | 0.642 | 0.570 |
| Original cosine | Query-level cosine all-support (α=.10; 1,000 cal) | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Original cosine | BY cosine (α=.10) | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Original cosine | BY cosine (α=.30) | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Original cosine | BY cosine (α=.50) | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Original cosine | BH cosine, ctx=10 (α=.10) | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Original cosine | BH cosine, ctx=10 (α=.30) | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Original cosine | BH cosine, ctx=10 (α=.50) | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Original cosine | BH cosine, ctx=10 (α=.90) | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Original cosine | BH cosine, ctx=10 (α=.99) | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Original cosine | BH cosine, ctx=20 (α=.99) | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Internal Qwen-7B | Cosine baseline (500 probe / 500 conformal cal; α=.10) | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Internal Qwen-7B | LM-head only (500 / 500; α=.10) | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Internal Qwen-7B | Hidden-state relevance probe only (500 / 500; α=.10) | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Internal Qwen-7B | LM-head + hidden-state probe (500 / 500; α=.10) | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Internal Qwen-7B | Cosine + LM-head + hidden-state probe (500 / 500; α=.10) | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Alpha-free | Query-level cosine F1-selected (1,000 cal; no α at test) | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Alpha-free | Query-level cosine budget-10 all-support (1,000 cal; no α at test) | pending | pending | pending | pending | pending | pending | pending | pending | pending |

## mmqa (calibration=1,000; test=100; split=`splits_khanh_27_09`)

| Family | Method | Chunks | Precision | Recall | Empty | Any support | All support | EM | F1 | Numeric |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed | Fixed Top-10 | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Fixed | Fixed Top-20 | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Legacy cosine proxy | CCE-style global positive-cosine proxy (α=.10) | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Legacy cosine proxy | CONFLARE-style global positive-cosine proxy (α=.10) | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Legacy cosine proxy | TRAQ-style loose positive-cosine proxy (α=.10) | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Literature | CCE Conformal-Embedding, Jina adaptation (α=.10; identical to CCE-style proxy) | 18.40 | 7.7% | 91.0% | 2.0% | 92.0% | 87.0% | 0.380 | 0.436 | 0.410 |
| Literature | CONFLARE source-question, Jina adaptation (α=.10) | 18.05 | 7.8% | 90.3% | 2.0% | 91.0% | 86.0% | 0.400 | 0.462 | 0.430 |
| Literature | TRAQ retrieval Bonferroni, Jina adaptation (α=.10; α_R=.05) | 19.50 | 7.5% | 94.2% | 2.0% | 94.0% | 91.0% | 0.390 | 0.435 | 0.400 |
| Original cosine | Query-level cosine all-support (α=.10; 1,000 cal) | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Original cosine | BY cosine (α=.10) | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Original cosine | BY cosine (α=.30) | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Original cosine | BY cosine (α=.50) | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Original cosine | BH cosine, ctx=10 (α=.10) | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Original cosine | BH cosine, ctx=10 (α=.30) | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Original cosine | BH cosine, ctx=10 (α=.50) | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Original cosine | BH cosine, ctx=10 (α=.90) | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Original cosine | BH cosine, ctx=10 (α=.99) | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Original cosine | BH cosine, ctx=20 (α=.99) | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Internal Qwen-7B | Cosine baseline (500 probe / 500 conformal cal; α=.10) | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Internal Qwen-7B | LM-head only (500 / 500; α=.10) | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Internal Qwen-7B | Hidden-state relevance probe only (500 / 500; α=.10) | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Internal Qwen-7B | LM-head + hidden-state probe (500 / 500; α=.10) | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Internal Qwen-7B | Cosine + LM-head + hidden-state probe (500 / 500; α=.10) | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Alpha-free | Query-level cosine F1-selected (1,000 cal; no α at test) | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Alpha-free | Query-level cosine budget-10 all-support (1,000 cal; no α at test) | pending | pending | pending | pending | pending | pending | pending | pending | pending |

## tatqa (calibration=1,000; test=100; split=`splits_khanh_27_09`)

| Family | Method | Chunks | Precision | Recall | Empty | Any support | All support | EM | F1 | Numeric |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed | Fixed Top-10 | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Fixed | Fixed Top-20 | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Legacy cosine proxy | CCE-style global positive-cosine proxy (α=.10) | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Legacy cosine proxy | CONFLARE-style global positive-cosine proxy (α=.10) | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Legacy cosine proxy | TRAQ-style loose positive-cosine proxy (α=.10) | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Literature | CCE Conformal-Embedding, Jina adaptation (α=.10; identical to CCE-style proxy) | 5.01 | 24.6% | 91.8% | 1.0% | 94.0% | 89.0% | 0.310 | 0.490 | 0.420 |
| Literature | CONFLARE source-question, Jina adaptation (α=.10) | 4.64 | 25.0% | 86.6% | 2.0% | 90.0% | 82.0% | 0.260 | 0.432 | 0.380 |
| Literature | TRAQ retrieval Bonferroni, Jina adaptation (α=.10; α_R=.05) | 5.09 | 24.6% | 93.3% | 1.0% | 95.0% | 91.0% | 0.320 | 0.500 | 0.430 |
| Original cosine | Query-level cosine all-support (α=.10; 1,000 cal) | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Original cosine | BY cosine (α=.10) | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Original cosine | BY cosine (α=.30) | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Original cosine | BY cosine (α=.50) | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Original cosine | BH cosine, ctx=10 (α=.10) | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Original cosine | BH cosine, ctx=10 (α=.30) | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Original cosine | BH cosine, ctx=10 (α=.50) | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Original cosine | BH cosine, ctx=10 (α=.90) | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Original cosine | BH cosine, ctx=10 (α=.99) | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Original cosine | BH cosine, ctx=20 (α=.99) | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Internal Qwen-7B | Cosine baseline (500 probe / 500 conformal cal; α=.10) | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Internal Qwen-7B | LM-head only (500 / 500; α=.10) | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Internal Qwen-7B | Hidden-state relevance probe only (500 / 500; α=.10) | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Internal Qwen-7B | LM-head + hidden-state probe (500 / 500; α=.10) | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Internal Qwen-7B | Cosine + LM-head + hidden-state probe (500 / 500; α=.10) | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Alpha-free | Query-level cosine F1-selected (1,000 cal; no α at test) | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Alpha-free | Query-level cosine budget-10 all-support (1,000 cal; no α at test) | pending | pending | pending | pending | pending | pending | pending | pending | pending |

## webqa (calibration=1,000; test=100; split=`splits_khanh_27_09`)

| Family | Method | Chunks | Precision | Recall | Empty | Any support | All support | EM | F1 | Numeric |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed | Fixed Top-10 | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Fixed | Fixed Top-20 | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Legacy cosine proxy | CCE-style global positive-cosine proxy (α=.10) | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Legacy cosine proxy | CONFLARE-style global positive-cosine proxy (α=.10) | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Legacy cosine proxy | TRAQ-style loose positive-cosine proxy (α=.10) | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Literature | CCE Conformal-Embedding, Jina adaptation (α=.10; identical to CCE-style proxy) | 23.36 | 6.5% | 86.3% | 0.0% | 91.0% | 80.0% | 0.110 | 0.260 | 0.140 |
| Literature | CONFLARE source-question, Jina adaptation (α=.10) | 21.60 | 6.8% | 83.4% | 0.0% | 89.0% | 75.0% | 0.110 | 0.255 | 0.140 |
| Literature | TRAQ retrieval Bonferroni, Jina adaptation (α=.10; α_R=.05) | 24.68 | 6.3% | 88.6% | 0.0% | 91.0% | 84.0% | 0.110 | 0.246 | 0.140 |
| Original cosine | Query-level cosine all-support (α=.10; 1,000 cal) | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Original cosine | BY cosine (α=.10) | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Original cosine | BY cosine (α=.30) | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Original cosine | BY cosine (α=.50) | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Original cosine | BH cosine, ctx=10 (α=.10) | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Original cosine | BH cosine, ctx=10 (α=.30) | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Original cosine | BH cosine, ctx=10 (α=.50) | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Original cosine | BH cosine, ctx=10 (α=.90) | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Original cosine | BH cosine, ctx=10 (α=.99) | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Original cosine | BH cosine, ctx=20 (α=.99) | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Internal Qwen-7B | Cosine baseline (500 probe / 500 conformal cal; α=.10) | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Internal Qwen-7B | LM-head only (500 / 500; α=.10) | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Internal Qwen-7B | Hidden-state relevance probe only (500 / 500; α=.10) | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Internal Qwen-7B | LM-head + hidden-state probe (500 / 500; α=.10) | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Internal Qwen-7B | Cosine + LM-head + hidden-state probe (500 / 500; α=.10) | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Alpha-free | Query-level cosine F1-selected (1,000 cal; no α at test) | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Alpha-free | Query-level cosine budget-10 all-support (1,000 cal; no α at test) | pending | pending | pending | pending | pending | pending | pending | pending | pending |

## Split and code provenance

- Archive supplied for this rerun: `splits_khanh_27_09.zip`; its SHA-256 is recorded above.
- Materialized qid manifests, canonically formatted from the archive: [splits](splits), including one calibration and one test manifest per dataset.
- Split audit: [SPLIT_INTEGRITY.json](splits/SPLIT_INTEGRITY.json).
- Literature selector/downstream runner: [run_literature_protocol_1000cal.py](../../run_literature_protocol_1000cal.py).
- Full-table renderer: [build_splits_khanh_27_09_literature_report.py](../../build_splits_khanh_27_09_literature_report.py).
- Shared frozen-candidate configuration and data mapping: [run_all_datasets_cosine_six_methods.py](../../run_all_datasets_cosine_six_methods.py).
- Exact model, decoding, archive, and input-profile configuration: [RUN_CONFIG.json](RUN_CONFIG.json).
- Raw completed selection and QA summary: [summary.json](summary.json).

The three literature rows are matched Jina adaptations, not full end-to-end replications of the original CCE, CONFLARE, or TRAQ systems. TRAQ reports its retrieval component only; it does not report TRAQ's semantic answer-set coverage procedure.
