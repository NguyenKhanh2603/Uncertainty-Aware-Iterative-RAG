# Alpha-free query-level cosine operating points: 1,000 calibration / 100 test

This is **not a conformal method**. The two new selectors choose a Jina cosine threshold from calibration labels, freeze it, and apply it to the disjoint test qids without receiving `alpha` at inference. The price is that neither selector has the query-level coverage guarantee of the alpha=.10 conformal reference.

- Frozen split manifests: [`research/internal_state_rag/results/all_datasets_cosine_six_methods_1000cal_100test_2026_09_26/splits`](../all_datasets_cosine_six_methods_1000cal_100test_2026_09_26/splits/)
- Candidate pool: frozen Jina cosine Top-30; per-query z-normalisation; deterministic Top-1 fallback.
- `F1-selected`: maximises micro evidence F1 on the 1,000 calibration qids. `Budget-10`: maximises conditional all-support coverage while calibration mean context is at most ten chunks.

## hotpotqa (calibration=1000, test=100)

| Method | Threshold | Chunks | Precision | Recall | Evidence F1 | Empty | Any support* | All support* |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Query-level conformal cosine (α=.10 reference) | conformal | 10.60 | 15.1% | 93.0% | 0.260 | 0.0% | 99.0% | 87.8% |
| Query-level cosine F1-selected (no α at test) | 2.0950 | 1.62 | 75.3% | 70.9% | 0.731 | 0.0% | 90.8% | 53.1% |
| Query-level cosine budget-10 all-support (no α at test) | -0.0311 | 9.89 | 16.2% | 93.0% | 0.276 | 0.0% | 99.0% | 87.8% |

*Any/all-support are conditional on at least one labelled support in Top-30; the test labels are used only for this evaluation.*

## mmqa (calibration=1000, test=100)

| Method | Threshold | Chunks | Precision | Recall | Evidence F1 | Empty | Any support* | All support* |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Query-level conformal cosine (α=.10 reference) | conformal | 9.05 | 12.0% | 96.5% | 0.214 | 0.0% | 100.0% | 95.7% |
| Query-level cosine F1-selected (no α at test) | 2.9657 | 1.04 | 71.2% | 65.5% | 0.682 | 0.0% | 79.6% | 62.4% |
| Query-level cosine budget-10 all-support (no α at test) | -0.0006 | 9.63 | 11.3% | 96.5% | 0.203 | 0.0% | 100.0% | 95.7% |

*Any/all-support are conditional on at least one labelled support in Top-30; the test labels are used only for this evaluation.*

## tatqa (calibration=1000, test=100)

| Method | Threshold | Chunks | Precision | Recall | Evidence F1 | Empty | Any support* | All support* |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Query-level conformal cosine (α=.10 reference) | conformal | 11.30 | 8.1% | 87.6% | 0.149 | 0.0% | 90.1% | 85.7% |
| Query-level cosine F1-selected (no α at test) | 2.7424 | 1.08 | 46.3% | 47.6% | 0.469 | 0.0% | 54.9% | 46.2% |
| Query-level cosine budget-10 all-support (no α at test) | 0.0975 | 10.05 | 9.0% | 85.7% | 0.162 | 0.0% | 89.0% | 83.5% |

*Any/all-support are conditional on at least one labelled support in Top-30; the test labels are used only for this evaluation.*

## webqa (calibration=1000, test=100)

| Method | Threshold | Chunks | Precision | Recall | Evidence F1 | Empty | Any support* | All support* |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Query-level conformal cosine (α=.10 reference) | conformal | 17.63 | 4.4% | 90.6% | 0.083 | 0.0% | 94.3% | 88.6% |
| Query-level cosine F1-selected (no α at test) | 2.1064 | 1.59 | 11.3% | 21.2% | 0.148 | 0.0% | 25.7% | 20.0% |
| Query-level cosine budget-10 all-support (no α at test) | 0.1224 | 10.18 | 6.0% | 71.8% | 0.111 | 0.0% | 77.1% | 67.1% |

*Any/all-support are conditional on at least one labelled support in Top-30; the test labels are used only for this evaluation.*

## Interpretation

An alpha-free row is an empirically selected utility operating point, not a confidence procedure. It is appropriate when the product objective is fixed (for example, maximum retrieval F1 or a ten-chunk context budget). Use the conformal row when the stated objective is a pre-declared all-support error level.
