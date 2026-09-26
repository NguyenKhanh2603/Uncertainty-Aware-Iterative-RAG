# Downstream diagnostic: alpha-free query-level cosine

The new rows use a shared deterministic Qwen direct-answer diagnostic over the exact 100 held-out qids. The α=.10 reference is reused from the completed literature-protocol pass only after exact qid validation; its selected context is identical. New predictions are generated only for the two alpha-free policies.

- Reference predictions: `research/internal_state_rag/results/literature_protocol_1000cal_100test_2026_09_26`
- Selection details and frozen thresholds: [`REPORT.md`](REPORT.md)
- This is a task metric, not a conformal coverage guarantee.

## hotpotqa (complete; calibration=1000, test=100)

| Method | Chunks | Evidence F1 | EM | Token F1 | Numeric |
|---|---:|---:|---:|---:|---:|
| Query-level conformal cosine (α=.10 reference) | 10.60 | 0.260 | 0.420 | 0.537 | 0.440 |
| Query-level cosine F1-selected (no α at test) | 1.62 | 0.731 | 0.430 | 0.534 | 0.440 |
| Query-level cosine budget-10 all-support (no α at test) | 9.89 | 0.276 | 0.400 | 0.508 | 0.420 |

## mmqa (complete; calibration=1000, test=100)

| Method | Chunks | Evidence F1 | EM | Token F1 | Numeric |
|---|---:|---:|---:|---:|---:|
| Query-level conformal cosine (α=.10 reference) | 9.05 | 0.214 | 0.510 | 0.552 | 0.530 |
| Query-level cosine F1-selected (no α at test) | 1.04 | 0.682 | 0.490 | 0.523 | 0.510 |
| Query-level cosine budget-10 all-support (no α at test) | 9.63 | 0.203 | 0.510 | 0.551 | 0.530 |

## tatqa (complete; calibration=1000, test=100)

| Method | Chunks | Evidence F1 | EM | Token F1 | Numeric |
|---|---:|---:|---:|---:|---:|
| Query-level conformal cosine (α=.10 reference) | 11.30 | 0.149 | 0.180 | 0.292 | 0.260 |
| Query-level cosine F1-selected (no α at test) | 1.08 | 0.469 | 0.130 | 0.234 | 0.220 |
| Query-level cosine budget-10 all-support (no α at test) | 10.05 | 0.162 | 0.160 | 0.275 | 0.250 |

## webqa (complete; calibration=1000, test=100)

| Method | Chunks | Evidence F1 | EM | Token F1 | Numeric |
|---|---:|---:|---:|---:|---:|
| Query-level conformal cosine (α=.10 reference) | 17.63 | 0.083 | 0.000 | 0.125 | 0.050 |
| Query-level cosine F1-selected (no α at test) | 1.59 | 0.148 | 0.000 | 0.115 | 0.030 |
| Query-level cosine budget-10 all-support (no α at test) | 10.18 | 0.111 | 0.000 | 0.120 | 0.050 |
