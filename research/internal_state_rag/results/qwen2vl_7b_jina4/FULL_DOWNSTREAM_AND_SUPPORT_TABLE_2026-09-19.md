# Full downstream and support-retention comparison

All methods use the same frozen Jina-v4 Top-30 candidates and the same 100
calibration / 100 test queries per dataset at alpha=0.10.  The downstream
generator is Qwen2-VL-7B with greedy decoding.

`P` is chunk precision among retained chunks. `Support R` is micro support
recall conditional on a support existing in Top-30. `Q-any` and `Q-all` are,
respectively, the fraction of those retrievable queries retaining at least one
or all labelled support chunks. `Empty` is over all 100 test queries. `Ctx
reduction` is relative to Top-30. All support statistics are from exactly the
selector used for answer generation.

## HotpotQA

| Method | EM | F1 | Num. acc. | Chunks | Ctx reduction | Empty | P | Support R | Q-any | Q-all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full Top-30 | 0.370 | 0.541 | 0.390 | 30.00 | 0.0% | 0.0% | 6.1% | 100.0% | 100.0% | 100.0% |
| BY cosine | 0.150 | 0.263 | 0.170 | 0.13 | 99.6% | 97.0% | 46.2% | 3.3% | 3.0% | 3.0% |
| CCE cosine | **0.430** | **0.580** | **0.450** | 11.19 | 62.7% | 1.0% | 14.9% | 90.8% | 99.0% | 84.0% |
| CONFLARE cosine adapter | 0.410 | 0.558 | 0.430 | 10.80 | 64.0% | 1.0% | 15.5% | 90.8% | 99.0% | 84.0% |
| TRAQ cosine retrieval adapter | 0.390 | 0.542 | 0.420 | 14.97 | 50.1% | 0.0% | 11.4% | 92.9% | 100.0% | 87.0% |
| Query-level cosine + internal fusion | 0.400 | 0.564 | 0.410 | **4.08** | **86.4%** | 0.0% | **44.4%** | **98.4%** | **100.0%** | **97.0%** |

## MMQA

| Method | EM | F1 | Num. acc. | Chunks | Ctx reduction | Empty | P | Support R | Q-any | Q-all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full Top-30 | 0.470 | 0.508 | 0.490 | 30.00 | 0.0% | 0.0% | 3.8% | 100.0% | 100.0% | 100.0% |
| BY cosine | 0.160 | 0.196 | 0.170 | 0.05 | 99.8% | 95.0% | **80.0%** | 3.5% | 4.3% | 2.2% |
| CCE cosine | **0.500** | **0.548** | **0.520** | 14.19 | 52.7% | 3.0% | 7.4% | 92.9% | 96.8% | 91.4% |
| CONFLARE cosine adapter | 0.490 | 0.531 | 0.510 | 13.02 | 56.6% | 3.0% | 8.0% | 92.0% | 96.8% | 90.3% |
| TRAQ cosine retrieval adapter | 0.480 | 0.531 | 0.500 | 20.43 | 31.9% | 1.0% | 5.4% | 97.3% | 100.0% | 96.8% |
| Query-level cosine + internal fusion | **0.500** | 0.544 | **0.520** | **4.75** | **84.2%** | 0.0% | 25.2% | **98.2%** | **100.0%** | **97.8%** |

## TAT-QA

| Method | EM | F1 | Num. acc. | Chunks | Ctx reduction | Empty | P | Support R | Q-any | Q-all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full Top-30 | 0.160 | 0.264 | 0.240 | 30.00 | 0.0% | 0.0% | 3.5% | 100.0% | 100.0% | 100.0% |
| BY cosine | 0.040 | 0.116 | 0.120 | 0.85 | 97.2% | 86.0% | 14.1% | 11.4% | 13.2% | 9.9% |
| CCE cosine | **0.160** | 0.263 | 0.240 | 20.10 | 33.0% | 5.0% | 4.7% | 89.5% | 91.2% | 89.0% |
| CONFLARE cosine adapter | **0.160** | **0.268** | 0.250 | 20.02 | 33.3% | 5.0% | 4.7% | 89.5% | 91.2% | 89.0% |
| TRAQ cosine retrieval adapter | 0.150 | 0.256 | 0.230 | 24.08 | 19.7% | 3.0% | 4.2% | **95.2%** | 94.5% | **94.5%** |
| Query-level cosine + internal fusion | **0.160** | 0.262 | **0.260** | **3.49** | **88.4%** | 0.0% | **29.9%** | 90.5% | **96.7%** | 89.0% |

## Reading the trade-off

BY cosine has high precision only because it almost always rejects every
candidate.  Its empty-context rate causes both support recall and answer
metrics to collapse.

The query-level fusion selector is the strongest compact-context option.  It
keeps 3.49--4.75 chunks/query, maintains 90.5--98.4% conditional support
recall, and matches CCE's EM on MMQA and TAT-QA.  CCE remains best on HotpotQA
EM/F1, but needs 11.19 chunks rather than 4.08.

CCE, CONFLARE, and TRAQ rows are retrieval-component comparisons sharing
cosine.  TRAQ does not include its complete answer-set conformal stage, and
CONFLARE replaces synthetic LLM questions with gold calibration supports.
