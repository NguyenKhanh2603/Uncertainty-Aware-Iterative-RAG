# Downstream QA: BY cosine, CCE, CONFLARE, and TRAQ retrieval components

## Protocol

- Generator: Qwen2-VL-7B-Instruct, greedy decoding, 24 new tokens.
- Retrieval candidates: frozen Jina-v4 cosine Top-30.
- Split: the same frozen 100-query calibration plan and 100-query test plan
  used by the retrieval benchmark for each dataset.
- `BY cosine`: false-score-bank p-values followed by candidate-wise
  Benjamini-Yekutieli at alpha=0.10.
- `CCE`: positive-support cosine split-conformal threshold.
- `CONFLARE`: official percentile-style retrieval threshold, with gold
  calibration supports in place of its synthetic LLM calibration questions.
- `TRAQ retrieval`: its alpha/2 retrieval threshold. This is **not** full
  TRAQ answer-set conformal prediction, which additionally needs calibrated
  generation samples and semantic-answer sets.
- MMQA images use the feature-extraction resolution contract: 3,136 to
  200,704 pixels.  Results use exact match (EM), token F1, and numerical
  accuracy, each averaged over the 100 test questions.

## Results at alpha=0.10

| Dataset | Method | EM | F1 | Numerical accuracy | Mean chunks | Empty contexts |
|---|---|---:|---:|---:|---:|---:|
| HotpotQA | Full Top-30 | 0.370 | 0.541 | 0.390 | 30.00 | 0% |
|  | BY cosine | 0.150 | 0.263 | 0.170 | 0.13 | 97% |
|  | CCE | **0.430** | **0.580** | **0.450** | 11.19 | 1% |
|  | CONFLARE adapter | 0.410 | 0.558 | 0.430 | 10.80 | 1% |
|  | TRAQ retrieval adapter | 0.390 | 0.542 | 0.420 | 14.97 | 0% |
| MMQA | Full Top-30 | 0.470 | 0.508 | 0.490 | 30.00 | 0% |
|  | BY cosine | 0.160 | 0.196 | 0.170 | 0.05 | 95% |
|  | CCE | **0.500** | **0.548** | **0.520** | 14.19 | 3% |
|  | CONFLARE adapter | 0.490 | 0.531 | 0.510 | 13.02 | 3% |
|  | TRAQ retrieval adapter | 0.480 | 0.531 | 0.500 | 20.43 | 1% |
| TAT-QA | Full Top-30 | 0.160 | 0.264 | 0.240 | 30.00 | 0% |
|  | BY cosine | 0.040 | 0.116 | 0.120 | 0.85 | 86% |
|  | CCE | 0.160 | 0.263 | 0.240 | 20.10 | 5% |
|  | CONFLARE adapter | 0.160 | **0.268** | **0.250** | 20.02 | 5% |
|  | TRAQ retrieval adapter | 0.150 | 0.256 | 0.230 | 24.08 | 3% |

The equally weighted three-dataset mean is: Full Top-30 EM 0.333/F1 0.438;
BY cosine EM 0.117/F1 0.192; CCE EM 0.363/F1 0.463; CONFLARE adapter
EM 0.353/F1 0.453; TRAQ retrieval adapter EM 0.340/F1 0.443.

## Interpretation

Candidate-wise BY cosine is far too conservative with this false-score bank:
it returns an empty context for 86--97% of test queries and its answer EM
collapses on every dataset.  This confirms the retrieval-level high-precision,
low-recall diagnosis in a downstream metric.

CCE and the controlled CONFLARE retrieval adapter prune 53--64% of HotpotQA
and MMQA candidates while preserving or improving downstream QA.  On TAT-QA,
they reduce context by about one third without a material EM change.  TRAQ's
retrieval allocation retains more context and is correspondingly less brittle,
but it does not beat CCE on this controlled QA comparison.

These downstream results compare retrieval selectors only.  They should not
be presented as an end-to-end reproduction of TRAQ or CONFLARE, because their
generation/synthetic-calibration stages are deliberately held out to keep the
candidate score and gold calibration data matched.

## Source artifacts

- HotpotQA: `downstream_conformal_baselines_a10_fixedcal/hotpotqa_summary.json`
  and `hotpotqa_predictions.jsonl`.
- MMQA: `downstream_conformal_baselines_a10_fixedcal_px200704/mmqa_summary.json`
  and `mmqa_predictions.jsonl`.
- TAT-QA: `downstream_conformal_baselines_a10_fixedcal_px200704/tatqa_summary.json`
  and `tatqa_predictions.jsonl`.
- Runner: `research/internal_state_rag/run_downstream_qa_conformal_baselines.py`.

WebQA is not included: this workspace has its cached selector features but
does not contain the matching raw questions and corpus required to generate
and score answers.  No WebQA EM has been inferred from retrieval labels.
