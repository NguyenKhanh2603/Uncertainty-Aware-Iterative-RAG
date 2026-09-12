# TATQA internal-state smoke result

## Protocol

- Model: locally cached `Qwen/Qwen2.5-3B-Instruct`, FP16, eager attention.
- Device: one NVIDIA A100 PCIe 40 GB.
- Data: 39 TATQA **calibration** questions from the official train split. Test
  questions were not used.
- Inclusion rule: exactly one gold answer and all retrieved labelled support
  occurs after rank 2 in the frozen BGE reranked top-30 log.
- Paired intervention:
  - `missing`: frozen top-2 chunks, which contain no labelled support;
  - `oracle`: the same top-2 chunks with retrieved gold-support chunks appended.
- Generation was greedy. Internal features teacher-forced the same gold answer
  under both contexts, which makes the latent comparison paired.
- Residual/logit-lens signals were collected at decoder layers
  `12,14,...,34,35`; attention remained separated by layer and head.

Raw results are in `results/tatqa_smoke_n39_seed17.json`.

## Results

| Metric | Missing support | Oracle support | Paired change |
|---|---:|---:|---:|
| Exact match | 17.95% | 20.51% | +2.56 pp |
| Token F1 | 20.98% | 24.27% | +3.29 pp |
| Numerical accuracy | 23.08% | 33.33% | +10.26 pp |

The answer-level changes are directional but uncertain at this sample size:

| Paired outcome | Improved | Same | Worse | Bootstrap 95% CI of mean change |
|---|---:|---:|---:|---:|
| Exact match | 2 | 36 | 1 | [-5.13, +10.26] pp |
| Token F1 | 7 | 30 | 2 | [-5.35, +12.54] pp |
| Numerical accuracy | 5 | 33 | 1 | [0.00, +23.08] pp |

Wilcoxon paired p-values are 0.564 for EM, 0.312 for F1, and 0.102 for
numerical accuracy. This run is a mechanism smoke test, not evidence of a final
QA improvement.

## Internal signal findings

Adding gold support increased the mean gold-token log probability in 33 of 39
questions (84.62%). The mean layer-averaged gain was +0.643 nats with a
bootstrap 95% interval of [+0.375, +0.937]. A one-sided binomial test against a
50% positive rate gives `p = 7.15e-6`.

The log-probability gain had a positive association with final F1 gain
(`Spearman rho = 0.396`, `p = 0.0126`). The model's latent distribution therefore
usually moves toward the gold answer when evidence is added, even when greedy
generation does not become correct.

The effect emerges mainly in late layers:

| Layer | Mean gold log-prob gain | Positive-gain questions |
|---:|---:|---:|
| 12 | -0.078 | 30.77% |
| 16 | -0.194 | 38.46% |
| 20 | +0.230 | 58.97% |
| 22 | +0.602 | 69.23% |
| 28 | +0.854 | 74.36% |
| 30 | +1.204 | 64.10% |
| 32 | +2.257 | 82.05% |
| 34 | +1.381 | 79.49% |
| 35 | +1.937 | 74.36% |

This layer trajectory contains information that final-layer perplexity or
embedding similarity cannot expose. Layer 32 was more consistently
evidence-responsive than the final layer in this run.

## Why raw attention is insufficient

The most support-focused head assigned a median 92.17% of its chunk attention
to labelled support; the range was 82.22% to 99.69%. Despite this apparent
saturation, its score was not significantly associated with F1 gain
(`rho = 0.187`, `p = 0.254`).

Mean support attention over all measured heads was less saturated and did
associate with F1 gain (`rho = 0.423`, `p = 0.0073`). This does not establish a
causal mechanism because total attention mass is affected by chunk length and
position. It does show that selecting a head simply because it attends almost
entirely to support is not enough.

The exact answer transitions were:

| Transition after adding gold support | Queries |
|---|---:|
| wrong -> wrong | 30 |
| wrong -> right | 2 |
| right -> right | 6 |
| right -> wrong | 1 |

The 30 `wrong -> wrong` cases are the key result. Their mean internal gold
log-prob gain was +0.735 and their mean maximum-head support attention was
92.17%. The model often receives the evidence and internally shifts toward the
answer but still fails to turn it into a correct output. These cases should be
labelled as reasoning/integration failures rather than retrieval failures.

## Interpretation

This smoke test supports two parts of the proposed direction:

1. Context-induced LM-head trajectories carry a measurable evidence signal,
   strongest in late but not necessarily final layers.
2. High attention to gold support does not establish evidence sufficiency or
   successful reasoning.

It does not yet establish that an internal-state stopping policy beats the
existing selector. A trained probe and causal head interventions are still
required. The next experiment should train the five-way counterfactual state
probe on calibration folds and evaluate whether it separates `retrieve_more`
from `reason_again` at matched false-stop risk.

## Limitations

- The 39 examples are a deliberately difficult subset and are not an unbiased
  estimate of TATQA performance.
- This run uses Qwen2.5-3B text-only as an available local smoke model, rather
  than the planned Qwen2-VL-7B.
- Internal log-probability is measured by teacher-forcing the gold answer. An
  inference-time probe must instead consume the model's own draft-answer states.
- Gold support was appended after the two distractors, so position and context
  length may contribute to the observed change.
- Chunk attention uses total mass and is not corrected for span length.
- No head masking or activation patching was performed, so the head observations
  are correlational.
