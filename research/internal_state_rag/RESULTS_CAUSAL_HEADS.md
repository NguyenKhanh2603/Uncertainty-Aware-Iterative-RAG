# Causal evidence-head masking result

## Question

Does high attention from answer tokens to gold-support chunks identify a sparse
set of heads that causally carries the answer?

## Protocol

- Reused the 39 hard TATQA calibration questions from the internal-signal smoke
  test. No test questions were used.
- Discovery/evaluation split: 20/19 questions, fixed seed 23.
- On discovery questions, ranked every measured `(layer, head)` by its mean
  fraction of chunk attention assigned to labelled support.
- Selected the top eight heads. Six were in layer 14 and two were in layer 18.
- During evaluation, zeroed each selected head's output immediately before that
  layer's attention output projection. The intervention applied during both
  prompt encoding and answer decoding.
- Compared top heads with eight random and eight bottom-ranked heads that had
  the exact same per-layer counts. Layer matching is essential because masking
  heads from later layers produced a misleading control in the first run.

Raw results are in
`results/causal_heads_layer_matched_d20_e19_seed23.json`.

## Selected heads

The eight support-attention heads were:

`(14,13), (14,15), (14,8), (18,8), (14,11), (14,10), (18,13), (14,12)`.

Their concentration in two middle layers is consistent with a sparse,
stage-specific context-integration mechanism. Concentration alone does not show
that the ranking criterion identifies a uniquely causal circuit.

## Gold-answer log-probability intervention

| Masked set | Mean delta | Median delta | Negative cases | Bootstrap 95% CI | Wilcoxon vs zero |
|---|---:|---:|---:|---:|---:|
| Top attention heads | -0.464 | -0.256 | 17/19 | [-0.840, -0.062] | 0.0071 |
| Layer-matched random | -0.256 | +0.000 | 9/19 | [-0.672, +0.131] | 0.4653 |
| Layer-matched bottom | -0.115 | +0.000 | 9/19 | [-0.870, +0.523] | 0.9843 |

Masking top heads reliably lowers gold-answer log probability in isolation. The
effect is more consistent than either control.

The stronger and necessary selectivity tests are inconclusive:

| Paired comparison | Mean difference | Top has larger negative effect | Bootstrap 95% CI | Wilcoxon p |
|---|---:|---:|---:|---:|
| Top minus random | -0.208 | 12/19 | [-0.719, +0.292] | 0.418 |
| Top minus bottom | -0.348 | 14/19 | [-1.181, +0.546] | 0.210 |

Therefore the experiment gives evidence that the selected middle-layer heads
participate in answer formation, but it does not establish that attention
fraction successfully distinguishes crucial heads from other heads in the same
layers.

## Generated-answer effect

| Condition | EM | F1 | Numerical accuracy |
|---|---:|---:|---:|
| No masking | 15.79% | 21.45% | 31.58% |
| Top-head masking | 15.79% | 23.62% | 31.58% |
| Random-head masking | 10.53% | 16.77% | 26.32% |
| Bottom-head masking | 15.79% | 21.01% | 31.58% |

Top-head masking does not reduce answer accuracy in this small evaluation. The
latent likelihood effect therefore does not yet translate into a task-level
causal effect. F1 is especially sensitive to answer verbosity in these examples.

## Conclusion

This experiment rejects the strongest naive hypothesis:

> The heads with the largest raw support-attention fraction are already the
> crucial evidence-use heads needed for a stopping decision.

The top set has a real effect on gold likelihood, but it is not significantly
larger than layer-matched controls and does not reduce generated-answer accuracy.
Raw attention remains useful for locating candidate layers and heads, not as the
final internal confidence score.

The next head score should measure contribution to the answer computation
itself. Two viable scores are:

1. direct logit attribution of each head output to generated answer tokens;
2. causal gold/draft log-probability delta from masking individual candidate
   heads on the discovery fold.

Head selection must remain separate from the evaluation fold. The stopping
probe should primarily use context-induced residual/logit trajectories, while
causally validated head features act as an additional grounding signal.

## Limitations

- Nineteen evaluation questions provide low power for a noisy head intervention.
- All questions are difficult cases with support below rank 2.
- Eight heads were masked jointly, so interactions among heads are unresolved.
- Masking modifies the complete prompt and answer computation; it does not
  isolate attention from answer tokens to support spans only.
- The experiment uses Qwen2.5-3B and teacher-forced gold likelihood.
