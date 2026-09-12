# DIRECTER plausibility for conformal chunk pruning

## What DIRECTER calls plausibility

DIRECTER does not train a plausibility probe. At decoding step `t`, it runs the
unmodified model to obtain `p_t` and an activation-steered model to obtain
`p_t_tilde`. Let

```text
i_raw   = argmax p_t
i_steer = argmax p_t_tilde
```

The steered token is accepted when

```text
p_t[i_steer] >= beta * p_t[i_raw].
```

The released implementation uses `beta = 0.5` by default. It computes the
candidate token from the steered logits, but scores that token under the raw
model. If the test fails, DIRECTER weakens the intervention by halving the
number of steered layers. If no intervention passes, it emits the raw model's
token.

This is a distribution-consistency gate. It asks whether an intervention's
proposed token was already plausible to the unmodified model. It does not ask
whether the token is correct, grounded, or supported by a document.

DIRECTER's intervention multiplies the KV-cache key vectors for instruction
tokens, normally by 100. A one-time sensitivity analysis perturbs one layer at a
time and ranks layers using cosine-distance disturbances in attention-block
representations. This layer ranking controls how steering is weakened; it is
separate from the probability-ratio plausibility test.

Primary references:

- Paper: <https://arxiv.org/abs/2603.06745>
- Official repository: <https://github.com/mjk0618/directer>
- Probability-ratio implementation:
  <https://github.com/mjk0618/directer/blob/ba4d50b5578e94d77a554e901f8b3e3cf4de3475/src/generator.py#L101-L114>

## Direct chunk-relevance adaptation

The first adaptation treats full context as DIRECTER's raw model and exact
answer-to-chunk masking as its intervention. For chunk `j`, at each
teacher-forced draft position:

```text
i_minus_j = argmax p(y_t | prefix, mask(chunk_j))
r_tj      = p_full(i_minus_j) / p_full(i_full)
```

Candidate relevance scores include the worst `-log(r_tj)`, mean `-log(r_tj)`,
fraction of positions where `r_tj < 0.5`, Top-1 change rate, and JS divergence
between full and masked distributions.

The intervention masks direct attention from answer positions to the candidate
chunk at every layer and head. Prompt tokens, answer prefix, and draft tokens
are held fixed.

## Experiment

- Qwen2.5-3B-Instruct in BF16 on one A100 40 GB.
- 24 official TATQA train questions: 12 support-at-rank-1/2 and 12 whose first
  support is after rank 2.
- 101 candidates: every support plus three highest-ranked false chunks.
- The 39-query internal-state set, 24-query causal set, and 120-query attention
  set were excluded. One query also appeared in the two-query execution smoke;
  the smoke result was not used to choose the method.
- No development calibration or test questions were used.
- `beta = 0.5`, matching DIRECTER's default.

Raw result:
`results/directer_plausibility_n24_balanced_seed71.json`.

## Support-ranking result

Candidate AUROC/AP use query-local score normalization. MRR and Top-1 are
unchanged by that normalization.

| Score | AUROC | AP | MRR | Top-1 support |
|---|---:|---:|---:|---:|
| BGE | 0.433 | **0.430** | 0.590 | 41.7% |
| All-layer attention mass | 0.554 | 0.379 | 0.608 | 41.7% |
| DIRECTER worst rejection | 0.499 | 0.326 | 0.559 | 33.3% |
| DIRECTER rejection rate | 0.534 | 0.327 | 0.601 | 41.7% |
| Full-vs-masked JS divergence | **0.585** | 0.396 | **0.639** | 41.7% |

The probability-ratio signal is random for support identification. At least one
token failed the `beta = 0.5` gate for 24.1% of support chunks and 16.7% of false
chunks. Mean rejection rate was 5.9% for supports and 6.5% for false chunks.
The model is sometimes more dependent on a distractor than on labelled support.

Adding the plausibility features to a leave-one-query-out linear support probe
did not consistently improve BGE plus attention. A simple veto on pruning the
lowest-ranked candidates rescued at most one support in this sample and usually
kept the same or more false chunks.

## Interpretation

DIRECTER plausibility should not be used as the conformal support-retention
score. It measures dependence on the current model behavior. If the model is
already following a false chunk, masking that distractor can be highly
implausible under the full-context distribution. A support-retention method
would then keep the distractor for exactly the wrong reason.

JS divergence is a somewhat better sensitivity measure than the binary or
ratio-based rejection features, but the 24-query result is too weak for it to
replace distributed attention or BGE.

## Useful application: a set-level output-fidelity veto

The DIRECTER mechanism still maps cleanly to a different role. Let the frozen
BGE-attention fusion propose a small context `C_K` from full retrieved context
`C_L`. During tentative decoding, compare the token proposed by `C_K` against
the full-context distribution:

```text
i_K = argmax p(y_t | prefix, C_K)
r_t = p(i_K | prefix, C_L) / max_i p(i | prefix, C_L)
```

If `r_t` is too low, restore the next chunk or expand `K`, then try again. This
mirrors DIRECTER's policy of weakening an intervention until its proposed token
is plausible. It answers:

> Did pruning move generation outside what the full-context model considered
> plausible?

It does not answer:

> Which chunk is relevant or factually correct?

This veto can therefore sit after the conformal chunk selector, but should not
replace it.

## How conformal fits

Conformal calibration can calibrate the continuous sequence statistic, for
example the minimum `r_t` over the first answer tokens. The calibration target
must be answer degradation after pruning, such as full-context-correct and
pruned-context-wrong. It is different from the all-support retention target
used by the current chunk selector.

A valid development experiment should use three disjoint roles:

1. train/freeze the BGE-attention chunk scorer;
2. calibrate its support-retention threshold and the plausibility veto on
   separate development queries;
3. evaluate final answer accuracy and retained tokens on untouched queries.

The fixed `beta = 0.5` from DIRECTER should be a baseline, not the final RAG
threshold. It was tuned for instruction steering and does not carry a
finite-sample RAG risk guarantee.

## Decision

- Do not add DIRECTER probability ratio as another chunk-relevance feature in
  the current conformal score.
- Keep position-controlled distributed attention plus BGE as the support
  selector.
- Test DIRECTER plausibility later as a set-level veto while expanding nested
  `K = 2, 5, 10, 20, L` contexts.
- Judge the veto by downstream EM/F1 preservation at matched retained-token
  budget, not by support AUROC.

The main computational limitation is that full and pruned distributions must be
available at decoding time. Apply the veto only to a proposed context set or to
borderline pruning decisions; running it independently for all Top-L chunks is
too expensive.
