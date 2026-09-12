# Full-Top-30 BGE plus internal-attention validation

## Decision

The larger experiment validates position-controlled internal attention as an
additional **chunk-ranking and conservative pruning signal**. The useful
configuration is the learned BGE-attention fusion at `alpha = 0.05`.

On the locked TATQA test-role sample, fusion improved every ranking endpoint
over BGE. At the frozen `alpha = 0.05` conformal threshold, it also retained all
labelled support for more queries while keeping fewer chunks and tokens. The
downstream answer experiment found no statistically distinguishable F1 or EM
change between fusion and BGE at that operating point.

This result does not establish that high attention is a causal explanation of
relevance. Earlier head-masking experiments did not validate that claim. Here,
distributed answer-to-chunk attention is used as a predictive internal feature.

## Frozen protocol

- Dataset: TATQA from the repository's official role-split bundle.
- Retriever reserve: all 30 candidates from the frozen BGE-reranked retrieval
  log. No labelled support was inserted into the candidate list.
- Generator: Qwen2.5-3B-Instruct in BF16 on one A100 40 GB.
- Scorer training: 120 calibration-role/source-train queries used by the earlier
  balanced pilot.
- Conformal calibration: 150 separate calibration-role/source-train queries.
- Method-development evaluation: 150 other calibration-role/source-train
  queries. These results were used to select `alpha = 0.05` as the useful
  operating point.
- Locked evaluation: 150 test-role/source-dev queries sampled with seed 211 only
  after features, model coefficients, thresholds, and alphas were frozen.
- Discovery queries from the earlier logit, head, and masking pilots were
  excluded from the development roles.

Each query first produces a short draft from the full Top-30 context. The model
then teacher-forces that same draft twice: once with the original candidate
order and once with the order reversed. For every chunk, the feature extractor
averages all-layer answer-to-chunk attention mass and attention fraction across
the two orders.

The learned scorer is a class-balanced logistic regression over query-local
z-scores:

```text
BGE score + mean attention mass + mean attention fraction
```

It is fit only on the 120 scorer-training queries. The frozen coefficients are
`[1.174, 3.221, -3.133]`; the intercept is `-1.208`. Conformal calibration uses
the minimum score among each query's support chunks, so query coverage means
that **all labelled support chunks** survive pruning.

## Retrieval ceiling and evaluated population

The calibration role contains 1,000 queries, of which 810 have labelled support
in Top-30. The test role contains 1,000 queries, of which 768 have support in
Top-30. Ranking and pruning metrics condition on this event because no
post-retrieval method can retain a support chunk that the retriever never found.

The locked 150-query test sample contains 113 easy queries whose first support
is at rank 1 or 2 and 37 hard queries whose first support is below rank 2. It has
4,500 candidates: 129 queries with one support, 20 with two, and one with three.

The conditional retrieval ceiling is therefore 76.8% on the complete test role.
Multiplying it by the observed `alpha = 0.05` conditional all-support coverage
gives an estimated full-pipeline support availability of 72.2% for BGE and
73.2% for fusion. Increasing `L`, improving first-stage retrieval, or both are
still required to address the other 23.2% of test queries.

## Locked test ranking

Candidate AP and AUROC pool candidates from the 150 evaluation queries. MRR and
Top-1 are computed per query.

| Scorer | AUROC | Candidate AP | MRR | Top-1 support |
|---|---:|---:|---:|---:|
| Cosine | 0.784 | 0.378 | 0.575 | 46.0% |
| BGE | 0.917 | 0.590 | 0.736 | 61.3% |
| Position-controlled attention | 0.916 | 0.636 | 0.754 | 67.3% |
| **BGE + internal attention** | **0.942** | **0.689** | **0.819** | **74.0%** |

The paired query bootstrap used 20,000 resamples:

| Fusion minus BGE | Mean difference | 95% bootstrap CI |
|---|---:|---:|
| Query average precision | +0.0720 | [+0.0281, +0.1155] |
| Reciprocal rank | +0.0825 | [+0.0383, +0.1262] |
| Top-1 support rate | +12.67 points | [+5.33, +20.00] |

The improvement is larger on the 37 hard queries:

| Scorer | Hard AP | Hard MRR | Hard Top-1 |
|---|---:|---:|---:|
| BGE | 0.103 | 0.214 | 0.0% |
| Position-controlled attention | 0.094 | 0.264 | 10.8% |
| **Fusion** | **0.211** | **0.444** | **27.0%** |

Attention by itself is not consistently better than BGE on hard candidate AP.
The gain comes from combining the two signals.

## Locked test conformal retention

Thresholds were selected once on the 150-query development calibration bank and
applied unchanged to test-role features.

| Alpha | Scorer | All-support coverage | Support recall | False chunks pruned | Candidates kept | Tokens kept | Mean chunks |
|---:|---|---:|---:|---:|---:|---:|---:|
| .05 | BGE | 94.00% | 94.19% | 48.01% | 53.60% | 58.29% | 16.08 |
| .05 | **Fusion** | **95.33%** | **95.93%** | **57.39%** | **44.64%** | **45.10%** | **13.39** |
| .10 | BGE | 86.67% | 87.79% | **86.09%** | **16.73%** | **18.90%** | **5.02** |
| .10 | Fusion | **89.33%** | **90.12%** | 84.96% | 17.91% | 20.10% | 5.37 |
| .20 | BGE | 64.67% | 65.12% | 96.60% | 5.76% | 6.48% | 1.73 |
| .20 | Fusion | **69.33%** | **72.09%** | **97.50%** | **5.16%** | **5.88%** | **1.55** |

At `alpha = 0.05`, fusion dominates BGE on this sample: it gains two net covered
queries, removes another 2.69 chunks and 532 context tokens per query, and has no
empty contexts. In paired terms, fusion rescues three queries that BGE harms,
harms one query that BGE covers, and both fail on six.

For the 37 hard queries at `alpha = 0.05`, coverage rises from 83.8% to 89.2%
while fusion keeps 0.92 fewer chunks on average. At `alpha = 0.10`, hard coverage
rises from 64.9% to 78.4%, but fusion keeps slightly more context overall and
loses downstream accuracy on easy queries. `alpha = 0.10` is therefore not the
recommended fusion setting. Both methods substantially miss the nominal target
at `alpha = 0.20`, indicating distribution shift and an overly aggressive small
calibration quantile.

## Downstream answer validation

The same Qwen model generated new answers from each pruned context. The stored
full-Top-30 draft is the reference context condition, not a gold answer. All
methods use greedy decoding with a 24-token cap.

| Method | EM | Token F1 | Numerical accuracy | Mean chunks | Mean context tokens |
|---|---:|---:|---:|---:|---:|
| Full Top-30 | 17.33% | 0.3435 | 29.33% | 30.00 | 4,034 |
| BGE, α=.05 | 18.67% | 0.3457 | 30.00% | 16.08 | 2,351 |
| **Fusion, α=.05** | **19.33%** | **0.3488** | 30.00% | **13.39** | **1,819** |
| BGE, α=.10 | 17.33% | 0.3354 | 28.00% | 5.02 | 763 |
| Fusion, α=.10 | 15.33% | 0.3232 | 28.67% | 5.37 | 811 |

At `alpha = 0.05`, fusion minus BGE is +0.0031 F1 with a 95% paired bootstrap CI
of `[-0.0370, +0.0444]`, and +0.67 EM points with CI
`[-3.33, +4.67]`. The experiment supports answer-quality preservation while
using less context; it does not show a statistically reliable accuracy gain.

The hard-query point estimate favors fusion at `alpha = 0.05`: F1 is 0.1849
versus 0.1331 for BGE and EM is 10.81% versus 5.41%. The paired hard-query F1
difference is +0.0518, but its CI `[-0.0448, +0.1622]` remains wide. At
`alpha = 0.10`, fusion improves hard-query F1 but reduces easy-query F1 enough to
lower the overall result. This reinforces the conservative `alpha = 0.05`
choice.

These are repository EM/F1 metrics rather than the official TATQA evaluator.
The local metric treats a list of gold spans as alternatives, which understates
multi-span performance. Absolute values should not be reported as official
TATQA scores; the paired comparisons remain useful because every method is
scored against the same labels.

## Robustness checks

- Reversing the candidate order gives a mean within-query Spearman correlation
  of 0.614 for attention mass; all 150 correlations are positive. Averaging the
  two orders reduces fixed prompt-position effects without erasing the signal.
- Query-local attention mass correlates only 0.184 with log chunk length on the
  locked analysis input.
- A BGE-plus-length logistic baseline obtains mean query AP 0.686, below BGE at
  0.718 and fusion at 0.791. Fusion's ranking gain is not explained by simply
  preferring longer chunks.
- Adding length to fusion slightly lowers mean query AP from 0.791 to 0.783.

## What is validated

For this model and dataset, the generated-answer attention trace contains
incremental information that BGE lacks. It is especially useful when the
support is deep in the Top-30 reserve. The continuous fusion score can be used
inside the existing conformal framework because it is trained before threshold
calibration and applied unchanged to test queries.

The supported pruning policy is:

1. retrieve and BGE-rerank Top-30;
2. generate a short full-context draft;
3. extract answer-to-chunk attention in original and reversed order;
4. score chunks with the frozen BGE-attention logistic model;
5. retain chunks above the frozen `alpha = 0.05` threshold;
6. generate the final answer from the retained chunks.

Attention-only pruning keeps too much context at conservative risk, and the
`alpha = 0.10` fusion setting does not improve the overall downstream tradeoff.

## Limits and next experiment

The result covers one dataset, one 3B model, one scorer-training split, and a
150-query locked test sample. The conformal bank has 150 queries even though the
bundle provides 1,000 development-role queries. Before a paper claim, extract
the remaining eligible development queries, calibrate once on the full frozen
bank, and evaluate all eligible test-role queries or repeat across several fixed
test samples.

The method also pays for a full-context draft plus two teacher-forced attention
passes before it saves tokens in final generation. The reported token reduction
is final-context compression, not end-to-end compute reduction. The next
ablation should test one-pass original-order attention, selected layers/heads,
and cached draft computation at matched wall time.

Finally, the signal remains correlational. Direct-logit attribution or
individual-head causal masking should still be used to identify a smaller
mechanistically supported head set. That experiment should be judged against
the validated full-attention fusion at matched all-support coverage and final
answer quality.

## Artifacts

- Development raw features and analysis:
  `results/full_top30_n420_seed101/`
- Locked test raw features, analysis, robustness checks, and downstream answers:
  `results/locked_test_top30_n150_seed211/`
- Full-Top-30 extractor: `run_full_topl_validation.py`
- Ranking and conformal analyzer: `analyze_full_topl_validation.py`
- Robustness analyzer: `analyze_full_topl_robustness.py`
- Downstream generator and paired analyzer: `run_pruned_answer_validation.py`
  and `analyze_pruned_answers.py`
