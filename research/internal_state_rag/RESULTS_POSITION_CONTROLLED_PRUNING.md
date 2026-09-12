# Position-controlled internal-attention pruning result

## Outcome

Generator-internal attention contains a useful chunk-retention signal on this
TATQA train-only pilot. The strongest simple score is mean answer-to-chunk
attention over all extracted layers and heads. Averaging the score from the
original and reversed context orders reduces prompt-position bias. A
query-grouped linear fusion of BGE and two attention features improves chunk
ranking further.

This result supports building a full development calibration bank. It is not a
test-set result or a conformal guarantee for the production Top-30 pool.

## Protocol

- Model: Qwen2.5-3B-Instruct in BF16 on one A100 40 GB.
- Data: 120 official TATQA training questions, with no test questions.
- Sampling: 60 easy questions whose first support is at retrieval rank 1-2 and
  60 hard questions whose first support is after rank 2.
- Exclusion: the 39 questions from the first internal-state experiment and 24
  questions from the chunk-causal experiment were excluded. Two questions also
  appeared in a two-query execution smoke test; no method choice was made from
  that smoke output.
- Candidate proxy: every labelled support plus the three highest-ranked false
  chunks, giving 490 candidates and 130 supports.
- Context: original Top-10 plus any candidate support outside Top-10.
- Draft: generated once from the original context order.
- Position control: teacher-force the same draft under the original order and
  the complete reverse order. Average each chunk's two attention scores.
- Internal score: mean attention mass from generated-answer positions to the
  aligned chunk tokens, averaged across heads and extracted layers 12, 14, ...,
  34, 35.

The reversed pass holds the answer tokens and chunk contents fixed. It changes
only context position, making it a direct control for beginning/end-of-prompt
bias.

Raw files:

- `results/attention_ranking_n120_balanced_seed47.json`
- `results/attention_position_control_n120_seed47.json`
- `results/attention_position_analysis_n120_seed47.json`

## Chunk-ranking result

All scores are standardized within query without using labels. The fusion is a
balanced logistic regression evaluated leave-one-query-out, so the model used
for each query never trains on that query.

| Score | AUROC | AP | MRR | Top-1 support |
|---|---:|---:|---:|---:|
| BGE | 0.529 | 0.445 | 0.613 | 45.0% |
| Position-controlled attention mass | 0.694 | 0.536 | 0.714 | 55.0% |
| BGE + attention mass + attention fraction | **0.777** | **0.560** | **0.766** | **60.8%** |

Against BGE, position-controlled attention gains 12 Top-1 questions and loses
zero. Its MRR improvement is +0.101 with a query-bootstrap 95% interval of
[+0.067, +0.138] and paired Wilcoxon p = 2.53e-8.

The leave-one-query-out fusion gains 23 Top-1 questions and loses four. Its MRR
improvement is +0.153, bootstrap 95% interval [+0.103, +0.205], with exact
paired sign-test p = 3.11e-4.

### By retrieval difficulty

| Stratum and score | AP | MRR | Top-1 support |
|---|---:|---:|---:|
| Easy: BGE | 0.880 | 0.950 | 90.0% |
| Easy: attention | **0.963** | **0.992** | **98.3%** |
| Easy: fusion | 0.919 | 0.967 | 93.3% |
| Hard: BGE | 0.155 | 0.276 | 0.0% |
| Hard: attention | 0.209 | 0.436 | 11.7% |
| Hard: fusion | **0.447** | **0.565** | **28.3%** |

The balanced easy/hard sampling is deliberately unlike the natural rank
distribution. These aggregate values therefore measure behavior across both
regimes; they are not estimates of deployment prevalence.

## Position and length controls

Chunk rankings under original and reversed order remain positively correlated:

| Internal score | Mean query Spearman | Median | Queries with positive correlation |
|---|---:|---:|---:|
| All-layer attention mass | 0.640 | 0.800 | 90.8% |
| All-layer attention fraction | 0.711 | 0.800 | 95.8% |

The original all-layer score and the original/reversed mean both outperform
BGE. This rejects the explanation that the entire gain comes from a fixed
beginning/end position. Order still matters materially, so a production scorer
must either average controlled orders or learn an explicit position correction.

Dividing attention by chunk token count performs poorly (Top-1 34.2% before the
order control). The successful signal is total evidence routing to a chunk,
rather than average attention per token. Attention fraction, which normalizes by
total context attention without rewarding absolute model confidence, retains
most of the gain.

## Conformal pruning pilot

Each of 100 trials uses disjoint, stratified query sets:

- 40 queries to train the linear fusion;
- 40 queries to calibrate a lower retention threshold;
- 40 queries for evaluation.

For each calibration query, the calibration statistic is the lowest score among
all of its support chunks. A chunk is kept when its score exceeds the calibrated
threshold. This targets 90% marginal probability of retaining every labelled
support in a query. Reported values are means across trials.

| Score | All-support query coverage | Support recall | False chunks pruned | Candidates kept |
|---|---:|---:|---:|---:|
| BGE | 89.9% | 90.7% | 0.7% | 97.0% |
| Attention | 91.2% | 91.9% | 9.0% | 91.2% |
| Fusion | 90.9% | 91.6% | **31.8%** | **74.4%** |

The fusion reaches approximately the same support coverage while pruning far
more false candidates than BGE. This is the first result in this branch that is
directly helpful to the conformal chunk-retention objective.

## What failed

The exact causal score does not work yet. On the separate 24-query pilot,
masking answer-to-chunk attention edges through every head in layers 14 and 18
gave draft causal AUROC 0.494 and gold causal AUROC 0.489. Absolute draft
influence reached only 0.570. In contrast, raw attention on the same examples
had AUROC 0.864.

Joint whole-head masking also lowered gold likelihood but did not beat
layer-matched controls. The current evidence rejects the sparse
"high-attention head equals crucial evidence head" hypothesis. The useful score
comes from distributed, all-layer routing, not the preselected layer-14/18 head
set.

## Decision

Proceed to the 1,000-query development-bank experiment with these predeclared
baselines:

1. query-normalized BGE;
2. position-controlled all-layer attention mass and fraction;
3. a frozen linear fusion of those three features.

Build candidates from the actual Top-L pool, rather than the labelled
support-plus-three-negative proxy. Train the fusion on a separate development
fold, calibrate the minimum-support threshold on the calibration fold, and
freeze both before evaluating official test queries. Report all-support
coverage, false-chunk pruning, retained tokens, and downstream answer EM/F1.

Direct-logit attribution remains a mechanistic follow-up. It should only replace
the attention features if it predicts exact edge-mask effects or improves
held-out pruning. The current exact edge intervention is too weak and noisy to
serve as the pruning score.

## Limitations

- The candidate proxy contains only three false chunks per query and uses gold
  support labels to assemble the experiment. It does not reproduce a full
  Top-30 deployment pool.
- The easy/hard sample is balanced and covers one dataset and one 3B model.
- The conformal pilot reuses the same 120 questions across repeated random
  trials. Its held-out partitions are clean within each trial, but it is not the
  final one-shot 1,000-query calibration certificate.
- The two-order score needs two teacher-forced passes after draft generation.
- Support retention improved; downstream answer quality after pruning has not
  yet been measured.
