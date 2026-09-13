# Conformal recall research package — 2026-09-13

This archive is a reproducible checkpoint of the repository after the clean
TAT-QA conformal pruning validation and the matched attention-only comparison.

## Main result

The final protocol uses 96 official-train queries to fit the frozen layer-30
linear relevance probe, 904 disjoint official-train queries to calibrate
query-level conformal thresholds, and all 1,000 official-dev test-role queries
for evaluation. Query-ID overlap between the three roles is zero.

At the all-support `alpha=0.10` operating point, the three-signal fusion keeps
3.39 of 30 candidates per retrievable query and reaches 91.4% mean support
recall. BGE keeps 5.73 candidates at 91.0% recall. At `alpha=0.05`, fusion keeps
6.30 candidates at 95.6% recall, compared with BGE's 15.16 candidates at 95.7%
recall.

The matched 150-query comparison confirms that answer-to-chunk attention is a
real relevance signal, but it is not an efficient high-recall pruner. At
`alpha=0.10`, position-controlled attention keeps 7.06 chunks at 86.6% micro
recall and 14.1% precision. The three-signal internal fusion keeps 2.70 chunks
at 84.9% recall and 36.0% precision on those exact test queries. Candidate-wise
cosine BY keeps only 0.12 chunks per query, leaves 93.3% empty, and reaches just
6.9% recall despite 52.2% precision.

A follow-up disjoint split trained the layer-30 probe on 452 queries and used
another 452 for conformal calibration. At `alpha=0.10`, retuned fusion keeps
3.23 chunks with 30.85% precision and 88.66% micro recall. Relative to the
96-query probe, paired bootstrap confirms fewer chunks and higher precision;
the small recall increase is not statistically resolved.

Top-30 contains support for 768 of 1,000 test queries, setting an unconditional
end-to-end query coverage ceiling of 76.8%. The observed conditional coverage
is slightly below its nominal target, so the report does not claim an absolute
90% or 95% guarantee across the official-train to official-dev shift.

## Entry points

- `research/internal_state_rag/MENTOR_BRIEF_CONFORMAL_RECALL.md`: concise
  Vietnamese analysis for mentor discussion.
- `research/internal_state_rag/FULL_PRUNING_COMPARISON_2026-09-13.md`: full
  cosine-BY, attention-only, BGE, and query-level internal-fusion table.
- `research/internal_state_rag/RESULTS_INTERNAL_CHUNK_SIGNALS.md`: full research
  report, including prior attention, LM-head, hidden-state, causal, and
  generation experiments.
- `research/internal_state_rag/analyze_pairwise_conformal_clean_split.py`: final
  clean-split evaluation.
- `research/internal_state_rag/analyze_attention_only_clean_conformal.py`:
  matched comparison and attention-only probe evaluation.
- `research/internal_state_rag/analyze_scaled_hidden_probe.py`: larger-probe
  split, retuned fusion, and paired-bootstrap analysis.
- `research/internal_state_rag/results/pairwise_conformal_clean_n96_cal904_test1000.json`:
  complete machine-readable result at alpha 0.20, 0.10, and 0.05.
- `research/internal_state_rag/results/pairwise_relevance_cal_fresh_n904/` and
  `research/internal_state_rag/results/pairwise_relevance_test_all_n1000/`: raw
  feature artifacts used by the final analysis.

## Method

The frozen candidate score is:

```text
z(BGE) + 0.20 z(Yes-vs-No LM-head)
       + 0.75 z(layer-30 relevance probe)
```

Qwen2.5-3B-Instruct remains frozen. Only an L2 logistic probe is trained. The
large feature runs used BF16 on an A100 40 GB; Python dependencies are managed
with the repository's uv environment.

## Validation and archive scope

`PYTHONPATH=. .venv/bin/pytest -q research/internal_state_rag/tests` passes all
19 tests. The ZIP is built from Git `HEAD`, so it includes tracked code, reports,
tests, and result artifacts while excluding `.git`, `.venv`, model/cache data,
and all untracked user files.
