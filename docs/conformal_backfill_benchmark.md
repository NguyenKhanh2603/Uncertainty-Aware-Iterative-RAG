# Conformal backfill benchmark

The implementation now separates three evidence-selection policies:

1. `fixed_top_k`: the original first K retrieved chunks.
2. `conformal_no_backfill`: only BY-accepted chunks inside the original Top-K.
3. `conformal_backfill`: keep those accepted Top-K chunks, then fill unused slots with
   BY-accepted candidates from ranks K+1 through L. An unaccepted chunk is never used merely
   to fill the context.

The benchmark reports evidence precision, conditional reserve support recall, support F1,
false-discovery diagnostics, chunks added by backfill, and queries where backfill restores at
least one support chunk. These are evidence-selection metrics, not downstream answer EM/F1.

## Run in Colab

Open `conformal_backfill_2_3_gpu_colab.ipynb`, set:

```python
MAX_QUESTIONS_PER_DATASET = 1000
```

and run all cells. The downloaded ZIP includes:

- `selection/backfill_benchmark/backfill_benchmark_summary.csv`
- `selection/backfill_benchmark/backfill_benchmark_summary.json`
- `selection/backfill_benchmark/backfill_benchmark_queries.jsonl.gz`

The 1000-question limit is per dataset, so four enabled datasets can produce up to 4000 total
queries. Retrieval/embedding uses the GPU; replaying p-values, alpha values, and the backfill
benchmark is CPU-only.

## Run from existing selection results

```bash
uv run python scripts/benchmark_conformal_backfill.py \
  --input selection/selection_decisions.jsonl.gz \
  --output-dir selection/backfill_benchmark \
  --alphas 0.01,0.025,0.05,0.10,0.20,0.30,0.50 \
  --max-context 10
```

The current BY-plus-at-most-K construction is experimental. The formal capped-risk proof and
the downstream generation comparison against attention pruning remain separate work items.
