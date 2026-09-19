#!/usr/bin/env bash
# Continue the resumable three-seed study after feature extraction finishes.
set -euo pipefail

root="research/internal_state_rag/results/qwen2vl_7b_jina4"
model="/workspace/hf_cache/hub/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/eed13092ef92e448dd6875b2a00151bd3f7db0ac"
while [[ $(find "$root/three_seed_features" -name features.npz 2>/dev/null | wc -l) -lt 18 ]]; do
  echo "$(date -u +%FT%TZ) waiting for feature extraction"
  sleep 60
done

bash research/internal_state_rag/run_three_seed_analysis.sh
for seed in 2027 2028; do
  for dataset in hotpotqa mmqa tatqa; do
    PYTHONPATH=src:. .venv-cu118/bin/python research/internal_state_rag/run_three_seed_downstream_comparison.py \
      --seed "$seed" --dataset "$dataset" --model "$model" \
      --feature-root "$root/three_seed_features" \
      --analysis-root "$root/three_seed_analysis" \
      --output-dir "$root/three_seed_downstream" \
      --max-new-tokens 24 --min-pixels 3136 --max-pixels 200704
  done
done
