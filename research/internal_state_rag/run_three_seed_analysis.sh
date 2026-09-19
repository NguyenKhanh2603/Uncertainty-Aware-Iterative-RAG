#!/usr/bin/env bash
set -euo pipefail

features="research/internal_state_rag/results/qwen2vl_7b_jina4/three_seed_features"
analysis="research/internal_state_rag/results/qwen2vl_7b_jina4/three_seed_analysis"
for seed in 2027 2028; do
  for dataset in hotpotqa mmqa tatqa; do
    out="$analysis/seed_${seed}/${dataset}/fusion_analysis.json"
    [[ -f "$out" ]] && { echo "[resume] seed=$seed dataset=$dataset"; continue; }
    PYTHONPATH=src:. .venv-cu118/bin/python research/internal_state_rag/analyze_qwen2vl_jina_ablation.py \
      --dataset "$dataset" \
      --probe-train "$features/seed_${seed}/$dataset/probe_train/features.npz" \
      --calibration "$features/seed_${seed}/$dataset/calibration/features.npz" \
      --test "$features/seed_${seed}/$dataset/test/features.npz" \
      --output "$out" \
      --predictions-output "$analysis/seed_${seed}/${dataset}/fusion_predictions.npz" \
      --alphas 0.1 --bootstrap-samples 10000
  done
done
