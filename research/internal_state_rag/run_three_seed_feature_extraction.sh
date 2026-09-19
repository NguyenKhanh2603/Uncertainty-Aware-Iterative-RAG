#!/usr/bin/env bash
set -euo pipefail

model_path="/workspace/hf_cache/hub/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/eed13092ef92e448dd6875b2a00151bd3f7db0ac"
revision="eed13092ef92e448dd6875b2a00151bd3f7db0ac"
plan_root="research/internal_state_rag/results/qwen2vl_7b_jina4/three_seed_feature_plans"
output_root="research/internal_state_rag/results/qwen2vl_7b_jina4/three_seed_features"

for seed in 2027 2028; do
  for dataset in hotpotqa mmqa tatqa; do
    bundle="data/conformal_global_run/official_bundle_role_split_${dataset}"
    for part in probe_train calibration test; do
      output_dir="$output_root/seed_${seed}/${dataset}/${part}"
      if [[ -f "$output_dir/features.npz" ]]; then
        echo "[resume] seed=$seed dataset=$dataset part=$part"
        continue
      fi
      PYTHONPATH=src:. .venv-cu118/bin/python research/internal_state_rag/run_qwen2vl_pairwise_features.py \
        --model "$model_path" \
        --model-revision "$revision" \
        --feature-manifest "$plan_root/seed_${seed}/${dataset}/${part}/manifest.json" \
        --questions "$bundle/${dataset}/questions.jsonl" \
        --corpus "$bundle/${dataset}/corpus.jsonl" \
        --bundle-root "$bundle" \
        --retrieval "research/internal_state_rag/results/qwen2vl_jina4/${dataset}_jina_v4_top30.jsonl.gz" \
        --output-dir "$output_dir" \
        --min-pixels 3136 --max-pixels 200704 \
        --text-batch-size 4 --image-batch-size 2
    done
  done
done
