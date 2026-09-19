#!/usr/bin/env bash
# Reuse seed-1's already fixed 100-query probe and calibration sets.
set -euo pipefail
model="/workspace/hf_cache/hub/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/eed13092ef92e448dd6875b2a00151bd3f7db0ac"
root="research/internal_state_rag/results/qwen2vl_7b_jina4"
plans="$root/full_test_1000_plans"
features="$root/full_test_1000_features"
analysis="$root/full_test_1000_analysis"
out="$root/full_test_1000_downstream"

PYTHONPATH=src:. .venv-cu118/bin/python research/internal_state_rag/prepare_full_test_1000_plans.py
for dataset in hotpotqa mmqa tatqa; do
  bundle="data/conformal_global_run/official_bundle_role_split_${dataset}"
  if [[ ! -f "$features/$dataset/features.npz" ]]; then
    PYTHONPATH=src:. .venv-cu118/bin/python research/internal_state_rag/run_qwen2vl_pairwise_features.py \
      --model "$model" --model-revision "$(basename "$model")" \
      --feature-manifest "$plans/$dataset/manifest.json" \
      --questions "$bundle/$dataset/questions.jsonl" --corpus "$bundle/$dataset/corpus.jsonl" \
      --bundle-root "$bundle" --retrieval "research/internal_state_rag/results/qwen2vl_jina4/${dataset}_jina_v4_top30.jsonl.gz" \
      --output-dir "$features/$dataset" --min-pixels 3136 --max-pixels 200704 --text-batch-size 4 --image-batch-size 2
  fi
  if [[ ! -f "$analysis/$dataset/fusion_predictions.npz" ]]; then
    PYTHONPATH=src:. .venv-cu118/bin/python research/internal_state_rag/analyze_qwen2vl_jina_ablation.py \
      --dataset "$dataset" \
      --probe-train "$root/fusion_features/$dataset/probe_train/features.npz" \
      --calibration "$root/fusion_features/$dataset/calibration/features.npz" \
      --test "$features/$dataset/features.npz" --output "$analysis/$dataset/fusion_analysis.json" \
      --predictions-output "$analysis/$dataset/fusion_predictions.npz" --alphas 0.1 --bootstrap-samples 10000
  fi
  PYTHONPATH=src:. .venv-cu118/bin/python research/internal_state_rag/run_three_seed_downstream_comparison.py \
    --seed 1 --dataset "$dataset" --model "$model" --feature-root "$root/fusion_features" \
    --analysis-root "$analysis" --fusion-predictions "$analysis/$dataset/fusion_predictions.npz" \
    --test-plan "$plans/$dataset/manifest.json" --output-dir "$out" \
    --max-new-tokens 24 --min-pixels 3136 --max-pixels 200704
done
