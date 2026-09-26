#!/usr/bin/env bash
# Resumable Qwen2-VL-7B internal-signal ablation for the inverse 1000/100 split.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

python_bin=".venv-cu118/bin/python"
model="/workspace/hf_cache/hub/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/eed13092ef92e448dd6875b2a00151bd3f7db0ac"
revision="eed13092ef92e448dd6875b2a00151bd3f7db0ac"
base="research/internal_state_rag/results/all_datasets_cosine_six_methods_1000cal_100test_2026_09_26/internal_signal_ablation_7b"

PYTHONPATH=src:. "$python_bin" research/internal_state_rag/prepare_inverse_1000cal_internal_signal_plans.py \
  --output-dir "$base"

for dataset in hotpotqa mmqa tatqa webqa; do
  case "$dataset" in
    hotpotqa)
      questions="data/official_seed42_hotpotqa_fixed/hotpotqa/questions.jsonl"
      corpus="data/official_seed42_hotpotqa_fixed/hotpotqa/corpus.jsonl"
      bundle="data/official_seed42_hotpotqa_fixed"
      retrieval="research/internal_state_rag/results/official_seed42_hotpotqa_cosine_top30.jsonl.gz"
      max_pixels=200704; image_batch_size=2
      ;;
    mmqa|tatqa)
      questions="data/conformal_global_run/official_bundle_role_split_${dataset}/${dataset}/questions.jsonl"
      corpus="data/conformal_global_run/official_bundle_role_split_${dataset}/${dataset}/corpus.jsonl"
      bundle="data/conformal_global_run/official_bundle_role_split_${dataset}"
      retrieval="research/internal_state_rag/results/qwen2vl_jina4/${dataset}_jina_v4_top30.jsonl.gz"
      max_pixels=200704; image_batch_size=2
      ;;
    webqa)
      questions="data/conformal_global_run/official_bundle_1000_webqa/webqa/questions.jsonl"
      corpus="data/conformal_global_run/official_bundle_1000_webqa/webqa/corpus.jsonl"
      bundle="data/conformal_global_run/official_bundle_1000_webqa"
      retrieval="research/internal_state_rag/results/qwen2vl_jina4/webqa_jina_v4_top30.jsonl.gz"
      max_pixels=262144; image_batch_size=4
      ;;
  esac

  roles=(probe_train conformal_calibration)
  # HotpotQA is freshly extracted because its historical cache had no manifest;
  # MMQA/TAT-QA/WebQA test artifacts were validated and materialized by prepare.
  if [[ ! -f "$base/features/$dataset/test/features.npz" ]]; then
    roles+=(test)
  fi
  for role in "${roles[@]}"; do
    features="$base/features/$dataset/$role/features.npz"
    if [[ ! -f "$features" ]]; then
      PYTHONPATH=src:. "$python_bin" research/internal_state_rag/run_qwen2vl_pairwise_features.py \
        --model "$model" --model-revision "$revision" \
        --feature-manifest "$base/splits/$dataset/${role}_manifest.json" \
        --questions "$questions" --corpus "$corpus" --bundle-root "$bundle" \
        --retrieval "$retrieval" --output-dir "$base/features/$dataset/$role" \
        --layers 3,7,11,15,19,23,27 --projection-dim 256 \
        --text-batch-size 4 --image-batch-size "$image_batch_size" \
        --max-length 1024 --min-pixels 3136 --max-pixels "$max_pixels" --checkpoint-every 1
    fi
  done

  analysis="$base/analysis/$dataset/fusion_analysis.json"
  if [[ ! -f "$analysis" ]]; then
    PYTHONPATH=src:. "$python_bin" research/internal_state_rag/analyze_qwen2vl_jina_ablation.py \
      --dataset "$dataset" \
      --probe-train "$base/features/$dataset/probe_train/features.npz" \
      --calibration "$base/features/$dataset/conformal_calibration/features.npz" \
      --test "$base/features/$dataset/test/features.npz" \
      --output "$analysis" \
      --predictions-output "$base/analysis/$dataset/fusion_predictions.npz" \
      --alphas 0.1 --bootstrap-samples 10000
  fi
done

PYTHONPATH=src:. "$python_bin" research/internal_state_rag/run_inverse_1000cal_internal_signal_ablation.py \
  --output-dir "$base" --model "$model" \
  --max-new-tokens 24 --min-pixels 3136 --max-pixels 200704
