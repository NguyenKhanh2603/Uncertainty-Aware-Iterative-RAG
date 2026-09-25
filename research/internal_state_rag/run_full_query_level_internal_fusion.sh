#!/usr/bin/env bash
# Build the matched full-test query-level cosine + internal-fusion column.
#
# Probe fitting uses the existing disjoint 100-query probe split; threshold
# calibration uses the existing disjoint 100-query calibration split.  The
# frozen test role is 1,000 queries for HotpotQA/MMQA/TAT-QA and 250 for WebQA.
# The script is resumable: pairwise features checkpoint per query and
# downstream predictions checkpoint per qid.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

python_bin=".venv-cu118/bin/python"
model="/workspace/hf_cache/hub/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/eed13092ef92e448dd6875b2a00151bd3f7db0ac"
revision="eed13092ef92e448dd6875b2a00151bd3f7db0ac"
base="research/internal_state_rag/results/qwen2vl_7b_jina4/full_test_internal_fusion_2026_09_25"
plans="$base/plans"
features="$base/features"
analysis="$base/analysis"
downstream="$base/downstream"
if (($#)); then
  datasets=("$@")
else
  datasets=(hotpotqa mmqa tatqa webqa)
fi

PYTHONPATH=src:. "$python_bin" research/internal_state_rag/prepare_full_internal_fusion_plans.py \
  --output-dir "$plans" --datasets "$(IFS=,; echo "${datasets[*]}")"

for dataset in "${datasets[@]}"; do
  case "$dataset" in
    hotpotqa)
      questions="data/official_seed42_hotpotqa_fixed/hotpotqa/questions.jsonl"
      corpus="data/official_seed42_hotpotqa_fixed/hotpotqa/corpus.jsonl"
      bundle="data/official_seed42_hotpotqa_fixed"
      retrieval="research/internal_state_rag/results/official_seed42_hotpotqa_cosine_top30.jsonl.gz"
      test_features="research/internal_state_rag/results/qwen2vl_7b_jina4/full_test_1000_features/hotpotqa/features.npz"
      max_pixels=200704; image_batch_size=2
      ;;
    mmqa|tatqa)
      questions="data/conformal_global_run/official_bundle_role_split_${dataset}/${dataset}/questions.jsonl"
      corpus="data/conformal_global_run/official_bundle_role_split_${dataset}/${dataset}/corpus.jsonl"
      bundle="data/conformal_global_run/official_bundle_role_split_${dataset}"
      retrieval="research/internal_state_rag/results/qwen2vl_jina4/${dataset}_jina_v4_top30.jsonl.gz"
      test_features="$features/$dataset/features.npz"
      max_pixels=200704; image_batch_size=2
      ;;
    webqa)
      questions="data/conformal_global_run/official_bundle_1000_webqa/webqa/questions.jsonl"
      corpus="data/conformal_global_run/official_bundle_1000_webqa/webqa/corpus.jsonl"
      bundle="data/conformal_global_run/official_bundle_1000_webqa"
      retrieval="research/internal_state_rag/results/qwen2vl_jina4/webqa_jina_v4_top30.jsonl.gz"
      test_features="$features/$dataset/features.npz"
      # Match WebQA's existing probe/calibration feature pixel cap.
      max_pixels=262144; image_batch_size=4
      ;;
    *) echo "unknown dataset: $dataset" >&2; exit 2 ;;
  esac

  if [[ "$dataset" != hotpotqa && ! -f "$test_features" ]]; then
    PYTHONPATH=src:. "$python_bin" research/internal_state_rag/run_qwen2vl_pairwise_features.py \
      --model "$model" --model-revision "$revision" \
      --feature-manifest "$plans/$dataset/manifest.json" \
      --questions "$questions" --corpus "$corpus" --bundle-root "$bundle" \
      --retrieval "$retrieval" --output-dir "$features/$dataset" \
      --layers 3,7,11,15,19,23,27 --projection-dim 256 \
      --text-batch-size 4 --image-batch-size "$image_batch_size" \
      --max-length 1024 --min-pixels 3136 --max-pixels "$max_pixels" --checkpoint-every 1
  fi

  if [[ ! -f "$analysis/$dataset/fusion_predictions.npz" ]]; then
    PYTHONPATH=src:. "$python_bin" research/internal_state_rag/analyze_qwen2vl_jina_ablation.py \
      --dataset "$dataset" \
      --probe-train "research/internal_state_rag/results/qwen2vl_7b_jina4/fusion_features/$dataset/probe_train/features.npz" \
      --calibration "research/internal_state_rag/results/qwen2vl_7b_jina4/fusion_features/$dataset/calibration/features.npz" \
      --test "$test_features" --output "$analysis/$dataset/fusion_analysis.json" \
      --predictions-output "$analysis/$dataset/fusion_predictions.npz" \
      --alphas 0.1 --bootstrap-samples 10000
  fi

  PYTHONPATH=src:. "$python_bin" research/internal_state_rag/run_all_datasets_cosine_six_methods.py \
    --datasets "$dataset" --methods query_level_cosine_internal_fusion_alpha_0.10 \
    --internal-fusion-predictions "$analysis/$dataset/fusion_predictions.npz" \
    --output-dir "$downstream/$dataset" --model "$model" \
    --max-new-tokens 24 --min-pixels 3136 --max-pixels 200704
done
