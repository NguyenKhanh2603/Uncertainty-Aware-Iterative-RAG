#!/usr/bin/env bash
set -euo pipefail
root=research/internal_state_rag/results
model=/workspace/hf_cache/hub/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/eed13092ef92e448dd6875b2a00151bd3f7db0ac
PYTHONPATH=src:. .venv-cu118/bin/python research/internal_state_rag/prepare_official_hotpot_feature_plans.py
for role in calibration test; do
  if [[ -f "$root/official_seed42_hotpot_features/$role/features.npz" ]]; then continue; fi
  PYTHONPATH=src:. .venv-cu118/bin/python research/internal_state_rag/run_qwen2vl_pairwise_features.py --model "$model" --model-revision "$(basename "$model")" --feature-manifest "$root/official_seed42_hotpotqa_plans/$role/manifest.json" --questions data/official_seed42_hotpotqa_fixed/hotpotqa/questions.jsonl --corpus data/official_seed42_hotpotqa_fixed/hotpotqa/corpus.jsonl --bundle-root data/official_seed42_hotpotqa_fixed --retrieval "$root/official_seed42_hotpotqa_cosine_top30.jsonl.gz" --output-dir "$root/official_seed42_hotpot_features/$role" --text-batch-size 4 --image-batch-size 2 --max-pixels 200704
done
