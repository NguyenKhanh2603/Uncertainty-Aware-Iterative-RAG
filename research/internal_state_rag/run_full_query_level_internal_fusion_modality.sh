#!/usr/bin/env bash
# Run the modality-conditioned query-level cosine + internal-fusion ablation.
#
# This script deliberately waits for the matched pooled run to complete before
# it uses the A100. It reuses its exact full-test features and its separately
# frozen 100-query probe-train and 100-query calibration roles. The primary
# rule is a Bonferroni modality allocation selected only with probe OOF scores.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

python_bin=".venv-cu118/bin/python"
model="/workspace/hf_cache/hub/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/eed13092ef92e448dd6875b2a00151bd3f7db0ac"
pooled="research/internal_state_rag/results/qwen2vl_7b_jina4/full_test_internal_fusion_2026_09_25_v2"
output="$pooled/modality_aware_bonferroni_v1"
if (($#)); then
  datasets=("$@")
else
  datasets=(hotpotqa mmqa tatqa webqa)
fi

is_complete() {
  "$python_bin" - "$1" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except (FileNotFoundError, json.JSONDecodeError):
    raise SystemExit(1)
raise SystemExit(0 if payload.get("status") == "complete" else 1)
PY
}

# The pooled run consumes the GPU during feature extraction and answer
# generation. Waiting here preserves that run exactly and prevents cache or GPU
# contention. Its per-dataset summary is written only after all answers finish.
while :; do
  ready=1
  for dataset in "${datasets[@]}"; do
    if ! is_complete "$pooled/downstream/$dataset/summary.json"; then
      ready=0
      break
    fi
  done
  if ((ready)); then
    break
  fi
  echo "$(date -Is) waiting for pooled internal-fusion run to finish" >&2
  sleep 60
done

for dataset in "${datasets[@]}"; do
  probe="research/internal_state_rag/results/qwen2vl_7b_jina4/fusion_features/$dataset/probe_train/features.npz"
  calibration="research/internal_state_rag/results/qwen2vl_7b_jina4/fusion_features/$dataset/calibration/features.npz"
  test="$pooled/features/$dataset/features.npz"
  pooled_analysis="$pooled/analysis/$dataset/fusion_analysis.json"
  analysis="$output/analysis/$dataset"
  downstream="$output/downstream/$dataset"

  if [[ ! -s "$analysis/modality_fusion_predictions.npz" ]]; then
    PYTHONPATH=src:. "$python_bin" research/internal_state_rag/analyze_modality_conditional_conformal.py \
      --probe-train "$probe" --calibration "$calibration" --test "$test" \
      --ablation-report "$pooled_analysis" \
      --output "$analysis/modality_fusion_analysis.json" \
      --predictions-output "$analysis/modality_fusion_predictions.npz" \
      --methods cosine_internal --alpha 0.10 --allocation-step 0.005 \
      --bootstrap-samples 10000
  fi

  if ! is_complete "$downstream/summary.json"; then
    PYTHONPATH=src:. "$python_bin" research/internal_state_rag/run_all_datasets_cosine_six_methods.py \
      --datasets "$dataset" \
      --methods query_level_cosine_internal_mondrian_bonferroni_alpha_0.10 \
      --internal-fusion-predictions "$analysis/modality_fusion_predictions.npz" \
      --internal-fusion-mask-key mask_cosine_internal_mondrian_bonferroni_probe_allocated \
      --internal-fusion-method-name query_level_cosine_internal_mondrian_bonferroni_alpha_0.10 \
      --output-dir "$downstream" --model "$model" \
      --max-new-tokens 24 --min-pixels 3136 --max-pixels 200704
  fi
done
