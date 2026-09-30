#!/usr/bin/env bash
# Resumable full official evaluation for CCE, CONFLARE, and TRAQ selectors.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH=src:.
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

PYTHON_BIN="${PYTHON_BIN:-.venv-cu118/bin/python}"
SPLIT_ROOT="research/internal_state_rag/splits/official_evaluable_test_2026_09_30"
BUNDLE_ROOT="data/official_evaluable_test_2026_09_30"
OUTPUT_ROOT="research/internal_state_rag/results/full_context_selection_eligible_full_2026_09_30"
RETRIEVAL_ROOT="$OUTPUT_ROOT/retrieval"
CACHE_ROOT="$OUTPUT_ROOT/embedding_cache"
QWEN_MODEL="/workspace/hf_cache/hub/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/eed13092ef92e448dd6875b2a00151bd3f7db0ac"
QWEN_REVISION="eed13092ef92e448dd6875b2a00151bd3f7db0ac"
JINA_MODEL="jinaai/jina-embeddings-v4"
JINA_REVISION="853c867b65b749f3c3c72a06868140d842e04f06"

mkdir -p "$RETRIEVAL_ROOT" "$CACHE_ROOT" "$OUTPUT_ROOT/logs"

for dataset in tatqa hotpotqa mmqa webqa; do
  echo "===== MATERIALIZE $dataset ====="
  "$PYTHON_BIN" scripts/build_official_evaluable_bundle.py \
    --split-root "$SPLIT_ROOT" --output-dir "$BUNDLE_ROOT" --datasets "$dataset"

  reuse_cache="data/zip_calibration_split_20_09/embedding_cache"
  if [[ "$dataset" == "webqa" ]]; then
    reuse_cache="data/zip_calibration_split_20_09_webqa/embedding_cache"
  fi
  text_batch_size=8
  if [[ "$dataset" == "hotpotqa" ]]; then
    # HotpotQA chunks are short text passages; the A100 safely supports the
    # larger batch and avoids spending most of the run in per-batch overhead.
    text_batch_size=32
  fi
  echo "===== SEED/ENCODE $dataset CORPUS ====="
  "$PYTHON_BIN" scripts/seed_official_evaluable_embedding_cache.py \
    --dataset "$dataset" --bundle-dir "$BUNDLE_ROOT" \
    --cache-dir "$CACHE_ROOT/$dataset" --reuse-cache-dir "$reuse_cache" \
    --device cuda --text-batch-size "$text_batch_size" \
    --image-batch-size 4 --checkpoint-every 128

  retrieval="$RETRIEVAL_ROOT/${dataset}_jina_v4_candidates.jsonl.gz"
  if [[ ! -f "${retrieval}.manifest.json" ]]; then
    echo "===== RETRIEVE $dataset ====="
    "$PYTHON_BIN" scripts/generate_conformal_retrieval_log.py \
      --bundle-dir "$BUNDLE_ROOT" --dataset "$dataset" --output "$retrieval" \
      --cache-dir "$CACHE_ROOT/$dataset" \
      --model "$JINA_MODEL" --model-revision "$JINA_REVISION" \
      --device cuda --dtype bfloat16 --truncate-dim 512 \
      --text-batch-size "$text_batch_size" --image-batch-size 4 --checkpoint-every 128 \
      --max-text-length 1024 --max-image-pixels 200704 \
      --top-l 30 --candidate-scope official_pool --retrieval-mode global \
      --split-policy bundle_roles --data-grade paper
  fi

  echo "===== SUPPORT METRICS $dataset ====="
  "$PYTHON_BIN" research/internal_state_rag/run_eligible_full_three_baselines.py \
    --output-dir "$OUTPUT_ROOT" --split-root "$SPLIT_ROOT" \
    --bundle-dir "$BUNDLE_ROOT" --retrieval-root "$RETRIEVAL_ROOT" \
    --datasets "$dataset" --selection-only

  echo "===== DOWNSTREAM $dataset ====="
  "$PYTHON_BIN" research/internal_state_rag/run_eligible_full_three_baselines.py \
    --output-dir "$OUTPUT_ROOT" --split-root "$SPLIT_ROOT" \
    --bundle-dir "$BUNDLE_ROOT" --retrieval-root "$RETRIEVAL_ROOT" \
    --datasets "$dataset" --model "$QWEN_MODEL" --model-revision "$QWEN_REVISION" \
    --max-new-tokens 24 --min-pixels 3136 --max-pixels 200704
done

echo "All four eligible-full datasets are complete."
