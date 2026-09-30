#!/usr/bin/env bash
# Build one fingerprint-consistent cosine bundle for BY and the three baselines.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH=src:.
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

PYTHON_BIN="${PYTHON_BIN:-.venv-cu118/bin/python}"
DATASETS="hotpotqa mmqa tatqa webqa"
BUNDLE_ROOT="data/precomputed_cosine_cal1000_official_test_2026_09_30"
RESULT_ROOT="research/internal_state_rag/results/precomputed_cosine_cal1000_official_test_2026_09_30"
RETRIEVAL_ROOT="$RESULT_ROOT/retrieval"
CACHE_ROOT="$RESULT_ROOT/embedding_cache"
SPLIT_ROOT="$RESULT_ROOT/splits"
OFFICIAL_RESULT_ROOT="research/internal_state_rag/results/full_context_selection_eligible_full_2026_09_30"
CALIBRATION_CACHE="data/zip_calibration_split_20_09/embedding_cache"
WEBQA_CALIBRATION_CACHE="data/zip_calibration_split_20_09_webqa/embedding_cache"
CALIBRATION_SPLITS="research/internal_state_rag/results/full_context_selection_splits_khanh_27_09_2026_09_27/splits"
TEST_SPLITS="research/internal_state_rag/splits/official_evaluable_test_2026_09_30"
JINA_MODEL="jinaai/jina-embeddings-v4"
JINA_REVISION="853c867b65b749f3c3c72a06868140d842e04f06"
QWEN_MODEL="/workspace/hf_cache/hub/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/eed13092ef92e448dd6875b2a00151bd3f7db0ac"
QWEN_REVISION="eed13092ef92e448dd6875b2a00151bd3f7db0ac"

mkdir -p "$RESULT_ROOT" "$RETRIEVAL_ROOT" "$CACHE_ROOT" "$SPLIT_ROOT"

echo "===== BUILD SHARED CALIBRATION+TEST BUNDLE ====="
"$PYTHON_BIN" scripts/build_calibration_official_test_bundle.py \
  --output-dir "$BUNDLE_ROOT" --datasets "${DATASETS// /,}"

for dataset in $DATASETS; do
  mkdir -p "$SPLIT_ROOT/$dataset"
  cp "$CALIBRATION_SPLITS/$dataset/calibration_manifest.json" "$SPLIT_ROOT/$dataset/"
  cp "$TEST_SPLITS/$dataset/test_manifest.json" "$SPLIT_ROOT/$dataset/"

  reuse_calibration="$CALIBRATION_CACHE"
  if [[ "$dataset" == "webqa" ]]; then
    reuse_calibration="$WEBQA_CALIBRATION_CACHE"
  fi
  text_batch_size=8
  image_batch_size=16
  if [[ "$dataset" == "hotpotqa" ]]; then
    text_batch_size=32
  fi

  echo "===== MERGE/ENCODE $dataset EMBEDDINGS ====="
  "$PYTHON_BIN" scripts/seed_official_evaluable_embedding_cache.py \
    --dataset "$dataset" --bundle-dir "$BUNDLE_ROOT" \
    --cache-dir "$CACHE_ROOT/$dataset" \
    --reuse-cache-dir "$reuse_calibration" \
    --reuse-cache-dir "$OFFICIAL_RESULT_ROOT/embedding_cache/$dataset" \
    --device cuda --text-batch-size "$text_batch_size" \
    --image-batch-size "$image_batch_size" --checkpoint-every 128

  retrieval="$RETRIEVAL_ROOT/${dataset}_top30_retrieval.jsonl.gz"
  if [[ ! -f "${retrieval}.manifest.json" ]]; then
    echo "===== PRECOMPUTE $dataset QUERY COSINES ====="
    "$PYTHON_BIN" scripts/generate_conformal_retrieval_log.py \
      --bundle-dir "$BUNDLE_ROOT" --dataset "$dataset" --output "$retrieval" \
      --cache-dir "$CACHE_ROOT/$dataset" \
      --model "$JINA_MODEL" --model-revision "$JINA_REVISION" \
      --device cuda --dtype bfloat16 --truncate-dim 512 \
      --text-batch-size "$text_batch_size" --image-batch-size "$image_batch_size" \
      --query-batch-size 64 --corpus-block-size 16384 --checkpoint-every 128 \
      --max-text-length 1024 --max-image-pixels 200704 \
      --top-l 30 --candidate-scope official_pool --retrieval-mode global \
      --split-policy bundle_roles --data-grade paper
  fi
  ln -sfn "${dataset}_top30_retrieval.jsonl.gz" \
    "$RETRIEVAL_ROOT/${dataset}_jina_v4_candidates.jsonl.gz"
done

inputs=()
for dataset in $DATASETS; do
  inputs+=("$RETRIEVAL_ROOT/${dataset}_top30_retrieval.jsonl.gz")
done

echo "===== BUILD FALSE-SCORE BANKS ====="
"$PYTHON_BIN" scripts/prepare_conformal_reference_banks.py \
  --input "${inputs[@]}" \
  --output "$RESULT_ROOT/combined_reference_banks.json.gz" \
  --rank-bins "1-3,4-10,11-30" --conditioning "dataset,modality" \
  --min-bank-size 1000 --allow-small-banks

echo "===== RUN BY ON OFFICIAL TEST ====="
"$PYTHON_BIN" scripts/run_conformal_backfill_selection.py \
  --input "${inputs[@]}" --bank "$RESULT_ROOT/combined_reference_banks.json.gz" \
  --output-dir "$RESULT_ROOT/by_selection" --split-role test \
  --alpha 0.10 --max-context 10 --allow-underpowered-banks --compare-pooled

echo "===== RUN THREE BASELINES + VALIDATE/RESUME DOWNSTREAM ====="
"$PYTHON_BIN" research/internal_state_rag/run_eligible_full_three_baselines.py \
  --output-dir "$OFFICIAL_RESULT_ROOT" --split-root "$TEST_SPLITS" \
  --bundle-dir "$BUNDLE_ROOT" --retrieval-root "$RETRIEVAL_ROOT" \
  --datasets "${DATASETS// /,}" --model "$QWEN_MODEL" \
  --model-revision "$QWEN_REVISION" --max-new-tokens 24 \
  --min-pixels 3136 --max-pixels 200704

mkdir -p "$RESULT_ROOT/three_baselines"
cp "$OFFICIAL_RESULT_ROOT/REPORT.md" "$RESULT_ROOT/three_baselines/REPORT.md"
cp "$OFFICIAL_RESULT_ROOT/summary.json" "$RESULT_ROOT/three_baselines/summary.json"

echo "===== EXPORT PORTABLE QUERYWISE BUNDLE ====="
"$PYTHON_BIN" scripts/export_precomputed_cosine_bundle.py \
  --retrieval-dir "$RETRIEVAL_ROOT" \
  --bank "$RESULT_ROOT/combined_reference_banks.json.gz" \
  --selection-dir "$RESULT_ROOT/by_selection" --split-root "$SPLIT_ROOT" \
  --output-dir "$RESULT_ROOT/portable_bundle" \
  --zip "$RESULT_ROOT/precomputed_cosine_cal1000_official_test_2026_09_30.zip"

echo "Precomputed cosine bundle, BY, three baselines, and downstream are complete."
