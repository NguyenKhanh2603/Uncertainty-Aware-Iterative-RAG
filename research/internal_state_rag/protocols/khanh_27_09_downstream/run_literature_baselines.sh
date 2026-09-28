#!/usr/bin/env bash
# Reproduce the three completed literature retrieval-adaptation rows.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv-cu118/bin/python}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2-VL-7B-Instruct}"
MODEL_REVISION="${MODEL_REVISION:-eed13092ef92e448dd6875b2a00151bd3f7db0ac}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/research/internal_state_rag/results/khanh_27_09_literature_reproduction}"
SPLIT_ROOT="$ROOT/research/internal_state_rag/results/full_context_selection_splits_khanh_27_09_2026_09_27/splits"
METHODS="cce_conformal_embedding_jina_alpha_0.10,conflare_source_question_jina_alpha_0.10,traq_retrieval_bonferroni_jina_alpha_0.10"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "PYTHON_BIN is not executable: $PYTHON_BIN" >&2
  exit 2
fi

cd "$ROOT"
PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON_BIN" -m research.internal_state_rag.run_literature_protocol_1000cal \
  --input-profile zip_20_09 \
  --plan-root "$SPLIT_ROOT" \
  --datasets "hotpotqa,mmqa,tatqa,webqa" \
  --methods "$METHODS" \
  --alpha 0.10 \
  --model "$MODEL_PATH" \
  --model-revision "$MODEL_REVISION" \
  --max-new-tokens 24 \
  --min-pixels 3136 \
  --max-pixels 200704 \
  --output-dir "$OUTPUT_DIR"
