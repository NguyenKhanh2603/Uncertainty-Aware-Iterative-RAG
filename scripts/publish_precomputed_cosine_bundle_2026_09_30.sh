#!/usr/bin/env bash
# Publish the completed portable score folder and route a large ZIP to Hugging Face.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

BRANCH="results/qwen2vl-jina4-four-datasets-2026-09-14"
RESULT_ROOT="research/internal_state_rag/results/precomputed_cosine_cal1000_official_test_2026_09_30"
PORTABLE="$RESULT_ROOT/portable_bundle"
ZIP_PATH="$RESULT_ROOT/precomputed_cosine_cal1000_official_test_2026_09_30.zip"
HF_REPO="danny2507/uncertainty-rag-precomputed-cosine-2026-09-30"
GITHUB_BASE="https://github.com/NguyenKhanh2603/Uncertainty-Aware-Iterative-RAG/tree/$BRANCH/$RESULT_ROOT"
GITHUB_BLOB="https://github.com/NguyenKhanh2603/Uncertainty-Aware-Iterative-RAG/blob/$BRANCH/$RESULT_ROOT"
GITHUB_LIMIT=$((50 * 1024 * 1024))

test -f "$PORTABLE/MANIFEST.json"
test -f "$ZIP_PATH"
zip_bytes="$(stat -c %s "$ZIP_PATH")"
publication="$RESULT_ROOT/PUBLICATION.md"

cat > "$publication" <<EOF
# Precomputed cosine publication

- [Portable GitHub folder]($GITHUB_BASE/portable_bundle)
- [Three-baseline report]($GITHUB_BLOB/three_baselines/REPORT.md)
- ZIP bytes: $zip_bytes
EOF

# The portable score files remain individually reviewable on GitHub.  Embedding
# caches, model outputs, and local corpus assets are deliberately excluded.
git add -f "$PORTABLE" "$RESULT_ROOT/three_baselines" "$RESULT_ROOT/by_selection/selection_summary.json" "$publication"

if (( zip_bytes <= GITHUB_LIMIT )); then
  git add -f "$ZIP_PATH"
  cat >> "$publication" <<EOF
- [ZIP on GitHub]($GITHUB_BLOB/$(basename "$ZIP_PATH")?raw=1)
EOF
  git add -f "$publication"
else
  hf repo create "$HF_REPO" --repo-type dataset --exist-ok
  hf upload "$HF_REPO" "$ZIP_PATH" "$(basename "$ZIP_PATH")" \
    --repo-type dataset \
    --commit-message "Add official-full precomputed cosine bundle" \
    --commit-description "Jina-v4 Top-30 cosine rows for 1,000 calibration queries plus the complete official labelled test splits; includes BY-ready reference banks and frozen split manifests."
  cat >> "$publication" <<EOF
- [ZIP on Hugging Face](https://huggingface.co/datasets/$HF_REPO/resolve/main/$(basename "$ZIP_PATH"))
EOF
  git add -f "$publication"
fi

git commit -m "results: publish official-full cosine bundle"
git push origin "$BRANCH"
