# Inverse calibration-size experiment: 1,000 calibration / 100 test

This is the inverse-scale companion to `all_datasets_cosine_six_methods_2026_09_24`: it calibrates every cosine-only selector on 1,000 disjoint queries per dataset and evaluates it on 100 held-out queries per dataset.

`SPLIT_INTEGRITY.json` and the per-dataset manifests record every qid and confirm zero calibration/test overlap. HotpotQA uses its official-seed calibration role. MMQA, TATQA, and WebQA each use the union of their frozen `calibration` and `development` roles, which contains exactly 1,000 queries; all four datasets use the first 100 lexicographically sorted qids from their frozen `test` role for evaluation.

The result table contains Fixed Top-K, CCE, CONFLARE, TRAQ-retrieval, query-level cosine, BY cosine, and BH cosine. It deliberately excludes the internal-fusion row: that method selects a hidden-state probe and fusion weights on an additional, disjoint probe-training set. Reusing its old 100-query calibration artifact would violate this inverse calibration protocol.

Run:

```bash
PYTHONPATH=src:. .venv-cu118/bin/python research/internal_state_rag/run_all_datasets_cosine_six_methods.py \
  --output-dir research/internal_state_rag/results/all_datasets_cosine_six_methods_1000cal_100test_2026_09_26 \
  --plan-root research/internal_state_rag/results/all_datasets_cosine_six_methods_1000cal_100test_2026_09_26/splits \
  --model /workspace/hf_cache/hub/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/eed13092ef92e448dd6875b2a00151bd3f7db0ac \
  --max-new-tokens 24 --min-pixels 3136 --max-pixels 200704
```
