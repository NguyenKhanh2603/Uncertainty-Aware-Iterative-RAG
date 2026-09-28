# Khánh 27-09 downstream comparison protocol

This folder fixes the downstream protocol for comparison on `splits_khanh_27_09.zip`. Each dataset has 1,000 calibration qids and 100 disjoint held-out test qids. Retrieval is frozen: the only input a comparison method may change is the subset of chunk IDs retained from its test-query candidate pool.

`PROTOCOL.json` is the machine-readable contract. The fixed generator is `Qwen/Qwen2-VL-7B-Instruct`, evaluated with greedy decoding, `max_new_tokens=24`, `min_pixels=3136`, and `max_pixels=200704`. The prompt is exactly:

```text
Answer using only the supplied context. Return only the short answer.
Question: {question}
```

The evaluator preserves frozen Jina retrieval-rank order after a method selects chunks. It supports text, table, and image chunks through Qwen2-VL. It reports exact match, token F1, and numerical accuracy against the stored gold answers.

## Run an external method

Write one JSONL row for every one of the 100 held-out qids of a dataset:

```json
{"qid":"5ac0630f5542992a796ded12","selected_chunk_ids":["hotpot:...","hotpot:..."]}
```

`selected_chunk_ids` must be a duplicate-free subset of that query's frozen Jina candidate pool. The evaluator validates all qids and chunk IDs before loading Qwen. It does not accept calibration qids, missing qids, or chunks from outside the candidate pool.

First validate an external selector without allocating the model:

```bash
PYTHONPATH=. .venv-cu118/bin/python \
  research/internal_state_rag/protocols/khanh_27_09_downstream/evaluate_external_selection.py \
  --dataset hotpotqa \
  --method my_method \
  --selection-jsonl /path/to/my_hotpotqa_selection.jsonl \
  --output-dir research/internal_state_rag/results/khanh_27_09_my_method/hotpotqa \
  --dry-run
```

Then omit `--dry-run` to run Qwen. The output directory is append-only and resumable: `predictions.jsonl`, `SUMMARY.json`, and `RUN_CONFIG.json` are written there. To compare all four datasets, run the same command once per dataset with that dataset's selection JSONL.

The evaluator pins model revision `eed13092ef92e448dd6875b2a00151bd3f7db0ac` by default; override it only when deliberately running a different model revision. The reported runs used the repository's CUDA-compatible `.venv-cu118/bin/python`. Set `PYTHON_BIN` or run the evaluator with another environment only when it has CUDA-enabled PyTorch, Qwen2-VL dependencies, and access to the Qwen checkpoint. Set `WEBQA_BUNDLE_ROOT` if WebQA's staged image/data bundle is stored somewhere other than `/dev/shm/uncertainty_rag_webqa_stage_20260922`.

## Reproduce the three existing retrieval-adaptation rows

```bash
PYTHON_BIN=.venv-cu118/bin/python \
MODEL_PATH=Qwen/Qwen2-VL-7B-Instruct \
research/internal_state_rag/protocols/khanh_27_09_downstream/run_literature_baselines.sh
```

This runs CCE Conformal-Embedding, CONFLARE source-question, and TRAQ retrieval Bonferroni using the same split, frozen Jina candidates, Qwen configuration, and output schema. It writes a fresh result directory unless `OUTPUT_DIR` is set.

## Frozen inputs

- Qid manifests: `research/internal_state_rag/results/full_context_selection_splits_khanh_27_09_2026_09_27/splits/`
- Retrieval logs: `research/internal_state_rag/results/zip_calibration_split_20_09/retrieval/`
- Completed literature result: `research/internal_state_rag/results/full_context_selection_splits_khanh_27_09_2026_09_27/REPORT.md`
- Original run settings: `research/internal_state_rag/results/full_context_selection_splits_khanh_27_09_2026_09_27/RUN_CONFIG.json`

The comparison runner evaluates context-selection output. It does not reretrieve or rerank documents.
