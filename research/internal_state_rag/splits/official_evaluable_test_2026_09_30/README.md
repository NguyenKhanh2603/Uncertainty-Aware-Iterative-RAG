# Full official evaluable test manifests — 2026-09-30

This folder freezes the complete official **public-label evaluation split** used for local
post-retrieval precision/recall and downstream QA evaluation.

| Dataset | Official source split used as local test | Queries |
|---|---|---:|
| HotpotQA | `distractor/validation` | 7,405 |
| MultiModalQA | `dev` | 2,441 |
| TAT-QA | `dev` | 1,668 |
| WebQA | `validation` | 4,966 |
| **Total** |  | **16,480** |

These are the full official splits with public answer and support labels. They are distinct
from the hidden/unlabelled benchmark test files and from the prior 100-query experimental
subset.

Each dataset directory contains:

- `test_manifest.json`: qids in official source order, pinned source revision, source-file
  SHA-256, qid count and qid-list SHA-256;
- `test_qids.txt`: one qid per line for shell scripts and long-running jobs.

`SPLIT_SUMMARY.json` records all counts and paths. No query text, answer, image or corpus asset
is duplicated in this folder.

## Rebuild and validate

From the repository root:

```bash
uv run --no-sync python \
  research/internal_state_rag/splits/official_evaluable_test_2026_09_30/build_manifests.py
```

The builder downloads only lightweight annotations plus the HotpotQA validation parquet. It
fails if a count differs from 7,405 / 2,441 / 1,668 / 4,966, if a qid is empty, or if a qid is
duplicated.
