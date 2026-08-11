"""Upload the reproducible WebQ UQ-result bundle to a Hugging Face dataset repo.

Run this script on the remote Linux machine after ``hf auth login``.  It uploads
only result artifacts and the standalone evaluation code; it never uploads a
token, model checkpoint, cached model, or the source WebQ data.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = ROOT / "results" / "webq_paper_baselines"
DEFAULT_CODE = ROOT / "eval" / "UQ_Eval"
DEFAULT_REPO = "danny2507/ragu-webq-mistral7b-uq-results"


DATASET_CARD = """---
language:
- en
tags:
- uncertainty-quantification
- retrieval-augmented-generation
- webquestions
- mistral-7b
license: mit
---

# RAGU WebQ Mistral-7B uncertainty results

Reproducibility artifacts for a WebQuestions uncertainty-quantification
comparison using the frozen RAGU 400-example seed-10 subset, top-5
Contriever-MSMARCO contexts, and `mistralai/Mistral-7B-Instruct-v0.3`.

## Contents

- `results/`: per-example JSONL outputs and summary JSON files for PPL,
  regular entropy, RAGU semantic entropy, this project's claim-level semantic
  uncertainty, token uncertainty, and p(True).
- `code/`: standalone scripts used to generate and score the artifacts.

## Protocol

All UQ scores consume the same saved greedy answer and ten sampled answers.
The primary label in these artifacts is RAGU raw `Acc` (normalized gold-answer
containment), not the paper's Qwen-based `AccLM`. The results therefore support
an internally paired comparison, but are not a direct reproduction of the
paper's AccLM Table-1 values.

The p(True) run uses regenerated seed-10 training demonstrations with raw-Acc
labels. It should be treated as a raw-Acc reproduction rather than the paper's
exact AccLM-supervised p(True) configuration.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload WebQ UQ artifacts to Hugging Face")
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--code-dir", type=Path, default=DEFAULT_CODE)
    parser.add_argument("--private", action="store_true", help="Create/update a private dataset repository")
    parser.add_argument("--revision", default="main")
    return parser.parse_args()


def copy_files(source: Path, destination: Path, suffixes: set[str]) -> None:
    for item in source.rglob("*"):
        if item.is_file() and item.suffix in suffixes:
            relative = item.relative_to(source)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)


def main() -> None:
    args = parse_args()
    if not args.results_dir.is_dir():
        raise FileNotFoundError(f"Results directory not found: {args.results_dir}")
    if not args.code_dir.is_dir():
        raise FileNotFoundError(f"Code directory not found: {args.code_dir}")

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="webq-uq-hf-") as temporary:
        staging = Path(temporary)
        (staging / "README.md").write_text(DATASET_CARD, encoding="utf-8")
        copy_files(args.results_dir, staging / "results", {".jsonl", ".json"})
        copy_files(args.code_dir, staging / "code", {".py", ".md"})
        api.upload_folder(
            repo_id=args.repo_id,
            repo_type="dataset",
            folder_path=str(staging),
            path_in_repo=".",
            revision=args.revision,
            commit_message="Upload WebQ Mistral-7B UQ evaluation artifacts",
        )
    print(f"Uploaded to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
