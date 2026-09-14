"""Create a deterministic query subset and its exact official candidate-union corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-bundle", type=Path, required=True)
    parser.add_argument("--train-queries", type=int, required=True)
    parser.add_argument("--heldout-queries", type=int, required=True)
    parser.add_argument("--heldout-split", default="validation")
    parser.add_argument("--seed", type=int, default=733)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def select(rows: list[dict], split: str, n: int, seed: int) -> list[dict]:
    eligible = [row for row in rows if row.get("metadata", {}).get("source_split") == split]
    if n > len(eligible):
        raise ValueError(f"Requested {n} {split} rows, only {len(eligible)} available")
    if n == len(eligible):
        return eligible
    indices = sorted(random.Random(seed).sample(range(len(eligible)), n))
    return [eligible[index] for index in indices]


def main() -> None:
    args = parse_args()
    source_dir = args.source_bundle / args.dataset
    questions = load_rows(source_dir / "questions.jsonl")
    selected = select(questions, "train", args.train_queries, args.seed) + select(
        questions, args.heldout_split, args.heldout_queries, args.seed + 1
    )
    candidate_ids = {
        str(chunk_id) for row in selected for chunk_id in row.get("candidate_ids", [])
    }
    corpus = [
        row
        for row in load_rows(source_dir / "corpus.jsonl")
        if str(row["id"]) in candidate_ids
    ]
    if {str(row["id"]) for row in corpus} != candidate_ids:
        raise ValueError("Selected questions reference corpus rows that are missing")

    output_dir = args.output_bundle / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    for row in corpus:
        if row.get("modality") == "image":
            content = Path(str(row["content"]))
            if not content.is_absolute():
                row["content"] = str((args.source_bundle / content).resolve())
    for name, rows in (("questions.jsonl", selected), ("corpus.jsonl", corpus)):
        with (output_dir / name).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    qids = [str(row["qid"]) for row in selected]
    manifest = {
        "schema_version": 1,
        "data_grade": "official_deterministic_subset",
        "dataset": args.dataset,
        "source_bundle": str(args.source_bundle.resolve()),
        "train_queries": args.train_queries,
        "heldout_queries": args.heldout_queries,
        "heldout_split": args.heldout_split,
        "seed": args.seed,
        "questions": len(selected),
        "corpus": len(corpus),
        "qid_sha256": hashlib.sha256("\n".join(qids).encode()).hexdigest(),
    }
    (args.output_bundle / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
