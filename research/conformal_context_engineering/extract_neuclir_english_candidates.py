"""Extract the English NeuCLIR documents needed by CoverageBench ranking lists.

The full NeuCLIR English translation is streamed; only documents that occur in a
chosen run are materialized.  This keeps the downstream common-context
comparison reproducible without duplicating the full corpus.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

DEFAULT_CORPUS = Path("data/external/neuclir1_english/data")
DEFAULT_RUN = Path(
    "data/external/coveragebench_neuclir/ranking/NeuCLIR/Initial-Retrieval/initial_qwen3.json"
)
DEFAULT_OUTPUT = Path(
    "data/external/neuclir1_english/candidates/coveragebench_initial_qwen3_top100.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = json.loads(args.run.read_text())
    wanted = {
        docid
        for scores in run.values()
        for docid, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)[: args.top_k]
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    found: set[str] = set()
    total = 0
    with args.output.open("w", encoding="utf-8") as out:
        for shard in sorted(args.corpus_dir.glob("*.jsonl.gz")):
            shard_found = 0
            with gzip.open(shard, "rt", encoding="utf-8") as lines:
                for line in lines:
                    total += 1
                    record = json.loads(line)
                    docid = record["id"]
                    if docid in wanted:
                        out.write(json.dumps(record, ensure_ascii=False) + "\n")
                        found.add(docid)
                        shard_found += 1
            print(f"{shard.name}: found {shard_found}; scanned {total}", flush=True)
    missing = sorted(wanted - found)
    summary = {
        "run": str(args.run),
        "top_k_per_topic": args.top_k,
        "topics": len(run),
        "requested_unique_documents": len(wanted),
        "extracted_documents": len(found),
        "missing_documents": len(missing),
        "missing_docids": missing,
        "output": str(args.output),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
