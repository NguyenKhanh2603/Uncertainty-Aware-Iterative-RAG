"""Chunk the complete MultiHop-RAG corpus into a deterministic BM25 collection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import read_json


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
DEFAULT_CORPUS = PROJECT_ROOT / "data" / "multihop_rag" / "corpus.json"
DEFAULT_OUTPUT = HERE / "artifacts" / "corpus_chunks.jsonl"


def chunk_words(words: list[str], size: int, overlap: int):
    step = size - overlap
    for start in range(0, len(words), step):
        part = words[start : start + size]
        if part:
            yield start, part
        if start + size >= len(words):
            break


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunk-words", type=int, default=256)
    parser.add_argument("--overlap-words", type=int, default=32)
    args = parser.parse_args()
    if args.chunk_words < 32 or not 0 <= args.overlap_words < args.chunk_words:
        parser.error("chunk size must be >= 32 and overlap must be in [0, chunk size)")

    documents = read_json(args.corpus)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open("w", encoding="utf-8") as output:
        for document_index, document in enumerate(documents):
            words = str(document.get("body", "")).split()
            for chunk_index, (_, chunk) in enumerate(
                chunk_words(words, args.chunk_words, args.overlap_words)
            ):
                row = {
                    "id": f"doc-{document_index}-chunk-{chunk_index}",
                    "document_index": document_index,
                    "chunk_index": chunk_index,
                    "title": document.get("title", ""),
                    "url": document.get("url", ""),
                    "source": document.get("source", ""),
                    "published_at": document.get("published_at", ""),
                    "text": " ".join(chunk),
                }
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1

    manifest = {
        "source": str(args.corpus.resolve()),
        "documents": len(documents),
        "chunks": count,
        "chunk_words": args.chunk_words,
        "overlap_words": args.overlap_words,
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

