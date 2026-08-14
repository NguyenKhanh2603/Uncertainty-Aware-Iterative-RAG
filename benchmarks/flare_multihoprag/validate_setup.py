"""Validate downloaded data, evidence labels, chunks, and local BM25 retrieval."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import BM25Corpus, read_json


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--queries", type=Path, default=PROJECT_ROOT / "data/multihop_rag/MultiHopRAG.json"
    )
    parser.add_argument(
        "--corpus", type=Path, default=PROJECT_ROOT / "data/multihop_rag/corpus.json"
    )
    parser.add_argument(
        "--chunks", type=Path, default=HERE / "artifacts/corpus_chunks.jsonl"
    )
    args = parser.parse_args()

    questions = read_json(args.queries)
    documents = read_json(args.corpus)
    if len(questions) != 2556:
        raise ValueError(f"Expected 2556 questions, found {len(questions)}")
    if len(documents) != 609:
        raise ValueError(f"Expected 609 corpus documents, found {len(documents)}")
    corpus_urls = {document["url"] for document in documents}
    missing = [
        evidence["url"]
        for question in questions
        for evidence in question["evidence_list"]
        if evidence["url"] not in corpus_urls
    ]
    if missing:
        raise ValueError(f"Found {len(missing)} evidence URLs absent from corpus")
    if not args.chunks.exists():
        raise FileNotFoundError(f"Prepared chunks missing: {args.chunks}")

    index = BM25Corpus.from_jsonl(args.chunks)
    example = next(question for question in questions if question["evidence_list"])
    results = index.search(example["query"], top_k=5)
    print(f"Validated questions: {len(questions)}")
    print(f"Validated corpus documents: {len(documents)}")
    print(f"Indexed chunks: {len(index.chunks)}")
    print("Example top-5 BM25 titles:")
    for rank, result in enumerate(results, 1):
        print(f"  {rank}. {result['title']} ({result['score']:.3f})")


if __name__ == "__main__":
    main()

