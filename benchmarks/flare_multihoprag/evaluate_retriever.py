"""Evaluate BM25 against gold document URLs over the entire shared corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import BM25Corpus, mean, read_json


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate whole-corpus BM25 retrieval")
    parser.add_argument(
        "--queries", type=Path, default=PROJECT_ROOT / "data/multihop_rag/MultiHopRAG.json"
    )
    parser.add_argument("--chunks", type=Path, default=HERE / "artifacts/corpus_chunks.jsonl")
    parser.add_argument("--output", type=Path, default=HERE / "artifacts/bm25_metrics.json")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 10])
    args = parser.parse_args()
    if not args.ks or min(args.ks) < 1:
        parser.error("ks must contain positive integers")

    questions = [
        row for row in read_json(args.queries)
        if row["question_type"] != "null_query"
    ]
    if args.max_examples is not None:
        questions = questions[: args.max_examples]
    index = BM25Corpus.from_jsonl(args.chunks)
    max_k = max(args.ks)
    recalls = {k: [] for k in args.ks}
    all_evidence = {k: [] for k in args.ks}
    reciprocal_ranks = []

    for question in questions:
        gold_urls = {evidence["url"] for evidence in question["evidence_list"]}
        results = index.search(question["query"], max_k)
        first_rank = None
        for rank, result in enumerate(results, 1):
            if result["url"] in gold_urls:
                first_rank = rank
                break
        reciprocal_ranks.append(1 / first_rank if first_rank else 0.0)
        for k in args.ks:
            retrieved_urls = {result["url"] for result in results[:k]}
            recalls[k].append(len(gold_urls & retrieved_urls) / len(gold_urls))
            all_evidence[k].append(float(gold_urls <= retrieved_urls))

    metrics = {
        "queries": len(questions),
        "corpus_documents": len({chunk["url"] for chunk in index.chunks}),
        "corpus_chunks": len(index.chunks),
        "mrr_at_max_k": mean(reciprocal_ranks),
        "recall": {f"@{k}": mean(recalls[k]) for k in args.ks},
        "all_evidence_success": {f"@{k}": mean(all_evidence[k]) for k in args.ks},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

