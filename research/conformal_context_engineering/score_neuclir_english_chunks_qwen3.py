"""Score each NeuCLIR English candidate chunk with Qwen3-Embedding-8B.

The output preserves a per-topic ranking of chunks from the same frozen Top-100
candidate documents released by CoverageBench.  It intentionally does not
create snippet relevance labels: public NeuCLIR labels are nugget-to-document.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as functional
from transformers import AutoModel, AutoTokenizer

DEFAULT_CHUNKS = Path(
    "data/external/neuclir1_english/candidates/coveragebench_initial_qwen3_top100_chunks_500_100.jsonl"
)
DEFAULT_TOPICS = Path("data/external/coveragebench_neuclir/topics/neuclir2024_topics.tsv")
DEFAULT_RUN = Path(
    "data/external/coveragebench_neuclir/ranking/NeuCLIR/Initial-Retrieval/initial_qwen3.json"
)
DEFAULT_OUTPUT = Path(
    "research/conformal_context_engineering/results/neuclir_english_qwen3_chunk_scores.json"
)
QUERY_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--topics", type=Path, default=DEFAULT_TOPICS)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--top-documents", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    return parser.parse_args()


def load_topics(path: Path) -> dict[str, str]:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            qid, title, narrative = line.rstrip("\n").split("\t", maxsplit=2)
            rows[qid] = f"{title}\n{narrative}"
    return rows


def last_token_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    positions = mask.sum(dim=1).sub(1).clamp_min(0)
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), positions]


@torch.inference_mode()
def encode(model, tokenizer, texts: list[str], batch_size: int, max_length: int, device: str) -> torch.Tensor:
    vectors = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start : start + batch_size], padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        ).to(device)
        output = model(**encoded)
        hidden = output.last_hidden_state if hasattr(output, "last_hidden_state") else output[0]
        vectors.append(functional.normalize(last_token_pool(hidden, encoded["attention_mask"]), p=2, dim=1).cpu())
    return torch.cat(vectors)


def main() -> None:
    args = arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; use the isolated CUDA-11.8 environment")
    chunks_by_doc: dict[str, list[dict]] = defaultdict(list)
    with args.chunks.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            chunks_by_doc[row["docid"]].append(row)
    topics = load_topics(args.topics)
    run = json.loads(args.run.read_text())
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="right", trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True
    ).to(device).eval()
    query_texts = [f"Instruct: {QUERY_INSTRUCTION}\nQuery: {topics[qid]}" for qid in run]
    query_vectors = encode(model, tokenizer, query_texts, args.batch_size, args.max_length, device)
    output: dict[str, list[dict]] = {}
    for query_index, qid in enumerate(run):
        docids = [
            docid for docid, _ in sorted(run[qid].items(), key=lambda item: (-item[1], item[0]))[: args.top_documents]
        ]
        rows = [chunk for docid in docids for chunk in chunks_by_doc[docid]]
        if not rows:
            raise RuntimeError(f"No chunks for {qid}")
        passage_texts = [f"{row['title']}\n{row['text']}" for row in rows]
        passage_vectors = encode(model, tokenizer, passage_texts, args.batch_size, args.max_length, device)
        scores = (passage_vectors @ query_vectors[query_index]).tolist()
        output[qid] = [
            {"chunk_id": row["chunk_id"], "docid": row["docid"], "score": score}
            for row, score in sorted(zip(rows, scores, strict=True), key=lambda item: (-item[1], item[0]["chunk_id"]))
        ]
        print(f"{qid}: {len(rows)} chunks", flush=True)
    artifact = {
        "status": "complete",
        "score_model": args.model,
        "query_instruction": QUERY_INSTRUCTION,
        "chunking": {"characters": 500, "overlap": 100},
        "candidate_source": str(args.run),
        "top_documents_per_query": args.top_documents,
        "scores": output,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
