"""Control prompt-position bias by reversing context before measuring attention."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from research.internal_state_rag import QwenInternalStateExtractor
from research.internal_state_rag.ranking import score_metrics
from research.internal_state_rag.run_attention_chunk_ranking import (
    attention_features,
    candidate_rows,
    write_json,
)
from research.internal_state_rag.run_chunk_attribution_mvp import build_context_rows
from research.internal_state_rag.run_tatqa_smoke import (
    ResearchTextClient,
    load_retrieval,
    load_rows,
    to_chunk,
)

ATTENTION_SCORES = [
    "attention_mass_focus",
    "attention_fraction_focus",
    "attention_density_focus",
    "attention_sqrt_density_focus",
    "attention_max_head_focus",
    "attention_mass_all_layers",
    "attention_fraction_all_layers",
]


def rows_by_qid(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["qid"]), []).append(row)
    return grouped


def parse_args() -> argparse.Namespace:
    root = Path("data/conformal_global_run")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--questions",
        type=Path,
        default=root / "official_bundle_role_split_tatqa/tatqa/questions.jsonl",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=root / "official_bundle_role_split_tatqa/tatqa/corpus.jsonl",
    )
    parser.add_argument(
        "--retrieval",
        type=Path,
        default=root / "retrieval/tatqa_top30_bge_reranked.jsonl.gz",
    )
    parser.add_argument("--n-false", type=int, default=3)
    parser.add_argument("--context-k", type=int, default=10)
    parser.add_argument("--max-answer-tokens", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = json.loads(args.source_results.read_text(encoding="utf-8"))
    source_rows = rows_by_qid(source["observations"])
    focus_layers = {int(layer) for layer in source["focus_layers"]}
    corpus = load_rows(args.corpus, "id")
    questions = load_rows(args.questions, "qid")
    retrieval = load_retrieval(args.retrieval)

    completed: list[dict[str, Any]] = []
    if args.output.exists():
        progress = json.loads(args.output.read_text(encoding="utf-8"))
        if progress.get("status") == "running":
            completed = progress.get("observations", [])
    completed_qids = {str(row["qid"]) for row in completed}

    client = ResearchTextClient(args.model, device="cuda", load_in_4bit=False)
    extractor = QwenInternalStateExtractor(client)
    observations = completed
    qids = list(source_rows)

    for query_index, qid in enumerate(qids, start=1):
        if qid in completed_qids:
            print(f"[{query_index}/{len(qids)}] resume skip {qid}", flush=True)
            continue
        original_records = source_rows[qid]
        question = questions[qid]
        rows = retrieval[qid]
        candidates = candidate_rows(rows, n_false=args.n_false)
        context_rows = build_context_rows(rows, candidates, context_k=args.context_k)
        original_positions = {
            str(row["chunk_id"]): index for index, row in enumerate(context_rows)
        }
        reversed_rows = list(reversed(context_rows))
        reversed_positions = {
            str(row["chunk_id"]): index for index, row in enumerate(reversed_rows)
        }
        chunks = [to_chunk(row, corpus) for row in reversed_rows]
        chunk_offsets = {chunk.id: index for index, chunk in enumerate(chunks)}
        prompt = (
            "Answer using only the supplied context. Return only the short answer.\n"
            f"Question: {question['question']}"
        )
        draft = str(original_records[0]["draft"])
        trace = extractor.extract(
            prompt, chunks, draft, max_answer_tokens=args.max_answer_tokens
        )
        alignment = client.align_and_prepare_inputs(prompt, chunks)
        spans = {span.chunk_id: span for span in alignment.batch_chunk_spans[0]}

        for original in original_records:
            chunk_id = str(original["chunk_id"])
            span = spans[chunk_id]
            token_count = (
                0
                if span.start is None or span.end is None
                else int(span.end) - int(span.start)
            )
            reversed_features = attention_features(
                trace,
                chunk_index=chunk_offsets[chunk_id],
                token_count=token_count,
                focus_layers=focus_layers,
            )
            record = {
                "qid": qid,
                "stratum": str(original["stratum"]),
                "chunk_id": chunk_id,
                "rank": int(original["rank"]),
                "is_support": bool(original["is_support"]),
                "bge_score": float(original["bge_score"]),
                "chunk_token_count": int(original["chunk_token_count"]),
                "original_context_position": original_positions[chunk_id],
                "reversed_context_position": reversed_positions[chunk_id],
            }
            for score_name in ATTENTION_SCORES:
                original_value = float(original[score_name])
                reversed_value = float(reversed_features[score_name])
                record[f"original_{score_name}"] = original_value
                record[f"reversed_{score_name}"] = reversed_value
                record[f"mean_{score_name}"] = (original_value + reversed_value) / 2
            observations.append(record)

        completed_qids.add(qid)
        print(
            f"[{query_index}/{len(qids)}] {qid} {original_records[0]['stratum']} "
            f"candidates={len(original_records)}",
            flush=True,
        )
        write_json(
            args.output,
            {
                "status": "running",
                "model": args.model,
                "source_results": str(args.source_results),
                "completed_queries": len(completed_qids),
                "focus_layers": sorted(focus_layers),
                "observations": observations,
            },
        )

    score_names = ["bge_score"]
    score_names.extend(
        f"{prefix}_{score_name}"
        for prefix in ("original", "reversed", "mean")
        for score_name in ATTENTION_SCORES
    )
    output = {
        "status": "complete",
        "model": args.model,
        "source_results": str(args.source_results),
        "split": "official_train_only",
        "n_queries": len(completed_qids),
        "n_candidates": len(observations),
        "focus_layers": sorted(focus_layers),
        "metrics": score_metrics(observations, score_names),
        "observations": observations,
    }
    write_json(args.output, output)
    summary = {key: value for key, value in output.items() if key != "observations"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
