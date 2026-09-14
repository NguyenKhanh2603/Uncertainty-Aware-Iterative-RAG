"""Rerun answer-to-chunk attention on the frozen clean calibration/test plans.

This is the original attention-only signal used by ``run_full_topl_validation``:
Qwen first drafts a short answer from the complete Top-L context, then a cached
teacher-forced pass measures attention from answer tokens to each chunk.  A
second pass reverses chunk order so the mean of both passes can control for
position bias.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from research.internal_state_rag import QwenInternalStateExtractor
from research.internal_state_rag.run_full_topl_validation import (
    append_jsonl,
    load_completed,
    scalar_attention_features,
)
from research.internal_state_rag.run_tatqa_smoke import (
    ResearchTextClient,
    generate_answer,
    load_retrieval,
    load_rows,
    to_chunk,
)


def parse_args() -> argparse.Namespace:
    root = Path("data/conformal_global_run")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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
    parser.add_argument("--max-answer-tokens", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--min-pixels", type=int, default=3136)
    parser.add_argument("--max-pixels", type=int, default=200704)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only the first N planned queries for a smoke run.",
    )
    return parser.parse_args()


def candidate_record(
    retrieval_row: dict[str, Any],
    *,
    original_trace: Any,
    reversed_trace: Any,
    original_spans: dict[str, Any],
) -> dict[str, Any]:
    chunk_id = str(retrieval_row["chunk_id"])
    span = original_spans[chunk_id]
    token_count = (
        0
        if span.start is None or span.end is None
        else int(span.end) - int(span.start)
    )
    original = scalar_attention_features(
        original_trace, chunk_id=chunk_id, token_count=token_count
    )
    reversed_features = scalar_attention_features(
        reversed_trace, chunk_id=chunk_id, token_count=token_count
    )
    return {
        "chunk_id": chunk_id,
        "rank": int(retrieval_row["rank"]),
        "modality": str(retrieval_row["modality"]),
        "is_support": retrieval_row["support_label"] == "support",
        "cosine_score": float(retrieval_row["cosine_score"]),
        "bge_score": float(retrieval_row["selection_score"]),
        "chunk_token_count": token_count,
        "original_attention_mass": original["mass"],
        "original_attention_fraction": original["fraction"],
        "reversed_attention_mass": reversed_features["mass"],
        "reversed_attention_fraction": reversed_features["fraction"],
        "mean_attention_mass": (original["mass"] + reversed_features["mass"]) / 2,
        "mean_attention_fraction": (
            original["fraction"] + reversed_features["fraction"]
        )
        / 2,
    }


def main() -> None:
    args = parse_args()
    source_manifest = json.loads(args.feature_manifest.read_text(encoding="utf-8"))
    plan = [str(value) for value in source_manifest["plan"]]
    if args.limit > 0:
        plan = plan[: args.limit]
    top_l = int(source_manifest["top_l"])
    input_role = str(source_manifest["input_role"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / "queries.jsonl"
    manifest_path = args.output_dir / "manifest.json"
    summary_path = args.output_dir / "summary.json"
    expected_manifest = {
        "status": "running",
        "method": "answer_token_to_chunk_attention_with_reversed_position_control",
        "model": args.model,
        "source_feature_manifest": str(args.feature_manifest),
        "input_role": input_role,
        "source_split": str(source_manifest["source_split"]),
        "top_l": top_l,
        "max_answer_tokens": args.max_answer_tokens,
        "max_new_tokens": args.max_new_tokens,
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
        "planned_queries": len(plan),
        "plan": plan,
    }
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        comparable = {key: existing.get(key) for key in expected_manifest if key != "status"}
        target = {key: value for key, value in expected_manifest.items() if key != "status"}
        if comparable != target:
            raise ValueError("Existing output manifest does not match this run")
    else:
        manifest_path.write_text(
            json.dumps(expected_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    questions = load_rows(args.questions, "qid")
    corpus = load_rows(args.corpus, "id")
    retrieval = load_retrieval(args.retrieval, split_role=input_role)
    completed = load_completed(rows_path)
    client = ResearchTextClient(args.model, device="cuda", load_in_4bit=False)
    if client.is_qwen_vl:
        client.processor.image_processor.min_pixels = args.min_pixels
        client.processor.image_processor.max_pixels = args.max_pixels
        # Prefix passes do not request attention matrices.  SDPA keeps those
        # long multimodal prefills tractable; Transformers falls back to eager
        # only for the one-token decode calls that request attention weights.
        client.model.config._attn_implementation = "sdpa"
        client.model.model.config._attn_implementation = "sdpa"
    extractor = QwenInternalStateExtractor(client)
    started = time.perf_counter()

    for query_index, qid in enumerate(plan, start=1):
        if qid in completed:
            print(f"[{query_index}/{len(plan)}] resume skip {qid}", flush=True)
            continue
        retrieval_rows = retrieval[qid][:top_l]
        if len(retrieval_rows) != top_l:
            raise ValueError(f"{qid} has {len(retrieval_rows)} candidates, expected {top_l}")
        question = questions[qid]
        chunks = [to_chunk(row, corpus) for row in retrieval_rows]
        prompt = (
            "Answer using only the supplied context. Return only the short answer.\n"
            f"Question: {question['question']}"
        )
        draft = generate_answer(client, prompt, chunks, max_new_tokens=args.max_new_tokens)
        if not draft:
            append_jsonl(rows_path, {"qid": qid, "status": "empty_draft"})
            print(f"[{query_index}/{len(plan)}] empty draft {qid}", flush=True)
            continue

        original_trace = extractor.extract(
            prompt, chunks, draft, max_answer_tokens=args.max_answer_tokens
        )
        original_alignment = client.align_and_prepare_inputs(prompt, chunks)
        original_spans = {
            span.chunk_id: span for span in original_alignment.batch_chunk_spans[0]
        }
        reversed_trace = extractor.extract(
            prompt, list(reversed(chunks)), draft, max_answer_tokens=args.max_answer_tokens
        )
        candidates = [
            candidate_record(
                row,
                original_trace=original_trace,
                reversed_trace=reversed_trace,
                original_spans=original_spans,
            )
            for row in retrieval_rows
        ]
        support_ranks = [row["rank"] for row in candidates if row["is_support"]]
        record = {
            "status": "complete",
            "qid": qid,
            "retrievable": bool(support_ranks),
            "stratum": (
                "unretrievable"
                if not support_ranks
                else "hard" if min(support_ranks) > 2 else "easy"
            ),
            "question": str(question["question"]),
            "draft": draft,
            "n_answer_tokens": len(original_trace.answer_token_ids),
            "support_ranks": support_ranks,
            "candidates": candidates,
        }
        append_jsonl(rows_path, record)
        completed.add(qid)
        elapsed = time.perf_counter() - started
        print(
            f"[{query_index}/{len(plan)}] {qid} support={support_ranks} "
            f"tokens={record['n_answer_tokens']} elapsed={elapsed:.1f}s "
            f"draft={draft[:36]!r}",
            flush=True,
        )

    rows_by_qid: dict[str, dict[str, Any]] = {}
    if rows_path.exists():
        for line in rows_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows_by_qid[str(row["qid"])] = row
    complete = sum(rows_by_qid.get(qid, {}).get("status") == "complete" for qid in plan)
    empty = sum(rows_by_qid.get(qid, {}).get("status") == "empty_draft" for qid in plan)
    final_status = "complete" if complete + empty == len(plan) else "partial"
    summary = {
        "status": final_status,
        "planned_queries": len(plan),
        "completed_queries": complete,
        "empty_draft_queries": empty,
        "elapsed_seconds_this_process": time.perf_counter() - started,
        "rows": str(rows_path),
        "manifest": str(manifest_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    expected_manifest["status"] = final_status
    manifest_path.write_text(
        json.dumps(expected_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
