"""Evaluate layerwise contrastive internal saliency for chunk pruning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from research.internal_state_rag.contrastive_saliency import extract_contrastive_saliency
from research.internal_state_rag.ranking import score_metrics
from research.internal_state_rag.run_attention_chunk_ranking import (
    candidate_rows,
    result_qids,
    stratified_cases,
)
from research.internal_state_rag.run_chunk_attribution_mvp import build_context_rows
from research.internal_state_rag.run_tatqa_smoke import (
    ResearchTextClient,
    generate_answer,
    load_retrieval,
    load_rows,
    to_chunk,
)


def write_progress(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    root = Path("data/conformal_global_run")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
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
    parser.add_argument("--n-easy", type=int, default=12)
    parser.add_argument("--n-hard", type=int, default=12)
    parser.add_argument("--n-false", type=int, default=3)
    parser.add_argument("--context-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=83)
    parser.add_argument(
        "--exclude-results",
        type=Path,
        action="append",
        default=[],
        help="Additional result files whose query IDs must not be sampled.",
    )
    parser.add_argument("--layers", default="0,3,6,9,12,15,18,21,24,27,30,33")
    parser.add_argument("--max-answer-tokens", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    layers = [int(value) for value in args.layers.split(",") if value.strip()]
    corpus = load_rows(args.corpus, "id")
    questions = load_rows(args.questions, "qid")
    retrieval = load_retrieval(args.retrieval)
    excluded = result_qids(
        [
            Path("research/internal_state_rag/results/tatqa_smoke_n39_seed17.json"),
            Path("research/internal_state_rag/results/chunk_attribution_n24_all_l14_l18_seed31.json"),
            Path("research/internal_state_rag/results/attention_ranking_n120_balanced_seed47.json"),
            *args.exclude_results,
        ]
    )
    cases = stratified_cases(
        questions,
        retrieval,
        excluded_qids=excluded,
        n_easy=args.n_easy,
        n_hard=args.n_hard,
        seed=args.seed,
    )
    client = ResearchTextClient(args.model, device="cuda", load_in_4bit=False)
    observations: list[dict[str, Any]] = []

    for query_index, (question, rows, stratum) in enumerate(cases, start=1):
        candidates = candidate_rows(rows, n_false=args.n_false)
        context_rows = build_context_rows(rows, candidates, context_k=args.context_k)
        chunks = [to_chunk(row, corpus) for row in context_rows]
        prompt = (
            "Answer using only the supplied context. Return only the short answer.\n"
            f"Question: {question['question']}"
        )
        draft = generate_answer(client, prompt, chunks, max_new_tokens=args.max_new_tokens)
        if not draft:
            continue
        trace = extract_contrastive_saliency(
            client,
            prompt,
            chunks,
            draft,
            max_answer_tokens=args.max_answer_tokens,
            layers=layers,
        )
        offsets = {chunk_id: index for index, chunk_id in enumerate(trace.chunk_ids)}
        query_rows = []
        for row in candidates:
            chunk_id = str(row["chunk_id"])
            chunk_index = offsets[chunk_id]
            record: dict[str, Any] = {
                "qid": str(question["qid"]),
                "stratum": stratum,
                "question": str(question["question"]),
                "draft": draft,
                "chunk_id": chunk_id,
                "rank": int(row["rank"]),
                "is_support": row["support_label"] == "support",
                "bge_score": float(row["selection_score"]),
                "cosine_score": float(row["cosine_score"]),
                "cti_js_mean": float(np.mean(trace.cti_js_divergence)),
                "selected_answer_tokens": len(trace.selected_answer_indices),
            }
            for name, values in trace.features.items():
                vector = values[:, chunk_index].astype(float)
                record[f"{name}_max_layer"] = float(vector.max())
                record[f"{name}_mean_layer"] = float(vector.mean())
                record[f"{name}_by_layer"] = vector.tolist()
            observations.append(record)
            query_rows.append(record)

        scalar_scores = [
            key
            for key in query_rows[0]
            if key in {"bge_score", "cosine_score"} or key.endswith(("_max_layer", "_mean_layer"))
        ]
        support = next(row for row in query_rows if row["is_support"])
        print(
            f"[{query_index}/{len(cases)}] {question['qid']} {stratum} "
            f"rank={support['rank']} cti={support['cti_js_mean']:.4g} "
            f"res={support['residual_grad_x_top5_max_layer']:.4g}",
            flush=True,
        )
        write_progress(
            args.output,
            {
                "status": "running",
                "model": args.model,
                "seed": args.seed,
                "layer_ids": layers,
                "completed_queries": query_index,
                "metrics": score_metrics(observations, scalar_scores),
                "observations": observations,
            },
        )

    scalar_scores = [
        key
        for key in observations[0]
        if key in {"bge_score", "cosine_score"} or key.endswith(("_max_layer", "_mean_layer"))
    ]
    output = {
        "status": "complete",
        "model": args.model,
        "seed": args.seed,
        "n_queries": len({row["qid"] for row in observations}),
        "n_candidates": len(observations),
        "layer_ids": layers,
        "metrics": score_metrics(observations, scalar_scores),
        "observations": observations,
    }
    write_progress(args.output, output)
    print(json.dumps({key: value for key, value in output.items() if key != "observations"}, indent=2))


if __name__ == "__main__":
    main()
