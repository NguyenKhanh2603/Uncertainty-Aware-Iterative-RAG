"""Compare internal chunk attribution with frozen retrieval scores on TATQA train."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from research.internal_state_rag import (
    QwenInternalStateExtractor,
    mask_qwen2_answer_chunk_edges,
)
from research.internal_state_rag.run_tatqa_smoke import (
    ResearchTextClient,
    generate_answer,
    load_retrieval,
    load_rows,
    to_chunk,
)

DEFAULT_HEADS = "14:13,14:15,14:8,18:8,14:11,14:10,18:13,14:12"


def parse_heads(specification: str) -> list[tuple[int, int]]:
    heads = []
    for item in specification.split(","):
        layer, head = item.strip().split(":", maxsplit=1)
        heads.append((int(layer), int(head)))
    if not heads:
        raise ValueError("At least one head is required.")
    return heads


def eligible_cases(
    questions: dict[str, dict[str, Any]],
    retrieval: dict[str, list[dict[str, Any]]],
    *,
    excluded_qids: set[str],
    seed: int,
    limit: int,
) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    cases = []
    for qid, rows in retrieval.items():
        question = questions.get(qid)
        supports = [row for row in rows if row["support_label"] == "support"]
        false_rows = [row for row in rows if row["support_label"] == "false"]
        if (
            qid in excluded_qids
            or question is None
            or question.get("metadata", {}).get("source_split") != "train"
            or len(question.get("gold_answers", [])) != 1
            or len(supports) != 1
            or len(false_rows) < 3
        ):
            continue
        cases.append((question, rows))
    if len(cases) < limit:
        raise ValueError(f"Only {len(cases)} eligible cases for requested limit {limit}.")
    return random.Random(seed).sample(cases, limit)


def select_candidate_rows(rows: list[dict[str, Any]], *, n_false: int) -> list[dict[str, Any]]:
    support = next(row for row in rows if row["support_label"] == "support")
    false_rows = [row for row in rows if row["support_label"] == "false"]
    same_modality = [row for row in false_rows if row["modality"] == support["modality"]]
    other = [row for row in false_rows if row["modality"] != support["modality"]]
    negatives = (same_modality + other)[:n_false]
    return [support, *negatives]


def build_context_rows(
    rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]], *, context_k: int
) -> list[dict[str, Any]]:
    context = list(rows[:context_k])
    existing = {str(row["chunk_id"]) for row in context}
    for row in candidate_rows:
        if str(row["chunk_id"]) not in existing:
            context.append(row)
            existing.add(str(row["chunk_id"]))
    return context


def trace_logprob(trace: Any) -> float:
    return float(trace.target_logprob[-1].mean())


def selected_head_attention(
    trace: Any,
    *,
    heads: list[tuple[int, int]],
    chunk_index: int,
) -> float:
    layer_offsets = {layer: index for index, layer in enumerate(trace.layer_ids)}
    values = [
        float(trace.attention_mass[layer_offsets[layer], head, chunk_index])
        for layer, head in heads
        if layer in layer_offsets
    ]
    return float(np.mean(values)) if values else 0.0


def ranking_metrics(rows: list[dict[str, Any]], score_name: str) -> dict[str, float]:
    labels = np.asarray([int(row["is_support"]) for row in rows])
    scores = np.asarray([float(row[score_name]) for row in rows])
    qids = sorted({str(row["qid"]) for row in rows})
    reciprocal_ranks = []
    top1 = []
    for qid in qids:
        candidates = [row for row in rows if str(row["qid"]) == qid]
        candidates.sort(key=lambda row: (-float(row[score_name]), int(row["rank"])))
        support_rank = next(
            index for index, row in enumerate(candidates, start=1) if row["is_support"]
        )
        reciprocal_ranks.append(1.0 / support_rank)
        top1.append(float(support_rank == 1))
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "query_mrr": float(np.mean(reciprocal_ranks)),
        "query_top1_support_rate": float(np.mean(top1)),
    }


def write_progress(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    root = Path("data/conformal_global_run")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
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
    parser.add_argument(
        "--exclude-results",
        type=Path,
        default=Path("research/internal_state_rag/results/tatqa_smoke_n39_seed17.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n", type=int, default=24)
    parser.add_argument("--n-false", type=int, default=3)
    parser.add_argument("--context-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--heads", default=DEFAULT_HEADS)
    parser.add_argument("--max-answer-tokens", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    heads = parse_heads(args.heads)
    source = json.loads(args.exclude_results.read_text(encoding="utf-8"))
    excluded_qids = {str(row["qid"]) for row in source["results"]}
    corpus = load_rows(args.corpus, "id")
    questions = load_rows(args.questions, "qid")
    retrieval = load_retrieval(args.retrieval)
    cases = eligible_cases(
        questions,
        retrieval,
        excluded_qids=excluded_qids,
        seed=args.seed,
        limit=args.n,
    )

    client = ResearchTextClient(args.model, device="cuda", load_in_4bit=False)
    extractor = QwenInternalStateExtractor(client)
    observations = []

    for query_index, (question, rows) in enumerate(cases, start=1):
        candidate_rows = select_candidate_rows(rows, n_false=args.n_false)
        context_rows = build_context_rows(rows, candidate_rows, context_k=args.context_k)
        chunks = [to_chunk(row, corpus) for row in context_rows]
        chunk_offsets = {chunk.id: index for index, chunk in enumerate(chunks)}
        prompt = (
            "Answer using only the supplied context. Return only the short answer.\n"
            f"Question: {question['question']}"
        )
        gold = str(question["gold_answers"][0])
        draft = generate_answer(
            client, prompt, chunks, max_new_tokens=args.max_new_tokens
        )
        if not draft:
            print(f"[{query_index}/{len(cases)}] skip empty draft {question['qid']}")
            continue

        baseline_gold = extractor.extract(
            prompt, chunks, gold, max_answer_tokens=args.max_answer_tokens
        )
        baseline_draft = extractor.extract(
            prompt, chunks, draft, max_answer_tokens=args.max_answer_tokens
        )
        baseline_gold_logprob = trace_logprob(baseline_gold)
        baseline_draft_logprob = trace_logprob(baseline_draft)
        alignment = client.align_and_prepare_inputs(prompt, chunks)
        prompt_length = int(alignment.inputs["input_ids"].shape[1])
        spans = {span.chunk_id: span for span in alignment.batch_chunk_spans[0]}

        query_observations = []
        for candidate in candidate_rows:
            chunk_id = str(candidate["chunk_id"])
            span = spans[chunk_id]
            if span.start is None or span.end is None:
                continue
            with mask_qwen2_answer_chunk_edges(
                client.model,
                heads,
                start=int(span.start),
                end=int(span.end),
                prompt_length=prompt_length,
            ):
                masked_gold = extractor.extract(
                    prompt, chunks, gold, max_answer_tokens=args.max_answer_tokens
                )
                masked_draft = extractor.extract(
                    prompt, chunks, draft, max_answer_tokens=args.max_answer_tokens
                )
            chunk_index = chunk_offsets[chunk_id]
            record = {
                "qid": str(question["qid"]),
                "question": str(question["question"]),
                "gold": gold,
                "draft": draft,
                "chunk_id": chunk_id,
                "rank": int(candidate["rank"]),
                "modality": str(candidate["modality"]),
                "is_support": candidate["support_label"] == "support",
                "cosine_score": float(candidate["cosine_score"]),
                "bge_score": float(candidate["selection_score"]),
                "selected_head_attention": selected_head_attention(
                    baseline_draft,
                    heads=heads,
                    chunk_index=chunk_index,
                ),
                "draft_causal_delta": baseline_draft_logprob - trace_logprob(masked_draft),
                "draft_absolute_influence": abs(
                    baseline_draft_logprob - trace_logprob(masked_draft)
                ),
                "gold_causal_delta": baseline_gold_logprob - trace_logprob(masked_gold),
            }
            observations.append(record)
            query_observations.append(record)

        support = next(row for row in query_observations if row["is_support"])
        print(
            f"[{query_index}/{len(cases)}] {question['qid']} "
            f"support rank={support['rank']} draft_delta={support['draft_causal_delta']:+.3f} "
            f"gold_delta={support['gold_causal_delta']:+.3f}",
            flush=True,
        )
        write_progress(
            args.output,
            {
                "status": "running",
                "model": args.model,
                "seed": args.seed,
                "heads": [list(head) for head in heads],
                "completed_queries": query_index,
                "observations": observations,
            },
        )

    score_names = [
        "cosine_score",
        "bge_score",
        "selected_head_attention",
        "draft_causal_delta",
        "draft_absolute_influence",
        "gold_causal_delta",
    ]
    metrics = {name: ranking_metrics(observations, name) for name in score_names}
    output = {
        "status": "complete",
        "model": args.model,
        "split": "official_train_only",
        "seed": args.seed,
        "n_queries": len({row["qid"] for row in observations}),
        "n_candidates": len(observations),
        "context_k": args.context_k,
        "heads": [list(head) for head in heads],
        "metrics": metrics,
        "observations": observations,
    }
    write_progress(args.output, output)
    summary = {key: value for key, value in output.items() if key != "observations"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
