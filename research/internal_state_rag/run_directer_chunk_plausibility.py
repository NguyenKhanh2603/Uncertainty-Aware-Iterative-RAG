"""Test DIRECTER-style plausibility after exact answer-to-chunk edge masking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from research.internal_state_rag import (
    QwenInternalStateExtractor,
    mask_qwen2_answer_chunk_edges,
)
from research.internal_state_rag.directer_plausibility import (
    distribution_plausibility,
    teacher_forced_final_logits,
)
from research.internal_state_rag.ranking import score_metrics
from research.internal_state_rag.run_attention_chunk_ranking import (
    attention_features,
    candidate_rows,
    result_qids,
    stratified_cases,
    write_json,
)
from research.internal_state_rag.run_chunk_attribution_mvp import build_context_rows
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
    parser.add_argument("--max-answer-tokens", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=71)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exclusions = [
        Path("research/internal_state_rag/results/tatqa_smoke_n39_seed17.json"),
        Path(
            "research/internal_state_rag/results/"
            "chunk_attribution_n24_all_l14_l18_seed31.json"
        ),
        Path(
            "research/internal_state_rag/results/"
            "attention_ranking_n120_balanced_seed47.json"
        ),
    ]
    corpus = load_rows(args.corpus, "id")
    questions = load_rows(args.questions, "qid")
    retrieval = load_retrieval(args.retrieval)
    cases = stratified_cases(
        questions,
        retrieval,
        excluded_qids=result_qids(exclusions),
        n_easy=args.n_easy,
        n_hard=args.n_hard,
        seed=args.seed,
    )

    client = ResearchTextClient(args.model, device="cuda", load_in_4bit=False)
    extractor = QwenInternalStateExtractor(client)
    num_layers = int(client.model.config.num_hidden_layers)
    num_heads = int(client.model.config.num_attention_heads)
    all_heads = [
        (layer, head) for layer in range(num_layers) for head in range(num_heads)
    ]
    focus_layers = {14, 18}
    observations = []

    for query_index, (question, rows, stratum) in enumerate(cases, start=1):
        candidates = candidate_rows(rows, n_false=args.n_false)
        context_rows = build_context_rows(rows, candidates, context_k=args.context_k)
        chunks = [to_chunk(row, corpus) for row in context_rows]
        chunk_offsets = {chunk.id: index for index, chunk in enumerate(chunks)}
        prompt = (
            "Answer using only the supplied context. Return only the short answer.\n"
            f"Question: {question['question']}"
        )
        draft = generate_answer(
            client, prompt, chunks, max_new_tokens=args.max_new_tokens
        )
        if not draft:
            continue
        attention_trace = extractor.extract(
            prompt, chunks, draft, max_answer_tokens=args.max_answer_tokens
        )
        raw_logits = teacher_forced_final_logits(
            extractor,
            prompt,
            chunks,
            draft,
            max_answer_tokens=args.max_answer_tokens,
        )
        alignment = client.align_and_prepare_inputs(prompt, chunks)
        prompt_length = int(alignment.inputs["input_ids"].shape[1])
        spans = {span.chunk_id: span for span in alignment.batch_chunk_spans[0]}

        for candidate in candidates:
            chunk_id = str(candidate["chunk_id"])
            span = spans[chunk_id]
            if span.start is None or span.end is None:
                continue
            with mask_qwen2_answer_chunk_edges(
                client.model,
                all_heads,
                start=int(span.start),
                end=int(span.end),
                prompt_length=prompt_length,
            ):
                intervened_logits = teacher_forced_final_logits(
                    extractor,
                    prompt,
                    chunks,
                    draft,
                    max_answer_tokens=args.max_answer_tokens,
                )
            plausibility = distribution_plausibility(
                raw_logits, intervened_logits, beta=args.beta
            )
            token_count = int(span.end) - int(span.start)
            features = attention_features(
                attention_trace,
                chunk_index=chunk_offsets[chunk_id],
                token_count=token_count,
                focus_layers=focus_layers,
            )
            observations.append(
                {
                    "qid": str(question["qid"]),
                    "stratum": stratum,
                    "chunk_id": chunk_id,
                    "rank": int(candidate["rank"]),
                    "is_support": candidate["support_label"] == "support",
                    "bge_score": float(candidate["selection_score"]),
                    "attention_mass_all_layers": features[
                        "attention_mass_all_layers"
                    ],
                    "directer_worst_rejection": plausibility.worst_rejection_score,
                    "directer_mean_rejection": float(
                        -np.log(max(plausibility.geometric_mean_probability_ratio, 1e-30))
                    ),
                    "directer_rejection_rate": plausibility.rejection_rate,
                    "top1_change_rate": plausibility.top1_change_rate,
                    "mean_js_divergence": plausibility.mean_js_divergence,
                    "min_probability_ratio": plausibility.min_probability_ratio,
                    "probability_ratios": plausibility.probability_ratios,
                }
            )

        print(
            f"[{query_index}/{len(cases)}] {question['qid']} {stratum} "
            f"candidates={len(candidates)} draft={draft[:45]!r}",
            flush=True,
        )
        write_json(
            args.output,
            {
                "status": "running",
                "model": args.model,
                "seed": args.seed,
                "beta": args.beta,
                "completed_queries": query_index,
                "observations": observations,
            },
        )

    score_names = [
        "bge_score",
        "attention_mass_all_layers",
        "directer_worst_rejection",
        "directer_mean_rejection",
        "directer_rejection_rate",
        "top1_change_rate",
        "mean_js_divergence",
    ]
    output = {
        "status": "complete",
        "model": args.model,
        "split": "official_train_only",
        "seed": args.seed,
        "beta": args.beta,
        "n_queries": len({row["qid"] for row in observations}),
        "n_candidates": len(observations),
        "intervention": "mask all answer-to-chunk edges at all layers and heads",
        "metrics": score_metrics(observations, score_names),
        "observations": observations,
    }
    write_json(args.output, output)
    summary = {key: value for key, value in output.items() if key != "observations"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
