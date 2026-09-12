"""Evaluate generator attention as a chunk-retention signal on TATQA train."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from research.internal_state_rag import QwenInternalStateExtractor
from research.internal_state_rag.ranking import score_metrics
from research.internal_state_rag.run_chunk_attribution_mvp import build_context_rows
from research.internal_state_rag.run_tatqa_smoke import (
    ResearchTextClient,
    generate_answer,
    load_retrieval,
    load_rows,
    to_chunk,
)


def result_qids(paths: list[Path]) -> set[str]:
    """Collect previously used queries from result files that exist."""

    qids = set()
    for path in paths:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("observations", payload.get("results", []))
        qids.update(str(row["qid"]) for row in rows)
    return qids


def stratified_cases(
    questions: dict[str, dict[str, Any]],
    retrieval: dict[str, list[dict[str, Any]]],
    *,
    excluded_qids: set[str],
    n_easy: int,
    n_hard: int,
    seed: int,
) -> list[tuple[dict[str, Any], list[dict[str, Any]], str]]:
    """Sample early-support and post-Top-2 queries independently."""

    strata: dict[str, list[tuple[dict[str, Any], list[dict[str, Any]], str]]] = {
        "easy": [],
        "hard": [],
    }
    for qid, rows in retrieval.items():
        question = questions.get(qid)
        supports = [row for row in rows if row["support_label"] == "support"]
        false_rows = [row for row in rows if row["support_label"] == "false"]
        if (
            qid in excluded_qids
            or question is None
            or question.get("metadata", {}).get("source_split") != "train"
            or not supports
            or len(false_rows) < 3
        ):
            continue
        stratum = "hard" if min(int(row["rank"]) for row in supports) > 2 else "easy"
        strata[stratum].append((question, rows, stratum))

    requested = {"easy": n_easy, "hard": n_hard}
    rng = random.Random(seed)
    selected = []
    for stratum, limit in requested.items():
        if len(strata[stratum]) < limit:
            raise ValueError(
                f"Only {len(strata[stratum])} {stratum} cases for requested {limit}."
            )
        selected.extend(rng.sample(strata[stratum], limit))
    rng.shuffle(selected)
    return selected


def candidate_rows(rows: list[dict[str, Any]], *, n_false: int) -> list[dict[str, Any]]:
    """Keep every labeled support and the highest-ranked hard negatives."""

    supports = [row for row in rows if row["support_label"] == "support"]
    false_rows = [row for row in rows if row["support_label"] == "false"][:n_false]
    return sorted([*supports, *false_rows], key=lambda row: int(row["rank"]))


def attention_features(
    trace: Any,
    *,
    chunk_index: int,
    token_count: int,
    focus_layers: set[int],
) -> dict[str, Any]:
    """Summarize answer-to-chunk attention with length and context controls."""

    mass = trace.attention_mass[:, :, chunk_index].astype(np.float64)
    totals = trace.attention_mass.sum(axis=-1).astype(np.float64)
    fractions = np.divide(mass, totals, out=np.zeros_like(mass), where=totals > 0)
    focus_indices = [
        index for index, layer in enumerate(trace.layer_ids) if layer in focus_layers
    ]
    if not focus_indices:
        raise ValueError("None of the focus layers were extracted")
    focus_mass = mass[focus_indices]
    focus_fraction = fractions[focus_indices]
    length = max(token_count, 1)
    return {
        "chunk_token_count": token_count,
        "attention_mass_focus": float(focus_mass.mean()),
        "attention_fraction_focus": float(focus_fraction.mean()),
        "attention_density_focus": float(focus_mass.mean() / length),
        "attention_sqrt_density_focus": float(focus_mass.mean() / np.sqrt(length)),
        "attention_max_head_focus": float(focus_mass.max()),
        "attention_mass_all_layers": float(mass.mean()),
        "attention_fraction_all_layers": float(fractions.mean()),
        "attention_mass_by_layer_head": mass.astype(np.float32).ravel().tolist(),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-easy", type=int, default=60)
    parser.add_argument("--n-hard", type=int, default=60)
    parser.add_argument("--n-false", type=int, default=3)
    parser.add_argument("--context-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--focus-layers", default="14,18")
    parser.add_argument("--max-answer-tokens", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument(
        "--exclude-results",
        type=Path,
        action="append",
        default=[],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    default_exclusions = [
        Path("research/internal_state_rag/results/tatqa_smoke_n39_seed17.json"),
        Path(
            "research/internal_state_rag/results/"
            "chunk_attribution_n24_all_l14_l18_seed31.json"
        ),
    ]
    exclusions = default_exclusions + args.exclude_results
    excluded_qids = result_qids(exclusions)
    focus_layers = {int(layer) for layer in args.focus_layers.split(",")}
    corpus = load_rows(args.corpus, "id")
    questions = load_rows(args.questions, "qid")
    retrieval = load_retrieval(args.retrieval)
    cases = stratified_cases(
        questions,
        retrieval,
        excluded_qids=excluded_qids,
        n_easy=args.n_easy,
        n_hard=args.n_hard,
        seed=args.seed,
    )

    client = ResearchTextClient(args.model, device="cuda", load_in_4bit=False)
    extractor = QwenInternalStateExtractor(client)
    observations = []
    layer_ids: list[int] | None = None

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
            print(f"[{query_index}/{len(cases)}] skip empty draft {question['qid']}")
            continue
        trace = extractor.extract(
            prompt, chunks, draft, max_answer_tokens=args.max_answer_tokens
        )
        layer_ids = trace.layer_ids
        alignment = client.align_and_prepare_inputs(prompt, chunks)
        spans = {span.chunk_id: span for span in alignment.batch_chunk_spans[0]}

        for candidate in candidates:
            chunk_id = str(candidate["chunk_id"])
            span = spans[chunk_id]
            token_count = (
                0
                if span.start is None or span.end is None
                else int(span.end) - int(span.start)
            )
            record = {
                "qid": str(question["qid"]),
                "stratum": stratum,
                "question": str(question["question"]),
                "draft": draft,
                "chunk_id": chunk_id,
                "rank": int(candidate["rank"]),
                "modality": str(candidate["modality"]),
                "is_support": candidate["support_label"] == "support",
                "cosine_score": float(candidate["cosine_score"]),
                "bge_score": float(candidate["selection_score"]),
                **attention_features(
                    trace,
                    chunk_index=chunk_offsets[chunk_id],
                    token_count=token_count,
                    focus_layers=focus_layers,
                ),
            }
            observations.append(record)

        support_ranks = [
            int(row["rank"])
            for row in candidates
            if row["support_label"] == "support"
        ]
        print(
            f"[{query_index}/{len(cases)}] {question['qid']} {stratum} "
            f"support_ranks={support_ranks} "
            f"draft={draft[:50]!r}",
            flush=True,
        )
        write_json(
            args.output,
            {
                "status": "running",
                "model": args.model,
                "seed": args.seed,
                "completed_queries": query_index,
                "layer_ids": layer_ids,
                "focus_layers": sorted(focus_layers),
                "observations": observations,
            },
        )

    score_names = [
        "cosine_score",
        "bge_score",
        "attention_mass_focus",
        "attention_fraction_focus",
        "attention_density_focus",
        "attention_sqrt_density_focus",
        "attention_max_head_focus",
        "attention_mass_all_layers",
        "attention_fraction_all_layers",
    ]
    output = {
        "status": "complete",
        "model": args.model,
        "split": "official_train_only",
        "seed": args.seed,
        "n_queries": len({row["qid"] for row in observations}),
        "n_candidates": len(observations),
        "n_easy": args.n_easy,
        "n_hard": args.n_hard,
        "context_k": args.context_k,
        "layer_ids": layer_ids,
        "focus_layers": sorted(focus_layers),
        "excluded_qids": len(excluded_qids),
        "metrics": score_metrics(observations, score_names),
        "observations": observations,
    }
    write_json(args.output, output)
    summary = {key: value for key, value in output.items() if key != "observations"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
