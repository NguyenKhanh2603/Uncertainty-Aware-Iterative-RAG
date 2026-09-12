"""Extract BGE and position-controlled internal attention on full Top-L contexts."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from research.internal_state_rag import QwenInternalStateExtractor
from research.internal_state_rag.run_attention_chunk_ranking import (
    attention_features,
    result_qids,
)
from research.internal_state_rag.run_tatqa_smoke import (
    ResearchTextClient,
    generate_answer,
    load_retrieval,
    load_rows,
    to_chunk,
)


def eligible_qids(
    questions: dict[str, dict[str, Any]],
    retrieval: dict[str, list[dict[str, Any]]],
) -> list[str]:
    """Return calibration-role train questions with support inside frozen Top-L."""

    return sorted(
        qid
        for qid, rows in retrieval.items()
        if qid in questions
        and questions[qid].get("metadata", {}).get("source_split") == "train"
        and any(row["support_label"] == "support" for row in rows)
    )


def make_split_plan(
    eligible: list[str],
    train_qids: list[str],
    excluded_qids: set[str],
    *,
    n_calibration: int,
    n_evaluation: int,
    seed: int,
) -> list[dict[str, str]]:
    """Freeze disjoint scorer-train, conformal-calibration, and evaluation roles."""

    eligible_set = set(eligible)
    train = list(dict.fromkeys(qid for qid in train_qids if qid in eligible_set))
    unavailable = set(train) | excluded_qids
    untouched = [qid for qid in eligible if qid not in unavailable]
    requested = n_calibration + n_evaluation
    if len(untouched) < requested:
        raise ValueError(f"Only {len(untouched)} untouched queries for {requested} requested")
    sampled = random.Random(seed).sample(untouched, requested)
    calibration = sampled[:n_calibration]
    evaluation = sampled[n_calibration:]
    plan = [
        *({"qid": qid, "role": "scorer_train"} for qid in train),
        *({"qid": qid, "role": "conformal_calibration"} for qid in calibration),
        *({"qid": qid, "role": "evaluation"} for qid in evaluation),
    ]
    random.Random(seed + 1).shuffle(plan)
    return plan


def load_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            completed.add(str(json.loads(line)["qid"]))
    return completed


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def scalar_attention_features(
    trace: Any, *, chunk_id: str, token_count: int
) -> dict[str, float]:
    index = trace.chunk_ids.index(chunk_id)
    features = attention_features(
        trace,
        chunk_index=index,
        token_count=token_count,
        focus_layers={14, 18},
    )
    return {
        "mass": float(features["attention_mass_all_layers"]),
        "fraction": float(features["attention_fraction_all_layers"]),
    }


def parse_args() -> argparse.Namespace:
    root = Path("data/conformal_global_run")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
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
    parser.add_argument(
        "--train-results",
        type=Path,
        default=Path(
            "research/internal_state_rag/results/"
            "attention_ranking_n120_balanced_seed47.json"
        ),
    )
    parser.add_argument("--n-calibration", type=int, default=150)
    parser.add_argument("--n-evaluation", type=int, default=150)
    parser.add_argument("--top-l", type=int, default=30)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--max-answer-tokens", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    rows_path = args.output_dir / "queries.jsonl"
    summary_path = args.output_dir / "summary.json"
    corpus = load_rows(args.corpus, "id")
    questions = load_rows(args.questions, "qid")
    retrieval = load_retrieval(args.retrieval)
    eligible = eligible_qids(questions, retrieval)
    train_payload = json.loads(args.train_results.read_text(encoding="utf-8"))
    train_qids = list(dict.fromkeys(str(row["qid"]) for row in train_payload["observations"]))
    discovery_exclusions = result_qids(
        [
            Path("research/internal_state_rag/results/tatqa_smoke_n39_seed17.json"),
            Path(
                "research/internal_state_rag/results/"
                "chunk_attribution_n24_all_l14_l18_seed31.json"
            ),
        ]
    )

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["seed"] != args.seed or manifest["top_l"] != args.top_l:
            raise ValueError("Existing manifest does not match requested seed/Top-L")
        plan = manifest["plan"]
    else:
        plan = make_split_plan(
            eligible,
            train_qids,
            discovery_exclusions,
            n_calibration=args.n_calibration,
            n_evaluation=args.n_evaluation,
            seed=args.seed,
        )
        manifest = {
            "dataset": "tatqa",
            "input_role": "calibration",
            "source_split": "train",
            "seed": args.seed,
            "top_l": args.top_l,
            "n_role_queries": len(retrieval),
            "n_queries_with_support_in_top_l": len(eligible),
            "conditional_retrieval_ceiling": len(eligible) / len(retrieval),
            "n_discovery_exclusions": len(discovery_exclusions),
            "plan": plan,
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    completed = load_completed(rows_path)
    client = ResearchTextClient(args.model, device="cuda", load_in_4bit=False)
    extractor = QwenInternalStateExtractor(client)

    for query_index, item in enumerate(plan, start=1):
        qid = str(item["qid"])
        if qid in completed:
            print(f"[{query_index}/{len(plan)}] resume skip {qid}", flush=True)
            continue
        question = questions[qid]
        retrieval_rows = retrieval[qid][: args.top_l]
        chunks = [to_chunk(row, corpus) for row in retrieval_rows]
        prompt = (
            "Answer using only the supplied context. Return only the short answer.\n"
            f"Question: {question['question']}"
        )
        draft = generate_answer(
            client, prompt, chunks, max_new_tokens=args.max_new_tokens
        )
        if not draft:
            append_jsonl(
                rows_path,
                {"qid": qid, "role": item["role"], "status": "empty_draft"},
            )
            continue
        original_trace = extractor.extract(
            prompt, chunks, draft, max_answer_tokens=args.max_answer_tokens
        )
        original_alignment = client.align_and_prepare_inputs(prompt, chunks)
        original_spans = {
            span.chunk_id: span for span in original_alignment.batch_chunk_spans[0]
        }
        reversed_chunks = list(reversed(chunks))
        reversed_trace = extractor.extract(
            prompt, reversed_chunks, draft, max_answer_tokens=args.max_answer_tokens
        )

        candidate_records = []
        for retrieval_row in retrieval_rows:
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
            candidate_records.append(
                {
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
                    "mean_attention_mass": (original["mass"] + reversed_features["mass"])
                    / 2,
                    "mean_attention_fraction": (
                        original["fraction"] + reversed_features["fraction"]
                    )
                    / 2,
                }
            )

        support_ranks = [row["rank"] for row in candidate_records if row["is_support"]]
        record = {
            "status": "complete",
            "qid": qid,
            "role": item["role"],
            "stratum": "hard" if min(support_ranks) > 2 else "easy",
            "question": str(question["question"]),
            "draft": draft,
            "n_answer_tokens": len(original_trace.answer_token_ids),
            "support_ranks": support_ranks,
            "candidates": candidate_records,
        }
        append_jsonl(rows_path, record)
        completed.add(qid)
        print(
            f"[{query_index}/{len(plan)}] {qid} {item['role']} "
            f"support={support_ranks} draft={draft[:42]!r}",
            flush=True,
        )

    complete_rows = load_completed(rows_path)
    summary = {
        "status": "complete" if len(complete_rows) == len(plan) else "partial",
        "model": args.model,
        "planned_queries": len(plan),
        "completed_queries": len(complete_rows),
        "rows": str(rows_path),
        "manifest": str(manifest_path),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
