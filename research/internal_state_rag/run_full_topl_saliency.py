"""Validate a frozen contrastive-saliency scorer on complete Top-L contexts."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from eval.metrics import exact_match, numerical_accuracy, token_f1
from research.internal_state_rag.contrastive_saliency import extract_contrastive_saliency
from research.internal_state_rag.ranking import ranking_metrics
from research.internal_state_rag.run_full_topl_validation import eligible_qids
from research.internal_state_rag.run_tatqa_smoke import (
    ResearchTextClient,
    generate_answer,
    load_retrieval,
    load_rows,
    to_chunk,
)


def grouped_z(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    scale = float(values.std())
    return (values - values.mean()) / (scale + 1e-7)


def excluded_qids(paths: list[Path]) -> set[str]:
    excluded: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "plan" in payload:
            excluded.update(str(row["qid"]) for row in payload["plan"])
        for row in payload.get("observations", []):
            excluded.add(str(row["qid"]))
    return excluded


def write_json(path: Path, payload: dict[str, Any]) -> None:
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
    parser.add_argument("--input-role", choices=("calibration", "test"), default="test")
    parser.add_argument("--source-split", default="dev")
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--top-l", type=int, default=30)
    parser.add_argument("--seed", type=int, default=307)
    parser.add_argument("--layer", type=int, default=33)
    parser.add_argument("--internal-weight", type=float, default=3.0)
    parser.add_argument("--max-answer-tokens", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--exclude", type=Path, action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    corpus = load_rows(args.corpus, "id")
    questions = load_rows(args.questions, "qid")
    retrieval = load_retrieval(args.retrieval, split_role=args.input_role)
    eligible = eligible_qids(questions, retrieval, source_split=args.source_split)
    excluded = excluded_qids(args.exclude)
    available = [qid for qid in eligible if qid not in excluded]
    if len(available) < args.n:
        raise ValueError(f"Only {len(available)} eligible unexcluded queries for n={args.n}")
    plan = random.Random(args.seed).sample(available, args.n)

    observations: list[dict[str, Any]] = []
    output: dict[str, Any] = {}
    if args.output.exists():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        if previous.get("plan") != plan:
            raise ValueError("Existing output has a different frozen query plan")
        observations = previous.get("observations", [])
        output = previous
    completed = {str(row["qid"]) for row in observations}
    client = ResearchTextClient(args.model, device="cuda", load_in_4bit=False)

    for index, qid in enumerate(plan, start=1):
        if qid in completed:
            continue
        question = questions[qid]
        retrieval_rows = retrieval[qid][: args.top_l]
        chunks = [to_chunk(row, corpus) for row in retrieval_rows]
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
            layers=[args.layer],
        )
        internal = trace.features["residual_grad_x_mean"][0].astype(np.float64)
        bge = np.asarray([float(row["selection_score"]) for row in retrieval_rows])
        fusion = grouped_z(bge) + args.internal_weight * grouped_z(internal)
        candidates = []
        for row, saliency, fused in zip(retrieval_rows, internal, fusion, strict=True):
            candidates.append(
                {
                    "chunk_id": str(row["chunk_id"]),
                    "rank": int(row["rank"]),
                    "modality": str(row["modality"]),
                    "is_support": row["support_label"] == "support",
                    "bge_score": float(row["selection_score"]),
                    "internal_score": float(saliency),
                    "fusion_score": float(fused),
                }
            )
        gold = [str(value) for value in question.get("gold_answers", [])]
        record = {
            "qid": qid,
            "stratum": "hard"
            if min(row["rank"] for row in candidates if row["is_support"]) > 2
            else "easy",
            "question": str(question["question"]),
            "gold_answers": gold,
            "draft": draft,
            "draft_em": exact_match(draft, gold),
            "draft_f1": token_f1(draft, gold),
            "draft_numerical_accuracy": numerical_accuracy(draft, gold),
            "cti_js_mean": float(np.mean(trace.cti_js_divergence)),
            "selected_answer_tokens": len(trace.selected_answer_indices),
            "candidates": candidates,
        }
        observations.append(record)
        completed.add(qid)
        flat = [
            {"qid": item["qid"], **candidate}
            for item in observations
            for candidate in item["candidates"]
        ]
        output = {
            "status": "running",
            "input_role": args.input_role,
            "source_split": args.source_split,
            "seed": args.seed,
            "top_l": args.top_l,
            "layer": args.layer,
            "internal_weight": args.internal_weight,
            "excluded_queries": len(excluded),
            "plan": plan,
            "completed_queries": len(observations),
            "metrics": {
                name: ranking_metrics(flat, np.asarray([row[name] for row in flat]))
                for name in ("bge_score", "internal_score", "fusion_score")
            },
            "observations": observations,
        }
        write_json(args.output, output)
        support = [row["rank"] for row in candidates if row["is_support"]]
        print(
            f"[{index}/{len(plan)}] {qid} support={support} "
            f"cti={record['cti_js_mean']:.4g} draft_f1={record['draft_f1']:.3f}",
            flush=True,
        )

    output["status"] = "complete" if len(observations) == len(plan) else "partial"
    write_json(args.output, output)
    print(json.dumps({key: value for key, value in output.items() if key != "observations"}, indent=2))


if __name__ == "__main__":
    main()
