"""Validate BGE and BGE-attention pruning with downstream answer generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from eval.metrics import exact_match, numerical_accuracy, token_f1
from research.internal_state_rag.analyze_full_topl_validation import load_observations
from research.internal_state_rag.analyze_position_control import conformal_threshold
from research.internal_state_rag.ranking import grouped_z_scores
from research.internal_state_rag.run_full_topl_validation import append_jsonl
from research.internal_state_rag.run_tatqa_smoke import (
    ResearchTextClient,
    generate_answer,
    load_rows,
)
from uncertainty_rag.modality.base import ContextChunk


def fit_pruning_scores(
    observations: list[dict[str, Any]],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Fit the frozen three-feature fusion using scorer-train observations only."""

    feature_names = [
        "bge_score",
        "mean_attention_mass",
        "mean_attention_fraction",
    ]
    features = np.column_stack(
        [grouped_z_scores(observations, name) for name in feature_names]
    )
    labels = np.asarray([int(row["is_support"]) for row in observations])
    roles = np.asarray([str(row["role"]) for row in observations])
    train = roles == "scorer_train"
    scaler = StandardScaler().fit(features[train])
    model = LogisticRegression(
        class_weight="balanced", random_state=0, max_iter=1_000
    ).fit(scaler.transform(features[train]), labels[train])
    return (
        {
            "bge": features[:, 0],
            "fusion": model.predict_proba(scaler.transform(features))[:, 1],
        },
        {
            "features": feature_names,
            "coefficients": model.coef_[0].tolist(),
            "intercept": float(model.intercept_[0]),
        },
    )


def score_lookup(
    observations: list[dict[str, Any]], scores: np.ndarray
) -> dict[tuple[str, str], float]:
    return {
        (str(row["qid"]), str(row["chunk_id"])): float(score)
        for row, score in zip(observations, scores)
    }


def prediction_metrics(prediction: str, gold_answers: list[str]) -> dict[str, float]:
    return {
        "em": exact_match(prediction, gold_answers),
        "f1": token_f1(prediction, gold_answers),
        "numerical_accuracy": numerical_accuracy(prediction, gold_answers),
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    methods = ["full_top30", "bge_a05", "fusion_a05", "bge_a10", "fusion_a10"]
    output: dict[str, Any] = {}
    for stratum in ("all", "easy", "hard"):
        selected = [
            row for row in records if stratum == "all" or row["stratum"] == stratum
        ]
        output[stratum] = {"n_queries": len(selected), "methods": {}}
        for method in methods:
            output[stratum]["methods"][method] = {
                metric: float(np.mean([row["metrics"][method][metric] for row in selected]))
                for metric in ("em", "f1", "numerical_accuracy")
            }
            output[stratum]["methods"][method].update(
                {
                    "mean_chunks": float(
                        np.mean([row["contexts"][method]["n_chunks"] for row in selected])
                    ),
                    "mean_tokens": float(
                        np.mean([row["contexts"][method]["n_tokens"] for row in selected])
                    ),
                }
            )
    return output


def parse_args() -> argparse.Namespace:
    root = Path("data/conformal_global_run")
    result_root = Path("research/internal_state_rag/results/full_top30_n420_seed101")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--rows", type=Path, default=result_root / "queries.jsonl")
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
        "--output", type=Path, default=result_root / "pruned_answers.jsonl"
    )
    parser.add_argument(
        "--summary", type=Path, default=result_root / "pruned_answer_summary.json"
    )
    parser.add_argument("--max-new-tokens", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    observations = load_observations(args.rows)
    score_values, fusion = fit_pruning_scores(observations)
    calibration_qids = {
        str(row["qid"])
        for row in observations
        if row["role"] == "conformal_calibration"
    }
    thresholds = {
        f"{name}_a{int(alpha * 100):02d}": conformal_threshold(
            observations, scores, calibration_qids, alpha
        )
        for alpha in (0.05, 0.1)
        for name, scores in score_values.items()
    }
    lookups = {
        name: score_lookup(observations, scores)
        for name, scores in score_values.items()
    }
    query_rows = [
        json.loads(line)
        for line in args.rows.read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("role") == "evaluation"
    ]
    questions = load_rows(args.questions, "qid")
    corpus = load_rows(args.corpus, "id")
    completed: dict[str, dict[str, Any]] = {}
    if args.output.exists():
        completed = {
            str(row["qid"]): row
            for row in (
                json.loads(line)
                for line in args.output.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        }

    client = ResearchTextClient(args.model, device="cuda", load_in_4bit=False)
    for index, query in enumerate(query_rows, start=1):
        qid = str(query["qid"])
        if qid in completed:
            print(f"[{index}/{len(query_rows)}] resume skip {qid}", flush=True)
            continue
        question = questions[qid]
        gold_answers = [str(value) for value in question["gold_answers"]]
        prompt = (
            "Answer using only the supplied context. Return only the short answer.\n"
            f"Question: {question['question']}"
        )
        predictions = {"full_top30": str(query["draft"])}
        contexts: dict[str, dict[str, Any]] = {
            "full_top30": {
                "n_chunks": len(query["candidates"]),
                "n_tokens": sum(
                    int(candidate["chunk_token_count"])
                    for candidate in query["candidates"]
                ),
                "all_support_kept": True,
            }
        }
        for key, threshold in thresholds.items():
            scorer = key.split("_a", maxsplit=1)[0]
            kept = [
                candidate
                for candidate in query["candidates"]
                if lookups[scorer][(qid, str(candidate["chunk_id"]))] >= threshold
            ]
            chunks = [
                ContextChunk(
                    id=str(candidate["chunk_id"]),
                    content=str(corpus[str(candidate["chunk_id"])]["content"]),
                    modality=str(candidate["modality"]),
                    metadata={"rank": int(candidate["rank"])},
                )
                for candidate in kept
            ]
            predictions[key] = generate_answer(
                client, prompt, chunks, max_new_tokens=args.max_new_tokens
            )
            contexts[key] = {
                "n_chunks": len(kept),
                "n_tokens": sum(int(candidate["chunk_token_count"]) for candidate in kept),
                "all_support_kept": all(
                    not candidate["is_support"] or candidate in kept
                    for candidate in query["candidates"]
                ),
            }
        record = {
            "qid": qid,
            "stratum": str(query["stratum"]),
            "answer_type": str(question.get("metadata", {}).get("answer_type", "")),
            "gold_answers": gold_answers,
            "predictions": predictions,
            "metrics": {
                name: prediction_metrics(prediction, gold_answers)
                for name, prediction in predictions.items()
            },
            "contexts": contexts,
        }
        append_jsonl(args.output, record)
        completed[qid] = record
        print(
            f"[{index}/{len(query_rows)}] {qid} "
            f"F1 full={record['metrics']['full_top30']['f1']:.3f} "
            f"bge10={record['metrics']['bge_a10']['f1']:.3f} "
            f"fusion10={record['metrics']['fusion_a10']['f1']:.3f}",
            flush=True,
        )

    ordered = [completed[str(query["qid"])] for query in query_rows]
    payload = {
        "status": "complete" if len(ordered) == len(query_rows) else "partial",
        "n_queries": len(ordered),
        "model": args.model,
        "max_new_tokens": args.max_new_tokens,
        "thresholds": thresholds,
        "fusion": fusion,
        "results": summarize(ordered),
    }
    args.summary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
