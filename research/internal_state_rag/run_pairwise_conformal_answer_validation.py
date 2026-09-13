"""Validate internal-fusion conformal pruning with generated TAT-QA answers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from eval.metrics import exact_match, numerical_accuracy, token_f1
from research.internal_state_rag.analyze_hidden_chunk_probe import query_z
from research.internal_state_rag.analyze_pairwise_conformal_pruning import probe_scores
from research.internal_state_rag.run_full_topl_validation import append_jsonl
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
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--fusion-analysis", type=Path, required=True)
    parser.add_argument("--conformal-analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=24)
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
    return parser.parse_args()


def prediction_metrics(prediction: str, gold_answers: list[str]) -> dict[str, float]:
    return {
        "em": exact_match(prediction, gold_answers),
        "f1": token_f1(prediction, gold_answers),
        "numerical_accuracy": numerical_accuracy(prediction, gold_answers),
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    methods = list(records[0]["predictions"])
    result = {}
    for method in methods:
        result[method] = {
            metric: float(np.mean([row["metrics"][method][metric] for row in records]))
            for metric in ("em", "f1", "numerical_accuracy")
        }
        result[method]["mean_chunks"] = float(
            np.mean([row["contexts"][method]["n_chunks"] for row in records])
        )
    for method in methods:
        if method == "full_top30":
            continue
        result[method]["f1_minus_full"] = float(
            np.mean(
                [
                    row["metrics"][method]["f1"]
                    - row["metrics"]["full_top30"]["f1"]
                    for row in records
                ]
            )
        )
    rng = np.random.default_rng(487)
    draws = rng.integers(0, len(records), size=(10_000, len(records)))
    for alpha in ("10", "05"):
        fusion_name, bge_name = f"fusion_a{alpha}", f"bge_a{alpha}"
        result[fusion_name]["paired_minus_bge"] = {}
        for metric in ("em", "f1", "numerical_accuracy"):
            differences = np.asarray(
                [
                    row["metrics"][fusion_name][metric]
                    - row["metrics"][bge_name][metric]
                    for row in records
                ]
            )
            boot = differences[draws].mean(axis=1)
            result[fusion_name]["paired_minus_bge"][metric] = {
                "mean": float(differences.mean()),
                "ci95_low": float(np.quantile(boot, 0.025)),
                "ci95_high": float(np.quantile(boot, 0.975)),
            }
    return result


def main() -> None:
    args = parse_args()
    train = np.load(args.train)
    test = np.load(args.test)
    if args.n < 1 or args.n > len(test["qids"]):
        raise ValueError("n lies outside the test feature set")
    fusion = json.loads(args.fusion_analysis.read_text(encoding="utf-8"))
    conformal = json.loads(args.conformal_analysis.read_text(encoding="utf-8"))
    layer, c = int(fusion["selected_layer"]), float(fusion["selected_c"])
    weights = fusion["selected_three_signal_weights"]
    _, test_probe = probe_scores(train, test, layer=layer, c=c)
    score_arrays = {
        "bge": query_z(test["bge_scores"].astype(np.float64)),
        "fusion": (
            query_z(test["bge_scores"].astype(np.float64))
            + float(weights["lm_head"])
            * query_z(test["lm_relevance_scores"].astype(np.float64))
            + float(weights["hidden_probe"]) * query_z(test_probe)
        ),
    }
    thresholds = {}
    for alpha in (0.1, 0.05):
        row = conformal["results"]["all_support"][str(alpha)]
        thresholds[f"bge_a{int(alpha * 100):02d}"] = float(
            row["bge"]["calibrated_threshold"]
        )
        thresholds[f"fusion_a{int(alpha * 100):02d}"] = float(
            row["three_signal_fusion"]["calibrated_threshold"]
        )

    qids = [str(value) for value in test["qids"][: args.n]]
    questions = load_rows(args.questions, "qid")
    corpus = load_rows(args.corpus, "id")
    retrieval = load_retrieval(args.retrieval, split_role="test")
    completed = {}
    if args.output.exists():
        completed = {
            str(row["qid"]): row
            for row in (
                json.loads(line)
                for line in args.output.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    client = ResearchTextClient(args.model, device="cuda", load_in_4bit=False)
    for query_index, qid in enumerate(qids):
        if qid in completed:
            print(f"[{query_index + 1}/{args.n}] resume-skip {qid}", flush=True)
            continue
        question = questions[qid]
        candidates = retrieval[qid][:30]
        chunks = [to_chunk(row, corpus) for row in candidates]
        prompt = (
            "Answer using only the supplied context. Return only the short answer.\n"
            f"Question: {question['question']}"
        )
        predictions = {
            "full_top30": generate_answer(
                client, prompt, chunks, max_new_tokens=args.max_new_tokens
            )
        }
        contexts = {"full_top30": {"n_chunks": 30}}
        for method, threshold in thresholds.items():
            scorer = "fusion" if method.startswith("fusion") else "bge"
            mask = score_arrays[scorer][query_index] >= threshold
            if not mask.any():
                mask[np.argmax(score_arrays[scorer][query_index])] = True
            kept_chunks = [chunk for chunk, keep in zip(chunks, mask, strict=True) if keep]
            predictions[method] = generate_answer(
                client, prompt, kept_chunks, max_new_tokens=args.max_new_tokens
            )
            contexts[method] = {"n_chunks": len(kept_chunks)}
        gold_answers = [str(value) for value in question["gold_answers"]]
        record = {
            "qid": qid,
            "gold_answers": gold_answers,
            "predictions": predictions,
            "metrics": {
                method: prediction_metrics(prediction, gold_answers)
                for method, prediction in predictions.items()
            },
            "contexts": contexts,
        }
        append_jsonl(args.output, record)
        completed[qid] = record
        print(
            f"[{query_index + 1}/{args.n}] {qid} "
            f"F1 full={record['metrics']['full_top30']['f1']:.3f} "
            f"bge05={record['metrics']['bge_a05']['f1']:.3f} "
            f"fusion05={record['metrics']['fusion_a05']['f1']:.3f}",
            flush=True,
        )
    ordered = [completed[qid] for qid in qids if qid in completed]
    payload = {
        "status": "complete" if len(ordered) == args.n else "partial",
        "n_queries": len(ordered),
        "planned_queries": args.n,
        "model": args.model,
        "thresholds": thresholds,
        "fusion_weights": weights,
        "results": summarize(ordered),
    }
    args.summary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
