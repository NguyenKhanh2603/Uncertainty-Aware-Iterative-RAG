"""Run downstream QA for the frozen query-level cosine + internal fusion mask.

The masks are loaded from the Qwen-7B fusion ablation artifact rather than
retuned here.  Thus probe training, fusion weights, and query-level conformal
calibration remain disjoint from the 100 frozen test queries.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from eval.metrics import exact_match, numerical_accuracy, token_f1
from research.internal_state_rag.run_downstream_qa_conformal_baselines import (
    QwenDirectAnswerGenerator,
    chunks,
    grouped_retrieval,
    iter_jsonl,
    keyed,
    paths,
)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def metric(prediction: str, gold: list[str]) -> dict[str, float]:
    return {
        "em": exact_match(prediction, gold),
        "f1": token_f1(prediction, gold),
        "numerical_accuracy": numerical_accuracy(prediction, gold),
    }


def summary(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    return {
        "n_queries": len(rows),
        **{
            name: float(np.mean([row["metrics"][name] for row in rows]))
            for name in ("em", "f1", "numerical_accuracy")
        },
        "mean_chunks": float(np.mean([row["context"]["n_chunks"] for row in rows])),
        "empty_rate": float(np.mean([row["context"]["n_chunks"] == 0 for row in rows])),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/conformal_global_run"))
    parser.add_argument("--artifact-root", type=Path, default=Path("research/internal_state_rag/results/qwen2vl_7b_jina4"))
    parser.add_argument("--datasets", default="hotpotqa,mmqa,tatqa")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--min-pixels", type=int, default=3136)
    parser.add_argument("--max-pixels", type=int, default=200704)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    alpha_key = str(args.alpha)
    method = "query_level_cosine_internal_fusion"
    client: QwenDirectAnswerGenerator | None = None
    manifest: dict[str, Any] = {
        "status": "running", "model": str(args.model), "method": method,
        "alpha": args.alpha, "min_pixels": args.min_pixels, "max_pixels": args.max_pixels,
        "datasets": {},
    }
    for dataset in (item.strip() for item in args.datasets.split(",") if item.strip()):
        questions_path, corpus_path, retrieval_path, bundle_root = paths(dataset, args.data_root)
        artifact = args.artifact_root / f"{dataset}_fusion_ablation_100_aligned_predictions.npz"
        analysis_path = args.artifact_root / f"{dataset}_fusion_ablation_100_aligned.json"
        if not all(path.is_file() for path in (questions_path, corpus_path, retrieval_path, artifact, analysis_path)):
            manifest["datasets"][dataset] = {"status": "unavailable_source_files"}
            continue
        saved = np.load(artifact)
        qids = [str(qid) for qid in saved["qids"].tolist()]
        masks = saved[f"mask_cosine_internal_alpha_{alpha_key}"].astype(bool)
        ids = saved["chunk_ids"].astype(str)
        questions = keyed(questions_path, "qid")
        corpus = keyed(corpus_path, "id")
        retrieval = grouped_retrieval(retrieval_path)["test"]
        output = args.output_dir / f"{dataset}_predictions.jsonl"
        done = {str(row["qid"]): row for row in iter_jsonl(output)} if output.exists() else {}
        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
        conformal = analysis["conformal"][alpha_key]["cosine_internal"]
        if client is None:
            client = QwenDirectAnswerGenerator(args.model, min_pixels=args.min_pixels, max_pixels=args.max_pixels)
        for index, qid in enumerate(qids, start=1):
            if qid in done:
                print(f"[{dataset} {index}/{len(qids)}] resume {qid}", flush=True)
                continue
            rows = retrieval[qid][:30]
            by_id = {str(chunk_id): bool(keep) for chunk_id, keep in zip(ids[index - 1], masks[index - 1], strict=True)}
            if {str(row["chunk_id"]) for row in rows} != set(by_id):
                raise ValueError(f"{dataset}/{qid}: frozen candidate IDs do not match fusion artifact")
            selected_mask = np.asarray([by_id[str(row["chunk_id"])] for row in rows], dtype=bool)
            selected = chunks(rows, selected_mask, corpus, bundle_root)
            question = questions[qid]
            prompt = "Answer using only the supplied context. Return only the short answer.\nQuestion: " + str(question["question"])
            prediction = client.generate(prompt, selected, max_new_tokens=args.max_new_tokens)
            gold = [str(value) for value in question["gold_answers"]]
            record = {
                "qid": qid, "gold_answers": gold, "prediction": prediction,
                "metrics": metric(prediction, gold),
                "context": {"n_chunks": len(selected), "chunk_ids": [chunk.id for chunk in selected]},
            }
            append_jsonl(output, record)
            done[qid] = record
            print(f"[{dataset} {index}/{len(qids)}] {qid} em={record['metrics']['em']:.0f}", flush=True)
        ordered = [done[qid] for qid in qids if qid in done]
        result = {
            "status": "complete" if len(ordered) == len(qids) else "partial",
            "dataset": dataset, "method": method, "alpha": args.alpha,
            "query_level_target": "all_support",
            "fusion_weights": analysis["selected_weights"]["cosine_internal"],
            "threshold": conformal["threshold"],
            "results": summary(ordered),
        }
        (args.output_dir / f"{dataset}_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        manifest["datasets"][dataset] = {"status": result["status"], "summary": str(args.output_dir / f"{dataset}_summary.json")}
    manifest["status"] = "complete"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
