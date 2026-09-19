"""Run the retrieval-pruning QA comparison for one independently sampled seed.

The feature plans define disjoint probe-train, calibration, and test query
sets.  Baseline selectors are calibrated from that seed's 100 calibration
queries; the fusion selector is fitted and calibrated by
``analyze_qwen2vl_jina_ablation.py`` on the same split.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from research.internal_state_rag.run_downstream_qa_conformal_baselines import (
    QwenDirectAnswerGenerator, append_jsonl, by_masks, chunks, grouped_retrieval,
    iter_jsonl, keyed, metrics, paths, selector_masks,
)

METHODS = ("full_top30", "by_cosine", "ecir_cce_embedding", "conflare_retrieval_adapter", "traq_retrieval_bonferroni_adapter", "query_level_cosine_lm_hidden")


def plan(path: Path) -> list[str]:
    return [str(qid) for qid in json.loads(path.read_text(encoding="utf-8"))["plan"]]


def selection_metrics(rows, masks):
    values = {}
    for name, selected in masks.items():
        counts = np.asarray([int(mask.sum()) for mask in selected], dtype=float)
        precisions, recalls, any_support, all_support = [], [], [], []
        for query_rows, mask in zip(rows, selected):
            labels = np.asarray([row["support_label"] == "support" for row in query_rows], dtype=bool)
            kept = labels[mask]
            precisions.append(float(kept.mean()) if len(kept) else 0.0)
            recalls.append(float(kept.sum() / labels.sum()) if labels.sum() else 1.0)
            any_support.append(float(kept.any()) if labels.sum() else 1.0)
            all_support.append(float(kept.sum() == labels.sum()) if labels.sum() else 1.0)
        values[name] = {"mean_chunks": float(counts.mean()), "empty_rate": float((counts == 0).mean()), "precision": float(np.mean(precisions)), "support_recall": float(np.mean(recalls)), "query_any_support": float(np.mean(any_support)), "query_all_support": float(np.mean(all_support))}
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/conformal_global_run"))
    parser.add_argument("--alpha", type=float, default=.1)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--min-pixels", type=int, default=3136)
    parser.add_argument("--max-pixels", type=int, default=200704)
    parser.add_argument("--test-plan", type=Path, help="Optional manifest for a larger fixed test set.")
    parser.add_argument("--fusion-predictions", type=Path, help="Optional NPZ from analysis on --test-plan.")
    args = parser.parse_args()

    base = args.feature_root / f"seed_{args.seed}" / args.dataset
    if not base.exists() and args.seed == 1:
        # The original completed seed predates the seed_{N} directory layout.
        base = args.feature_root / args.dataset
    calibration_qids = plan(base / "calibration" / "manifest.json")
    test_qids = plan(args.test_plan) if args.test_plan else plan(base / "test" / "manifest.json")
    qpath, cpath, rpath, bundle = paths(args.dataset, args.data_root)
    questions, corpus, retrieval = keyed(qpath, "qid"), keyed(cpath, "id"), grouped_retrieval(rpath)
    calibration = [retrieval["calibration"][qid] for qid in calibration_qids]
    test = [retrieval["test"][qid] for qid in test_qids]
    masks, thresholds = selector_masks(calibration, test, args.alpha)

    pred_path = args.fusion_predictions or (args.analysis_root / f"seed_{args.seed}" / args.dataset / "fusion_predictions.npz")
    prediction_data = np.load(pred_path)
    if [str(q) for q in prediction_data["qids"].tolist()] != test_qids:
        raise ValueError("fusion prediction qids do not match feature plan")
    fusion = prediction_data["mask_cosine_internal_alpha_0.1"].astype(bool)
    # Verify feature array chunk ordering against frozen retrieval, then reuse mask.
    for expected, rows in zip(prediction_data["chunk_ids"], test):
        if [str(x) for x in expected.tolist()] != [str(row["chunk_id"]) for row in rows]:
            raise ValueError("fusion candidate ordering does not match retrieval log")
    masks["query_level_cosine_lm_hidden"] = list(fusion)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"seed_{args.seed}_{args.dataset}_predictions.jsonl"
    completed = {str(row["qid"]): row for row in iter_jsonl(output)} if output.exists() else {}
    client = QwenDirectAnswerGenerator(args.model, min_pixels=args.min_pixels, max_pixels=args.max_pixels)
    for i, (qid, rows) in enumerate(zip(test_qids, test), 1):
        if qid in completed:
            continue
        question, gold = questions[qid], [str(x) for x in questions[qid]["gold_answers"]]
        prompt = "Answer using only the supplied context. Return only the short answer.\nQuestion: " + str(question["question"])
        predictions, contexts = {}, {}
        # Methods with identical selected chunk IDs share the deterministic decode.
        answer_cache = {}
        for name in METHODS:
            selected = chunks(rows, masks[name][i - 1], corpus, bundle)
            key = tuple(chunk.id for chunk in selected)
            prediction = answer_cache.get(key)
            if prediction is None:
                prediction = client.generate(prompt, selected, max_new_tokens=args.max_new_tokens)
                answer_cache[key] = prediction
            predictions[name] = prediction
            contexts[name] = {"n_chunks": len(selected), "chunk_ids": list(key)}
        record = {"qid": qid, "gold_answers": gold, "predictions": predictions, "metrics": {name: metrics(value, gold) for name, value in predictions.items()}, "contexts": contexts}
        append_jsonl(output, record); completed[qid] = record
        print(f"[{args.dataset} seed={args.seed} {i}/{len(test)}] {qid}", flush=True)
    ordered = [completed[qid] for qid in test_qids]
    qa = {name: {metric: float(np.mean([row["metrics"][name][metric] for row in ordered])) for metric in ("em", "f1", "numerical_accuracy")} for name in METHODS}
    for name in METHODS:
        qa[name].update(selection_metrics(test, {name: masks[name]})[name])
    result = {"status": "complete", "seed": args.seed, "dataset": args.dataset, "alpha": args.alpha, "n_test": len(test_qids), "thresholds": thresholds, "results": qa}
    (args.output_dir / f"seed_{args.seed}_{args.dataset}_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
