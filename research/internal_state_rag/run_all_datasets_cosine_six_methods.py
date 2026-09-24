"""Evaluate cosine conformal context selectors on all four QA datasets.

Every selector for a dataset receives its frozen Top-30 cosine log and the
same disjoint 100-query calibration plan.  The six selector families are CCE,
CONFLARE, TRAQ retrieval, candidate-level BY, query-level cosine conformal,
and candidate-level BH.  BH's rank cap is explicitly experimental.

The downstream prediction files are append-only.  Re-running the command
resumes the first query missing from each dataset, then refreshes REPORT.md.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from eval.metrics import exact_match, numerical_accuracy, token_f1
from research.internal_state_rag.run_downstream_qa_conformal_baselines import (
    QwenDirectAnswerGenerator,
    append_jsonl,
    chunks,
    iter_jsonl,
    keyed,
)
from research.internal_state_rag.run_hotpotqa_cosine_six_methods import (
    benjamini_hochberg,
    bh_mask,
    false_bank,
    finite_threshold,
    grouped,
    iter_retrieval,
    mask_from_indices,
    p_values,
    read_plan,
    selection_summary,
)
from uncertainty_rag.core.conformal_selection import benjamini_yekutieli


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    retrieval: Path
    calibration_plan: Path
    test_plan: Path | None
    questions: Path
    corpus: Path
    bundle_root: Path


ROOT = Path("data/conformal_global_run")
FEATURES = Path("research/internal_state_rag/results/qwen2vl_7b_jina4/fusion_features")
FULL_PLANS = Path("research/internal_state_rag/results/qwen2vl_7b_jina4/full_test_1000_plans")
JINA_LOGS = Path("research/internal_state_rag/results/qwen2vl_jina4")
CONFIGS = {
    "hotpotqa": DatasetConfig(
        "hotpotqa",
        Path("research/internal_state_rag/results/official_seed42_hotpotqa_cosine_top30.jsonl.gz"),
        Path("research/internal_state_rag/results/official_seed42_hotpotqa_plans/calibration/manifest.json"),
        Path("research/internal_state_rag/results/official_seed42_hotpotqa_plans/test/manifest.json"),
        Path("data/official_seed42_hotpotqa_fixed/hotpotqa/questions.jsonl"),
        Path("data/official_seed42_hotpotqa_fixed/hotpotqa/corpus.jsonl"),
        Path("data/official_seed42_hotpotqa_fixed"),
    ),
    "mmqa": DatasetConfig(
        "mmqa", JINA_LOGS / "mmqa_jina_v4_top30.jsonl.gz",
        FEATURES / "mmqa/calibration/manifest.json", FULL_PLANS / "mmqa/manifest.json",
        ROOT / "official_bundle_role_split_mmqa/mmqa/questions.jsonl",
        ROOT / "official_bundle_role_split_mmqa/mmqa/corpus.jsonl",
        ROOT / "official_bundle_role_split_mmqa",
    ),
    "tatqa": DatasetConfig(
        "tatqa", JINA_LOGS / "tatqa_jina_v4_top30.jsonl.gz",
        FEATURES / "tatqa/calibration/manifest.json", FULL_PLANS / "tatqa/manifest.json",
        ROOT / "official_bundle_role_split_tatqa/tatqa/questions.jsonl",
        ROOT / "official_bundle_role_split_tatqa/tatqa/corpus.jsonl",
        ROOT / "official_bundle_role_split_tatqa",
    ),
    "webqa": DatasetConfig(
        "webqa", JINA_LOGS / "webqa_jina_v4_top30.jsonl.gz",
        FEATURES / "webqa/calibration/manifest.json", None,
        ROOT / "official_bundle_1000_webqa/webqa/questions.jsonl",
        ROOT / "official_bundle_1000_webqa/webqa/corpus.jsonl",
        ROOT / "official_bundle_1000_webqa",
    ),
}


def query_z(scores: np.ndarray) -> np.ndarray:
    mean = scores.mean(axis=1, keepdims=True)
    return (scores - mean) / (scores.std(axis=1, keepdims=True) + 1e-8)


def query_level_cosine_mask(
    calibration: list[list[dict[str, Any]]], test: list[list[dict[str, Any]]], alpha: float
) -> tuple[list[np.ndarray], dict[str, float | int]]:
    cal_scores = np.asarray([[float(row["cosine_score"]) for row in rows] for rows in calibration])
    cal_labels = np.asarray([[row["support_label"] == "support" for row in rows] for rows in calibration])
    retrievable = cal_labels.any(axis=1)
    if not retrievable.any():
        raise ValueError("Query-level calibration has no retrievable queries")
    critical = np.asarray([
        row[label].min() for row, label in zip(query_z(cal_scores)[retrievable], cal_labels[retrievable], strict=True)
    ])
    order = min(len(critical), math.ceil((len(critical) + 1) * (1 - alpha)))
    threshold = float(np.sort(-critical)[order - 1] * -1)
    test_scores = query_z(np.asarray([[float(row["cosine_score"]) for row in rows] for rows in test]))
    masks = test_scores >= threshold
    empty = ~masks.any(axis=1)
    if empty.any():
        masks[np.flatnonzero(empty), np.argmax(test_scores[empty], axis=1)] = True
    return list(masks), {
        "threshold": threshold,
        "finite_sample_order": order,
        "retrievable_calibration_queries": int(retrievable.sum()),
        "top1_fallback_queries": int(empty.sum()),
    }


def all_masks(
    calibration: list[list[dict[str, Any]]], test: list[list[dict[str, Any]]]
) -> tuple[dict[str, list[np.ndarray]], dict[str, Any]]:
    bank = false_bank(calibration)
    support = np.asarray([
        float(row["cosine_score"]) for rows in calibration for row in rows
        if row["support_label"] == "support"
    ])
    if not len(support):
        raise ValueError("Calibration has no support evidence")
    thresholds: dict[str, Any] = {
        "cce_alpha_0.10": finite_threshold(support, 0.10),
        "conflare_alpha_0.10": float(np.percentile(support, 10)),
        "traq_alpha_0.10": float(np.quantile(support, 0.05, method="lower")),
        "false_score_bank_size": int(len(bank)), "support_score_bank_size": int(len(support)),
    }
    masks: dict[str, list[np.ndarray]] = {
        "fixed_top10": [np.asarray([int(row["rank"]) <= 10 for row in rows]) for rows in test],
        "fixed_top20": [np.asarray([int(row["rank"]) <= 20 for row in rows]) for rows in test],
        "cce_alpha_0.10": [np.asarray([float(row["cosine_score"]) >= thresholds["cce_alpha_0.10"] for row in rows]) for rows in test],
        "conflare_alpha_0.10": [np.asarray([float(row["cosine_score"]) >= thresholds["conflare_alpha_0.10"] for row in rows]) for rows in test],
        "traq_alpha_0.10": [np.asarray([float(row["cosine_score"]) >= thresholds["traq_alpha_0.10"] for row in rows]) for rows in test],
    }
    query_masks, query_details = query_level_cosine_mask(calibration, test, 0.10)
    masks["query_level_cosine_alpha_0.10"] = query_masks
    thresholds["query_level_cosine_alpha_0.10"] = query_details
    for alpha in (0.10, 0.30, 0.50):
        masks[f"by_cosine_alpha_{alpha:.2f}"] = [
            mask_from_indices(len(rows), benjamini_yekutieli(p_values(rows, bank), alpha).rejected_indices)
            for rows in test
        ]
    for alpha in (0.10, 0.30, 0.50, 0.90, 0.99):
        masks[f"bh_cosine_alpha_{alpha:.2f}_ctx10"] = [
            bh_mask(rows, p_values(rows, bank), alpha, 10) for rows in test
        ]
    masks["bh_cosine_alpha_0.99_ctx20"] = [
        bh_mask(rows, p_values(rows, bank), 0.99, 20) for rows in test
    ]
    return masks, thresholds


def metric_summary(records: Sequence[dict[str, Any]], methods: Sequence[str]) -> dict[str, dict[str, float]]:
    return {
        method: {
            metric: float(np.mean([row["metrics"][method][metric] for row in records]))
            for metric in ("em", "f1", "numerical_accuracy")
        }
        for method in methods
    }


def make_report(output: Path, results: dict[str, dict[str, Any]], meta: dict[str, Any]) -> None:
    lines = [
        "# Four-dataset cosine conformal selector comparison", "",
        "Every row within a dataset uses the same frozen Top-30 cosine candidates and its disjoint 100-query calibration plan. Query-level cosine is the pure per-query z-scored cosine threshold with a deterministic Top-1 empty-context fallback. BH context caps are experimental rank-ordered policies and have no claimed capped-procedure FDR guarantee.",
        "",
    ]
    for dataset, result in results.items():
        state = result["status"]
        lines.extend([f"## {dataset} ({state}; n={result['test_queries']})", "", "| Method | Chunks | Precision | Recall | Empty | Any support | All support | EM | F1 | Numeric |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
        for method, values in result["selection"].items():
            qa = result.get("downstream", {}).get(method)
            tail = (f"{qa['em']:.3f} | {qa['f1']:.3f} | {qa['numerical_accuracy']:.3f} |" if qa else "pending | pending | pending |")
            lines.append(f"| {method} | {values['mean_chunks']:.2f} | {values['precision']:.1%} | {values['support_recall']:.1%} | {values['empty_rate']:.1%} | {values['query_any_support']:.1%} | {values['query_all_support']:.1%} | {tail}")
        lines.append("")
    lines += ["## Protocol", "", "```json", json.dumps(meta, indent=2), "```", ""]
    output.write_text("\n".join(lines), encoding="utf-8")


def load_dataset(config: DatasetConfig) -> tuple[list[str], list[list[dict[str, Any]]], list[str], list[list[dict[str, Any]]]]:
    retrieval = grouped(iter_retrieval(config.retrieval))
    calibration_qids = read_plan(config.calibration_plan)
    test_qids = read_plan(config.test_plan) if config.test_plan else sorted(retrieval["test"])
    calibration = [retrieval["calibration"][qid] for qid in calibration_qids]
    test = [retrieval["test"][qid] for qid in test_qids]
    if any(len(rows) != 30 for rows in calibration + test):
        raise ValueError(f"{config.name}: expected exactly 30 candidates per planned query")
    return calibration_qids, calibration, test_qids, test


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", default="hotpotqa,mmqa,tatqa,webqa")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--selection-only", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--min-pixels", type=int, default=3136)
    parser.add_argument("--max-pixels", type=int, default=200704)
    args = parser.parse_args()
    names = [name.strip() for name in args.datasets.split(",") if name.strip()]
    if not names or any(name not in CONFIGS for name in names):
        raise ValueError(f"datasets must be a non-empty subset of {tuple(CONFIGS)}")
    if not args.selection_only and args.model is None:
        raise ValueError("--model is required unless --selection-only is given")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state: dict[str, dict[str, Any]] = {}
    pending: list[tuple[DatasetConfig, list[str], list[list[dict[str, Any]]], dict[str, list[np.ndarray]]]] = []
    for name in names:
        config = CONFIGS[name]
        calibration_qids, calibration, test_qids, test = load_dataset(config)
        masks, thresholds = all_masks(calibration, test)
        state[name] = {
            "status": "selection_complete" if args.selection_only else "running",
            "calibration_queries": len(calibration_qids), "test_queries": len(test_qids),
            "thresholds": thresholds, "selection": selection_summary(test, masks),
        }
        pending.append((config, test_qids, test, masks))
    meta = {"selector_families": ["CCE", "CONFLARE", "TRAQ retrieval", "BY cosine", "query-level cosine", "BH cosine"], "top_l": 30, "calibration_queries_per_dataset": 100, "BH_note": "BH normally needs independence or suitable positive dependence; its rank cap is experimental."}
    make_report(args.output_dir / "REPORT.md", state, meta)
    (args.output_dir / "selection_summary.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    if args.selection_only:
        return
    generator = QwenDirectAnswerGenerator(args.model, min_pixels=args.min_pixels, max_pixels=args.max_pixels)
    for config, test_qids, test, masks in pending:
        questions, corpus = keyed(config.questions, "qid"), keyed(config.corpus, "id")
        prediction_path = args.output_dir / f"{config.name}_downstream_predictions.jsonl"
        done = {str(row["qid"]): row for row in iter_jsonl(prediction_path)} if prediction_path.exists() else {}
        methods = list(masks)
        for number, (qid, rows) in enumerate(zip(test_qids, test, strict=True), start=1):
            if qid in done:
                continue
            item = questions[qid]
            gold = [str(answer) for answer in item["gold_answers"]]
            prompt = "Answer using only the supplied context. Return only the short answer.\nQuestion: " + str(item["question"])
            answer_cache: dict[tuple[str, ...], str] = {}
            predictions, contexts = {}, {}
            for method in methods:
                selected = chunks(rows, masks[method][number - 1], corpus, config.bundle_root)
                key = tuple(chunk.id for chunk in selected)
                answer = answer_cache.get(key)
                if answer is None:
                    answer = generator.generate(prompt, selected, max_new_tokens=args.max_new_tokens)
                    answer_cache[key] = answer
                predictions[method] = answer
                contexts[method] = {"n_chunks": len(selected), "chunk_ids": list(key)}
            record = {"qid": qid, "gold_answers": gold, "predictions": predictions, "contexts": contexts, "metrics": {method: {"em": exact_match(answer, gold), "f1": token_f1(answer, gold), "numerical_accuracy": numerical_accuracy(answer, gold)} for method, answer in predictions.items()}}
            append_jsonl(prediction_path, record)
            done[qid] = record
            print(f"[{config.name} {number}/{len(test_qids)}] {qid}", flush=True)
        records = [done[qid] for qid in test_qids]
        state[config.name]["status"] = "complete"
        state[config.name]["downstream"] = metric_summary(records, methods)
        (args.output_dir / f"{config.name}_summary.json").write_text(json.dumps(state[config.name], indent=2) + "\n", encoding="utf-8")
        make_report(args.output_dir / "REPORT.md", state, meta)
        (args.output_dir / "summary.json").write_text(json.dumps({"status": "running", "datasets": state, "protocol": meta}, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "summary.json").write_text(json.dumps({"status": "complete", "datasets": state, "protocol": meta}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
