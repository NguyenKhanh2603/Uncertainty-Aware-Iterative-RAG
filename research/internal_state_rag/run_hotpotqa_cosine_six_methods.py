"""Compare cosine-only conformal context selectors on HotpotQA.

All selectors operate on one frozen Top-L cosine retrieval log and the same
100-query calibration plan.  The sixth selector is Benjamini--Hochberg (BH)
over candidate-level conformal p-values.  Its optional rank-ordered context
cap is an experimental engineering policy; BH's usual FDR conditions do not
automatically transfer to the capped selection procedure.

The runner is append-only and resumable.  First use ``--selection-only`` to
write the selection report, then run without it for deterministic Qwen
downstream QA.  Re-running either mode resumes existing prediction JSONL.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from eval.metrics import exact_match, numerical_accuracy, token_f1
from research.internal_state_rag.run_downstream_qa_conformal_baselines import (
    QwenDirectAnswerGenerator,
    append_jsonl,
    chunks,
    iter_jsonl,
    keyed,
)
from uncertainty_rag.core.conformal_selection import (
    backfill_rejected_indices,
    benjamini_yekutieli,
)


def benjamini_hochberg(p_values: Sequence[float], alpha: float) -> tuple[int, ...]:
    """Return BH rejections in input order.

    BH controls FDR under independence or suitable positive dependence.  This
    function deliberately does not make the stronger BY dependence claim.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    ordered = sorted(range(len(p_values)), key=lambda index: (p_values[index], index))
    cutoff = 0
    for rank, index in enumerate(ordered, start=1):
        if p_values[index] <= rank * alpha / len(p_values):
            cutoff = rank
    accepted = set(ordered[:cutoff])
    return tuple(index for index in range(len(p_values)) if index in accepted)


def iter_retrieval(path: Path) -> Iterable[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def read_plan(path: Path) -> list[str]:
    return [str(qid) for qid in json.loads(path.read_text(encoding="utf-8"))["plan"]]


def grouped(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    result: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        result[str(row["split_role"])][str(row["qid"])].append(row)
    for by_qid in result.values():
        for items in by_qid.values():
            items.sort(key=lambda item: (int(item["rank"]), str(item["chunk_id"])))
    return result


def false_bank(calibration: Sequence[Sequence[dict[str, Any]]]) -> np.ndarray:
    scores = np.asarray(
        [float(row["cosine_score"]) for rows in calibration for row in rows if row["support_label"] == "false"],
        dtype=float,
    )
    if not len(scores):
        raise ValueError("calibration has no false evidence scores")
    return np.sort(scores)


def p_values(rows: Sequence[dict[str, Any]], bank: np.ndarray) -> list[float]:
    return [
        float((1 + len(bank) - np.searchsorted(bank, float(row["cosine_score"]), side="left")) / (len(bank) + 1))
        for row in rows
    ]


def finite_threshold(scores: np.ndarray, alpha: float) -> float:
    ordered = np.sort(scores)
    return float(ordered[min(len(ordered) - 1, int(math.floor((len(ordered) + 1) * alpha)))])


def mask_from_indices(length: int, indices: Sequence[int]) -> np.ndarray:
    value = np.zeros(length, dtype=bool)
    value[list(indices)] = True
    return value


def bh_mask(rows: Sequence[dict[str, Any]], values: Sequence[float], alpha: float, cap: int) -> np.ndarray:
    rejected = benjamini_hochberg(values, alpha)
    selected, _ = backfill_rejected_indices(
        [int(row["rank"]) for row in rows], [str(row["chunk_id"]) for row in rows], rejected, cap
    )
    return mask_from_indices(len(rows), selected)


def all_masks(
    calibration: list[list[dict[str, Any]]],
    test: list[list[dict[str, Any]]],
    fusion_mask: np.ndarray,
) -> tuple[dict[str, list[np.ndarray]], dict[str, Any]]:
    bank = false_bank(calibration)
    support = np.asarray(
        [float(row["cosine_score"]) for rows in calibration for row in rows if row["support_label"] == "support"],
        dtype=float,
    )
    if not len(support):
        raise ValueError("calibration has no support evidence scores")
    masks: dict[str, list[np.ndarray]] = {
        "fixed_top10": [np.asarray([int(row["rank"]) <= 10 for row in rows]) for rows in test],
        "fixed_top20": [np.asarray([int(row["rank"]) <= 20 for row in rows]) for rows in test],
        "cce_alpha_0.10": [],
        "conflare_alpha_0.10": [],
        "traq_alpha_0.10": [],
        "query_level_cosine_internal_alpha_0.10": list(fusion_mask.astype(bool)),
    }
    thresholds = {
        "cce_alpha_0.10": finite_threshold(support, 0.10),
        "conflare_alpha_0.10": float(np.percentile(support, 10)),
        "traq_alpha_0.10": float(np.quantile(support, 0.05, method="lower")),
        "false_score_bank_size": int(len(bank)),
        "support_score_bank_size": int(len(support)),
    }
    for name, threshold in list(thresholds.items()):
        if name.endswith("0.10") and name != "traq_alpha_0.10":
            masks[name] = [np.asarray([float(row["cosine_score"]) >= threshold for row in rows]) for rows in test]
    masks["traq_alpha_0.10"] = [np.asarray([float(row["cosine_score"]) >= thresholds["traq_alpha_0.10"] for row in rows]) for rows in test]
    for alpha in (0.10, 0.30, 0.50):
        masks[f"by_cosine_alpha_{alpha:.2f}"] = [
            mask_from_indices(len(rows), benjamini_yekutieli(p_values(rows, bank), alpha).rejected_indices)
            for rows in test
        ]
    for alpha in (0.10, 0.30, 0.50, 0.90, 0.99):
        masks[f"bh_cosine_alpha_{alpha:.2f}_ctx10"] = [bh_mask(rows, p_values(rows, bank), alpha, 10) for rows in test]
    masks["bh_cosine_alpha_0.99_ctx20"] = [bh_mask(rows, p_values(rows, bank), 0.99, 20) for rows in test]
    return masks, thresholds


def selection_summary(test: list[list[dict[str, Any]]], masks: dict[str, list[np.ndarray]]) -> dict[str, dict[str, float]]:
    result = {}
    for method, method_masks in masks.items():
        kept = supports = total_supports = empty = 0
        macro_precision = macro_recall = any_support = all_support = 0.0
        for rows, mask in zip(test, method_masks):
            labels = np.asarray([row["support_label"] == "support" for row in rows], dtype=bool)
            selected = labels[mask]
            kept += int(mask.sum()); supports += int(selected.sum()); total_supports += int(labels.sum())
            empty += int(not mask.any())
            macro_precision += float(selected.mean()) if len(selected) else 0.0
            recall = float(selected.sum() / labels.sum()) if labels.sum() else 1.0
            macro_recall += recall; any_support += float(selected.any() if labels.sum() else True)
            all_support += float(selected.sum() == labels.sum())
        q = len(test)
        result[method] = {
            "mean_chunks": kept / q, "empty_rate": empty / q,
            "precision": supports / kept if kept else 0.0,
            "support_recall": supports / total_supports if total_supports else 0.0,
            "macro_precision": macro_precision / q, "macro_support_recall": macro_recall / q,
            "query_any_support": any_support / q, "query_all_support": all_support / q,
            "selected_chunks": kept, "selected_supports": supports, "reserve_supports": total_supports,
        }
    return result


def metric_summary(records: list[dict[str, Any]], methods: Sequence[str]) -> dict[str, dict[str, float]]:
    return {
        method: {
            "em": float(np.mean([row["metrics"][method]["em"] for row in records])),
            "f1": float(np.mean([row["metrics"][method]["f1"] for row in records])),
            "numerical_accuracy": float(np.mean([row["metrics"][method]["numerical_accuracy"] for row in records])),
        }
        for method in methods
    }


def report(output: Path, selection: dict[str, dict[str, float]], downstream: dict[str, dict[str, float]] | None, meta: dict[str, Any]) -> None:
    lines = [
        "# HotpotQA cosine conformal selector comparison", "",
        "Frozen cosine Top-30 candidates; 100 disjoint calibration queries; 1,000 official HotpotQA dev queries selected by seed 42. BH caps are experimental rank-ordered backfill policies, so no capped-procedure FDR guarantee is claimed.", "",
        "| Method | Chunks | Precision | Recall | Empty | Any support | All support | EM | F1 | Numeric |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, value in selection.items():
        qa = (downstream or {}).get(name, {})
        lines.append(
            f"| {name} | {value['mean_chunks']:.2f} | {value['precision']:.1%} | {value['support_recall']:.1%} | {value['empty_rate']:.1%} | {value['query_any_support']:.1%} | {value['query_all_support']:.1%} | "
            + (f"{qa.get('em', 0):.3f} | {qa.get('f1', 0):.3f} | {qa.get('numerical_accuracy', 0):.3f} |" if downstream else "pending | pending | pending |")
        )
    lines += ["", "## Calibration", "", "```json", json.dumps(meta, indent=2), "```", ""]
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--calibration-plan", type=Path, required=True)
    parser.add_argument("--test-plan", type=Path, required=True)
    parser.add_argument("--fusion-predictions", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--selection-only", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--min-pixels", type=int, default=3136)
    parser.add_argument("--max-pixels", type=int, default=200704)
    args = parser.parse_args()

    calibration_qids, test_qids = read_plan(args.calibration_plan), read_plan(args.test_plan)
    retrieval = grouped(iter_retrieval(args.retrieval))
    calibration = [retrieval["calibration"][qid] for qid in calibration_qids]
    test = [retrieval["test"][qid] for qid in test_qids]
    if any(len(rows) != 30 for rows in calibration + test):
        raise ValueError("all planned queries must have exactly frozen Top-30 candidates")
    prediction_data = np.load(args.fusion_predictions)
    if [str(qid) for qid in prediction_data["qids"]] != test_qids:
        raise ValueError("fusion predictions and test plan qid ordering disagree")
    for expected, rows in zip(prediction_data["chunk_ids"], test):
        if [str(x) for x in expected] != [str(row["chunk_id"]) for row in rows]:
            raise ValueError("fusion predictions and retrieval chunk ordering disagree")
    masks, thresholds = all_masks(calibration, test, prediction_data["mask_cosine_internal_alpha_0.1"])
    selection = selection_summary(test, masks)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"calibration_queries": len(calibration), "test_queries": len(test), "thresholds": thresholds, "methods": list(masks), "bh_note": "BH is calculated from the same candidate false-score p-values as BY. BH context caps are experimental."}
    summary_path = args.output_dir / "selection_summary.json"
    summary_path.write_text(json.dumps({"metadata": metadata, "selection": selection}, indent=2) + "\n", encoding="utf-8")
    prediction_path = args.output_dir / "downstream_predictions.jsonl"
    if args.selection_only:
        report(args.output_dir / "REPORT.md", selection, None, metadata)
        return
    if args.model is None:
        raise ValueError("--model is required unless --selection-only is used")
    questions, corpus = keyed(args.questions, "qid"), keyed(args.corpus, "id")
    done = {str(row["qid"]): row for row in iter_jsonl(prediction_path)} if prediction_path.exists() else {}
    generator = QwenDirectAnswerGenerator(args.model, min_pixels=args.min_pixels, max_pixels=args.max_pixels)
    methods = list(masks)
    for number, (qid, rows) in enumerate(zip(test_qids, test), start=1):
        if qid in done:
            continue
        item = questions[qid]; gold = [str(answer) for answer in item["gold_answers"]]
        prompt = "Answer using only the supplied context. Return only the short answer.\nQuestion: " + str(item["question"])
        answer_cache: dict[tuple[str, ...], str] = {}
        predictions, contexts = {}, {}
        for method in methods:
            selected = chunks(rows, masks[method][number - 1], corpus, args.bundle_root)
            key = tuple(chunk.id for chunk in selected)
            answer = answer_cache.get(key)
            if answer is None:
                answer = generator.generate(prompt, selected, max_new_tokens=args.max_new_tokens)
                answer_cache[key] = answer
            predictions[method] = answer
            contexts[method] = {"n_chunks": len(selected), "chunk_ids": list(key)}
        record = {"qid": qid, "gold_answers": gold, "predictions": predictions, "contexts": contexts, "metrics": {method: {"em": exact_match(answer, gold), "f1": token_f1(answer, gold), "numerical_accuracy": numerical_accuracy(answer, gold)} for method, answer in predictions.items()}}
        append_jsonl(prediction_path, record); done[qid] = record
        print(f"[hotpotqa {number}/{len(test_qids)}] {qid}", flush=True)
    ordered = [done[qid] for qid in test_qids]
    downstream = metric_summary(ordered, methods)
    summary = {"status": "complete", "metadata": metadata, "selection": selection, "downstream": downstream}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report(args.output_dir / "REPORT.md", selection, downstream, metadata)


if __name__ == "__main__":
    main()
