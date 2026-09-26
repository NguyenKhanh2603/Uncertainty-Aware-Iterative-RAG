"""Run shared Qwen QA diagnostics for alpha-free query-level cosine selectors.

The alpha=.10 conformal reference is deliberately *reused* from the completed
literature-protocol downstream run after exact qid validation.  This avoids
spending another GPU pass on an identical selected context.  This program only
generates the two new alpha-free contexts and reports them beside that frozen
reference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from eval.metrics import exact_match, numerical_accuracy, token_f1
from research.internal_state_rag.run_all_datasets_cosine_six_methods import (
    CONFIGS,
    load_dataset,
    query_level_cosine_mask,
    validate_downstream_inputs,
)
from research.internal_state_rag.run_alpha_free_query_level_1000cal import mask_from_threshold, score_arrays
from research.internal_state_rag.run_downstream_qa_conformal_baselines import (
    QwenDirectAnswerGenerator,
    append_jsonl,
    chunks,
    iter_jsonl,
)


DATASETS = ("hotpotqa", "mmqa", "tatqa", "webqa")
REFERENCE_KEY = "query_level_all_support_cosine_alpha_0.10"
METHODS = (
    "query_level_conformal_alpha_0.10_reference",
    "query_level_f1_calibrated",
    "query_level_budget10_all_support",
)
NEW_METHODS = METHODS[1:]
DISPLAY = {
    "query_level_conformal_alpha_0.10_reference": "Query-level conformal cosine (α=.10 reference)",
    "query_level_f1_calibrated": "Query-level cosine F1-selected (no α at test)",
    "query_level_budget10_all_support": "Query-level cosine budget-10 all-support (no α at test)",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def mean_metrics(records: Sequence[dict[str, Any]], methods: Sequence[str]) -> dict[str, dict[str, float]]:
    return {
        method: {
            metric: float(np.mean([record["metrics"][method][metric] for record in records]))
            for metric in ("em", "f1", "numerical_accuracy")
        }
        for method in methods
    }


def source_reference(
    source: Path, dataset: str, expected_qids: Sequence[str]
) -> dict[str, float]:
    """Load the identical alpha=.10 Qwen reference and validate qid identity."""

    summary = read_json(source / f"{dataset}_summary.json")
    if summary.get("status") != "complete":
        raise RuntimeError(f"{dataset}: source reference is not complete")
    prediction_path = source / f"{dataset}_downstream_predictions.jsonl"
    records = list(iter_jsonl(prediction_path))
    observed = [str(record["qid"]) for record in records]
    if observed != list(expected_qids):
        raise RuntimeError(f"{dataset}: source reference qids do not match the frozen test manifest")
    for record in records:
        if REFERENCE_KEY not in record.get("metrics", {}):
            raise RuntimeError(f"{dataset}: source reference lacks {REFERENCE_KEY}")
    return dict(summary["downstream"][REFERENCE_KEY])


def masks_for_dataset(
    calibration: list[list[dict[str, Any]]],
    test: list[list[dict[str, Any]]],
    result: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Reconstruct thresholds frozen in the committed alpha-free selection artifact."""

    test_scores, _labels = score_arrays(test)
    conformal, _audit = query_level_cosine_mask(calibration, test, 0.10)
    masks = {"query_level_conformal_alpha_0.10_reference": np.asarray(conformal, dtype=bool)}
    for method in NEW_METHODS:
        threshold = float(result["methods"][method]["threshold"])
        masks[method] = mask_from_threshold(test_scores, threshold)
    expected_shape = (len(test), len(test[0]))
    if any(value.shape != expected_shape for value in masks.values()):
        raise RuntimeError("Reconstructed mask shape does not match frozen candidate order")
    return masks


def write_report(output: Path, state: dict[str, dict[str, Any]], source: Path) -> None:
    lines = [
        "# Downstream diagnostic: alpha-free query-level cosine",
        "",
        "The new rows use a shared deterministic Qwen direct-answer diagnostic over the exact 100 held-out qids. The α=.10 reference is reused from the completed literature-protocol pass only after exact qid validation; its selected context is identical. New predictions are generated only for the two alpha-free policies.",
        "",
        f"- Reference predictions: `{source}`",
        "- Selection details and frozen thresholds: [`REPORT.md`](REPORT.md)",
        "- This is a task metric, not a conformal coverage guarantee.",
        "",
    ]
    for dataset in DATASETS:
        item = state[dataset]
        lines.extend(
            [
                f"## {dataset} ({item['status']}; calibration={item['calibration_queries']}, test={item['test_queries']})",
                "",
                "| Method | Chunks | Evidence F1 | EM | Token F1 | Numeric |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for method in METHODS:
            selection = item["selection"][method]
            qa = item.get("downstream", {}).get(method)
            qa_cells = (
                f"{qa['em']:.3f} | {qa['f1']:.3f} | {qa['numerical_accuracy']:.3f}"
                if qa
                else "pending | pending | pending"
            )
            lines.append(
                f"| {DISPLAY[method]} | {selection['mean_chunks']:.2f} | "
                f"{selection['evidence_f1']:.3f} | {qa_cells} |"
            )
        lines.append("")
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection-dir",
        type=Path,
        default=Path(
            "research/internal_state_rag/results/"
            "alpha_free_query_level_1000cal_100test_2026_09_26"
        ),
    )
    parser.add_argument(
        "--plan-root",
        type=Path,
        default=Path(
            "research/internal_state_rag/results/"
            "all_datasets_cosine_six_methods_1000cal_100test_2026_09_26/splits"
        ),
    )
    parser.add_argument(
        "--source-downstream-dir",
        type=Path,
        default=Path(
            "research/internal_state_rag/results/"
            "literature_protocol_1000cal_100test_2026_09_26"
        ),
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--min-pixels", type=int, default=3136)
    parser.add_argument("--max-pixels", type=int, default=200704)
    args = parser.parse_args()
    datasets = tuple(value.strip() for value in args.datasets.split(",") if value.strip())
    if not datasets or any(dataset not in DATASETS for dataset in datasets):
        raise ValueError(f"datasets must be a non-empty subset of {DATASETS}")

    selection_state = read_json(args.selection_dir / "selection_summary.json")
    state: dict[str, dict[str, Any]] = {}
    pending: list[tuple[str, Any, list[str], list[list[dict[str, Any]]], dict[str, np.ndarray], dict[str, Any], dict[str, float]]] = []
    for dataset in datasets:
        calibration_plan = args.plan_root / dataset / "calibration_manifest.json"
        test_plan = args.plan_root / dataset / "test_manifest.json"
        calibration_qids, calibration, test_qids, test = load_dataset(
            CONFIGS[dataset], calibration_plan=calibration_plan, test_plan=test_plan
        )
        result = selection_state[dataset]
        if len(calibration_qids) != 1000 or len(test_qids) != 100:
            raise RuntimeError(f"{dataset}: expected 1,000 calibration / 100 test qids")
        if result["calibration_queries"] != len(calibration_qids) or result["test_queries"] != len(test_qids):
            raise RuntimeError(f"{dataset}: selection artifact split counts disagree")
        masks = masks_for_dataset(calibration, test, result)
        reference = source_reference(args.source_downstream_dir, dataset, test_qids)
        state[dataset] = {
            "status": "running",
            "calibration_queries": len(calibration_qids),
            "test_queries": len(test_qids),
            "selection": {
                method: result["methods"][method]["test_selection"] for method in METHODS
            },
            "downstream": {"query_level_conformal_alpha_0.10_reference": reference},
        }
        pending.append((dataset, CONFIGS[dataset], test_qids, test, masks, result, reference))

    write_report(args.selection_dir / "DOWNSTREAM_REPORT.md", state, args.source_downstream_dir)
    resolved = {
        config.name: validate_downstream_inputs(config, test_qids, test)
        for _dataset, config, test_qids, test, _masks, _result, _reference in pending
    }
    generator = QwenDirectAnswerGenerator(
        args.model, min_pixels=args.min_pixels, max_pixels=args.max_pixels
    )
    for dataset, config, test_qids, test, masks, _result, reference in pending:
        questions, corpus = resolved[config.name]
        prediction_path = args.selection_dir / f"{dataset}_downstream_predictions.jsonl"
        completed = {str(record["qid"]): record for record in iter_jsonl(prediction_path)} if prediction_path.exists() else {}
        unexpected = set(completed).difference(test_qids)
        if unexpected:
            raise RuntimeError(f"{dataset}: prediction file contains qids outside frozen test plan")
        for index, (qid, rows) in enumerate(zip(test_qids, test, strict=True), start=1):
            if qid in completed:
                continue
            item = questions[qid]
            gold = [str(answer) for answer in item["gold_answers"]]
            prompt = (
                "Answer using only the supplied context. Return only the short answer.\nQuestion: "
                + str(item["question"])
            )
            cache: dict[tuple[str, ...], str] = {}
            predictions: dict[str, str] = {}
            contexts: dict[str, dict[str, Any]] = {}
            for method in NEW_METHODS:
                selected = chunks(rows, masks[method][index - 1], corpus, config.bundle_root)
                key = tuple(chunk.id for chunk in selected)
                answer = cache.get(key)
                if answer is None:
                    answer = generator.generate(prompt, selected, max_new_tokens=args.max_new_tokens)
                    cache[key] = answer
                predictions[method] = answer
                contexts[method] = {"n_chunks": len(selected), "chunk_ids": list(key)}
            record = {
                "qid": qid,
                "gold_answers": gold,
                "predictions": predictions,
                "contexts": contexts,
                "metrics": {
                    method: {
                        "em": exact_match(answer, gold),
                        "f1": token_f1(answer, gold),
                        "numerical_accuracy": numerical_accuracy(answer, gold),
                    }
                    for method, answer in predictions.items()
                },
            }
            append_jsonl(prediction_path, record)
            completed[qid] = record
            print(f"[{dataset} {index}/{len(test_qids)}] {qid}", flush=True)
        records = [completed[qid] for qid in test_qids]
        state[dataset]["status"] = "complete"
        state[dataset]["downstream"].update(mean_metrics(records, NEW_METHODS))
        state[dataset]["downstream"]["query_level_conformal_alpha_0.10_reference"] = reference
        (args.selection_dir / f"{dataset}_alpha_free_downstream_summary.json").write_text(
            json.dumps(state[dataset], indent=2) + "\n", encoding="utf-8"
        )
        write_report(args.selection_dir / "DOWNSTREAM_REPORT.md", state, args.source_downstream_dir)
        (args.selection_dir / "downstream_summary.json").write_text(
            json.dumps({"status": "running", "datasets": state}, indent=2) + "\n",
            encoding="utf-8",
        )
    (args.selection_dir / "downstream_summary.json").write_text(
        json.dumps({"status": "complete", "datasets": state}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
