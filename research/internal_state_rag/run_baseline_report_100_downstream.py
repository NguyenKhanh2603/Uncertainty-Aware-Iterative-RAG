"""Run end-to-end QA for the nine selectors in baseline_comparison_report_100_queries_detailed.

The source report did not retain a separate test-qid manifest. Its published
Fixed Top-10 values identify the deterministic source selection: the first 100
lexicographically sorted qids having ``split_role == 'test'`` in each frozen
Jina-v4 Top-30 log. This runner materializes that recovered manifest, validates
its Fixed Top-10 row against the report, and evaluates every reported context
selector with greedy Qwen2-VL-7B answer generation.

Calibration uses the exact committed 100-qid manifests. BH p-values use the
calibration false-score bank matched by modality, matching the report's stated
``--conditioning dataset,modality`` procedure. The post-BH cap keeps the
smallest p-values (rank breaks ties), as in the report's simulate_bh.py.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import defaultdict
from dataclasses import dataclass
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
from research.internal_state_rag.run_hotpotqa_cosine_six_methods import (
    finite_threshold,
    selection_summary,
)


DATASETS = ("hotpotqa", "mmqa", "tatqa", "webqa")
SPLITS = Path("research/internal_state_rag/results/all_datasets_cosine_six_methods_2026_09_24/splits")
LOGS = Path("research/internal_state_rag/results/qwen2vl_jina4")

# The selection metrics published in baseline_comparison_report_100_queries_detailed.md.
# They provide a guard that the recovered test plan has not drifted.
PUBLISHED_FIXED_TOP10 = {
    "hotpotqa": {"mean_chunks": 10.00, "precision": 0.177, "support_recall": 0.962},
    "mmqa": {"mean_chunks": 10.00, "precision": 0.109, "support_recall": 0.965},
    "tatqa": {"mean_chunks": 10.00, "precision": 0.089, "support_recall": 0.848},
    "webqa": {"mean_chunks": 10.00, "precision": 0.058, "support_recall": 0.682},
}


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    questions: Path
    corpus: Path
    bundle_root: Path


CONFIGS = {
    "hotpotqa": DatasetConfig(
        "hotpotqa",
        Path("data/official_seed42_hotpotqa_fixed/hotpotqa/questions.jsonl"),
        Path("data/official_seed42_hotpotqa_fixed/hotpotqa/corpus.jsonl"),
        Path("data/official_seed42_hotpotqa_fixed"),
    ),
    "mmqa": DatasetConfig(
        "mmqa",
        Path("data/conformal_global_run/official_bundle_role_split_mmqa/mmqa/questions.jsonl"),
        Path("data/conformal_global_run/official_bundle_role_split_mmqa/mmqa/corpus.jsonl"),
        Path("data/conformal_global_run/official_bundle_role_split_mmqa"),
    ),
    "tatqa": DatasetConfig(
        "tatqa",
        Path("data/conformal_global_run/official_bundle_role_split_tatqa/tatqa/questions.jsonl"),
        Path("data/conformal_global_run/official_bundle_role_split_tatqa/tatqa/corpus.jsonl"),
        Path("data/conformal_global_run/official_bundle_role_split_tatqa"),
    ),
    "webqa": DatasetConfig(
        "webqa",
        Path("data/conformal_global_run/official_bundle_1000_webqa/webqa/questions.jsonl"),
        Path("data/conformal_global_run/official_bundle_1000_webqa/webqa/corpus.jsonl"),
        Path("data/conformal_global_run/official_bundle_1000_webqa"),
    ),
}

METHODS = (
    "fixed_top10",
    "fixed_top20",
    "cce_cosine_alpha_0.10",
    "conflare_cosine_alpha_0.10",
    "traq_cosine_alpha_0.10",
    "bh_modality_cosine_alpha_0.10_ctx10",
    "bh_modality_cosine_alpha_0.90_ctx10",
    "bh_modality_cosine_alpha_0.99_ctx10",
    "bh_modality_cosine_alpha_0.99_ctx20",
)


def iter_retrieval(path: Path) -> Iterable[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def grouped_retrieval(
    path: Path,
) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], dict[str, list[dict[str, Any]]]]:
    result: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    all_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in iter_retrieval(path):
        result[str(row["split_role"])][str(row["qid"])].append(row)
        all_rows[str(row["qid"])].append(row)
    for by_qid in result.values():
        for rows in by_qid.values():
            rows.sort(key=lambda row: (int(row["rank"]), str(row["chunk_id"])))
    for rows in all_rows.values():
        rows.sort(key=lambda row: (int(row["rank"]), str(row["chunk_id"])))
    return result, dict(all_rows)


def read_plan(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    plan = [str(qid) for qid in payload["plan"]]
    if len(plan) != len(set(plan)):
        raise ValueError(f"Duplicate qids in {path}")
    return plan


def false_banks_by_modality(calibration: Sequence[Sequence[dict[str, Any]]]) -> dict[str, np.ndarray]:
    values: dict[str, list[float]] = defaultdict(list)
    for rows in calibration:
        for row in rows:
            if row["support_label"] != "support":
                values[str(row["modality"])].append(float(row["cosine_score"]))
    banks = {modality: np.sort(np.asarray(scores, dtype=float)) for modality, scores in values.items()}
    if not banks or any(not len(scores) for scores in banks.values()):
        raise ValueError("Every reported modality needs a non-empty calibration false-score bank")
    return banks


def conformal_p_value(score: float, false_scores: np.ndarray) -> float:
    return float(
        (1 + len(false_scores) - np.searchsorted(false_scores, score, side="left"))
        / (len(false_scores) + 1)
    )


def bh_rejections(p_values: Sequence[float], alpha: float) -> tuple[int, ...]:
    """Return the BH rejection prefix in increasing p-value/rank order."""

    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    order = sorted(range(len(p_values)), key=lambda index: (p_values[index], index))
    cutoff = 0
    for rank, index in enumerate(order, start=1):
        if p_values[index] <= alpha * rank / len(order):
            cutoff = rank
    return tuple(order[:cutoff])


def bh_modality_mask(rows: Sequence[dict[str, Any]], banks: dict[str, np.ndarray], alpha: float, cap: int) -> np.ndarray:
    p_values = [
        conformal_p_value(float(row["cosine_score"]), banks[str(row["modality"])])
        for row in rows
    ]
    selected = bh_rejections(p_values, alpha)[:cap]
    mask = np.zeros(len(rows), dtype=bool)
    mask[list(selected)] = True
    return mask


def selector_masks(calibration: list[list[dict[str, Any]]], test: list[list[dict[str, Any]]]) -> tuple[dict[str, list[np.ndarray]], dict[str, Any]]:
    support = np.asarray(
        [float(row["cosine_score"]) for rows in calibration for row in rows if row["support_label"] == "support"],
        dtype=float,
    )
    if not len(support):
        raise ValueError("Calibration contains no support evidence")
    thresholds = {
        "cce_cosine_alpha_0.10": finite_threshold(support, 0.10),
        "conflare_cosine_alpha_0.10": float(np.percentile(support, 10)),
        "traq_cosine_alpha_0.10": float(np.quantile(support, 0.05, method="lower")),
    }
    banks = false_banks_by_modality(calibration)
    masks: dict[str, list[np.ndarray]] = {
        "fixed_top10": [np.asarray([int(row["rank"]) <= 10 for row in rows]) for rows in test],
        "fixed_top20": [np.asarray([int(row["rank"]) <= 20 for row in rows]) for rows in test],
    }
    for name, threshold in thresholds.items():
        masks[name] = [
            np.asarray([float(row["cosine_score"]) >= threshold for row in rows]) for rows in test
        ]
    for alpha, cap in ((0.10, 10), (0.90, 10), (0.99, 10), (0.99, 20)):
        masks[f"bh_modality_cosine_alpha_{alpha:.2f}_ctx{cap}"] = [
            bh_modality_mask(rows, banks, alpha, cap) for rows in test
        ]
    return masks, {
        "support_score_bank_size": int(len(support)),
        "thresholds": thresholds,
        "false_score_bank_sizes_by_modality": {key: int(len(value)) for key, value in sorted(banks.items())},
        "bh": {
            "conditioning": ["dataset", "modality"],
            "procedure": "BH p-value step-up, then smallest-p-value context cap; rank breaks ties",
            "note": "The post-BH cap is an experimental context policy; no capped-procedure FDR guarantee is claimed.",
        },
    }


def metric_summary(records: Sequence[dict[str, Any]], methods: Sequence[str]) -> dict[str, dict[str, float]]:
    return {
        method: {
            "em": float(np.mean([row["metrics"][method]["em"] for row in records])),
            "f1": float(np.mean([row["metrics"][method]["f1"] for row in records])),
            "numerical_accuracy": float(np.mean([row["metrics"][method]["numerical_accuracy"] for row in records])),
        }
        for method in methods
    }


def materialize_split(output_dir: Path, dataset: str, calibration_qids: list[str], test_qids: list[str]) -> None:
    split_dir = output_dir / "splits" / dataset
    split_dir.mkdir(parents=True, exist_ok=True)
    documents = {
        "calibration_manifest.json": {
            "dataset": dataset,
            "role": "calibration",
            "n_queries": len(calibration_qids),
            "source_manifest": str(SPLITS / dataset / "calibration_manifest.json"),
            "plan": calibration_qids,
        },
        "test_manifest.json": {
            "dataset": dataset,
            "role": "test",
            "n_queries": len(test_qids),
            "source": "first lexicographically sorted split_role=test qids in frozen Jina-v4 Top-30 log",
            "source_retrieval": str(LOGS / f"{dataset}_jina_v4_top30.jsonl.gz"),
            "plan": test_qids,
        },
    }
    for name, content in documents.items():
        (split_dir / name).write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")


def validate_published_fixed_top10(dataset: str, summary: dict[str, dict[str, float]]) -> None:
    expected = PUBLISHED_FIXED_TOP10[dataset]
    observed = summary["fixed_top10"]
    # Chunks are exact. Percentages in the source report are rounded to 0.1pp.
    if not math.isclose(observed["mean_chunks"], expected["mean_chunks"], abs_tol=1e-12):
        raise ValueError(f"{dataset}: recovered test plan has wrong Top-10 context size")
    for key in ("precision", "support_recall"):
        if round(observed[key] * 100, 1) != round(expected[key] * 100, 1):
            raise ValueError(
                f"{dataset}: recovered test plan does not reproduce published Top-10 {key}: "
                f"{observed[key]:.4f} != {expected[key]:.4f}"
            )


def write_report(path: Path, results: dict[str, dict[str, Any]]) -> None:
    lines = [
        "# Downstream QA for the 100-query baseline-comparison report", "",
        "This rerun reconstructs the report's test plan from the frozen Jina-v4 Top-30 logs and verifies the published Fixed Top-10 row before Qwen generation. Every row uses greedy Qwen2-VL-7B-Instruct output with at most 24 new tokens. CCE, CONFLARE, and TRAQ are retrieval adapters; BH is the report's modality-conditioned selection component.",
        "",
    ]
    for dataset, result in results.items():
        lines.extend([
            f"## {dataset} ({result['status']}; n={result['test_queries']})", "",
            "| Method | Chunks | Precision | Recall | Empty | Any support | All support | EM | F1 | Numeric |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for method, values in result["selection"].items():
            qa = result.get("downstream", {}).get(method)
            tail = (
                f"{qa['em']:.3f} | {qa['f1']:.3f} | {qa['numerical_accuracy']:.3f} |"
                if qa else "pending | pending | pending |"
            )
            lines.append(
                f"| {method} | {values['mean_chunks']:.2f} | {values['precision']:.1%} | "
                f"{values['support_recall']:.1%} | {values['empty_rate']:.1%} | "
                f"{values['query_any_support']:.1%} | {values['query_all_support']:.1%} | {tail}"
            )
        lines.extend(["", "### Calibration and selection metadata", "", "```json", json.dumps(result["metadata"], indent=2), "```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def load_dataset(dataset: str, test_queries: int) -> tuple[list[str], list[list[dict[str, Any]]], list[str], list[list[dict[str, Any]]]]:
    retrieval, all_rows = grouped_retrieval(LOGS / f"{dataset}_jina_v4_top30.jsonl.gz")
    calibration_qids = read_plan(SPLITS / dataset / "calibration_manifest.json")
    test_qids = sorted(retrieval["test"])[:test_queries]
    if len(test_qids) != test_queries:
        raise ValueError(f"{dataset}: only {len(test_qids)} frozen test queries are available")
    # The report's manifest assigns logical roles.  In the historical HotpotQA
    # Jina log, 51 of those calibration qids retained a stale ``development``
    # row label; selecting them by manifest qid reproduces the report's own
    # filtering step without relabeling or dropping their frozen candidates.
    calibration = [all_rows[qid] for qid in calibration_qids]
    test = [all_rows[qid] for qid in test_qids]
    if any(len(rows) != 30 for rows in calibration + test):
        raise ValueError(f"{dataset}: all report rows must have exactly 30 candidates")
    return calibration_qids, calibration, test_qids, test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--test-queries", type=int, default=100)
    parser.add_argument("--selection-only", action="store_true")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--min-pixels", type=int, default=3136)
    parser.add_argument("--max-pixels", type=int, default=200704)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = tuple(value.strip() for value in args.datasets.split(",") if value.strip())
    methods = tuple(value.strip() for value in args.methods.split(",") if value.strip())
    if not datasets or set(datasets).difference(DATASETS):
        raise ValueError(f"--datasets must be a non-empty subset of {DATASETS}")
    if not methods or set(methods).difference(METHODS):
        raise ValueError(f"--methods must be a non-empty subset of {METHODS}")
    if args.test_queries != 100:
        raise ValueError("This runner is locked to the report's recovered 100-query test plan")
    if not args.selection_only and args.model is None:
        raise ValueError("--model is required unless --selection-only is supplied")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pending: list[tuple[str, list[str], list[list[dict[str, Any]]], dict[str, list[np.ndarray]], DatasetConfig]] = []
    results: dict[str, dict[str, Any]] = {}
    for dataset in datasets:
        calibration_qids, calibration, test_qids, test = load_dataset(dataset, args.test_queries)
        masks, metadata = selector_masks(calibration, test)
        masks = {method: masks[method] for method in methods}
        summary = selection_summary(test, masks)
        if "fixed_top10" in masks:
            validate_published_fixed_top10(dataset, summary)
        materialize_split(args.output_dir, dataset, calibration_qids, test_qids)
        results[dataset] = {
            "status": "selection_complete" if args.selection_only else "running",
            "test_queries": len(test_qids),
            "calibration_queries": len(calibration_qids),
            "selection": summary,
            "metadata": metadata,
        }
        pending.append((dataset, test_qids, test, masks, CONFIGS[dataset]))

    write_report(args.output_dir / "REPORT.md", results)
    (args.output_dir / "selection_summary.json").write_text(
        json.dumps({"status": "selection_complete", "datasets": results}, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.selection_only:
        return

    generator = QwenDirectAnswerGenerator(args.model, min_pixels=args.min_pixels, max_pixels=args.max_pixels)
    for dataset, qids, test, masks, config in pending:
        questions, corpus = keyed(config.questions, "qid"), keyed(config.corpus, "id")
        prediction_path = args.output_dir / dataset / "downstream_predictions.jsonl"
        done = {str(row["qid"]): row for row in iter_jsonl(prediction_path)} if prediction_path.exists() else {}
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        for number, (qid, rows) in enumerate(zip(qids, test, strict=True), start=1):
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
            done[qid] = record
            print(f"[{dataset} {number}/{len(qids)}] {qid}", flush=True)
        ordered = [done[qid] for qid in qids]
        results[dataset]["status"] = "complete"
        results[dataset]["downstream"] = metric_summary(ordered, methods)
        (args.output_dir / dataset / "summary.json").write_text(
            json.dumps(results[dataset], indent=2) + "\n", encoding="utf-8"
        )
        write_report(args.output_dir / "REPORT.md", results)
        (args.output_dir / "summary.json").write_text(
            json.dumps({"status": "running", "datasets": results}, indent=2) + "\n",
            encoding="utf-8",
        )
    (args.output_dir / "summary.json").write_text(
        json.dumps({"status": "complete", "datasets": results}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
