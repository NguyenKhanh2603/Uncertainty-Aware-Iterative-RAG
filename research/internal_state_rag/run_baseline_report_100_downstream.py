"""Run end-to-end QA for the nine selectors in baseline_comparison_report_100_queries_detailed.

The definitive protocol is the committed 800-row CSV in
``research/internal_state_rag/protocols/baseline_report_100``:
100 calibration and 100 held-out test qids per dataset.  The runner reads that
CSV directly, rejects duplicate or overlapping qids, materializes the exact
plans, and evaluates every reported context selector with greedy Qwen2-VL-7B
answer generation.  It never infers a split from a retrieval-row role label.

BH p-values use the calibration false-score bank matched by modality, matching
the report's stated ``--conditioning dataset,modality`` procedure. The post-BH
cap keeps the smallest p-values (rank breaks ties), as in the report's
simulate_bh.py.
"""
from __future__ import annotations

import argparse
import csv
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
SPLIT_CSV_PATH = Path(
    "research/internal_state_rag/protocols/baseline_report_100/"
    "report_100_calibration_100_test_question_ids_2026-09-19.csv"
)
LOGS = Path("research/internal_state_rag/results/qwen2vl_jina4")

# The selection metrics published in baseline_comparison_report_100_queries_detailed.md.
# They provide a guard that the recovered test plan has not drifted.
PUBLISHED_FIXED_TOP10 = {
    "hotpotqa": {"mean_chunks": 10.00, "precision": 0.177, "support_recall": 0.962},
    "mmqa": {"mean_chunks": 10.00, "precision": 0.109, "support_recall": 0.965},
    "tatqa": {"mean_chunks": 10.00, "precision": 0.089, "support_recall": 0.848},
    "webqa": {"mean_chunks": 10.00, "precision": 0.058, "support_recall": 0.682},
}

# These are transcribed from baseline_comparison_report_100_queries_detailed.md.
# The audit is descriptive: downstream generation always uses the materialized
# masks, never an aggregate number copied from this table.
PUBLISHED_SELECTION = {
    "hotpotqa": {
        "fixed_top10": (10.00, 0.177, 0.962, 0.000),
        "fixed_top20": (20.00, 0.092, 0.995, 0.000),
        "cce_cosine_alpha_0.10": (11.19, 0.149, 0.908, 0.010),
        "conflare_cosine_alpha_0.10": (10.80, 0.155, 0.908, 0.010),
        "traq_cosine_alpha_0.10": (14.97, 0.114, 0.929, 0.000),
        "bh_modality_cosine_alpha_0.10_ctx10": (0.74, 0.635, 0.255, 0.660),
        "bh_modality_cosine_alpha_0.90_ctx10": (8.19, 0.186, 0.826, 0.110),
        "bh_modality_cosine_alpha_0.99_ctx10": (9.90, 0.178, 0.957, 0.010),
        "bh_modality_cosine_alpha_0.99_ctx20": (19.80, 0.092, 0.989, 0.010),
    },
    "mmqa": {
        "fixed_top10": (10.00, 0.109, 0.965, 0.000),
        "fixed_top20": (20.00, 0.055, 0.973, 0.000),
        "cce_cosine_alpha_0.10": (14.19, 0.074, 0.929, 0.030),
        "conflare_cosine_alpha_0.10": (13.02, 0.080, 0.920, 0.030),
        "traq_cosine_alpha_0.10": (20.43, 0.054, 0.973, 0.010),
        "bh_modality_cosine_alpha_0.10_ctx10": (0.21, 0.286, 0.053, 0.930),
        "bh_modality_cosine_alpha_0.90_ctx10": (7.79, 0.118, 0.814, 0.150),
        "bh_modality_cosine_alpha_0.99_ctx10": (9.41, 0.109, 0.912, 0.050),
        "bh_modality_cosine_alpha_0.99_ctx20": (18.81, 0.055, 0.920, 0.050),
    },
    "tatqa": {
        "fixed_top10": (10.00, 0.089, 0.848, 0.000),
        "fixed_top20": (20.00, 0.051, 0.962, 0.000),
        "cce_cosine_alpha_0.10": (20.10, 0.047, 0.895, 0.050),
        "conflare_cosine_alpha_0.10": (20.02, 0.047, 0.895, 0.050),
        "traq_cosine_alpha_0.10": (24.08, 0.042, 0.952, 0.030),
        "bh_modality_cosine_alpha_0.10_ctx10": (1.15, 0.122, 0.133, 0.770),
        "bh_modality_cosine_alpha_0.90_ctx10": (8.39, 0.088, 0.705, 0.110),
        "bh_modality_cosine_alpha_0.99_ctx10": (9.70, 0.089, 0.819, 0.030),
        "bh_modality_cosine_alpha_0.99_ctx20": (19.40, 0.051, 0.943, 0.030),
    },
    "webqa": {
        "fixed_top10": (10.00, 0.058, 0.682, 0.000),
        "fixed_top20": (20.00, 0.040, 0.929, 0.000),
        "cce_cosine_alpha_0.10": (21.98, 0.035, 0.918, 0.080),
        "conflare_cosine_alpha_0.10": (21.71, 0.035, 0.906, 0.080),
        "traq_cosine_alpha_0.10": (26.16, 0.032, 0.976, 0.020),
        "bh_modality_cosine_alpha_0.10_ctx10": (0.33, 0.030, 0.012, 0.950),
        "bh_modality_cosine_alpha_0.90_ctx10": (7.69, 0.064, 0.576, 0.230),
        "bh_modality_cosine_alpha_0.99_ctx10": (9.10, 0.066, 0.706, 0.090),
        "bh_modality_cosine_alpha_0.99_ctx20": (18.20, 0.041, 0.871, 0.090),
    },
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
        # These qids are from the historical Jina development/test role split,
        # not the later official-seed42 HotpotQA test bundle.
        Path("data/conformal_global_run/official_bundle_role_split_hotpotqa/hotpotqa/questions.jsonl"),
        Path("data/conformal_global_run/official_bundle_role_split_hotpotqa/hotpotqa/corpus.jsonl"),
        Path("data/conformal_global_run/official_bundle_role_split_hotpotqa"),
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


def csv_split_qids(dataset: str) -> tuple[list[str], list[str]]:
    """Read the authoritative calibration/test plans from the supplied CSV."""

    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")
    if not SPLIT_CSV_PATH.is_file():
        raise FileNotFoundError(f"Missing protocol CSV: {SPLIT_CSV_PATH}")
    with SPLIT_CSV_PATH.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    required = {"Dataset", "SplitRole", "QueryID"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{SPLIT_CSV_PATH} is missing required columns {sorted(required)}")
    plans = {
        role: [str(row["QueryID"]) for row in rows if row["Dataset"] == dataset and row["SplitRole"] == role]
        for role in ("calibration", "test")
    }
    for role, qids in plans.items():
        if len(qids) != 100 or len(qids) != len(set(qids)):
            raise ValueError(f"{dataset}/{role}: expected 100 unique qids in {SPLIT_CSV_PATH}")
    overlap = set(plans["calibration"]).intersection(plans["test"])
    if overlap:
        raise ValueError(f"{dataset}: calibration/test qids overlap: {sorted(overlap)[:3]}")
    return plans["calibration"], plans["test"]


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
            "source_protocol": str(SPLIT_CSV_PATH),
            "plan": calibration_qids,
        },
        "test_manifest.json": {
            "dataset": dataset,
            "role": "test",
            "n_queries": len(test_qids),
            "source_protocol": str(SPLIT_CSV_PATH),
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


def audit_published_selection(dataset: str, summary: dict[str, dict[str, float]]) -> dict[str, Any]:
    """Compare materialized masks with the report's rounded selection table."""

    fields = ("mean_chunks", "precision", "support_recall", "empty_rate")
    rows = {}
    for method, published in PUBLISHED_SELECTION[dataset].items():
        if method not in summary:
            continue
        observed = summary[method]
        expected = dict(zip(fields, published, strict=True))
        rounded_observed = {
            "mean_chunks": round(observed["mean_chunks"], 2),
            "precision": round(observed["precision"] * 100, 1) / 100,
            "support_recall": round(observed["support_recall"] * 100, 1) / 100,
            "empty_rate": round(observed["empty_rate"] * 100, 1) / 100,
        }
        rows[method] = {
            "published": expected,
            "recomputed": rounded_observed,
            "matches_published_rounding": all(
                math.isclose(rounded_observed[field], expected[field], abs_tol=1e-12)
                for field in fields
            ),
        }
    mismatches = [method for method, row in rows.items() if not row["matches_published_rounding"]]
    return {
        "matches_all_evaluated_rows": not mismatches,
        "evaluated_methods": list(rows),
        "mismatched_methods": mismatches,
        "rows": rows,
    }


def write_report(path: Path, results: dict[str, dict[str, Any]]) -> None:
    lines = [
        "# Downstream QA for the 100-query baseline-comparison report", "",
        "**Source selection report:** [baseline_comparison_report_100_queries_detailed.md](https://github.com/NguyenKhanh2603/Uncertainty-Aware-Iterative-RAG/blob/docs/add-detailed-report/baseline_comparison_report_100_queries_detailed.md).<br>",
        "**Authoritative split:** [`report_100_calibration_100_test_question_ids_2026-09-19.csv`](../../protocols/baseline_report_100/report_100_calibration_100_test_question_ids_2026-09-19.csv).<br>",
        "**Runner:** [`run_baseline_report_100_downstream.py`](../../run_baseline_report_100_downstream.py).<br>",
        "**Splits:** [`splits/`](splits/) contains the exact 100 calibration qids and 100 held-out test qids used here for each dataset. Both plans come directly from `report_100_calibration_100_test_question_ids_2026-09-19.csv` in the supplied protocol archive.<br>",
        "**Leakage audit:** [`CSV_SPLIT_INTEGRITY.json`](CSV_SPLIT_INTEGRITY.json) verifies 100 unique calibration qids, 100 unique test qids, and zero within-dataset overlap for every dataset.",
        "",
        "This rerun reads the source report's explicit CSV protocol, verifies that calibration and held-out test qids are disjoint, then joins those qids to the frozen Jina-v4 Top-30 retrieval logs. Every row uses greedy Qwen2-VL-7B-Instruct output with at most 24 new tokens. CCE, CONFLARE, and TRAQ are retrieval adapters; BH is the source report's modality-conditioned selection component: false-score calibration banks are stratified by `{dataset, modality}`, then BH is applied across the candidate set of each query.",
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
        audit = result["metadata"]["published_selection_audit"]
        if audit["matches_all_evaluated_rows"]:
            lines.extend(["", "The recomputed selector masks match every rounded selection row published in the source report."])
        else:
            lines.extend([
                "",
                "The following rows do not reproduce from the available frozen inputs and are **not** represented as historical downstream results: "
                + ", ".join(f"`{method}`" for method in audit["mismatched_methods"]) + ".",
            ])
        lines.extend(["", "### Calibration and selection metadata", "", "```json", json.dumps(result["metadata"], indent=2), "```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def validate_downstream_inputs(
    config: DatasetConfig,
    qids: Sequence[str],
    test: Sequence[Sequence[dict[str, Any]]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Fail before model loading if a frozen qid, chunk, or image is absent."""

    questions, corpus = keyed(config.questions, "qid"), keyed(config.corpus, "id")
    missing_questions = [qid for qid in qids if qid not in questions]
    if missing_questions:
        raise ValueError(f"{config.name}: question bundle is missing qids {missing_questions[:3]}")
    missing_chunks = sorted(
        {str(row["chunk_id"]) for rows in test for row in rows}.difference(corpus)
    )
    if missing_chunks:
        raise ValueError(f"{config.name}: corpus is missing chunks {missing_chunks[:3]}")
    # This also resolves image paths for every frozen candidate before Qwen
    # loads, preventing a multi-hour run from failing on a late image row.
    for rows in test:
        chunks(rows, np.ones(len(rows), dtype=bool), corpus, config.bundle_root)
    return questions, corpus


def load_dataset(dataset: str, test_queries: int) -> tuple[list[str], list[list[dict[str, Any]]], list[str], list[list[dict[str, Any]]]]:
    _retrieval, all_rows = grouped_retrieval(LOGS / f"{dataset}_jina_v4_top30.jsonl.gz")
    calibration_qids, test_qids = csv_split_qids(dataset)
    if test_queries != len(test_qids):
        raise ValueError(f"{dataset}: requested {test_queries} but protocol contains {len(test_qids)} test qids")
    missing = [qid for qid in calibration_qids + test_qids if qid not in all_rows]
    if missing:
        raise ValueError(f"{dataset}: frozen retrieval is missing protocol qids {missing[:3]}")
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
        metadata["published_selection_audit"] = audit_published_selection(dataset, summary)
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

    resolved_inputs = {
        dataset: validate_downstream_inputs(config, qids, test)
        for dataset, qids, test, _masks, config in pending
    }
    generator = QwenDirectAnswerGenerator(args.model, min_pixels=args.min_pixels, max_pixels=args.max_pixels)
    for dataset, qids, test, masks, config in pending:
        questions, corpus = resolved_inputs[dataset]
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
