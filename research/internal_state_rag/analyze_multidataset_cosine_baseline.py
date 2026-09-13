"""Compute dense-cosine baselines on the exact internal-fusion splits."""

from __future__ import annotations

import gzip
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from research.internal_state_rag.analyze_hidden_chunk_probe import query_z
from research.internal_state_rag.analyze_pairwise_conformal_clean_split import (
    conditional_metrics,
    conformal_threshold,
    end_to_end_metrics,
    fixed_k_metrics,
    keep_mask,
)
from uncertainty_rag.core.conformal_selection import benjamini_yekutieli


ROOT = Path("research/internal_state_rag/results")


def retrieval_rows(path: Path) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            grouped[str(row["qid"])].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: (int(row["rank"]), str(row["chunk_id"])))
    return grouped


def aligned_cosine(features: np.lib.npyio.NpzFile, grouped: dict[str, list[dict]]) -> np.ndarray:
    top_l = features["labels"].shape[1]
    scores = []
    for qid in features["qids"].astype(str):
        rows = grouped[qid][:top_l]
        if len(rows) != top_l:
            raise ValueError(f"{qid} has {len(rows)} candidates, expected {top_l}")
        scores.append([float(row["cosine_score"]) for row in rows])
    return np.asarray(scores, dtype=np.float64)


def aligned_rows(qids: list[str], grouped: dict[str, list[dict]]) -> list[list[dict]]:
    aligned = []
    for qid in qids:
        rows = grouped[qid]
        if len(rows) != 30:
            raise ValueError(f"{qid} has {len(rows)} candidates, expected 30")
        aligned.append(rows)
    return aligned


def selection_metrics(labels: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Metrics for selectors that may return an empty context."""

    retrievable = labels.any(axis=1)
    y = labels[retrievable]
    selected = mask[retrievable]
    retained = (y & selected).sum(axis=1)
    support_total = y.sum(axis=1)
    kept = selected.sum(axis=1)
    selected_total = int(kept.sum())
    return {
        "queries": int(len(y)),
        "support_chunks_total": int(support_total.sum()),
        "support_chunks_retained": int(retained.sum()),
        "mean_chunks_kept": float(kept.mean()),
        "chunk_precision_among_kept": (
            float(retained.sum() / selected_total) if selected_total else 0.0
        ),
        "micro_support_recall": float(retained.sum() / support_total.sum()),
        "mean_support_recall": float(np.mean(retained / support_total)),
        "query_any_support_coverage": float(np.mean(retained > 0)),
        "query_all_support_coverage": float(np.mean(retained == support_total)),
        "empty_query_rate": float(np.mean(kept == 0)),
    }


def false_score_banks(
    calibration_rows: list[list[dict]], *, conditioning: str
) -> dict[str, np.ndarray]:
    banks: dict[str, list[float]] = defaultdict(list)
    for rows in calibration_rows:
        for row in rows:
            if row["support_label"] != "false":
                continue
            key = str(row["modality"]) if conditioning == "modality" else "pooled"
            banks[key].append(float(row["cosine_score"]))
    return {key: np.sort(values) for key, values in banks.items()}


def proposal_by_mask(
    test_rows: list[list[dict]],
    banks: dict[str, np.ndarray],
    *,
    conditioning: str,
    alpha: float,
    max_context: int = 10,
) -> np.ndarray:
    """Exact proposal selector: false-bank p-values, BY, then cosine Top-K cap."""

    mask = np.zeros((len(test_rows), 30), dtype=bool)
    for query_index, rows in enumerate(test_rows):
        p_values = []
        for row in rows:
            key = str(row["modality"]) if conditioning == "modality" else "pooled"
            bank = banks[key]
            score = float(row["cosine_score"])
            count_greater_or_equal = len(bank) - int(np.searchsorted(bank, score, side="left"))
            p_values.append((1.0 + count_greater_or_equal) / (len(bank) + 1.0))
        rejected = benjamini_yekutieli(p_values, alpha).rejected_indices
        selected = sorted(
            rejected,
            key=lambda index: (int(rows[index]["cosine_rank"]), str(rows[index]["chunk_id"])),
        )[:max_context]
        mask[query_index, selected] = True
    return mask


def support_threshold(scores: list[float], alpha: float) -> float:
    """Finite-sample lower support-score threshold used by the prior coverage baseline."""

    ordered = np.sort(np.asarray(scores, dtype=np.float64))
    order = math.floor(alpha * (len(ordered) + 1))
    if order < 1:
        return float("-inf")
    return float(ordered[min(order, len(ordered)) - 1])


def support_coverage_mask(
    calibration_rows: list[list[dict]],
    test_rows: list[list[dict]],
    *,
    alpha: float,
    max_context: int = 10,
    min_group_size: int = 30,
) -> tuple[np.ndarray, dict[str, object]]:
    """Prior support-bank cosine baseline with modality thresholds and a Top-K cap."""

    pooled_supports: list[float] = []
    grouped_supports: dict[str, list[float]] = defaultdict(list)
    for rows in calibration_rows:
        for row in rows:
            if row["support_label"] == "support":
                score = float(row["cosine_score"])
                pooled_supports.append(score)
                grouped_supports[str(row["modality"])].append(score)
    pooled_threshold = support_threshold(pooled_supports, alpha)
    thresholds = {
        modality: (
            support_threshold(scores, alpha) if len(scores) >= min_group_size else pooled_threshold
        )
        for modality, scores in grouped_supports.items()
    }
    mask = np.zeros((len(test_rows), 30), dtype=bool)
    for query_index, rows in enumerate(test_rows):
        eligible = [
            index
            for index, row in enumerate(rows)
            if float(row["cosine_score"])
            >= thresholds.get(str(row["modality"]), pooled_threshold)
        ]
        selected = sorted(
            eligible,
            key=lambda index: (int(rows[index]["cosine_rank"]), str(rows[index]["chunk_id"])),
        )[:max_context]
        mask[query_index, selected] = True
    audit = {
        "target_candidate_support_coverage": 1.0 - alpha,
        "max_context": max_context,
        "min_group_size": min_group_size,
        "pooled_support_count": len(pooled_supports),
        "pooled_threshold": pooled_threshold,
        "modality_support_counts": {
            modality: len(scores) for modality, scores in grouped_supports.items()
        },
        "modality_thresholds": thresholds,
    }
    return mask, audit


def evaluate_original_cosine_methods(
    calibration_rows: list[list[dict]], test_rows: list[list[dict]]
) -> dict[str, object]:
    labels = np.asarray(
        [[row["support_label"] == "support" for row in rows] for rows in test_rows],
        dtype=bool,
    )
    false_banks = {
        conditioning: false_score_banks(calibration_rows, conditioning=conditioning)
        for conditioning in ("pooled", "modality")
    }
    results: dict[str, object] = {}
    for alpha in (0.2, 0.1, 0.05):
        methods: dict[str, object] = {}
        for conditioning in ("pooled", "modality"):
            by_mask = proposal_by_mask(
                test_rows,
                false_banks[conditioning],
                conditioning=conditioning,
                alpha=alpha,
            )
            methods[f"proposal_false_bank_by_{conditioning}"] = selection_metrics(
                labels, by_mask
            )
        coverage_mask, audit = support_coverage_mask(
            calibration_rows, test_rows, alpha=alpha
        )
        methods["support_bank_coverage_k10"] = {
            **selection_metrics(labels, coverage_mask),
            "calibration": audit,
        }
        results[str(alpha)] = methods
    return results


def evaluate(
    calibration_labels: np.ndarray,
    calibration_scores: np.ndarray,
    test_labels: np.ndarray,
    test_scores: np.ndarray,
) -> dict:
    cal_retrievable = calibration_labels.any(axis=1)
    test_retrievable = test_labels.any(axis=1)
    methods = {
        "cosine_raw": (calibration_scores, test_scores),
        "cosine_query_z": (query_z(calibration_scores), query_z(test_scores)),
    }
    result = {
        "test_ranking": {
            name: fixed_k_metrics(test_labels, scores[1]) for name, scores in methods.items()
        },
        "conformal": {},
    }
    for alpha in (0.2, 0.1, 0.05):
        result["conformal"][str(alpha)] = {}
        for name, (cal_score, test_score) in methods.items():
            threshold, order = conformal_threshold(
                cal_score[cal_retrievable],
                calibration_labels[cal_retrievable],
                alpha=alpha,
                coverage_target="all_support",
            )
            mask = keep_mask(test_score, threshold)
            result["conformal"][str(alpha)][name] = {
                "threshold": threshold,
                "finite_sample_order": order,
                "conditional_on_retrievable": conditional_metrics(
                    test_labels[test_retrievable], mask[test_retrievable]
                ),
                "end_to_end": end_to_end_metrics(test_labels, mask),
            }
    return result


def main() -> None:
    tatqa_source = np.load(ROOT / "pairwise_relevance_cal_fresh_n904/features.npz")
    tatqa_test = np.load(ROOT / "pairwise_relevance_test_all_n1000/features.npz")
    tatqa_rows = retrieval_rows(
        Path("data/conformal_global_run/retrieval/tatqa_top30_bge_reranked.jsonl.gz")
    )
    permutation = np.random.default_rng(733).permutation(len(tatqa_source["labels"]))
    tatqa_cal_indices = permutation[len(permutation) // 2 :]
    tatqa_source_cosine = aligned_cosine(tatqa_source, tatqa_rows)
    tatqa = evaluate(
        tatqa_source["labels"][tatqa_cal_indices].astype(bool),
        tatqa_source_cosine[tatqa_cal_indices],
        tatqa_test["labels"].astype(bool),
        aligned_cosine(tatqa_test, tatqa_rows),
    )
    tatqa["original_cosine_methods"] = evaluate_original_cosine_methods(
        aligned_rows(
            sorted(
                qid for qid, rows in tatqa_rows.items() if rows[0]["split_role"] == "calibration"
            ),
            tatqa_rows,
        ),
        aligned_rows(tatqa_test["qids"].astype(str).tolist(), tatqa_rows),
    )

    hotpot_cal = np.load(ROOT / "hotpotqa_pairwise_cal_n500/features.npz")
    hotpot_test = np.load(ROOT / "hotpotqa_pairwise_test_n1000/features.npz")
    hotpot_rows = retrieval_rows(ROOT / "hotpotqa_top30_bge_reranked.jsonl.gz")
    hotpot = evaluate(
        hotpot_cal["labels"].astype(bool),
        aligned_cosine(hotpot_cal, hotpot_rows),
        hotpot_test["labels"].astype(bool),
        aligned_cosine(hotpot_test, hotpot_rows),
    )
    hotpot["original_cosine_methods"] = evaluate_original_cosine_methods(
        aligned_rows(hotpot_cal["qids"].astype(str).tolist(), hotpot_rows),
        aligned_rows(hotpot_test["qids"].astype(str).tolist(), hotpot_rows),
    )

    mmqa_cal = np.load(ROOT / "mmqa_text_table_pairwise_cal_n523/features.npz")
    mmqa_test = np.load(ROOT / "mmqa_text_table_pairwise_test_n300/features.npz")
    mmqa_rows = retrieval_rows(ROOT / "mmqa_text_table_top30_bge_reranked.jsonl.gz")
    mmqa = evaluate(
        mmqa_cal["labels"].astype(bool),
        aligned_cosine(mmqa_cal, mmqa_rows),
        mmqa_test["labels"].astype(bool),
        aligned_cosine(mmqa_test, mmqa_rows),
    )
    mmqa["original_cosine_methods"] = evaluate_original_cosine_methods(
        aligned_rows(mmqa_cal["qids"].astype(str).tolist(), mmqa_rows),
        aligned_rows(mmqa_test["qids"].astype(str).tolist(), mmqa_rows),
    )

    output = {
        "status": "complete",
        "note": (
            "original_cosine_methods contains the proposal's false-bank conformal "
            "p-values + BY + K=10 and the later support-bank coverage + K=10 baseline. "
            "The conformal field is an uncapped query-worst-support ablation and must "
            "not be labelled as the original cosine-conformal proposal."
        ),
        "datasets": {"tatqa": tatqa, "hotpotqa": hotpot, "mmqa_text_table": mmqa},
    }
    destination = ROOT / "multidataset_cosine_baselines.json"
    destination.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
