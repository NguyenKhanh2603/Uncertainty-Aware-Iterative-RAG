"""Compute dense-cosine baselines on the exact internal-fusion splits."""

from __future__ import annotations

import gzip
import json
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

    hotpot_cal = np.load(ROOT / "hotpotqa_pairwise_cal_n500/features.npz")
    hotpot_test = np.load(ROOT / "hotpotqa_pairwise_test_n1000/features.npz")
    hotpot_rows = retrieval_rows(ROOT / "hotpotqa_top30_bge_reranked.jsonl.gz")
    hotpot = evaluate(
        hotpot_cal["labels"].astype(bool),
        aligned_cosine(hotpot_cal, hotpot_rows),
        hotpot_test["labels"].astype(bool),
        aligned_cosine(hotpot_test, hotpot_rows),
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

    output = {
        "status": "complete",
        "note": (
            "These are query-level all-support split-conformal baselines. They are "
            "separate from the proposal's candidate-wise BY-FDR rule."
        ),
        "datasets": {"tatqa": tatqa, "hotpotqa": hotpot, "mmqa_text_table": mmqa},
    }
    destination = ROOT / "multidataset_cosine_baselines.json"
    destination.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
