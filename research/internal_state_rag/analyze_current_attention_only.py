"""Evaluate the original attention-only signals on one dataset's clean roles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from research.internal_state_rag.analyze_attention_only_clean_conformal import (
    learned_attention_scores,
    load_attention,
)
from research.internal_state_rag.analyze_pairwise_conformal_clean_split import (
    conditional_metrics,
    conformal_threshold,
    end_to_end_metrics,
    fixed_k_metrics,
    keep_mask,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--probe-train", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alphas", default="0.2,0.1,0.05")
    return parser.parse_args()


def qid_set(data: dict) -> set[str]:
    return set(data["qids"].astype(str).tolist())


def evaluate_method(
    cal_scores: np.ndarray,
    test_scores: np.ndarray,
    cal_labels: np.ndarray,
    test_labels: np.ndarray,
    alphas: list[float],
) -> dict[str, object]:
    cal_retrievable = cal_labels.any(axis=1)
    test_retrievable = test_labels.any(axis=1)
    output = {
        "ranking": fixed_k_metrics(test_labels, test_scores),
        "conformal": {},
    }
    for alpha in alphas:
        threshold, order = conformal_threshold(
            cal_scores[cal_retrievable],
            cal_labels[cal_retrievable],
            alpha=alpha,
            coverage_target="all_support",
        )
        mask = keep_mask(test_scores, threshold)
        output["conformal"][str(alpha)] = {
            "threshold": threshold,
            "finite_sample_order": order,
            "conditional_on_retrievable": conditional_metrics(
                test_labels[test_retrievable], mask[test_retrievable]
            ),
            "end_to_end": end_to_end_metrics(test_labels, mask),
        }
    return output


def main() -> None:
    args = parse_args()
    alphas = [float(value) for value in args.alphas.split(",")]
    train = load_attention(args.probe_train)
    calibration = load_attention(args.calibration)
    test = load_attention(args.test)
    overlaps = {
        "probe_calibration": len(qid_set(train) & qid_set(calibration)),
        "probe_test": len(qid_set(train) & qid_set(test)),
        "calibration_test": len(qid_set(calibration) & qid_set(test)),
    }
    if any(overlaps.values()):
        raise ValueError(f"Query overlap across roles: {overlaps}")
    learned_cal, learned_test, learned_metadata = learned_attention_scores(
        train, calibration, test
    )
    cal_scores = {
        **calibration["scores"],
        "attention_learned_probe": learned_cal,
    }
    test_scores = {**test["scores"], "attention_learned_probe": learned_test}
    methods = {
        name: evaluate_method(
            cal_scores[name],
            test_scores[name],
            calibration["labels"],
            test["labels"],
            alphas,
        )
        for name in cal_scores
    }
    output = {
        "status": "complete",
        "dataset": args.dataset,
        "method": (
            "draft-answer token attention to Top-L chunks; original and reversed "
            "chunk order; no cosine/reranker/hidden/LM features"
        ),
        "split": {
            "probe_train_queries": int(len(train["labels"])),
            "calibration_queries": int(len(calibration["labels"])),
            "test_queries": int(len(test["labels"])),
            "test_retrievable": int(test["labels"].any(axis=1).sum()),
            "top_l": int(test["labels"].shape[1]),
            "overlaps": overlaps,
        },
        "learned_probe": learned_metadata,
        "methods": methods,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
