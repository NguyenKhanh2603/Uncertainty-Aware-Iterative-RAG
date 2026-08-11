"""Paired DeLong test for two AUROCs on the same WebQ examples.

Higher score must mean a higher probability that the answer is incorrect.  The
implementation follows the fast DeLong covariance estimator used for paired ROC
curves; no RAGU or main-project imports are required.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def midranks(values: np.ndarray) -> np.ndarray:
    """Return one-indexed average ranks, assigning tied scores their midrank."""
    order = np.argsort(values)
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[start:end] = 0.5 * (start + 1 + end)
        start = end
    result = np.empty(len(values), dtype=float)
    result[order] = ranks
    return result


def fast_delong(predictions: np.ndarray, positive_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Return AUROCs and their paired covariance matrix.

    ``predictions`` has shape (number_of_scores, positives + negatives), with
    positives ordered first.
    """
    score_count, total_count = predictions.shape
    negative_count = total_count - positive_count
    if positive_count == 0 or negative_count == 0:
        raise ValueError("DeLong requires both incorrect and correct answers")
    positive = predictions[:, :positive_count]
    negative = predictions[:, positive_count:]
    tx = np.empty((score_count, positive_count), dtype=float)
    ty = np.empty((score_count, negative_count), dtype=float)
    tz = np.empty((score_count, total_count), dtype=float)
    for index in range(score_count):
        tx[index] = midranks(positive[index])
        ty[index] = midranks(negative[index])
        tz[index] = midranks(predictions[index])
    aucs = tz[:, :positive_count].sum(axis=1) / positive_count / negative_count - (positive_count + 1.0) / (2.0 * negative_count)
    v01 = (tz[:, :positive_count] - tx) / negative_count
    v10 = 1.0 - (tz[:, positive_count:] - ty) / positive_count
    sx = np.atleast_2d(np.cov(v01))
    sy = np.atleast_2d(np.cov(v10))
    covariance = sx / positive_count + sy / negative_count
    return aucs, covariance


def paired_delong(labels_incorrect: np.ndarray, first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    order = np.argsort(-labels_incorrect)
    positives = int(labels_incorrect.sum())
    predictions = np.vstack((first[order], second[order]))
    aucs, covariance = fast_delong(predictions, positives)
    difference = float(aucs[0] - aucs[1])
    variance = float(np.array([1.0, -1.0]) @ covariance @ np.array([1.0, -1.0]))
    if variance <= 0:
        raise ValueError("Non-positive DeLong variance; the two score vectors may be identical")
    z_score = difference / math.sqrt(variance)
    # Two-sided normal-test p-value. erfc(|z| / sqrt(2)) = 2 * NormalSF(|z|).
    p_value = math.erfc(abs(z_score) / math.sqrt(2.0))
    return {
        "auroc_first": float(aucs[0]),
        "auroc_second": float(aucs[1]),
        "difference_first_minus_second": difference,
        "z_score": z_score,
        "p_value_two_sided": p_value,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired two-sided DeLong AUROC comparison")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--first", default="ours_semantic_uncertainty", help="First uncertainty JSON field")
    parser.add_argument("--second", default="semantic_entropy", help="Second uncertainty JSON field")
    parser.add_argument("--correct-label", default="correct_acc", help="0/1 field where one means correct")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.input)
    missing = [
        index for index, row in enumerate(rows, start=1)
        if args.first not in row or args.second not in row or args.correct_label not in row
    ]
    if missing:
        raise ValueError(f"Required fields are missing on JSONL line {missing[0]}")
    correct = np.asarray([int(row[args.correct_label]) for row in rows], dtype=int)
    result = paired_delong(
        1 - correct,
        np.asarray([float(row[args.first]) for row in rows]),
        np.asarray([float(row[args.second]) for row in rows]),
    )
    result.update({"examples": len(rows), "first": args.first, "second": args.second, "positive_label": "incorrect"})
    print(json.dumps(result, indent=2))
    if result["p_value_two_sided"] < 0.05 and result["difference_first_minus_second"] > 0:
        print("Conclusion: first score is significantly better (two-sided paired DeLong, p < 0.05).")
    elif result["p_value_two_sided"] < 0.05:
        print("Conclusion: second score is significantly better (two-sided paired DeLong, p < 0.05).")
    else:
        print("Conclusion: no statistically significant AUROC difference at alpha = 0.05.")


if __name__ == "__main__":
    main()
