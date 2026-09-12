"""Evaluation helpers for query-grouped chunk-retention scores."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


def grouped_z_scores(
    rows: list[dict[str, Any]], score_name: str
) -> np.ndarray:
    """Standardize a score within each query without using support labels."""

    result = np.zeros(len(rows), dtype=np.float64)
    qids = sorted({str(row["qid"]) for row in rows})
    for qid in qids:
        indices = [index for index, row in enumerate(rows) if str(row["qid"]) == qid]
        values = np.asarray([float(rows[index][score_name]) for index in indices])
        scale = float(values.std())
        normalized = values - values.mean()
        if scale > 0:
            normalized /= scale
        result[indices] = normalized
    return result


def ranking_metrics(
    rows: list[dict[str, Any]], scores: np.ndarray
) -> dict[str, float]:
    """Measure candidate separation and first-support rank per query."""

    labels = np.asarray([int(row["is_support"]) for row in rows])
    values = np.asarray(scores, dtype=np.float64)
    if len(values) != len(rows):
        raise ValueError("scores and rows must have equal length")
    reciprocal_ranks = []
    top1 = []
    for qid in sorted({str(row["qid"]) for row in rows}):
        indices = [index for index, row in enumerate(rows) if str(row["qid"]) == qid]
        indices.sort(key=lambda index: (-values[index], int(rows[index]["rank"])))
        support_rank = next(
            position
            for position, index in enumerate(indices, start=1)
            if rows[index]["is_support"]
        )
        reciprocal_ranks.append(1.0 / support_rank)
        top1.append(float(support_rank == 1))
    return {
        "auroc": float(roc_auc_score(labels, values)),
        "average_precision": float(average_precision_score(labels, values)),
        "query_mrr": float(np.mean(reciprocal_ranks)),
        "query_top1_support_rate": float(np.mean(top1)),
    }


def leave_one_query_out_logistic(
    rows: list[dict[str, Any]], feature_names: list[str]
) -> np.ndarray:
    """Return held-out predictions from a group-safe linear fusion probe."""

    if not feature_names:
        raise ValueError("At least one feature is required")
    features = np.column_stack(
        [grouped_z_scores(rows, feature_name) for feature_name in feature_names]
    )
    labels = np.asarray([int(row["is_support"]) for row in rows])
    groups = np.asarray([str(row["qid"]) for row in rows])
    predictions = np.empty(len(rows), dtype=np.float64)
    for qid in sorted(set(groups)):
        train = groups != qid
        held_out = ~train
        scaler = StandardScaler().fit(features[train])
        model = LogisticRegression(
            class_weight="balanced", random_state=0, max_iter=1_000
        ).fit(scaler.transform(features[train]), labels[train])
        predictions[held_out] = model.predict_proba(
            scaler.transform(features[held_out])
        )[:, 1]
    return predictions


def score_metrics(
    rows: list[dict[str, Any]], score_names: list[str]
) -> dict[str, dict[str, float]]:
    """Evaluate raw scores and their query-local standardized forms."""

    output = {}
    for score_name in score_names:
        raw = np.asarray([float(row[score_name]) for row in rows])
        output[score_name] = ranking_metrics(rows, raw)
        output[f"{score_name}_query_z"] = ranking_metrics(
            rows, grouped_z_scores(rows, score_name)
        )
    return output


def bootstrap_query_metric_difference(
    rows: list[dict[str, Any]],
    left_scores: np.ndarray,
    right_scores: np.ndarray,
    metric: Callable[[list[dict[str, Any]], np.ndarray], float],
    *,
    seed: int = 0,
    samples: int = 2_000,
) -> dict[str, float]:
    """Bootstrap a paired score difference with queries as the sampling unit."""

    qids = sorted({str(row["qid"]) for row in rows})
    by_qid = {
        qid: [index for index, row in enumerate(rows) if str(row["qid"]) == qid]
        for qid in qids
    }
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(samples):
        sampled = rng.choice(qids, size=len(qids), replace=True)
        sample_rows = []
        sample_left = []
        sample_right = []
        for draw_index, qid in enumerate(sampled):
            for index in by_qid[str(qid)]:
                copied = dict(rows[index])
                copied["qid"] = f"{draw_index}:{qid}"
                sample_rows.append(copied)
                sample_left.append(left_scores[index])
                sample_right.append(right_scores[index])
        differences.append(
            metric(sample_rows, np.asarray(sample_left))
            - metric(sample_rows, np.asarray(sample_right))
        )
    low, high = np.quantile(differences, [0.025, 0.975])
    return {
        "mean": float(np.mean(differences)),
        "ci95_low": float(low),
        "ci95_high": float(high),
    }
