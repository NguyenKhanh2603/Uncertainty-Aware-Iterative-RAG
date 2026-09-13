"""Fit a frozen linear chunk-relevance probe and evaluate it on another split."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def query_z(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return (values - values.mean(axis=1, keepdims=True)) / (
        values.std(axis=1, keepdims=True) + 1e-7
    )


def query_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    aps, reciprocal_ranks, top1 = [], [], []
    for query_labels, query_scores in zip(labels, scores, strict=True):
        ordered = query_labels[np.argsort(-query_scores, kind="stable")]
        positions = np.flatnonzero(ordered) + 1
        aps.append(float(np.mean(np.arange(1, len(positions) + 1) / positions)))
        reciprocal_ranks.append(float(1 / positions[0]))
        top1.append(float(positions[0] == 1))
    return {
        "mean_query_ap": float(np.mean(aps)),
        "mrr": float(np.mean(reciprocal_ranks)),
        "top1_support_rate": float(np.mean(top1)),
    }


def make_probe(c: float) -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c,
            class_weight="balanced",
            solver="liblinear",
            max_iter=3_000,
            random_state=0,
        ),
    )


def bootstrap_difference(
    labels: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    *,
    seed: int,
    samples: int = 10_000,
) -> dict[str, dict[str, float]]:
    per_query = []
    for query_labels, a, b in zip(labels, left, right, strict=True):
        left_metrics = query_metrics(query_labels[None, :], a[None, :])
        right_metrics = query_metrics(query_labels[None, :], b[None, :])
        per_query.append(
            [
                left_metrics["mean_query_ap"] - right_metrics["mean_query_ap"],
                left_metrics["mrr"] - right_metrics["mrr"],
                left_metrics["top1_support_rate"] - right_metrics["top1_support_rate"],
            ]
        )
    differences = np.asarray(per_query)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(labels), size=(samples, len(labels)))
    boot = differences[draws].mean(axis=1)
    names = ("mean_query_ap", "mrr", "top1_support_rate")
    return {
        name: {
            "mean": float(differences[:, index].mean()),
            "ci95_low": float(np.quantile(boot[:, index], 0.025)),
            "ci95_high": float(np.quantile(boot[:, index], 0.975)),
        }
        for index, name in enumerate(names)
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=433)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calibration = np.load(args.calibration)
    test = np.load(args.test)
    if not np.array_equal(calibration["layer_ids"], test["layer_ids"]):
        raise ValueError("Calibration and test layer IDs differ")
    layers = calibration["layer_ids"].astype(int).tolist()
    train_features = calibration["features"].astype(np.float32)
    train_labels = calibration["labels"].astype(bool)
    test_features = test["features"].astype(np.float32)
    test_labels = test["labels"].astype(bool)
    groups = np.repeat(np.arange(len(train_labels)), train_labels.shape[1])
    folds = list(GroupKFold(5).split(groups, train_labels.ravel(), groups))
    c_values = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)

    candidates = []
    for layer_index, layer in enumerate(layers):
        x = train_features[:, :, layer_index, :].reshape(-1, train_features.shape[-1])
        for c in c_values:
            oof = np.zeros(train_labels.size, dtype=np.float64)
            for train_indices, held_indices in folds:
                probe = make_probe(c).fit(x[train_indices], train_labels.ravel()[train_indices])
                oof[held_indices] = probe.predict_proba(x[held_indices])[:, 1]
            metrics = query_metrics(train_labels, oof.reshape(train_labels.shape))
            candidates.append((metrics["mean_query_ap"], metrics["mrr"], layer_index, c, oof))
    _, _, selected_index, selected_c, train_probe_scores = max(
        candidates, key=lambda row: (row[0], row[1])
    )
    train_x = train_features[:, :, selected_index, :].reshape(
        -1, train_features.shape[-1]
    )
    test_x = test_features[:, :, selected_index, :].reshape(-1, test_features.shape[-1])
    probe = make_probe(selected_c).fit(train_x, train_labels.ravel())
    test_probe_scores = probe.predict_proba(test_x)[:, 1].reshape(test_labels.shape)
    train_probe_scores = train_probe_scores.reshape(train_labels.shape)

    train_bge = calibration["bge_scores"].astype(np.float64)
    test_bge = test["bge_scores"].astype(np.float64)
    weights = (0.0, 0.01, 0.025, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0)
    fusion_candidates = []
    for weight in weights:
        score = query_z(train_bge) + weight * query_z(train_probe_scores)
        metrics = query_metrics(train_labels, score)
        fusion_candidates.append((metrics["mean_query_ap"], metrics["mrr"], weight))
    _, _, selected_weight = max(fusion_candidates)
    test_fusion = query_z(test_bge) + selected_weight * query_z(test_probe_scores)

    output = {
        "status": "complete",
        "selected_layer": layers[selected_index],
        "selected_c": selected_c,
        "selected_internal_weight": selected_weight,
        "calibration_queries": int(len(train_labels)),
        "test_queries": int(len(test_labels)),
        "calibration": {
            "bge": query_metrics(train_labels, train_bge),
            "probe_oof": query_metrics(train_labels, train_probe_scores),
            "fusion_oof": query_metrics(
                train_labels,
                query_z(train_bge) + selected_weight * query_z(train_probe_scores),
            ),
        },
        "test": {
            "bge": query_metrics(test_labels, test_bge),
            "probe": query_metrics(test_labels, test_probe_scores),
            "fusion": query_metrics(test_labels, test_fusion),
            "probe_minus_bge": bootstrap_difference(
                test_labels, test_probe_scores, test_bge, seed=args.seed
            ),
            "fusion_minus_bge": bootstrap_difference(
                test_labels, test_fusion, test_bge, seed=args.seed + 1
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        main()
