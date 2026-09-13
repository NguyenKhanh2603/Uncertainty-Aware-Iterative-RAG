"""Evaluate zero-shot LM-head and trained pairwise relevance signals."""

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

from research.internal_state_rag.analyze_causal_value_probe import retention_metrics
from research.internal_state_rag.analyze_hidden_chunk_probe import (
    bootstrap_difference,
    query_metrics,
    query_z,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=457)
    return parser.parse_args()


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


def evaluate(labels: np.ndarray, scores: np.ndarray) -> dict[str, object]:
    output: dict[str, object] = query_metrics(labels, scores)
    output["retention"] = retention_metrics(labels, scores)
    return output


def bootstrap_retention_difference(
    labels: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    *,
    seed: int,
    samples: int = 10_000,
) -> dict[str, dict[str, dict[str, float]]]:
    """Paired query bootstrap for Top-K support coverage and recall."""

    ks = (1, 3, 5, 10)
    differences = np.empty((len(labels), len(ks), 2), dtype=np.float64)
    for query_index, (query_labels, left_scores, right_scores) in enumerate(
        zip(labels, left, right, strict=True)
    ):
        support_total = query_labels.sum()
        for k_index, k in enumerate(ks):
            left_kept = np.argsort(-left_scores, kind="stable")[:k]
            right_kept = np.argsort(-right_scores, kind="stable")[:k]
            differences[query_index, k_index, 0] = float(
                query_labels[left_kept].any()
            ) - float(query_labels[right_kept].any())
            differences[query_index, k_index, 1] = (
                query_labels[left_kept].sum() - query_labels[right_kept].sum()
            ) / support_total
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(labels), size=(samples, len(labels)))
    boot = differences[draws].mean(axis=1)
    output = {}
    for k_index, k in enumerate(ks):
        output[f"top_{k}"] = {}
        for metric_index, name in enumerate(
            ("query_support_coverage", "mean_support_recall")
        ):
            output[f"top_{k}"][name] = {
                "mean": float(differences[:, k_index, metric_index].mean()),
                "ci95_low": float(np.quantile(boot[:, k_index, metric_index], 0.025)),
                "ci95_high": float(np.quantile(boot[:, k_index, metric_index], 0.975)),
            }
    return output


def main() -> None:
    args = parse_args()
    train = np.load(args.train)
    test = np.load(args.test)
    if not np.array_equal(train["layer_ids"], test["layer_ids"]):
        raise ValueError("Train and test layer IDs differ")
    layers = train["layer_ids"].astype(int).tolist()
    train_features = train["features"].astype(np.float32)
    test_features = test["features"].astype(np.float32)
    train_labels = train["labels"].astype(bool)
    test_labels = test["labels"].astype(bool)
    groups = np.repeat(np.arange(len(train_labels)), train_labels.shape[1])
    folds = list(GroupKFold(5).split(groups, train_labels.ravel(), groups))
    c_values = (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)

    candidates = []
    for layer_index, layer in enumerate(layers):
        x = train_features[:, :, layer_index, :].reshape(-1, train_features.shape[-1])
        for c in c_values:
            oof = np.zeros(train_labels.size, dtype=np.float64)
            for train_indices, held_indices in folds:
                probe = make_probe(c).fit(x[train_indices], train_labels.ravel()[train_indices])
                oof[held_indices] = probe.predict_proba(x[held_indices])[:, 1]
            scores = oof.reshape(train_labels.shape)
            metrics = query_metrics(train_labels, scores)
            candidates.append(
                {
                    "layer_index": layer_index,
                    "layer": layer,
                    "c": c,
                    "metrics": metrics,
                    "scores": scores,
                }
            )
    selected = max(
        candidates,
        key=lambda row: (row["metrics"]["mean_query_ap"], row["metrics"]["mrr"]),
    )
    layer_index, selected_c = int(selected["layer_index"]), float(selected["c"])
    train_x = train_features[:, :, layer_index, :].reshape(-1, train_features.shape[-1])
    test_x = test_features[:, :, layer_index, :].reshape(-1, test_features.shape[-1])
    probe = make_probe(selected_c).fit(train_x, train_labels.ravel())
    probe_test = probe.predict_proba(test_x)[:, 1].reshape(test_labels.shape)
    probe_oof = np.asarray(selected["scores"])

    train_bge = train["bge_scores"].astype(np.float64)
    test_bge = test["bge_scores"].astype(np.float64)
    train_lm = train["lm_relevance_scores"].astype(np.float64)
    test_lm = test["lm_relevance_scores"].astype(np.float64)
    weights = (0.0, 0.01, 0.025, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)

    def select_weight(extra: np.ndarray) -> float:
        choices = []
        for weight in weights:
            score = query_z(train_bge) + weight * query_z(extra)
            metrics = query_metrics(train_labels, score)
            choices.append((metrics["mean_query_ap"], metrics["mrr"], weight))
        return float(max(choices)[2])

    lm_weight = select_weight(train_lm)
    probe_weight = select_weight(probe_oof)
    lm_fusion_test = query_z(test_bge) + lm_weight * query_z(test_lm)
    probe_fusion_test = query_z(test_bge) + probe_weight * query_z(probe_test)
    multi_choices = []
    for lm_candidate in weights:
        for probe_candidate in weights:
            score = (
                query_z(train_bge)
                + lm_candidate * query_z(train_lm)
                + probe_candidate * query_z(probe_oof)
            )
            metrics = query_metrics(train_labels, score)
            multi_choices.append(
                (metrics["mean_query_ap"], metrics["mrr"], lm_candidate, probe_candidate)
            )
    _, _, multi_lm_weight, multi_probe_weight = max(multi_choices)
    multi_fusion_test = (
        query_z(test_bge)
        + multi_lm_weight * query_z(test_lm)
        + multi_probe_weight * query_z(probe_test)
    )

    scores = {
        "bge": test_bge,
        "zero_shot_lm_head": test_lm,
        "zero_shot_lm_head_fusion": lm_fusion_test,
        "supervised_hidden_probe": probe_test,
        "supervised_hidden_probe_fusion": probe_fusion_test,
        "three_signal_fusion": multi_fusion_test,
    }
    test_output = {}
    for offset, (name, values) in enumerate(scores.items()):
        test_output[name] = evaluate(test_labels, values)
        if name != "bge":
            test_output[name]["minus_bge"] = bootstrap_difference(
                test_labels, values, test_bge, seed=args.seed + offset
            )
            test_output[name]["retention_minus_bge"] = bootstrap_retention_difference(
                test_labels,
                values,
                test_bge,
                seed=args.seed + 100 + offset,
            )

    output = {
        "status": "complete",
        "method": "explicit_pairwise_relevance_prompt",
        "train_queries": int(len(train_labels)),
        "test_queries": int(len(test_labels)),
        "selected_layer": layers[layer_index],
        "selected_c": selected_c,
        "selected_lm_fusion_weight": lm_weight,
        "selected_probe_fusion_weight": probe_weight,
        "selected_three_signal_weights": {
            "lm_head": multi_lm_weight,
            "hidden_probe": multi_probe_weight,
        },
        "train_oof": {
            "bge": evaluate(train_labels, train_bge),
            "zero_shot_lm_head": evaluate(train_labels, train_lm),
            "supervised_hidden_probe": evaluate(train_labels, probe_oof),
            "three_signal_fusion": evaluate(
                train_labels,
                query_z(train_bge)
                + multi_lm_weight * query_z(train_lm)
                + multi_probe_weight * query_z(probe_oof),
            ),
        },
        "test": test_output,
        "selection_candidates": [
            {
                "layer": row["layer"],
                "c": row["c"],
                "metrics": row["metrics"],
            }
            for row in candidates
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    printable = dict(output)
    printable.pop("selection_candidates")
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        main()
