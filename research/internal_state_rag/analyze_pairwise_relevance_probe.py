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

    scores = {
        "bge": test_bge,
        "zero_shot_lm_head": test_lm,
        "zero_shot_lm_head_fusion": lm_fusion_test,
        "supervised_hidden_probe": probe_test,
        "supervised_hidden_probe_fusion": probe_fusion_test,
    }
    test_output = {}
    for offset, (name, values) in enumerate(scores.items()):
        test_output[name] = evaluate(test_labels, values)
        if name != "bge":
            test_output[name]["minus_bge"] = bootstrap_difference(
                test_labels, values, test_bge, seed=args.seed + offset
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
        "train_oof": {
            "bge": evaluate(train_labels, train_bge),
            "zero_shot_lm_head": evaluate(train_labels, train_lm),
            "supervised_hidden_probe": evaluate(train_labels, probe_oof),
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
