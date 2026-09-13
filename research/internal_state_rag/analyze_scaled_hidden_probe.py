"""Train the hidden relevance probe on a larger disjoint split and recalibrate."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

from research.internal_state_rag.analyze_hidden_chunk_probe import (
    make_probe,
    query_metrics,
    query_z,
)
from research.internal_state_rag.analyze_pairwise_conformal_clean_split import (
    conditional_metrics,
    conformal_threshold,
    fixed_k_metrics,
    keep_mask,
)


WEIGHTS = (0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-features", type=Path, required=True)
    parser.add_argument("--test-features", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=733)
    parser.add_argument("--layer", type=int, default=30)
    parser.add_argument("--c", type=float, default=0.1)
    parser.add_argument("--alphas", default="0.2,0.1,0.05")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = np.load(args.calibration_features)
    test = np.load(args.test_features)
    layers = source["layer_ids"].astype(int).tolist()
    layer_index = layers.index(args.layer)
    n_queries, top_l = source["labels"].shape
    permutation = np.random.default_rng(args.seed).permutation(n_queries)
    split = n_queries // 2
    train_indices = permutation[:split]
    calibration_indices = permutation[split:]

    source_x = source["features"][:, :, layer_index, :].astype(np.float32)
    source_y = source["labels"].astype(bool)
    test_x = test["features"][:, :, layer_index, :].astype(np.float32)
    test_y = test["labels"].astype(bool)
    train_x = source_x[train_indices]
    train_y = source_y[train_indices]
    groups = np.repeat(np.arange(len(train_indices)), top_l)
    oof = np.zeros(train_y.size, dtype=np.float64)
    for fit_rows, held_rows in GroupKFold(5).split(
        train_x.reshape(-1, train_x.shape[-1]), train_y.ravel(), groups
    ):
        model = make_probe(args.c).fit(
            train_x.reshape(-1, train_x.shape[-1])[fit_rows], train_y.ravel()[fit_rows]
        )
        oof[held_rows] = model.predict_proba(
            train_x.reshape(-1, train_x.shape[-1])[held_rows]
        )[:, 1]
    oof = oof.reshape(train_y.shape)
    final_probe = make_probe(args.c).fit(
        train_x.reshape(-1, train_x.shape[-1]), train_y.ravel()
    )
    calibration_probe = final_probe.predict_proba(
        source_x[calibration_indices].reshape(-1, source_x.shape[-1])
    )[:, 1].reshape(len(calibration_indices), top_l)
    test_probe = final_probe.predict_proba(
        test_x.reshape(-1, test_x.shape[-1])
    )[:, 1].reshape(test_y.shape)

    train_signals = {
        "bge": query_z(source["bge_scores"][train_indices].astype(np.float64)),
        "lm_head": query_z(
            source["lm_relevance_scores"][train_indices].astype(np.float64)
        ),
        "hidden": query_z(oof),
    }
    calibration_signals = {
        "bge": query_z(
            source["bge_scores"][calibration_indices].astype(np.float64)
        ),
        "lm_head": query_z(
            source["lm_relevance_scores"][calibration_indices].astype(np.float64)
        ),
        "hidden": query_z(calibration_probe),
    }
    test_signals = {
        "bge": query_z(test["bge_scores"].astype(np.float64)),
        "lm_head": query_z(test["lm_relevance_scores"].astype(np.float64)),
        "hidden": query_z(test_probe),
    }
    train_retrievable = train_y.any(axis=1)
    tuning = []
    for lm_weight in WEIGHTS:
        for hidden_weight in WEIGHTS:
            score = (
                train_signals["bge"]
                + lm_weight * train_signals["lm_head"]
                + hidden_weight * train_signals["hidden"]
            )
            metrics = query_metrics(
                train_y[train_retrievable], score[train_retrievable]
            )
            tuning.append(
                {
                    "lm_head": lm_weight,
                    "hidden_probe": hidden_weight,
                    **metrics,
                }
            )
    selected = max(
        tuning,
        key=lambda row: (
            row["mean_query_ap"],
            row["mrr"],
            -(row["lm_head"] + row["hidden_probe"]),
        ),
    )
    score_configs = {
        "bge": {"lm_head": 0.0, "hidden_probe": 0.0},
        "fixed_96_query_weights": {"lm_head": 0.2, "hidden_probe": 0.75},
        "retuned_452_query_weights": {
            "lm_head": float(selected["lm_head"]),
            "hidden_probe": float(selected["hidden_probe"]),
        },
    }

    def combine(signals: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
        return (
            signals["bge"]
            + weights["lm_head"] * signals["lm_head"]
            + weights["hidden_probe"] * signals["hidden"]
        )

    cal_y = source_y[calibration_indices]
    cal_retrievable = cal_y.any(axis=1)
    test_retrievable = test_y.any(axis=1)
    calibration_scores = {
        name: combine(calibration_signals, weights)
        for name, weights in score_configs.items()
    }
    test_scores = {
        name: combine(test_signals, weights) for name, weights in score_configs.items()
    }
    results = {}
    for alpha in [float(value) for value in args.alphas.split(",")]:
        results[str(alpha)] = {}
        for name in score_configs:
            threshold, order = conformal_threshold(
                calibration_scores[name][cal_retrievable],
                cal_y[cal_retrievable],
                alpha=alpha,
                coverage_target="all_support",
            )
            mask = keep_mask(test_scores[name], threshold)
            results[str(alpha)][name] = {
                "threshold": threshold,
                "finite_sample_order": order,
                "conditional_on_retrievable": conditional_metrics(
                    test_y[test_retrievable], mask[test_retrievable]
                ),
            }

    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    output = {
        "status": "complete",
        "method": "larger_disjoint_hidden_probe_training_split",
        "split": {
            "probe_training_queries_total": int(len(train_indices)),
            "probe_training_queries_retrievable": int(train_retrievable.sum()),
            "conformal_queries_total": int(len(calibration_indices)),
            "conformal_queries_retrievable": int(cal_retrievable.sum()),
            "test_queries_total": int(len(test_y)),
            "test_queries_retrievable": int(test_retrievable.sum()),
            "query_overlap": int(
                len(
                    set(source["qids"][train_indices].tolist())
                    & set(source["qids"][calibration_indices].tolist())
                )
            ),
        },
        "probe": {"layer": args.layer, "c": args.c},
        "selected_weights": selected,
        "test_ranking": {
            name: fixed_k_metrics(test_y, score) for name, score in test_scores.items()
        },
        "conformal": results,
        "reference_96_probe_714_conformal": {
            alpha: reference["conformal"]["all_support"][alpha][
                "three_signal_fusion"
            ]
            for alpha in ("0.2", "0.1", "0.05")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        main()
