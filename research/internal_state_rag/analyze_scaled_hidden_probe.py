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
    parser.add_argument("--old-probe-train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=733)
    parser.add_argument("--layer", type=int, default=30)
    parser.add_argument("--c", type=float, default=0.1)
    parser.add_argument("--alphas", default="0.2,0.1,0.05")
    return parser.parse_args()


def paired_bootstrap(
    labels: np.ndarray,
    new_mask: np.ndarray,
    old_mask: np.ndarray,
    *,
    seed: int,
    samples: int = 10_000,
) -> dict[str, dict[str, float]]:
    """Bootstrap query-level differences, including micro precision and recall."""

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(labels), size=(samples, len(labels)))

    support_per_query = labels.sum(axis=1)

    def query_values(mask: np.ndarray) -> tuple[np.ndarray, ...]:
        retained = (labels & mask).sum(axis=1)
        return (
            mask.sum(axis=1),
            retained,
            support_per_query,
            retained == support_per_query,
        )

    def statistics(mask: np.ndarray, indices: np.ndarray) -> np.ndarray:
        kept_q, retained_q, support_q, coverage_q = query_values(mask)
        retained = retained_q[indices].sum(axis=1)
        kept = kept_q[indices].sum(axis=1)
        support = support_q[indices].sum(axis=1)
        all_coverage = coverage_q[indices].mean(axis=1)
        return np.column_stack(
            [
                kept_q[indices].mean(axis=1),
                retained / kept,
                retained / support,
                all_coverage,
            ]
        )

    difference = statistics(new_mask, draws) - statistics(old_mask, draws)
    point_draw = np.arange(len(labels))[None, :]
    point = (statistics(new_mask, point_draw) - statistics(old_mask, point_draw))[0]
    names = (
        "mean_chunks_kept",
        "chunk_precision",
        "micro_support_recall",
        "query_all_support_coverage",
    )
    return {
        name: {
            "difference": float(point[index]),
            "ci95_low": float(np.quantile(difference[:, index], 0.025)),
            "ci95_high": float(np.quantile(difference[:, index], 0.975)),
        }
        for index, name in enumerate(names)
    }


def main() -> None:
    args = parse_args()
    source = np.load(args.calibration_features)
    test = np.load(args.test_features)
    old_train = np.load(args.old_probe_train)
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
    old_layer_index = old_train["layer_ids"].astype(int).tolist().index(args.layer)
    old_train_x = old_train["features"][:, :, old_layer_index, :].astype(np.float32)
    old_model = make_probe(args.c).fit(
        old_train_x.reshape(-1, old_train_x.shape[-1]),
        old_train["labels"].astype(bool).ravel(),
    )
    old_cal_probe = old_model.predict_proba(
        source_x.reshape(-1, source_x.shape[-1])
    )[:, 1].reshape(source_y.shape)
    old_test_probe = old_model.predict_proba(
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
    old_cal_score = (
        query_z(source["bge_scores"].astype(np.float64))
        + 0.2 * query_z(source["lm_relevance_scores"].astype(np.float64))
        + 0.75 * query_z(old_cal_probe)
    )
    old_test_score = (
        query_z(test["bge_scores"].astype(np.float64))
        + 0.2 * query_z(test["lm_relevance_scores"].astype(np.float64))
        + 0.75 * query_z(old_test_probe)
    )
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
    bootstrap_results = {}
    for alpha_index, alpha in enumerate(
        [float(value) for value in args.alphas.split(",")]
    ):
        results[str(alpha)] = {}
        masks = {}
        for name in score_configs:
            threshold, order = conformal_threshold(
                calibration_scores[name][cal_retrievable],
                cal_y[cal_retrievable],
                alpha=alpha,
                coverage_target="all_support",
            )
            mask = keep_mask(test_scores[name], threshold)
            masks[name] = mask
            results[str(alpha)][name] = {
                "threshold": threshold,
                "finite_sample_order": order,
                "conditional_on_retrievable": conditional_metrics(
                    test_y[test_retrievable], mask[test_retrievable]
                ),
            }
        old_threshold, _ = conformal_threshold(
            old_cal_score[source_y.any(axis=1)],
            source_y[source_y.any(axis=1)],
            alpha=alpha,
            coverage_target="all_support",
        )
        old_mask = keep_mask(old_test_score, old_threshold)
        bootstrap_results[str(alpha)] = paired_bootstrap(
            test_y[test_retrievable],
            masks["retuned_452_query_weights"][test_retrievable],
            old_mask[test_retrievable],
            seed=args.seed + alpha_index,
        )

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
        "retuned_452_minus_old_96_paired_bootstrap": bootstrap_results,
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
