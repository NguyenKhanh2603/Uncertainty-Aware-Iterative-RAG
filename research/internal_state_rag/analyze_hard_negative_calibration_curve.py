"""Measure hard-negative probe scaling and conformal calibration sample efficiency."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import warnings
from pathlib import Path
from typing import Any

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


HARD_NEGATIVE_COUNTS: tuple[int | None, ...] = (None, 3, 5, 10, 15)
LM_WEIGHTS = (0.0, 0.05, 0.1, 0.2, 0.35)
HIDDEN_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--test-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=733)
    parser.add_argument("--sweep-seed", type=int, default=1291)
    parser.add_argument("--layer", type=int, default=30)
    parser.add_argument("--c", type=float, default=0.1)
    parser.add_argument("--train-sizes", default="32,64,128,256,452")
    parser.add_argument("--calibration-sizes", default="20,32,64,128,256,363")
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--alphas", default="0.2,0.1,0.05")
    return parser.parse_args()


def hard_negative_mask(
    labels: np.ndarray, bge_scores: np.ndarray, hard_negatives: int | None
) -> np.ndarray:
    """Keep all supports and either all or each query's highest-BGE negatives."""

    labels = np.asarray(labels, dtype=bool)
    if hard_negatives is None:
        return np.ones_like(labels, dtype=bool)
    if hard_negatives < 1:
        raise ValueError("hard_negatives must be positive or None")
    mask = labels.copy()
    for row_index, (row_labels, row_scores) in enumerate(
        zip(labels, bge_scores, strict=True)
    ):
        negatives = np.flatnonzero(~row_labels)
        order = negatives[np.argsort(-row_scores[negatives], kind="stable")]
        mask[row_index, order[:hard_negatives]] = True
    return mask


def fit_probe(
    x: np.ndarray,
    labels: np.ndarray,
    bge_scores: np.ndarray,
    *,
    hard_negatives: int | None,
    c: float,
) -> object:
    chosen = hard_negative_mask(labels, bge_scores, hard_negatives)
    return make_probe(c).fit(x[chosen], labels[chosen])


def oof_probe_scores(
    x: np.ndarray,
    labels: np.ndarray,
    bge_scores: np.ndarray,
    *,
    hard_negatives: int | None,
    c: float,
) -> np.ndarray:
    groups = np.arange(len(labels))
    folds = GroupKFold(5).split(groups, groups=groups)
    result = np.zeros_like(bge_scores, dtype=np.float64)
    for fit_queries, held_queries in folds:
        model = fit_probe(
            x[fit_queries],
            labels[fit_queries],
            bge_scores[fit_queries],
            hard_negatives=hard_negatives,
            c=c,
        )
        held_x = x[held_queries]
        result[held_queries] = model.predict_proba(
            held_x.reshape(-1, held_x.shape[-1])
        )[:, 1].reshape(len(held_queries), held_x.shape[1])
    return result


def combine_scores(
    bge: np.ndarray,
    lm_head: np.ndarray,
    hidden: np.ndarray,
    *,
    lm_weight: float,
    hidden_weight: float,
) -> np.ndarray:
    return (
        query_z(bge)
        + lm_weight * query_z(lm_head)
        + hidden_weight * query_z(hidden)
    )


def tune_fusion(
    labels: np.ndarray,
    bge: np.ndarray,
    lm_head: np.ndarray,
    hidden: np.ndarray,
) -> dict[str, float]:
    retrievable = labels.any(axis=1)
    candidates = []
    for lm_weight in LM_WEIGHTS:
        for hidden_weight in HIDDEN_WEIGHTS:
            scores = combine_scores(
                bge,
                lm_head,
                hidden,
                lm_weight=lm_weight,
                hidden_weight=hidden_weight,
            )
            candidates.append(
                {
                    "lm_head": lm_weight,
                    "hidden_probe": hidden_weight,
                    **query_metrics(labels[retrievable], scores[retrievable]),
                }
            )
    return max(
        candidates,
        key=lambda row: (
            row["mean_query_ap"],
            row["mrr"],
            -(row["lm_head"] + row["hidden_probe"]),
        ),
    )


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "p025": float(np.quantile(array, 0.025)),
        "p50": float(np.quantile(array, 0.5)),
        "p975": float(np.quantile(array, 0.975)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def subset_hash(qids: np.ndarray) -> str:
    payload = "\n".join(sorted(str(qid) for qid in qids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def calibration_sweep(
    labels: np.ndarray,
    qids: np.ndarray,
    calibration_scores: dict[str, np.ndarray],
    test_labels: np.ndarray,
    test_scores: dict[str, np.ndarray],
    *,
    sizes: list[int],
    alphas: list[float],
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    retrievable_indices = np.flatnonzero(labels.any(axis=1))
    retrievable_labels = labels[retrievable_indices]
    retrievable_qids = qids[retrievable_indices]
    retrievable_scores = {
        name: scores[retrievable_indices] for name, scores in calibration_scores.items()
    }
    test_retrievable = test_labels.any(axis=1)
    test_labels = test_labels[test_retrievable]
    test_scores = {
        name: scores[test_retrievable] for name, scores in test_scores.items()
    }
    pool_size = len(retrievable_indices)
    if not sizes or min(sizes) < 1 or max(sizes) > pool_size:
        raise ValueError(f"Calibration sizes must be within [1, {pool_size}]")
    rng = np.random.default_rng(seed)
    permutations = [rng.permutation(pool_size) for _ in range(repeats)]
    result: dict[str, Any] = {
        "seed": seed,
        "repeats": repeats,
        "nested_subsets_within_repeat": True,
        "retrievable_pool_queries": pool_size,
        "retrievable_pool_qid_sha256": subset_hash(retrievable_qids),
        "sizes": {},
    }
    metric_names = (
        "mean_chunks_kept",
        "chunk_precision_among_kept",
        "micro_support_recall",
        "mean_support_recall",
        "query_any_support_coverage",
        "query_all_support_coverage",
    )
    for size in sizes:
        repeat_count = 1 if size == pool_size else repeats
        rows = []
        for repeat_index, permutation in enumerate(permutations[:repeat_count]):
            chosen = permutation[:size] if size < pool_size else np.arange(pool_size)
            row: dict[str, Any] = {
                "repeat": repeat_index,
                "calibration_queries": size,
                "subset_qid_sha256": subset_hash(retrievable_qids[chosen]),
                "alphas": {},
            }
            for alpha in alphas:
                alpha_row = {}
                for method in calibration_scores:
                    threshold, order = conformal_threshold(
                        retrievable_scores[method][chosen],
                        retrievable_labels[chosen],
                        alpha=alpha,
                        coverage_target="all_support",
                    )
                    metrics = conditional_metrics(
                        test_labels, keep_mask(test_scores[method], threshold)
                    )
                    alpha_row[method] = {
                        "threshold": threshold,
                        "finite_sample_order": order,
                        **metrics,
                    }
                row["alphas"][str(alpha)] = alpha_row
            rows.append(row)
        aggregates: dict[str, Any] = {}
        for alpha in alphas:
            aggregates[str(alpha)] = {}
            for method in calibration_scores:
                method_rows = [row["alphas"][str(alpha)][method] for row in rows]
                aggregates[str(alpha)][method] = {
                    "threshold": summarize([row["threshold"] for row in method_rows]),
                    **{
                        metric: summarize([row[metric] for row in method_rows])
                        for metric in metric_names
                    },
                    "fraction_meeting_nominal_all_support_coverage": float(
                        np.mean(
                            [
                                row["query_all_support_coverage"] >= 1 - alpha
                                for row in method_rows
                            ]
                        )
                    ),
                    "finite_sample_order": sorted(
                        {int(row["finite_sample_order"]) for row in method_rows}
                    ),
                }
        result["sizes"][str(size)] = {"aggregate": aggregates, "repeat_log": rows}
    return result


def main() -> None:
    args = parse_args()
    source = np.load(args.features)
    test = np.load(args.test_features)
    layer_ids = source["layer_ids"].astype(int).tolist()
    if test["layer_ids"].astype(int).tolist() != layer_ids:
        raise ValueError("Feature files use different layer IDs")
    layer_index = layer_ids.index(args.layer)
    source_x = source["features"][:, :, layer_index, :].astype(np.float32)
    source_y = source["labels"].astype(bool)
    source_bge = source["bge_scores"].astype(np.float64)
    source_lm = source["lm_relevance_scores"].astype(np.float64)
    test_x = test["features"][:, :, layer_index, :].astype(np.float32)
    test_y = test["labels"].astype(bool)
    test_bge = test["bge_scores"].astype(np.float64)
    test_lm = test["lm_relevance_scores"].astype(np.float64)

    permutation = np.random.default_rng(args.seed).permutation(len(source_y))
    split = len(source_y) // 2
    train_indices = permutation[:split]
    calibration_indices = permutation[split:]
    train_x, train_y = source_x[train_indices], source_y[train_indices]
    train_bge, train_lm = source_bge[train_indices], source_lm[train_indices]
    calibration_x, calibration_y = (
        source_x[calibration_indices],
        source_y[calibration_indices],
    )

    mining_selection = []
    for hard_count in HARD_NEGATIVE_COUNTS:
        oof = oof_probe_scores(
            train_x,
            train_y,
            train_bge,
            hard_negatives=hard_count,
            c=args.c,
        )
        selected_weights = tune_fusion(train_y, train_bge, train_lm, oof)
        mining_selection.append(
            {
                "name": "all_negatives" if hard_count is None else f"top_{hard_count}_bge_negatives",
                "hard_negatives_per_query": hard_count,
                "oof_probe_ranking": query_metrics(
                    train_y[train_y.any(axis=1)], oof[train_y.any(axis=1)]
                ),
                "selected_weights": selected_weights,
            }
        )
    selected = max(
        mining_selection,
        key=lambda row: (
            row["selected_weights"]["mean_query_ap"],
            row["selected_weights"]["mrr"],
            math.inf
            if row["hard_negatives_per_query"] is None
            else -row["hard_negatives_per_query"],
        ),
    )
    selected_hard_count = selected["hard_negatives_per_query"]
    weights = selected["selected_weights"]

    model = fit_probe(
        train_x,
        train_y,
        train_bge,
        hard_negatives=selected_hard_count,
        c=args.c,
    )
    calibration_hidden = model.predict_proba(
        calibration_x.reshape(-1, calibration_x.shape[-1])
    )[:, 1].reshape(calibration_y.shape)
    test_hidden = model.predict_proba(test_x.reshape(-1, test_x.shape[-1]))[:, 1].reshape(
        test_y.shape
    )
    calibration_fusion = combine_scores(
        source_bge[calibration_indices],
        source_lm[calibration_indices],
        calibration_hidden,
        lm_weight=float(weights["lm_head"]),
        hidden_weight=float(weights["hidden_probe"]),
    )
    test_fusion = combine_scores(
        test_bge,
        test_lm,
        test_hidden,
        lm_weight=float(weights["lm_head"]),
        hidden_weight=float(weights["hidden_probe"]),
    )

    train_sizes = [int(value) for value in args.train_sizes.split(",")]
    learning_curve = []
    for train_size in train_sizes:
        if not 1 <= train_size <= len(train_y):
            raise ValueError(f"Invalid train size: {train_size}")
        size_model = fit_probe(
            train_x[:train_size],
            train_y[:train_size],
            train_bge[:train_size],
            hard_negatives=selected_hard_count,
            c=args.c,
        )
        size_test_hidden = size_model.predict_proba(
            test_x.reshape(-1, test_x.shape[-1])
        )[:, 1].reshape(test_y.shape)
        size_test_fusion = combine_scores(
            test_bge,
            test_lm,
            size_test_hidden,
            lm_weight=float(weights["lm_head"]),
            hidden_weight=float(weights["hidden_probe"]),
        )
        learning_curve.append(
            {
                "probe_training_queries": train_size,
                "retrievable_training_queries": int(
                    train_y[:train_size].any(axis=1).sum()
                ),
                "selected_training_chunks": int(
                    hard_negative_mask(
                        train_y[:train_size],
                        train_bge[:train_size],
                        selected_hard_count,
                    ).sum()
                ),
                "probe_test_ranking": fixed_k_metrics(test_y, size_test_hidden),
                "fusion_test_ranking": fixed_k_metrics(test_y, size_test_fusion),
            }
        )

    calibration_scores = {
        "bge": query_z(source_bge[calibration_indices]),
        "hard_negative_fusion": calibration_fusion,
    }
    test_scores = {"bge": query_z(test_bge), "hard_negative_fusion": test_fusion}
    alphas = [float(value) for value in args.alphas.split(",")]
    calibration_sizes = [int(value) for value in args.calibration_sizes.split(",")]
    sweep = calibration_sweep(
        calibration_y,
        source["qids"][calibration_indices],
        calibration_scores,
        test_y,
        test_scores,
        sizes=calibration_sizes,
        alphas=alphas,
        repeats=args.repeats,
        seed=args.sweep_seed,
    )
    output = {
        "status": "complete",
        "method": "hard_negative_hidden_probe_and_calibration_sample_efficiency",
        "split": {
            "seed": args.seed,
            "probe_train_queries": len(train_indices),
            "probe_train_retrievable": int(train_y.any(axis=1).sum()),
            "calibration_queries": len(calibration_indices),
            "calibration_retrievable": int(calibration_y.any(axis=1).sum()),
            "test_queries": len(test_y),
            "test_retrievable": int(test_y.any(axis=1).sum()),
            "train_calibration_overlap": int(
                len(
                    set(source["qids"][train_indices].tolist())
                    & set(source["qids"][calibration_indices].tolist())
                )
            ),
        },
        "probe": {"layer": args.layer, "c": args.c},
        "hard_negative_selection_oof": mining_selection,
        "selected_hard_negative_configuration": selected,
        "test_ranking": {
            "bge": fixed_k_metrics(test_y, test_scores["bge"]),
            "hard_negative_probe": fixed_k_metrics(test_y, test_hidden),
            "hard_negative_fusion": fixed_k_metrics(
                test_y, test_scores["hard_negative_fusion"]
            ),
        },
        "probe_training_learning_curve": learning_curve,
        "calibration_sample_efficiency": sweep,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.predictions,
        calibration_qids=source["qids"][calibration_indices],
        calibration_labels=calibration_y,
        test_qids=test["qids"],
        test_labels=test_y,
        calibration_bge=calibration_scores["bge"].astype(np.float32),
        calibration_fusion=calibration_fusion.astype(np.float32),
        test_bge=test_scores["bge"].astype(np.float32),
        test_fusion=test_fusion.astype(np.float32),
        selected_hard_negatives=np.asarray(
            -1 if selected_hard_count is None else selected_hard_count
        ),
    )
    compact = {
        "selected": selected,
        "test_ranking": output["test_ranking"],
        "learning_curve": [
            {
                "probe_training_queries": row["probe_training_queries"],
                "fusion_ranking": row["fusion_test_ranking"]["ranking"],
            }
            for row in learning_curve
        ],
        "calibration_aggregate": {
            size: row["aggregate"] for size, row in sweep["sizes"].items()
        },
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        main()
