"""Build conformal chunk keep-sets from the pairwise internal-state fusion."""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

from research.internal_state_rag.analyze_hidden_chunk_probe import query_z
from research.internal_state_rag.analyze_pairwise_relevance_probe import make_probe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--fusion-analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alphas", default="0.2,0.1,0.05")
    parser.add_argument("--seed", type=int, default=467)
    return parser.parse_args()


def conformal_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    alpha: float,
    coverage_target: str,
) -> tuple[float, int]:
    """Calibrate a score threshold with the finite-sample split-conformal rule."""

    if coverage_target == "any_support":
        critical = np.asarray([row[y].max() for row, y in zip(scores, labels, strict=True)])
    elif coverage_target == "all_support":
        critical = np.asarray([row[y].min() for row, y in zip(scores, labels, strict=True)])
    else:
        raise ValueError(f"Unknown coverage target {coverage_target}")
    nonconformity = np.sort(-critical)
    order = min(len(nonconformity), math.ceil((len(nonconformity) + 1) * (1 - alpha)))
    return float(-nonconformity[order - 1]), order


def keep_mask(scores: np.ndarray, threshold: float) -> np.ndarray:
    mask = scores >= threshold
    empty = ~mask.any(axis=1)
    if empty.any():
        best = np.argmax(scores[empty], axis=1)
        mask[np.flatnonzero(empty), best] = True
    return mask


def keep_metrics(labels: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    kept = mask.sum(axis=1)
    retained_support = (labels & mask).sum(axis=1)
    support_total = labels.sum(axis=1)
    return {
        "mean_chunks_kept": float(kept.mean()),
        "median_chunks_kept": float(np.median(kept)),
        "p90_chunks_kept": float(np.quantile(kept, 0.9)),
        "fraction_of_top_l_kept": float(mask.mean()),
        "chunk_precision_among_kept": float((labels & mask).sum() / mask.sum()),
        "mean_support_recall": float(np.mean(retained_support / support_total)),
        "query_any_support_coverage": float(np.mean(retained_support > 0)),
        "query_all_support_coverage": float(np.mean(retained_support == support_total)),
    }


def paired_bootstrap(
    labels: np.ndarray,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    *,
    seed: int,
    samples: int = 10_000,
) -> dict[str, dict[str, float]]:
    support_total = labels.sum(axis=1)
    left_support = (labels & left_mask).sum(axis=1)
    right_support = (labels & right_mask).sum(axis=1)
    per_query = np.column_stack(
        [
            left_mask.sum(axis=1) - right_mask.sum(axis=1),
            left_support / support_total - right_support / support_total,
            (left_support > 0).astype(float) - (right_support > 0).astype(float),
            (left_support == support_total).astype(float)
            - (right_support == support_total).astype(float),
        ]
    )
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(labels), size=(samples, len(labels)))
    boot = per_query[draws].mean(axis=1)
    names = (
        "mean_chunks_kept",
        "mean_support_recall",
        "query_any_support_coverage",
        "query_all_support_coverage",
    )
    return {
        name: {
            "mean": float(per_query[:, index].mean()),
            "ci95_low": float(np.quantile(boot[:, index], 0.025)),
            "ci95_high": float(np.quantile(boot[:, index], 0.975)),
        }
        for index, name in enumerate(names)
    }


def probe_scores(
    train: np.lib.npyio.NpzFile,
    test: np.lib.npyio.NpzFile,
    *,
    layer: int,
    c: float,
) -> tuple[np.ndarray, np.ndarray]:
    layer_index = train["layer_ids"].astype(int).tolist().index(layer)
    train_x = train["features"][:, :, layer_index, :].astype(np.float32)
    test_x = test["features"][:, :, layer_index, :].astype(np.float32)
    train_labels = train["labels"].astype(bool)
    flat_x = train_x.reshape(-1, train_x.shape[-1])
    flat_labels = train_labels.ravel()
    groups = np.repeat(np.arange(len(train_labels)), train_labels.shape[1])
    oof = np.empty(len(flat_labels), dtype=np.float64)
    for train_indices, held_indices in GroupKFold(5).split(groups, flat_labels, groups):
        model = make_probe(c).fit(flat_x[train_indices], flat_labels[train_indices])
        oof[held_indices] = model.predict_proba(flat_x[held_indices])[:, 1]
    model = make_probe(c).fit(flat_x, flat_labels)
    test_prediction = model.predict_proba(
        test_x.reshape(-1, test_x.shape[-1])
    )[:, 1]
    return oof.reshape(train_labels.shape), test_prediction.reshape(test_x.shape[:2])


def main() -> None:
    args = parse_args()
    train = np.load(args.train)
    test = np.load(args.test)
    analysis = json.loads(args.fusion_analysis.read_text(encoding="utf-8"))
    alphas = [float(value) for value in args.alphas.split(",")]
    layer, c = int(analysis["selected_layer"]), float(analysis["selected_c"])
    weights = analysis["selected_three_signal_weights"]
    train_probe, test_probe = probe_scores(train, test, layer=layer, c=c)
    train_scores = {
        "bge": query_z(train["bge_scores"].astype(np.float64)),
        "three_signal_fusion": (
            query_z(train["bge_scores"].astype(np.float64))
            + float(weights["lm_head"])
            * query_z(train["lm_relevance_scores"].astype(np.float64))
            + float(weights["hidden_probe"]) * query_z(train_probe)
        ),
    }
    test_scores = {
        "bge": query_z(test["bge_scores"].astype(np.float64)),
        "three_signal_fusion": (
            query_z(test["bge_scores"].astype(np.float64))
            + float(weights["lm_head"])
            * query_z(test["lm_relevance_scores"].astype(np.float64))
            + float(weights["hidden_probe"]) * query_z(test_probe)
        ),
    }
    train_labels = train["labels"].astype(bool)
    test_labels = test["labels"].astype(bool)
    results = {}
    for target in ("any_support", "all_support"):
        results[target] = {}
        for alpha_index, alpha in enumerate(alphas):
            row = {}
            masks = {}
            for name in train_scores:
                threshold, order = conformal_threshold(
                    train_scores[name], train_labels, alpha=alpha, coverage_target=target
                )
                mask = keep_mask(test_scores[name], threshold)
                masks[name] = mask
                row[name] = {
                    "calibrated_threshold": threshold,
                    "finite_sample_order": order,
                    "test": keep_metrics(test_labels, mask),
                }
            row["fusion_minus_bge"] = paired_bootstrap(
                test_labels,
                masks["three_signal_fusion"],
                masks["bge"],
                seed=args.seed + alpha_index + (100 if target == "all_support" else 0),
            )
            results[target][str(alpha)] = row
    output = {
        "status": "complete",
        "method": "split_conformal_threshold_on_query_critical_support_score",
        "calibration_note": (
            "Probe predictions are group-OOF on the 96 calibration queries. A separate "
            "held-out calibration bank is required for a formal deployment guarantee."
        ),
        "calibration_queries": int(len(train_labels)),
        "test_queries": int(len(test_labels)),
        "top_l": int(test_labels.shape[1]),
        "probe_layer": layer,
        "probe_c": c,
        "fusion_weights": weights,
        "alphas": alphas,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        main()
