"""Evaluate query-level conformal chunk pruning on disjoint train/cal/test splits.

The relevance probe and fusion weights are frozen from a prior probe-training run.
Conformal thresholds are calibrated only on queries whose Top-L candidate set contains
at least one labelled support chunk.  Results therefore report both conditional
selection quality and unconditional end-to-end coverage, including the retrieval
ceiling caused by queries with no support in Top-L.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path

import numpy as np

from research.internal_state_rag.analyze_causal_value_probe import retention_metrics
from research.internal_state_rag.analyze_hidden_chunk_probe import query_metrics, query_z
from research.internal_state_rag.analyze_pairwise_relevance_probe import make_probe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-train", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--fusion-analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alphas", default="0.2,0.1,0.05")
    parser.add_argument("--seed", type=int, default=487)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    return parser.parse_args()


def conformal_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    alpha: float,
    coverage_target: str,
) -> tuple[float, int]:
    """Return a finite-sample split-conformal keep threshold.

    Every calibration query must contain at least one support candidate.  For
    ``any_support`` the critical score is the best support; for ``all_support`` it
    is the worst support.  Keeping candidates at or above the returned threshold
    targets the corresponding query-level coverage event.
    """

    if len(scores) == 0 or np.any(labels.sum(axis=1) == 0):
        raise ValueError("Calibration rows must all contain support in Top-L")
    if coverage_target == "any_support":
        critical = np.asarray([row[y].max() for row, y in zip(scores, labels, strict=True)])
    elif coverage_target == "all_support":
        critical = np.asarray([row[y].min() for row, y in zip(scores, labels, strict=True)])
    else:
        raise ValueError(f"Unknown coverage target: {coverage_target}")
    nonconformity = np.sort(-critical)
    order = min(len(nonconformity), math.ceil((len(nonconformity) + 1) * (1 - alpha)))
    return float(-nonconformity[order - 1]), order


def keep_mask(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Apply a threshold with deterministic Top-1 fallback for an empty set."""

    mask = scores >= threshold
    empty = ~mask.any(axis=1)
    if empty.any():
        best = np.argmax(scores[empty], axis=1)
        mask[np.flatnonzero(empty), best] = True
    return mask


def conditional_metrics(labels: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Selection metrics on queries with at least one support in Top-L."""

    support_total = labels.sum(axis=1)
    if np.any(support_total == 0):
        raise ValueError("Conditional metrics require retrievable queries")
    retained = (labels & mask).sum(axis=1)
    kept = mask.sum(axis=1)
    return {
        "queries": int(len(labels)),
        "mean_chunks_kept": float(kept.mean()),
        "median_chunks_kept": float(np.median(kept)),
        "p90_chunks_kept": float(np.quantile(kept, 0.9)),
        "fraction_of_top_l_kept": float(mask.mean()),
        "chunk_precision_among_kept": float((labels & mask).sum() / mask.sum()),
        "mean_support_recall": float(np.mean(retained / support_total)),
        "query_any_support_coverage": float(np.mean(retained > 0)),
        "query_all_support_coverage": float(np.mean(retained == support_total)),
    }


def end_to_end_metrics(labels: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Metrics over every test query, counting unretrievable queries as failures."""

    retrievable = labels.any(axis=1)
    retained = (labels & mask).sum(axis=1)
    support_total = labels.sum(axis=1)
    return {
        "queries": int(len(labels)),
        "retrievable_queries": int(retrievable.sum()),
        "retrieval_query_coverage_ceiling": float(retrievable.mean()),
        "mean_chunks_kept": float(mask.sum(axis=1).mean()),
        "chunk_precision_among_kept": float((labels & mask).sum() / mask.sum()),
        "query_any_support_coverage": float(np.mean(retained > 0)),
        "query_all_support_coverage": float(
            np.mean(retrievable & (retained == support_total))
        ),
    }


def fixed_k_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, object]:
    """Ranking and Top-K retention metrics on retrievable queries."""

    retrievable = labels.any(axis=1)
    return {
        "queries": int(retrievable.sum()),
        "ranking": query_metrics(labels[retrievable], scores[retrievable]),
        "retention": retention_metrics(labels[retrievable], scores[retrievable]),
    }


def paired_bootstrap(
    labels: np.ndarray,
    fusion_mask: np.ndarray,
    bge_mask: np.ndarray,
    *,
    seed: int,
    samples: int,
) -> dict[str, dict[str, float]]:
    """Paired query bootstrap of fusion minus BGE on retrievable queries."""

    support_total = labels.sum(axis=1)
    if np.any(support_total == 0):
        raise ValueError("Bootstrap rows must be retrievable")
    fusion_support = (labels & fusion_mask).sum(axis=1)
    bge_support = (labels & bge_mask).sum(axis=1)
    per_query = np.column_stack(
        [
            fusion_mask.sum(axis=1) - bge_mask.sum(axis=1),
            fusion_support / support_total - bge_support / support_total,
            (fusion_support > 0).astype(float) - (bge_support > 0).astype(float),
            (fusion_support == support_total).astype(float)
            - (bge_support == support_total).astype(float),
        ]
    )
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(labels), size=(samples, len(labels)))
    bootstrap = per_query[draws].mean(axis=1)
    names = (
        "mean_chunks_kept",
        "mean_support_recall",
        "query_any_support_coverage",
        "query_all_support_coverage",
    )
    return {
        name: {
            "mean": float(per_query[:, index].mean()),
            "ci95_low": float(np.quantile(bootstrap[:, index], 0.025)),
            "ci95_high": float(np.quantile(bootstrap[:, index], 0.975)),
        }
        for index, name in enumerate(names)
    }


def probe_predictions(
    probe_train: np.lib.npyio.NpzFile,
    targets: list[np.lib.npyio.NpzFile],
    *,
    layer: int,
    c: float,
) -> list[np.ndarray]:
    """Fit the frozen probe on its training split and predict disjoint targets."""

    layers = probe_train["layer_ids"].astype(int).tolist()
    layer_index = layers.index(layer)
    train_features = probe_train["features"][:, :, layer_index, :].astype(np.float32)
    train_labels = probe_train["labels"].astype(bool)
    model = make_probe(c).fit(
        train_features.reshape(-1, train_features.shape[-1]), train_labels.ravel()
    )
    predictions = []
    for target in targets:
        if target["layer_ids"].astype(int).tolist() != layers:
            raise ValueError("Feature files use different layer IDs")
        features = target["features"][:, :, layer_index, :].astype(np.float32)
        scores = model.predict_proba(features.reshape(-1, features.shape[-1]))[:, 1]
        predictions.append(scores.reshape(features.shape[:2]))
    return predictions


def qid_set(data: np.lib.npyio.NpzFile) -> set[str]:
    return {str(value) for value in data["qids"].tolist()}


def main() -> None:
    args = parse_args()
    train = np.load(args.probe_train)
    calibration = np.load(args.calibration)
    test = np.load(args.test)
    analysis = json.loads(args.fusion_analysis.read_text(encoding="utf-8"))
    alphas = [float(value) for value in args.alphas.split(",")]
    layer = int(analysis["selected_layer"])
    c = float(analysis["selected_c"])
    weights = analysis["selected_three_signal_weights"]

    qids = {"probe_train": qid_set(train), "calibration": qid_set(calibration), "test": qid_set(test)}
    overlaps = {
        "probe_train_calibration": len(qids["probe_train"] & qids["calibration"]),
        "probe_train_test": len(qids["probe_train"] & qids["test"]),
        "calibration_test": len(qids["calibration"] & qids["test"]),
    }
    if any(overlaps.values()):
        raise ValueError(f"Query splits overlap: {overlaps}")

    calibration_probe, test_probe = probe_predictions(
        train, [calibration, test], layer=layer, c=c
    )
    calibration_scores = {
        "bge": query_z(calibration["bge_scores"].astype(np.float64)),
        "three_signal_fusion": (
            query_z(calibration["bge_scores"].astype(np.float64))
            + float(weights["lm_head"])
            * query_z(calibration["lm_relevance_scores"].astype(np.float64))
            + float(weights["hidden_probe"]) * query_z(calibration_probe)
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
    calibration_labels = calibration["labels"].astype(bool)
    test_labels = test["labels"].astype(bool)
    calibration_retrievable = calibration_labels.any(axis=1)
    test_retrievable = test_labels.any(axis=1)

    ranking = {
        name: fixed_k_metrics(test_labels, scores) for name, scores in test_scores.items()
    }
    results: dict[str, object] = {}
    for target_index, target in enumerate(("any_support", "all_support")):
        target_results = {}
        for alpha_index, alpha in enumerate(alphas):
            row: dict[str, object] = {}
            masks = {}
            for name in ("bge", "three_signal_fusion"):
                threshold, order = conformal_threshold(
                    calibration_scores[name][calibration_retrievable],
                    calibration_labels[calibration_retrievable],
                    alpha=alpha,
                    coverage_target=target,
                )
                mask = keep_mask(test_scores[name], threshold)
                masks[name] = mask
                row[name] = {
                    "calibrated_threshold": threshold,
                    "finite_sample_order": order,
                    "conditional_on_retrievable": conditional_metrics(
                        test_labels[test_retrievable], mask[test_retrievable]
                    ),
                    "end_to_end_all_test_queries": end_to_end_metrics(test_labels, mask),
                }
            row["fusion_minus_bge_conditional"] = paired_bootstrap(
                test_labels[test_retrievable],
                masks["three_signal_fusion"][test_retrievable],
                masks["bge"][test_retrievable],
                seed=args.seed + 10 * target_index + alpha_index,
                samples=args.bootstrap_samples,
            )
            target_results[str(alpha)] = row
        results[target] = target_results

    output = {
        "status": "complete",
        "method": "disjoint_split_query_level_conformal_chunk_pruning",
        "guarantee_scope": (
            "Query-level coverage is calibrated conditional on at least one labelled "
            "support chunk occurring in Top-L. End-to-end coverage also includes upstream "
            "retrieval failures and cannot exceed the reported retrieval ceiling."
        ),
        "splits": {
            "probe_train_queries": int(len(train["labels"])),
            "calibration_queries_total": int(len(calibration_labels)),
            "calibration_queries_retrievable": int(calibration_retrievable.sum()),
            "test_queries_total": int(len(test_labels)),
            "test_queries_retrievable": int(test_retrievable.sum()),
            "query_id_overlaps": overlaps,
        },
        "top_l": int(test_labels.shape[1]),
        "test_retrieval_query_coverage_ceiling": float(test_retrievable.mean()),
        "probe_layer": layer,
        "probe_c": c,
        "fusion_weights": weights,
        "alphas": alphas,
        "ranking_on_retrievable_test_queries": ranking,
        "conformal": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        main()
