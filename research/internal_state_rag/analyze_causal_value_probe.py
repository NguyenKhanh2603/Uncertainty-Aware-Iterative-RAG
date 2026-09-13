"""Train a frozen-state probe against chunk causal-value rankings.

Hyperparameters are selected only on the first development queries.  The final
eight causal-labelled queries remain a teacher-label holdout.  A model refit on
all causal-labelled queries is then evaluated for support ranking on the
separate test feature set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from research.internal_state_rag.analyze_hidden_chunk_probe import (
    bootstrap_difference,
    query_metrics,
    query_z,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--causal-labels", type=Path, required=True)
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--test-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--development-queries", type=int, default=24)
    parser.add_argument("--seed", type=int, default=443)
    return parser.parse_args()


def within_query_ranks(values: np.ndarray) -> np.ndarray:
    """Map each query's utility values to centered percentile ranks."""

    output = np.empty_like(values, dtype=np.float64)
    for index, row in enumerate(values):
        output[index] = (rankdata(row, method="average") - 1) / (len(row) - 1) - 0.5
    return output


def features_for(
    hidden: np.ndarray,
    bge: np.ndarray,
    ranks: np.ndarray,
    *,
    variant: str,
    layer_index: int,
) -> np.ndarray:
    query_features = np.stack(
        [query_z(bge), -query_z(ranks.astype(np.float64))], axis=-1
    )
    if variant == "bge":
        return query_features
    layer_hidden = hidden[:, :, layer_index, :].astype(np.float32)
    if variant == "internal":
        return layer_hidden
    if variant == "bge_internal":
        return np.concatenate([layer_hidden, query_features], axis=-1)
    raise ValueError(f"Unknown variant {variant}")


def mean_query_spearman(target: np.ndarray, prediction: np.ndarray) -> float:
    correlations = []
    for truth, score in zip(target, prediction, strict=True):
        correlation = float(spearmanr(truth, score).statistic)
        correlations.append(0.0 if np.isnan(correlation) else correlation)
    return float(np.mean(correlations))


def causal_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    reciprocal_ranks, top1, top3 = [], [], []
    for truth, score in zip(target, prediction, strict=True):
        true_best = int(np.argmax(truth))
        order = np.argsort(-score, kind="stable")
        position = int(np.flatnonzero(order == true_best)[0]) + 1
        reciprocal_ranks.append(1 / position)
        top1.append(position == 1)
        top3.append(position <= 3)
    return {
        "mean_query_spearman": mean_query_spearman(target, prediction),
        "top_causal_chunk_mrr": float(np.mean(reciprocal_ranks)),
        "top_causal_chunk_top1": float(np.mean(top1)),
        "top_causal_chunk_top3": float(np.mean(top3)),
        "rank_target_mse": float(np.mean((target - prediction) ** 2)),
    }


def retention_metrics(
    labels: np.ndarray, scores: np.ndarray, ks: tuple[int, ...] = (1, 3, 5, 10)
) -> dict[str, dict[str, float]]:
    result = {}
    for k in ks:
        any_support, support_recall = [], []
        for query_labels, query_scores in zip(labels, scores, strict=True):
            kept = np.argsort(-query_scores, kind="stable")[:k]
            any_support.append(bool(query_labels[kept].any()))
            support_recall.append(float(query_labels[kept].sum() / query_labels.sum()))
        result[f"top_{k}"] = {
            "query_support_coverage": float(np.mean(any_support)),
            "mean_support_recall": float(np.mean(support_recall)),
        }
    return result


def fit_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    alpha: float,
) -> np.ndarray:
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(train_x.reshape(-1, train_x.shape[-1]), train_y.ravel())
    return model.predict(test_x.reshape(-1, test_x.shape[-1])).reshape(test_x.shape[:2])


def select_configuration(
    hidden: np.ndarray,
    bge: np.ndarray,
    ranks: np.ndarray,
    target: np.ndarray,
    layers: list[int],
    development_queries: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    alphas = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
    groups = np.repeat(np.arange(development_queries), target.shape[1])
    folds = list(GroupKFold(4).split(groups, target[:development_queries].ravel(), groups))
    candidates = []
    for variant in ("bge", "internal", "bge_internal"):
        layer_indices = [0] if variant == "bge" else list(range(len(layers)))
        for layer_index in layer_indices:
            x = features_for(
                hidden[:development_queries],
                bge[:development_queries],
                ranks[:development_queries],
                variant=variant,
                layer_index=layer_index,
            )
            flat_x, flat_y = x.reshape(-1, x.shape[-1]), target[:development_queries].ravel()
            for alpha in alphas:
                prediction = np.empty_like(flat_y)
                for train_indices, held_indices in folds:
                    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
                    model.fit(flat_x[train_indices], flat_y[train_indices])
                    prediction[held_indices] = model.predict(flat_x[held_indices])
                prediction = prediction.reshape(development_queries, target.shape[1])
                metrics = causal_metrics(target[:development_queries], prediction)
                candidates.append(
                    {
                        "variant": variant,
                        "layer_index": layer_index,
                        "layer": None if variant == "bge" else layers[layer_index],
                        "alpha": alpha,
                        "development_oof": metrics,
                    }
                )
    best_by_variant = {}
    for variant in ("bge", "internal", "bge_internal"):
        choices = [row for row in candidates if row["variant"] == variant]
        best_by_variant[variant] = max(
            choices,
            key=lambda row: (
                row["development_oof"]["mean_query_spearman"],
                row["development_oof"]["top_causal_chunk_mrr"],
            ),
        )
    return best_by_variant, candidates


def main() -> None:
    args = parse_args()
    causal = json.loads(args.causal_labels.read_text(encoding="utf-8"))
    train = np.load(args.train_features)
    test = np.load(args.test_features)
    qids = [str(value) for value in train["qids"]]
    causal_qids = [str(row["qid"]) for row in causal["queries"]]
    if qids[: len(causal_qids)] != causal_qids:
        raise ValueError("Causal queries do not match the feature-array prefix")
    if args.development_queries >= len(causal_qids):
        raise ValueError("At least one causal-labelled query must remain held out")
    if not np.array_equal(train["layer_ids"], test["layer_ids"]):
        raise ValueError("Train and test feature layers differ")

    n = len(causal_qids)
    hidden = train["features"][:n].astype(np.float32)
    bge = train["bge_scores"][:n].astype(np.float64)
    ranks = train["ranks"][:n].astype(np.float64)
    labels = train["labels"][:n].astype(bool)
    causal_values = np.asarray(
        [[chunk["gold_logprob_drop"] for chunk in row["chunks"]] for row in causal["queries"]],
        dtype=np.float64,
    )
    for index, row in enumerate(causal["queries"]):
        observed_ranks = [chunk["rank"] for chunk in row["chunks"]]
        if observed_ranks != ranks[index].astype(int).tolist():
            raise ValueError(f"Candidate order mismatch for {row['qid']}")
    target = within_query_ranks(causal_values)
    layers = train["layer_ids"].astype(int).tolist()
    selected, candidates = select_configuration(
        hidden, bge, ranks, target, layers, args.development_queries
    )

    holdout_slice = slice(args.development_queries, n)
    causal_holdout = {}
    test_support = {
        "raw_bge": query_metrics(test["labels"].astype(bool), test["bge_scores"]),
    }
    test_support["raw_bge"]["retention"] = retention_metrics(
        test["labels"].astype(bool), test["bge_scores"]
    )
    test_predictions = {}
    for variant, configuration in selected.items():
        layer_index = int(configuration["layer_index"])
        train_x = features_for(
            hidden,
            bge,
            ranks,
            variant=variant,
            layer_index=layer_index,
        )
        holdout_prediction = fit_predict(
            train_x[: args.development_queries],
            target[: args.development_queries],
            train_x[holdout_slice],
            float(configuration["alpha"]),
        )
        causal_holdout[variant] = causal_metrics(target[holdout_slice], holdout_prediction)

        test_x = features_for(
            test["features"].astype(np.float32),
            test["bge_scores"].astype(np.float64),
            test["ranks"].astype(np.float64),
            variant=variant,
            layer_index=layer_index,
        )
        prediction = fit_predict(
            train_x, target, test_x, float(configuration["alpha"])
        )
        test_predictions[variant] = prediction
        test_support[variant] = query_metrics(test["labels"].astype(bool), prediction)
        test_support[variant]["retention"] = retention_metrics(
            test["labels"].astype(bool), prediction
        )
        test_support[variant]["minus_bge"] = bootstrap_difference(
            test["labels"].astype(bool),
            prediction,
            test["bge_scores"].astype(np.float64),
            seed=args.seed + len(test_predictions),
        )

    raw_teacher = {
        "bge_support_ranking": query_metrics(labels, bge),
        "causal_value_support_ranking": query_metrics(labels, causal_values),
        "support_positive_causal_fraction": float((causal_values[labels] > 0).mean()),
        "non_support_positive_causal_fraction": float((causal_values[~labels] > 0).mean()),
        "support_mean_causal_value": float(causal_values[labels].mean()),
        "non_support_mean_causal_value": float(causal_values[~labels].mean()),
    }
    output = {
        "status": "complete",
        "method": "ridge_probe_of_within_query_causal_value_rank",
        "causal_queries": n,
        "development_queries": args.development_queries,
        "causal_holdout_queries": n - args.development_queries,
        "test_queries": int(len(test["qids"])),
        "selection_rule": "maximum grouped-OOF mean per-query Spearman on development queries",
        "selected": selected,
        "raw_teacher": raw_teacher,
        "causal_holdout": causal_holdout,
        "test_support_ranking": test_support,
        "all_selection_candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    printable = dict(output)
    printable.pop("all_selection_candidates")
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
