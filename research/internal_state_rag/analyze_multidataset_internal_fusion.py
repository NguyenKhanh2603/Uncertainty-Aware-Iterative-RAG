"""Evaluate internal relevance signals and conformal pruning on a new dataset.

The probe-training, conformal-calibration, and test query sets must be disjoint.
Hyperparameters and fusion weights are selected only with group-wise out-of-fold
predictions on the probe-training queries.  A TAT-QA-trained probe is included as
a zero-shot transfer control.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

from research.internal_state_rag.analyze_hidden_chunk_probe import (
    bootstrap_difference,
    make_probe,
    query_metrics,
    query_z,
)
from research.internal_state_rag.analyze_pairwise_conformal_clean_split import (
    conditional_metrics,
    conformal_threshold,
    end_to_end_metrics,
    fixed_k_metrics,
    keep_mask,
    paired_bootstrap,
)


C_VALUES = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
WEIGHTS = (0.0, 0.01, 0.025, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--probe-train", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--tatqa-probe-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions-output", type=Path, required=True)
    parser.add_argument("--alphas", default="0.2,0.1,0.05")
    parser.add_argument(
        "--candidate-layers",
        default="",
        help="Comma-separated subset to tune; empty evaluates every stored layer.",
    )
    parser.add_argument(
        "--c-values",
        default=",".join(str(value) for value in C_VALUES),
        help="Comma-separated logistic-probe C values.",
    )
    parser.add_argument("--seed", type=int, default=733)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    return parser.parse_args()


def validate_inputs(
    train: np.lib.npyio.NpzFile,
    calibration: np.lib.npyio.NpzFile,
    test: np.lib.npyio.NpzFile,
) -> None:
    if not (
        np.array_equal(train["layer_ids"], calibration["layer_ids"])
        and np.array_equal(train["layer_ids"], test["layer_ids"])
    ):
        raise ValueError("Layer IDs differ across train/calibration/test")
    if len({train["labels"].shape[1], calibration["labels"].shape[1], test["labels"].shape[1]}) != 1:
        raise ValueError("Top-L differs across train/calibration/test")
    qid_sets = [set(part["qids"].astype(str).tolist()) for part in (train, calibration, test)]
    if any(qid_sets[left] & qid_sets[right] for left, right in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Query overlap across train/calibration/test")


def select_probe(
    features: np.ndarray,
    labels: np.ndarray,
    layers: list[int],
    *,
    candidate_layers: set[int],
    c_values: tuple[float, ...],
) -> tuple[int, float, np.ndarray, list[dict[str, float]]]:
    """Select a layer and regularization using query-grouped OOF predictions."""

    groups = np.repeat(np.arange(len(labels)), labels.shape[1])
    folds = list(GroupKFold(5).split(groups, labels.ravel(), groups))
    retrievable = labels.any(axis=1)
    candidates: list[tuple[float, float, int, float, np.ndarray]] = []
    audit: list[dict[str, float]] = []
    for layer_index, layer in enumerate(layers):
        if layer not in candidate_layers:
            continue
        x = features[:, :, layer_index, :].reshape(-1, features.shape[-1])
        for c in c_values:
            oof = np.zeros(labels.size, dtype=np.float64)
            for fit_rows, held_rows in folds:
                model = make_probe(c).fit(x[fit_rows], labels.ravel()[fit_rows])
                oof[held_rows] = model.predict_proba(x[held_rows])[:, 1]
            scores = oof.reshape(labels.shape)
            metrics = query_metrics(labels[retrievable], scores[retrievable])
            audit.append({"layer": layer, "c": c, **metrics})
            candidates.append(
                (metrics["mean_query_ap"], metrics["mrr"], layer_index, c, scores)
            )
    _, _, layer_index, c, scores = max(
        candidates, key=lambda row: (row[0], row[1], -row[3])
    )
    return layer_index, c, scores, audit


def fit_predict_probe(
    train: np.lib.npyio.NpzFile,
    targets: list[np.lib.npyio.NpzFile],
    *,
    layer_index: int,
    c: float,
    train_indices: np.ndarray | None = None,
) -> list[np.ndarray]:
    features = train["features"].astype(np.float32)
    labels = train["labels"].astype(bool)
    if train_indices is not None:
        features = features[train_indices]
        labels = labels[train_indices]
    x = features[:, :, layer_index, :].reshape(-1, features.shape[-1])
    model = make_probe(c).fit(x, labels.ravel())
    outputs = []
    for target in targets:
        target_x = target["features"][:, :, layer_index, :].astype(np.float32)
        outputs.append(
            model.predict_proba(target_x.reshape(-1, target_x.shape[-1]))[:, 1].reshape(
                target_x.shape[:2]
            )
        )
    return outputs


def tune_fusion(
    labels: np.ndarray,
    bge: np.ndarray,
    lm_head: np.ndarray,
    hidden: np.ndarray,
) -> dict[str, dict[str, float]]:
    retrievable = labels.any(axis=1)
    signals = {"bge": query_z(bge), "lm_head": query_z(lm_head), "hidden": query_z(hidden)}

    def choose(configs: list[tuple[float, float]]) -> dict[str, float]:
        rows = []
        for lm_weight, hidden_weight in configs:
            score = (
                signals["bge"]
                + lm_weight * signals["lm_head"]
                + hidden_weight * signals["hidden"]
            )
            rows.append(
                {
                    "lm_head": lm_weight,
                    "hidden_probe": hidden_weight,
                    **query_metrics(labels[retrievable], score[retrievable]),
                }
            )
        return max(
            rows,
            key=lambda row: (
                row["mean_query_ap"],
                row["mrr"],
                -(row["lm_head"] + row["hidden_probe"]),
            ),
        )

    return {
        "bge_lm": choose([(weight, 0.0) for weight in WEIGHTS]),
        "bge_hidden": choose([(0.0, weight) for weight in WEIGHTS]),
        "bge_lm_hidden": choose([(left, right) for left in WEIGHTS for right in WEIGHTS]),
    }


def score_methods(
    part: np.lib.npyio.NpzFile,
    hidden: np.ndarray,
    transfer_hidden: np.ndarray,
    weights: dict[str, dict[str, float]],
) -> dict[str, np.ndarray]:
    bge = query_z(part["bge_scores"].astype(np.float64))
    lm_head = query_z(part["lm_relevance_scores"].astype(np.float64))
    hidden = query_z(hidden)
    transfer_hidden = query_z(transfer_hidden)

    def fuse(name: str) -> np.ndarray:
        row = weights[name]
        return bge + row["lm_head"] * lm_head + row["hidden_probe"] * hidden

    return {
        "bge": bge,
        "lm_head": lm_head,
        "hidden_probe": hidden,
        "bge_lm": fuse("bge_lm"),
        "bge_hidden": fuse("bge_hidden"),
        "bge_lm_hidden": fuse("bge_lm_hidden"),
        "tatqa_transfer_hidden": transfer_hidden,
        "tatqa_fixed_fusion": bge + 0.05 * lm_head + 1.5 * transfer_hidden,
    }


def main() -> None:
    args = parse_args()
    train = np.load(args.probe_train)
    calibration = np.load(args.calibration)
    test = np.load(args.test)
    tatqa = np.load(args.tatqa_probe_source)
    validate_inputs(train, calibration, test)

    layers = train["layer_ids"].astype(int).tolist()
    candidate_layers = (
        {int(value) for value in args.candidate_layers.split(",") if value.strip()}
        if args.candidate_layers
        else set(layers)
    )
    unknown_layers = candidate_layers - set(layers)
    if unknown_layers:
        raise ValueError(f"Candidate layers absent from features: {sorted(unknown_layers)}")
    c_values = tuple(float(value) for value in args.c_values.split(",") if value.strip())
    if not c_values or any(value <= 0 for value in c_values):
        raise ValueError("--c-values must contain positive values")
    train_features = train["features"].astype(np.float32)
    train_labels = train["labels"].astype(bool)
    layer_index, selected_c, train_hidden_oof, probe_audit = select_probe(
        train_features,
        train_labels,
        layers,
        candidate_layers=candidate_layers,
        c_values=c_values,
    )
    cal_hidden, test_hidden = fit_predict_probe(
        train, [calibration, test], layer_index=layer_index, c=selected_c
    )

    transfer_layer = 30
    transfer_index = tatqa["layer_ids"].astype(int).tolist().index(transfer_layer)
    target_transfer_index = layers.index(transfer_layer)
    if transfer_index != target_transfer_index:
        raise ValueError("Layer layouts differ between TAT-QA and target features")
    permutation = np.random.default_rng(733).permutation(len(tatqa["labels"]))
    tatqa_train_indices = permutation[: len(tatqa["labels"]) // 2]
    train_transfer, cal_transfer, test_transfer = fit_predict_probe(
        tatqa,
        [train, calibration, test],
        layer_index=transfer_index,
        c=0.1,
        train_indices=tatqa_train_indices,
    )

    selected_weights = tune_fusion(
        train_labels,
        train["bge_scores"].astype(np.float64),
        train["lm_relevance_scores"].astype(np.float64),
        train_hidden_oof,
    )
    train_scores = score_methods(
        train, train_hidden_oof, train_transfer, selected_weights
    )
    calibration_scores = score_methods(
        calibration, cal_hidden, cal_transfer, selected_weights
    )
    test_scores = score_methods(test, test_hidden, test_transfer, selected_weights)

    cal_labels = calibration["labels"].astype(bool)
    test_labels = test["labels"].astype(bool)
    train_retrievable = train_labels.any(axis=1)
    cal_retrievable = cal_labels.any(axis=1)
    test_retrievable = test_labels.any(axis=1)
    alphas = [float(value) for value in args.alphas.split(",")]
    conformal: dict[str, dict[str, object]] = {}
    masks_for_save: dict[str, np.ndarray] = {}
    bootstrap: dict[str, object] = {}
    for alpha_index, alpha in enumerate(alphas):
        alpha_key = str(alpha)
        conformal[alpha_key] = {}
        masks = {}
        for name, cal_score in calibration_scores.items():
            threshold, order = conformal_threshold(
                cal_score[cal_retrievable],
                cal_labels[cal_retrievable],
                alpha=alpha,
                coverage_target="all_support",
            )
            mask = keep_mask(test_scores[name], threshold)
            masks[name] = mask
            masks_for_save[f"mask_{name}_alpha_{alpha_key}"] = mask
            conformal[alpha_key][name] = {
                "threshold": threshold,
                "finite_sample_order": order,
                "conditional_on_retrievable": conditional_metrics(
                    test_labels[test_retrievable], mask[test_retrievable]
                ),
                "end_to_end": end_to_end_metrics(test_labels, mask),
            }
        bootstrap[alpha_key] = paired_bootstrap(
            test_labels[test_retrievable],
            masks["bge_lm_hidden"][test_retrievable],
            masks["bge"][test_retrievable],
            seed=args.seed + alpha_index,
            samples=args.bootstrap_samples,
        )

    ranking = {
        name: fixed_k_metrics(test_labels, score) for name, score in test_scores.items()
    }
    ranking_bootstrap = {
        name: bootstrap_difference(
            test_labels[test_retrievable],
            score[test_retrievable],
            test_scores["bge"][test_retrievable],
            seed=args.seed + index,
            samples=args.bootstrap_samples,
        )
        for index, (name, score) in enumerate(test_scores.items())
        if name != "bge"
    }
    output = {
        "status": "complete",
        "dataset": args.dataset,
        "protocol": "disjoint_probe_train_conformal_calibration_official_test",
        "split": {
            "probe_train_total": int(len(train_labels)),
            "probe_train_retrievable": int(train_retrievable.sum()),
            "calibration_total": int(len(cal_labels)),
            "calibration_retrievable": int(cal_retrievable.sum()),
            "test_total": int(len(test_labels)),
            "test_retrievable": int(test_retrievable.sum()),
            "top_l": int(test_labels.shape[1]),
        },
        "in_domain_probe": {
            "selected_layer": layers[layer_index],
            "selected_c": selected_c,
            "selection": "5-fold query-grouped OOF mean AP",
            "candidate_layers": sorted(candidate_layers),
            "candidate_c_values": list(c_values),
            "candidate_audit": probe_audit,
        },
        "tatqa_transfer_probe": {
            "layer": transfer_layer,
            "c": 0.1,
            "training_queries": int(len(tatqa_train_indices)),
            "training_index_policy": "seed733 permutation first half of n904",
        },
        "selected_fusion_weights": selected_weights,
        "train_oof_ranking": {
            name: query_metrics(train_labels[train_retrievable], score[train_retrievable])
            for name, score in train_scores.items()
        },
        "test_ranking": ranking,
        "test_ranking_minus_bge_paired_bootstrap": ranking_bootstrap,
        "conformal": conformal,
        "bge_lm_hidden_minus_bge_conformal_paired_bootstrap": bootstrap,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.predictions_output,
        qids=test["qids"],
        labels=test_labels,
        **{f"score_{name}": score for name, score in test_scores.items()},
        **masks_for_save,
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        main()
