"""Use internal-state summaries to adapt conformal pruning depth per query.

A regressor trained only on the probe role predicts each query's worst support
score from candidate-score distributions.  Subtracting that predicted difficulty
from every candidate score leaves the within-query ranking unchanged but lets the
single conformal threshold retain more candidates for queries predicted to be
hard.  Final calibration and test remain disjoint from regressor selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from research.internal_state_rag.analyze_modality_conditional_conformal import (
    modality_recall,
)
from research.internal_state_rag.analyze_multidataset_internal_fusion import (
    fit_predict_probe,
    select_probe,
    validate_inputs,
)
from research.internal_state_rag.analyze_pairwise_conformal_clean_split import (
    conditional_metrics,
    conformal_threshold,
    end_to_end_metrics,
    keep_mask,
    paired_bootstrap,
)
from research.internal_state_rag.analyze_qwen2vl_jina_ablation import make_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-train", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--ablation-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions-output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2029)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    return parser.parse_args()


def distribution_features(scores: np.ndarray) -> np.ndarray:
    quantiles = np.quantile(scores, (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0), axis=1).T
    ordered = np.sort(scores, axis=1)[:, ::-1]
    gaps = np.column_stack(
        (
            ordered[:, 0] - ordered[:, 1],
            ordered[:, 1] - ordered[:, 2],
            ordered[:, 0] - ordered[:, 4],
            ordered[:, 0] - ordered[:, 9],
        )
    )
    shifted = scores - scores.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    entropy = -(probabilities * np.log(probabilities + 1e-12)).sum(axis=1)
    return np.column_stack((quantiles, gaps, entropy))


def modality_features(
    scores: np.ndarray,
    modalities: np.ndarray,
    modality_names: list[str],
) -> np.ndarray:
    columns = []
    for modality in modality_names:
        rows = modalities == modality
        count = rows.sum(axis=1)
        safe_count = np.maximum(count, 1)
        masked = np.where(rows, scores, -np.inf)
        maximum = masked.max(axis=1)
        maximum[count == 0] = 0.0
        mean = np.where(rows, scores, 0.0).sum(axis=1) / safe_count
        centered = np.where(rows, scores - mean[:, None], 0.0)
        std = np.sqrt((centered**2).sum(axis=1) / safe_count)
        columns.extend((count / scores.shape[1], maximum, mean, std))
    return np.column_stack(columns)


def rank_agreement(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    features = [np.mean(left * right, axis=1)]
    for k in (1, 3, 5):
        left_top = np.argpartition(left, -k, axis=1)[:, -k:]
        right_top = np.argpartition(right, -k, axis=1)[:, -k:]
        overlap = np.asarray(
            [len(set(a.tolist()) & set(b.tolist())) / k for a, b in zip(left_top, right_top)]
        )
        features.append(overlap)
    return np.column_stack(features)


def query_features(
    scores: dict[str, np.ndarray],
    modalities: np.ndarray,
    modality_names: list[str],
    *,
    include_internal: bool,
) -> np.ndarray:
    blocks = [
        distribution_features(scores["jina_m0_reranker"]),
        distribution_features(scores["jina_v4_cosine"]),
        modality_features(scores["jina_m0_reranker"], modalities, modality_names),
    ]
    if include_internal:
        blocks.extend(
            (
                distribution_features(scores["hidden_probe_only"]),
                distribution_features(scores["lm_head_only"]),
                modality_features(scores["hidden_probe_only"], modalities, modality_names),
                modality_features(scores["lm_head_only"], modalities, modality_names),
                rank_agreement(
                    scores["jina_m0_reranker"], scores["hidden_probe_only"]
                ),
                rank_agreement(scores["jina_m0_reranker"], scores["lm_head_only"]),
            )
        )
    return np.column_stack(blocks).astype(np.float64)


def make_regressor(name: str, seed: int):
    if name.startswith("ridge_"):
        alpha = float(name.removeprefix("ridge_"))
        return make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    if name == "gbr_depth1":
        return GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.03,
            max_depth=1,
            min_samples_leaf=20,
            loss="huber",
            random_state=seed,
        )
    if name == "gbr_depth2":
        return GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.03,
            max_depth=2,
            min_samples_leaf=20,
            loss="huber",
            random_state=seed,
        )
    if name == "hist_gbr":
        return HistGradientBoostingRegressor(
            max_iter=100,
            learning_rate=0.05,
            max_leaf_nodes=7,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=seed,
        )
    raise ValueError(f"Unknown regressor: {name}")


def critical_support_scores(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    if np.any(~labels.any(axis=1)):
        raise ValueError("Critical scores require retrievable queries")
    return np.asarray([row[y].min() for row, y in zip(scores, labels, strict=True)])


def select_regressor(
    features: np.ndarray,
    base_scores: np.ndarray,
    labels: np.ndarray,
    *,
    alpha: float,
    seed: int,
) -> tuple[str, list[dict[str, float]]]:
    target = critical_support_scores(base_scores, labels)
    folds = list(KFold(5, shuffle=True, random_state=seed).split(features))
    names = ("ridge_1", "ridge_10", "ridge_100", "gbr_depth1", "gbr_depth2", "hist_gbr")
    audit = []
    for model_index, name in enumerate(names):
        predictions = np.empty(len(target), dtype=np.float64)
        for fit_rows, held_rows in folds:
            model = make_regressor(name, seed + model_index)
            model.fit(features[fit_rows], target[fit_rows])
            predictions[held_rows] = model.predict(features[held_rows])
        adapted = base_scores - predictions[:, None]
        threshold, _ = conformal_threshold(
            adapted, labels, alpha=alpha, coverage_target="all_support"
        )
        mask = keep_mask(adapted, threshold)
        metrics = conditional_metrics(labels, mask)
        correlation = float(np.corrcoef(target, predictions)[0, 1])
        audit.append(
            {
                "model": name,
                "critical_score_mae": float(np.mean(np.abs(target - predictions))),
                "critical_score_correlation": correlation,
                "oof_mean_chunks_kept": metrics["mean_chunks_kept"],
                "oof_micro_support_recall": metrics["micro_support_recall"],
                "oof_query_all_support_coverage": metrics[
                    "query_all_support_coverage"
                ],
            }
        )
    best = min(
        audit,
        key=lambda row: (
            row["oof_mean_chunks_kept"],
            row["critical_score_mae"],
            -row["critical_score_correlation"],
        ),
    )
    return str(best["model"]), audit


def evaluate(
    cal_scores: np.ndarray,
    test_scores: np.ndarray,
    cal_labels: np.ndarray,
    test_labels: np.ndarray,
    test_modalities: np.ndarray,
    *,
    alpha: float,
) -> tuple[dict[str, object], np.ndarray]:
    cal_retrievable = cal_labels.any(axis=1)
    test_retrievable = test_labels.any(axis=1)
    threshold, order = conformal_threshold(
        cal_scores[cal_retrievable],
        cal_labels[cal_retrievable],
        alpha=alpha,
        coverage_target="all_support",
    )
    mask = keep_mask(test_scores, threshold)
    return {
        "threshold": threshold,
        "finite_sample_order": order,
        "conditional_on_retrievable": conditional_metrics(
            test_labels[test_retrievable], mask[test_retrievable]
        ),
        "end_to_end": end_to_end_metrics(test_labels, mask),
        "test_support_by_modality": modality_recall(
            test_labels[test_retrievable],
            mask[test_retrievable],
            test_modalities[test_retrievable],
        ),
    }, mask


def main() -> None:
    args = parse_args()
    train = np.load(args.probe_train)
    calibration = np.load(args.calibration)
    test = np.load(args.test)
    validate_inputs(train, calibration, test)
    report = json.loads(args.ablation_report.read_text(encoding="utf-8"))
    layers = train["layer_ids"].astype(int).tolist()
    selected_layer = int(report["probe"]["selected_layer"])
    selected_c = float(report["probe"]["selected_c"])
    layer_index = layers.index(selected_layer)

    _, _, train_hidden_oof, _ = select_probe(
        train["features"].astype(np.float32),
        train["labels"].astype(bool),
        layers,
        candidate_layers={selected_layer},
        c_values=(selected_c,),
    )
    cal_hidden, test_hidden = fit_predict_probe(
        train, [calibration, test], layer_index=layer_index, c=selected_c
    )
    weights = report["selected_weights"]
    train_scores = make_scores(train, train_hidden_oof, weights)
    cal_scores = make_scores(calibration, cal_hidden, weights)
    test_scores = make_scores(test, test_hidden, weights)
    train_labels = train["labels"].astype(bool)
    cal_labels = calibration["labels"].astype(bool)
    test_labels = test["labels"].astype(bool)
    train_modalities = train["modalities"].astype(str)
    cal_modalities = calibration["modalities"].astype(str)
    test_modalities = test["modalities"].astype(str)
    modalities = sorted(set(train_modalities.ravel().tolist()))
    train_retrievable = train_labels.any(axis=1)
    test_retrievable = test_labels.any(axis=1)

    feature_sets = {}
    for name, include_internal in (("external", False), ("external_internal", True)):
        feature_sets[name] = {
            "train": query_features(
                train_scores, train_modalities, modalities, include_internal=include_internal
            ),
            "calibration": query_features(
                cal_scores, cal_modalities, modalities, include_internal=include_internal
            ),
            "test": query_features(
                test_scores, test_modalities, modalities, include_internal=include_internal
            ),
        }

    results = {}
    selection_audit = {}
    masks_to_save = {}
    offsets_to_save = {}
    for method_index, method in enumerate(("jina_m0_reranker", "reranker_internal")):
        raw_result, raw_mask = evaluate(
            cal_scores[method],
            test_scores[method],
            cal_labels,
            test_labels,
            test_modalities,
            alpha=args.alpha,
        )
        method_results = {"raw_global": raw_result}
        masks_to_save[f"mask_{method}_raw_global"] = raw_mask
        selection_audit[method] = {}
        for feature_index, (feature_name, parts) in enumerate(feature_sets.items()):
            selected_model, audit = select_regressor(
                parts["train"][train_retrievable],
                train_scores[method][train_retrievable],
                train_labels[train_retrievable],
                alpha=args.alpha,
                seed=args.seed + 100 * method_index + feature_index,
            )
            selection_audit[method][feature_name] = {
                "selected_model": selected_model,
                "candidate_audit": audit,
            }
            target = critical_support_scores(
                train_scores[method][train_retrievable], train_labels[train_retrievable]
            )
            model = make_regressor(selected_model, args.seed + method_index)
            model.fit(parts["train"][train_retrievable], target)
            cal_offset = model.predict(parts["calibration"])
            test_offset = model.predict(parts["test"])
            adapted_cal = cal_scores[method] - cal_offset[:, None]
            adapted_test = test_scores[method] - test_offset[:, None]
            result, mask = evaluate(
                adapted_cal,
                adapted_test,
                cal_labels,
                test_labels,
                test_modalities,
                alpha=args.alpha,
            )
            result["selected_regressor"] = selected_model
            result["paired_bootstrap_minus_raw"] = paired_bootstrap(
                test_labels[test_retrievable],
                mask[test_retrievable],
                raw_mask[test_retrievable],
                seed=args.seed + 1000 * method_index + feature_index,
                samples=args.bootstrap_samples,
            )
            method_results[f"adaptive_{feature_name}"] = result
            masks_to_save[f"mask_{method}_adaptive_{feature_name}"] = mask
            offsets_to_save[f"offset_{method}_adaptive_{feature_name}"] = test_offset
        results[method] = method_results

    output = {
        "status": "complete",
        "dataset": report["dataset"],
        "alpha": args.alpha,
        "protocol": {
            "difficulty_target": "worst support score within a retrievable query",
            "adapted_candidate_score": "candidate score minus predicted query difficulty",
            "ranking_effect": "none within each query",
            "regressor_selection": "5-fold OOF on probe-training queries",
            "conformal_calibration": "disjoint calibration queries",
            "evaluation": "official held-out test queries",
            "selected_layer": selected_layer,
            "selected_c": selected_c,
        },
        "selection_audit": selection_audit,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        args.predictions_output,
        qids=test["qids"],
        chunk_ids=test["chunk_ids"],
        modalities=test_modalities,
        labels=test_labels,
        **masks_to_save,
        **offsets_to_save,
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
