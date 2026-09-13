"""Test query-adaptive conformal thresholds for internal-fusion chunk pruning."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from research.internal_state_rag.analyze_hidden_chunk_probe import query_z
from research.internal_state_rag.analyze_pairwise_conformal_clean_split import (
    conditional_metrics,
    keep_mask,
    probe_predictions,
)


def distribution_features(signals: dict[str, np.ndarray]) -> np.ndarray:
    """Build label-free query features from candidate score distributions."""

    blocks = []
    for values in signals.values():
        ordered = np.sort(values, axis=1)[:, ::-1]
        quantiles = np.quantile(values, [0.0, 0.1, 0.25, 0.5, 0.75, 1.0], axis=1).T
        logits = values - values.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        entropy = -(probs * np.log(probs + 1e-12)).sum(axis=1, keepdims=True)
        blocks.extend(
            [
                ordered[:, [0, 1, 2, 4, 9]],
                ordered[:, [0]] - ordered[:, [1]],
                ordered[:, [1]] - ordered[:, [2]],
                quantiles,
                entropy,
            ]
        )
    names = list(signals)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            correlation = np.asarray(
                [
                    np.corrcoef(a, b)[0, 1]
                    for a, b in zip(signals[left], signals[right], strict=True)
                ]
            )[:, None]
            blocks.append(np.nan_to_num(correlation))
    return np.column_stack(blocks)


def critical_scores(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    if np.any(labels.sum(axis=1) == 0):
        raise ValueError("Critical scores require retrievable queries")
    return np.asarray([row[y].min() for row, y in zip(scores, labels, strict=True)])


def correction_quantile(residuals: np.ndarray, alpha: float) -> tuple[float, int]:
    ordered = np.sort(np.asarray(residuals, dtype=np.float64))
    order = min(len(ordered), math.ceil((len(ordered) + 1) * (1 - alpha)))
    return float(ordered[order - 1]), order


def adaptive_mask(scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return keep_mask(scores - thresholds[:, None], 0.0)


def models(seed: int) -> dict[str, Any]:
    return {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "gradient_boosting": GradientBoostingRegressor(
            loss="huber",
            n_estimators=200,
            max_depth=2,
            min_samples_leaf=15,
            learning_rate=0.03,
            random_state=seed,
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=12,
            max_features=0.7,
            n_jobs=-1,
            random_state=seed,
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-train", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--fusion-analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=619)
    parser.add_argument("--alphas", default="0.2,0.1,0.05")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train = np.load(args.probe_train)
    calibration = np.load(args.calibration)
    test = np.load(args.test)
    analysis = json.loads(args.fusion_analysis.read_text(encoding="utf-8"))
    layer = int(analysis["selected_layer"])
    c = float(analysis["selected_c"])
    weights = analysis["selected_three_signal_weights"]
    calibration_probe, test_probe = probe_predictions(
        train, [calibration, test], layer=layer, c=c
    )

    def signal_arrays(data: Any, probe: np.ndarray) -> dict[str, np.ndarray]:
        bge = query_z(data["bge_scores"].astype(np.float64))
        lm_head = query_z(data["lm_relevance_scores"].astype(np.float64))
        hidden = query_z(probe)
        fusion = (
            bge
            + float(weights["lm_head"]) * lm_head
            + float(weights["hidden_probe"]) * hidden
        )
        return {"bge": bge, "lm_head": lm_head, "hidden": hidden, "fusion": fusion}

    cal_signals = signal_arrays(calibration, calibration_probe)
    test_signals = signal_arrays(test, test_probe)
    cal_labels_all = calibration["labels"].astype(bool)
    test_labels_all = test["labels"].astype(bool)
    cal_retrievable = np.flatnonzero(cal_labels_all.any(axis=1))
    test_retrievable = test_labels_all.any(axis=1)

    rng = np.random.default_rng(args.seed)
    shuffled = rng.permutation(cal_retrievable)
    split = len(shuffled) // 2
    predictor_indices = shuffled[:split]
    conformal_indices = shuffled[split:]
    predictor_labels = cal_labels_all[predictor_indices]
    conformal_labels = cal_labels_all[conformal_indices]
    predictor_scores = cal_signals["fusion"][predictor_indices]
    conformal_scores = cal_signals["fusion"][conformal_indices]
    test_scores = test_signals["fusion"]
    predictor_target = critical_scores(predictor_scores, predictor_labels)
    conformal_target = critical_scores(conformal_scores, conformal_labels)
    predictor_x = distribution_features(
        {name: values[predictor_indices] for name, values in cal_signals.items()}
    )
    conformal_x = distribution_features(
        {name: values[conformal_indices] for name, values in cal_signals.items()}
    )
    test_x = distribution_features(test_signals)

    cv = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    candidates = models(args.seed)
    cv_mae = {
        name: float(
            -cross_val_score(
                model,
                predictor_x,
                predictor_target,
                cv=cv,
                scoring="neg_mean_absolute_error",
                n_jobs=-1,
            ).mean()
        )
        for name, model in candidates.items()
    }
    selected_name = min(cv_mae, key=cv_mae.get)
    selected = candidates[selected_name].fit(predictor_x, predictor_target)
    conformal_prediction = selected.predict(conformal_x)
    test_prediction = selected.predict(test_x)
    predictor_fit = selected.predict(predictor_x)
    residuals = conformal_prediction - conformal_target
    alphas = [float(value) for value in args.alphas.split(",")]

    results = {}
    for alpha in alphas:
        correction, order = correction_quantile(residuals, alpha)
        adaptive_thresholds = test_prediction - correction
        adaptive = adaptive_mask(test_scores, adaptive_thresholds)

        global_critical = critical_scores(conformal_scores, conformal_labels)
        global_nonconformity = np.sort(-global_critical)
        global_order = min(
            len(global_nonconformity),
            math.ceil((len(global_nonconformity) + 1) * (1 - alpha)),
        )
        global_threshold = float(-global_nonconformity[global_order - 1])
        global_mask = keep_mask(test_scores, global_threshold)
        results[str(alpha)] = {
            "global_same_357_calibration_queries": {
                "threshold": global_threshold,
                "finite_sample_order": global_order,
                "conditional_on_retrievable": conditional_metrics(
                    test_labels_all[test_retrievable], global_mask[test_retrievable]
                ),
            },
            "adaptive_threshold": {
                "correction": correction,
                "finite_sample_order": order,
                "threshold_mean": float(adaptive_thresholds.mean()),
                "threshold_std": float(adaptive_thresholds.std()),
                "conditional_on_retrievable": conditional_metrics(
                    test_labels_all[test_retrievable], adaptive[test_retrievable]
                ),
            },
        }

    output = {
        "status": "complete",
        "method": "split_conformal_query_adaptive_critical_score_regression",
        "score": "three_signal_fusion",
        "split": {
            "predictor_train_queries": int(len(predictor_indices)),
            "conformal_calibration_queries": int(len(conformal_indices)),
            "test_queries_total": int(len(test_labels_all)),
            "test_queries_retrievable": int(test_retrievable.sum()),
        },
        "adaptive_features": int(predictor_x.shape[1]),
        "model_cv_mae": cv_mae,
        "selected_model": selected_name,
        "predictor_train_mae": float(mean_absolute_error(predictor_target, predictor_fit)),
        "conformal_prediction_mae": float(
            mean_absolute_error(conformal_target, conformal_prediction)
        ),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
