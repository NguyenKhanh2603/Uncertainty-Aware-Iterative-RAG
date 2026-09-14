"""Learn modality-specific score maps before query-level conformal pruning.

Score maps are fitted on the probe-training role.  Hidden-probe scores on that
role are out-of-fold, preventing the modality map from seeing in-sample probe
predictions.  A single all-support conformal threshold is then fitted on the
disjoint calibration role and evaluated once on test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

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
from research.internal_state_rag.analyze_modality_conditional_conformal import (
    modality_recall,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-train", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--ablation-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions-output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1559)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    return parser.parse_args()


def fit_score_maps(
    scores: np.ndarray,
    labels: np.ndarray,
    modalities: np.ndarray,
) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    maps: dict[str, dict[str, object]] = {
        "modality_platt": {},
        "modality_isotonic": {},
        "modality_negative_cdf": {},
    }
    audit: dict[str, object] = {}
    for modality in sorted(set(modalities.ravel().astype(str))):
        rows = modalities == modality
        x = scores[rows].astype(np.float64)
        y = labels[rows].astype(int)
        if len(np.unique(y)) != 2:
            raise ValueError(f"Modality {modality!r} needs positive and negative examples")
        platt = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000).fit(
            x[:, None], y
        )
        isotonic = IsotonicRegression(out_of_bounds="clip").fit(x, y)
        negative_scores = np.sort(x[y == 0])
        maps["modality_platt"][modality] = platt
        maps["modality_isotonic"][modality] = isotonic
        maps["modality_negative_cdf"][modality] = negative_scores
        audit[modality] = {
            "candidates": int(len(x)),
            "support_candidates": int(y.sum()),
            "platt_coefficient": float(platt.coef_[0, 0]),
            "platt_intercept": float(platt.intercept_[0]),
            "isotonic_thresholds": int(len(isotonic.X_thresholds_)),
            "negative_cdf_reference_candidates": int(len(negative_scores)),
        }
    return maps, audit


def apply_score_map(
    scores: np.ndarray,
    modalities: np.ndarray,
    fitted: dict[str, object],
    kind: str,
) -> np.ndarray:
    output = np.empty(scores.shape, dtype=np.float64)
    for modality, model in fitted.items():
        rows = modalities == modality
        x = scores[rows].astype(np.float64)
        if kind == "modality_platt":
            output[rows] = model.predict_proba(x[:, None])[:, 1]
        elif kind == "modality_isotonic":
            output[rows] = model.predict(x)
        elif kind == "modality_negative_cdf":
            output[rows] = np.searchsorted(model, x, side="right") / (len(model) + 1)
        else:
            raise ValueError(f"Unknown score map: {kind}")
    return output


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
    if not 0 < args.alpha < 1:
        raise ValueError("alpha must be between zero and one")
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
    test_retrievable = test_labels.any(axis=1)

    results = {}
    maps_audit = {}
    masks_to_save = {}
    for method_index, method in enumerate(("jina_m0_reranker", "reranker_internal")):
        method_results = {}
        raw_result, raw_mask = evaluate(
            cal_scores[method],
            test_scores[method],
            cal_labels,
            test_labels,
            test_modalities,
            alpha=args.alpha,
        )
        method_results["raw_global"] = raw_result
        masks_to_save[f"mask_{method}_raw_global"] = raw_mask
        fitted_maps, maps_audit[method] = fit_score_maps(
            train_scores[method], train_labels, train_modalities
        )
        for map_index, (kind, fitted) in enumerate(fitted_maps.items()):
            transformed_cal = apply_score_map(
                cal_scores[method], cal_modalities, fitted, kind
            )
            transformed_test = apply_score_map(
                test_scores[method], test_modalities, fitted, kind
            )
            result, mask = evaluate(
                transformed_cal,
                transformed_test,
                cal_labels,
                test_labels,
                test_modalities,
                alpha=args.alpha,
            )
            result["paired_bootstrap_minus_raw"] = paired_bootstrap(
                test_labels[test_retrievable],
                mask[test_retrievable],
                raw_mask[test_retrievable],
                seed=args.seed + 100 * method_index + map_index,
                samples=args.bootstrap_samples,
            )
            method_results[kind] = result
            masks_to_save[f"mask_{method}_{kind}"] = mask
        results[method] = method_results

    output = {
        "status": "complete",
        "dataset": report["dataset"],
        "alpha": args.alpha,
        "protocol": {
            "score_map_fit": "probe-training role with OOF hidden-probe scores",
            "threshold_fit": "disjoint calibration role",
            "evaluation": "official held-out test role",
            "coverage_target": "single query-level all-support conformal threshold",
            "selected_layer": selected_layer,
            "selected_c": selected_c,
        },
        "score_map_audit": maps_audit,
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
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
