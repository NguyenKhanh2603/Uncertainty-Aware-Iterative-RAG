"""Compare global and support-modality-conditional conformal chunk pruning.

The alpha allocation is fixed using only out-of-fold probe predictions.  Final
thresholds are fitted on the disjoint calibration split and evaluated once on
the test split.  The Bonferroni variants allocate a query-level failure budget
across modalities, while the same-alpha variant is a diagnostic marginal rule.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold

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
    parser.add_argument("--allocation-step", type=float, default=0.005)
    parser.add_argument(
        "--global-alpha-grid",
        default="0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.045,0.05,0.06,0.07,0.08,0.09,0.1,0.12,0.15,0.2",
        help="Global thresholds used for a descriptive budget-matched frontier.",
    )
    parser.add_argument("--seed", type=int, default=941)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    return parser.parse_args()


def modality_thresholds(
    scores: np.ndarray,
    labels: np.ndarray,
    modalities: np.ndarray,
    alpha_by_modality: dict[str, float],
) -> tuple[dict[str, float], dict[str, int], dict[str, int]]:
    thresholds = {}
    orders = {}
    calibration_queries = {}
    for modality, alpha in alpha_by_modality.items():
        modality_labels = labels & (modalities == modality)
        rows = modality_labels.any(axis=1)
        if not rows.any():
            raise ValueError(f"No calibration query has {modality!r} support")
        threshold, order = conformal_threshold(
            scores[rows],
            modality_labels[rows],
            alpha=alpha,
            coverage_target="all_support",
        )
        thresholds[modality] = threshold
        orders[modality] = order
        calibration_queries[modality] = int(rows.sum())
    return thresholds, orders, calibration_queries


def modality_keep_mask(
    scores: np.ndarray,
    modalities: np.ndarray,
    thresholds: dict[str, float],
) -> np.ndarray:
    mask = np.zeros(scores.shape, dtype=bool)
    for modality, threshold in thresholds.items():
        candidates = modalities == modality
        mask |= candidates & (scores >= threshold)
    empty = ~mask.any(axis=1)
    if empty.any():
        best = np.argmax(scores[empty], axis=1)
        mask[np.flatnonzero(empty), best] = True
    return mask


def modality_recall(
    labels: np.ndarray,
    mask: np.ndarray,
    modalities: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    output = {}
    for modality in sorted(set(modalities.ravel().astype(str))):
        support = labels & (modalities == modality)
        total = int(support.sum())
        retained = int((support & mask).sum())
        output[modality] = {
            "support_total": total,
            "support_retained": retained,
            "support_recall": float(retained / total) if total else math.nan,
        }
    return output


def allocation_grid(modalities: list[str], alpha: float, step: float):
    units = round(alpha / step)
    if not math.isclose(units * step, alpha, rel_tol=0, abs_tol=1e-10):
        raise ValueError("alpha must be an integer multiple of allocation-step")
    if units < len(modalities):
        raise ValueError("allocation-step is too large to give every modality alpha > 0")
    for cuts in itertools.combinations(range(1, units), len(modalities) - 1):
        endpoints = (0, *cuts, units)
        allocation = {
            modality: (endpoints[index + 1] - endpoints[index]) * step
            for index, modality in enumerate(modalities)
        }
        yield allocation


def select_allocation(
    scores: np.ndarray,
    labels: np.ndarray,
    modalities: np.ndarray,
    support_modalities: list[str],
    *,
    alpha: float,
    step: float,
    seed: int,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    folds = list(KFold(5, shuffle=True, random_state=seed).split(scores))
    audit = []
    for allocation in allocation_grid(support_modalities, alpha, step):
        held_masks = np.zeros(labels.shape, dtype=bool)
        for fit_rows, held_rows in folds:
            thresholds, _, _ = modality_thresholds(
                scores[fit_rows], labels[fit_rows], modalities[fit_rows], allocation
            )
            held_masks[held_rows] = modality_keep_mask(
                scores[held_rows], modalities[held_rows], thresholds
            )
        retrievable = labels.any(axis=1)
        metrics = conditional_metrics(labels[retrievable], held_masks[retrievable])
        audit.append({"allocation": allocation, **metrics})
    best = min(
        audit,
        key=lambda row: (
            row["mean_chunks_kept"],
            -row["query_all_support_coverage"],
            -row["micro_support_recall"],
        ),
    )
    return dict(best["allocation"]), audit


def evaluate_rule(
    cal_scores: np.ndarray,
    test_scores: np.ndarray,
    cal_labels: np.ndarray,
    test_labels: np.ndarray,
    cal_modalities: np.ndarray,
    test_modalities: np.ndarray,
    alpha_by_modality: dict[str, float],
) -> tuple[dict[str, object], np.ndarray]:
    thresholds, orders, counts = modality_thresholds(
        cal_scores, cal_labels, cal_modalities, alpha_by_modality
    )
    cal_mask = modality_keep_mask(cal_scores, cal_modalities, thresholds)
    mask = modality_keep_mask(test_scores, test_modalities, thresholds)
    cal_retrievable = cal_labels.any(axis=1)
    retrievable = test_labels.any(axis=1)
    return {
        "alpha_by_modality": alpha_by_modality,
        "thresholds": thresholds,
        "finite_sample_orders": orders,
        "calibration_queries_by_support_modality": counts,
        "calibration_mean_chunks_kept": float(
            cal_mask[cal_retrievable].sum(axis=1).mean()
        ),
        "conditional_on_retrievable": conditional_metrics(
            test_labels[retrievable], mask[retrievable]
        ),
        "end_to_end": end_to_end_metrics(test_labels, mask),
        "test_support_by_modality": modality_recall(
            test_labels[retrievable], mask[retrievable], test_modalities[retrievable]
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

    # Recreate only the selected probe's OOF predictions; do not retune on test.
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
    support_modalities = sorted(
        set(cal_modalities[cal_labels].tolist()) | set(test_modalities[test_labels].tolist())
    )
    global_alpha_grid = sorted(
        {float(value) for value in args.global_alpha_grid.split(",") if value.strip()}
    )
    if not global_alpha_grid or any(not 0 < value < 1 for value in global_alpha_grid):
        raise ValueError("global-alpha-grid values must be between zero and one")

    methods = [
        "jina_v4_cosine",
        "cosine_internal",
        "jina_m0_reranker",
        "reranker_internal",
    ]
    equal_allocation = {
        modality: args.alpha / len(support_modalities) for modality in support_modalities
    }
    marginal_allocation = {modality: args.alpha for modality in support_modalities}
    results = {}
    masks_to_save = {}
    allocation_audit = {}
    retrievable = test_labels.any(axis=1)
    for method_index, method in enumerate(methods):
        threshold, order = conformal_threshold(
            cal_scores[method][cal_labels.any(axis=1)],
            cal_labels[cal_labels.any(axis=1)],
            alpha=args.alpha,
            coverage_target="all_support",
        )
        global_mask = keep_mask(test_scores[method], threshold)
        global_cal_mask = keep_mask(cal_scores[method], threshold)
        cal_retrievable = cal_labels.any(axis=1)
        global_result = {
            "alpha": args.alpha,
            "threshold": threshold,
            "finite_sample_order": order,
            "calibration_mean_chunks_kept": float(
                global_cal_mask[cal_retrievable].sum(axis=1).mean()
            ),
            "conditional_on_retrievable": conditional_metrics(
                test_labels[retrievable], global_mask[retrievable]
            ),
            "end_to_end": end_to_end_metrics(test_labels, global_mask),
            "test_support_by_modality": modality_recall(
                test_labels[retrievable],
                global_mask[retrievable],
                test_modalities[retrievable],
            ),
        }
        learned_allocation, audit = select_allocation(
            train_scores[method],
            train_labels,
            train_modalities,
            support_modalities,
            alpha=args.alpha,
            step=args.allocation_step,
            seed=args.seed + method_index,
        )
        allocation_audit[method] = audit
        rules = {"global": global_result}
        masks = {"global": global_mask}
        for rule_name, allocation in (
            ("mondrian_same_alpha", marginal_allocation),
            ("mondrian_bonferroni_equal", equal_allocation),
            ("mondrian_bonferroni_probe_allocated", learned_allocation),
        ):
            rules[rule_name], masks[rule_name] = evaluate_rule(
                cal_scores[method],
                test_scores[method],
                cal_labels,
                test_labels,
                cal_modalities,
                test_modalities,
                allocation,
            )
            rules[rule_name]["paired_bootstrap_minus_global"] = paired_bootstrap(
                test_labels[retrievable],
                masks[rule_name][retrievable],
                global_mask[retrievable],
                seed=args.seed + 100 * method_index + len(rules),
                samples=args.bootstrap_samples,
            )

        global_frontier = []
        for frontier_alpha in global_alpha_grid:
            frontier_threshold, frontier_order = conformal_threshold(
                cal_scores[method][cal_retrievable],
                cal_labels[cal_retrievable],
                alpha=frontier_alpha,
                coverage_target="all_support",
            )
            frontier_cal_mask = keep_mask(cal_scores[method], frontier_threshold)
            frontier_test_mask = keep_mask(test_scores[method], frontier_threshold)
            global_frontier.append(
                {
                    "alpha": frontier_alpha,
                    "threshold": frontier_threshold,
                    "finite_sample_order": frontier_order,
                    "calibration_mean_chunks_kept": float(
                        frontier_cal_mask[cal_retrievable].sum(axis=1).mean()
                    ),
                    "conditional_on_retrievable": conditional_metrics(
                        test_labels[retrievable], frontier_test_mask[retrievable]
                    ),
                    "test_support_by_modality": modality_recall(
                        test_labels[retrievable],
                        frontier_test_mask[retrievable],
                        test_modalities[retrievable],
                    ),
                }
            )
        for rule_name in (
            "mondrian_bonferroni_equal",
            "mondrian_bonferroni_probe_allocated",
        ):
            target = rules[rule_name]["calibration_mean_chunks_kept"]
            matched = min(
                global_frontier,
                key=lambda row: abs(row["calibration_mean_chunks_kept"] - target),
            )
            rules[rule_name]["descriptive_global_budget_match"] = matched
        rules["global_alpha_frontier"] = global_frontier
        results[method] = rules
        for rule_name, mask in masks.items():
            masks_to_save[f"mask_{method}_{rule_name}"] = mask

    output = {
        "status": "complete",
        "dataset": report["dataset"],
        "alpha": args.alpha,
        "support_modalities": support_modalities,
        "protocol": {
            "selection": (
                "5-fold OOF probe queries choose Bonferroni alpha allocation by "
                "minimum mean chunks; disjoint calibration fits thresholds; test is held out"
            ),
            "global_target": "query-level all-support coverage",
            "mondrian_same_alpha": (
                "diagnostic marginal modality coverage; no joint query guarantee"
            ),
            "mondrian_bonferroni": (
                "sum of modality failure budgets equals global alpha; union-bound joint target"
            ),
            "selected_layer": selected_layer,
            "selected_c": selected_c,
            "allocation_step": args.allocation_step,
            "global_alpha_grid": global_alpha_grid,
            "budget_match_note": (
                "nearest global point is selected by calibration mean chunks and is a "
                "descriptive efficiency comparison, not a separately guaranteed target"
            ),
        },
        "results": results,
        "probe_allocation_audit": allocation_audit,
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
