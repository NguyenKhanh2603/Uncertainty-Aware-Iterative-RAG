"""Analyze position-controlled attention and a conformal pruning pilot."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest, spearmanr, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from research.internal_state_rag.ranking import (
    grouped_z_scores,
    leave_one_query_out_logistic,
    ranking_metrics,
)

FEATURES = [
    "bge_score",
    "mean_attention_mass_all_layers",
    "mean_attention_fraction_all_layers",
]


def subset_metrics(
    rows: list[dict[str, Any]], scores: np.ndarray
) -> dict[str, dict[str, float]]:
    output = {}
    for stratum in ("all", "easy", "hard"):
        indices = [
            index
            for index, row in enumerate(rows)
            if stratum == "all" or row["stratum"] == stratum
        ]
        selected_rows = [rows[index] for index in indices]
        output[stratum] = ranking_metrics(selected_rows, scores[indices])
    return output


def query_outcomes(
    rows: list[dict[str, Any]], scores: np.ndarray
) -> dict[str, tuple[bool, float]]:
    outcomes = {}
    for qid in sorted({str(row["qid"]) for row in rows}):
        indices = [index for index, row in enumerate(rows) if str(row["qid"]) == qid]
        indices.sort(key=lambda index: (-scores[index], int(rows[index]["rank"])))
        rank = next(
            position
            for position, index in enumerate(indices, start=1)
            if rows[index]["is_support"]
        )
        outcomes[qid] = (rank == 1, 1.0 / rank)
    return outcomes


def paired_comparison(
    rows: list[dict[str, Any]],
    candidate_scores: np.ndarray,
    baseline_scores: np.ndarray,
    *,
    seed: int,
) -> dict[str, Any]:
    candidate = query_outcomes(rows, candidate_scores)
    baseline = query_outcomes(rows, baseline_scores)
    qids = sorted(candidate)
    gains = sum(candidate[qid][0] and not baseline[qid][0] for qid in qids)
    losses = sum(baseline[qid][0] and not candidate[qid][0] for qid in qids)
    differences = np.asarray(
        [candidate[qid][1] - baseline[qid][1] for qid in qids]
    )
    rng = np.random.default_rng(seed)
    bootstrap = np.asarray(
        [rng.choice(differences, len(differences), replace=True).mean() for _ in range(10_000)]
    )
    return {
        "top1_gains": gains,
        "top1_losses": losses,
        "top1_exact_sign_test_p": float(
            binomtest(gains, gains + losses, 0.5).pvalue
        ),
        "mrr_difference": float(differences.mean()),
        "mrr_wilcoxon_p": float(wilcoxon(differences).pvalue),
        "mrr_bootstrap_ci95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
    }


def order_stability(rows: list[dict[str, Any]], score_name: str) -> dict[str, float]:
    correlations = []
    for qid in sorted({str(row["qid"]) for row in rows}):
        selected = [row for row in rows if str(row["qid"]) == qid]
        original = [float(row[f"original_{score_name}"]) for row in selected]
        reversed_values = [float(row[f"reversed_{score_name}"]) for row in selected]
        correlations.append(float(spearmanr(original, reversed_values).statistic))
    values = np.asarray(correlations)
    return {
        "mean_query_spearman": float(np.nanmean(values)),
        "median_query_spearman": float(np.nanmedian(values)),
        "fraction_positive": float(np.nanmean(values > 0)),
    }


def conformal_threshold(
    rows: list[dict[str, Any]], scores: np.ndarray, qids: set[str], alpha: float
) -> float:
    """Calibrate a threshold on each query's lowest support score."""

    groups = np.asarray([str(row["qid"]) for row in rows])
    labels = np.asarray([bool(row["is_support"]) for row in rows])
    minima = []
    for qid in qids:
        indices = np.flatnonzero((groups == qid) & labels)
        minima.append(float(scores[indices].min()))
    # The k-th smallest support minimum allows at most k/(n+1) lower-tail errors.
    k = max(1, math.floor(alpha * (len(minima) + 1)))
    return float(np.sort(minima)[k - 1])


def pruning_metrics(
    rows: list[dict[str, Any]], scores: np.ndarray, qids: set[str], threshold: float
) -> dict[str, float]:
    groups = np.asarray([str(row["qid"]) for row in rows])
    labels = np.asarray([bool(row["is_support"]) for row in rows])
    selected = np.asarray([qid in qids for qid in groups])
    kept = scores[selected] >= threshold
    selected_labels = labels[selected]
    all_support_coverage = []
    for qid in qids:
        indices = np.flatnonzero((groups == qid) & labels)
        all_support_coverage.append(bool(np.all(scores[indices] >= threshold)))
    return {
        "query_all_support_coverage": float(np.mean(all_support_coverage)),
        "support_chunk_recall": float(np.mean(kept[selected_labels])),
        "false_chunk_pruning_rate": float(np.mean(~kept[~selected_labels])),
        "candidate_keep_rate": float(np.mean(kept)),
    }


def three_way_conformal_trials(
    rows: list[dict[str, Any]],
    *,
    alpha: float,
    trials: int,
    seed: int,
) -> dict[str, Any]:
    """Train fusion, calibrate threshold, and evaluate on disjoint query thirds."""

    groups = np.asarray([str(row["qid"]) for row in rows])
    labels = np.asarray([int(row["is_support"]) for row in rows])
    qids = sorted(set(groups))
    strata = {
        qid: str(next(row["stratum"] for row in rows if str(row["qid"]) == qid))
        for qid in qids
    }
    features = np.column_stack(
        [grouped_z_scores(rows, feature_name) for feature_name in FEATURES]
    )
    base_scores = {"bge": features[:, 0], "attention": features[:, 1]}
    rng = np.random.default_rng(seed)
    collected: dict[str, list[dict[str, float]]] = {
        "bge": [],
        "attention": [],
        "fusion": [],
    }

    for _ in range(trials):
        parts: list[list[str]] = [[], [], []]
        for stratum in ("easy", "hard"):
            stratum_qids = np.asarray([qid for qid in qids if strata[qid] == stratum])
            rng.shuffle(stratum_qids)
            thirds = np.array_split(stratum_qids, 3)
            for index, third in enumerate(thirds):
                parts[index].extend(str(qid) for qid in third)
        train_qids, calibration_qids, evaluation_qids = map(set, parts)
        train = np.asarray([qid in train_qids for qid in groups])
        scaler = StandardScaler().fit(features[train])
        model = LogisticRegression(
            class_weight="balanced", random_state=0, max_iter=1_000
        ).fit(scaler.transform(features[train]), labels[train])
        scores = {
            **base_scores,
            "fusion": model.predict_proba(scaler.transform(features))[:, 1],
        }
        for name, values in scores.items():
            threshold = conformal_threshold(
                rows, values, calibration_qids, alpha
            )
            collected[name].append(
                pruning_metrics(rows, values, evaluation_qids, threshold)
            )

    output = {}
    for name, trial_rows in collected.items():
        output[name] = {}
        for metric_name in trial_rows[0]:
            values = np.asarray([row[metric_name] for row in trial_rows])
            output[name][metric_name] = {
                "mean": float(values.mean()),
                "p10": float(np.quantile(values, 0.1)),
                "p90": float(np.quantile(values, 0.9)),
            }
    return {
        "alpha": alpha,
        "trials": trials,
        "split": "40 train / 40 calibration / 40 evaluation; 20 easy + 20 hard each",
        "features": FEATURES,
        "metrics": output,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=55)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    rows = payload["observations"]
    raw_scores = {
        feature_name: grouped_z_scores(rows, feature_name)
        for feature_name in FEATURES
    }
    fusion_scores = leave_one_query_out_logistic(rows, FEATURES)
    all_scores = {**raw_scores, "fusion_leave_one_query_out": fusion_scores}
    bge_scores = raw_scores["bge_score"]
    analysis = {
        "input": str(args.input),
        "n_queries": len({str(row["qid"]) for row in rows}),
        "n_candidates": len(rows),
        "ranking": {
            name: subset_metrics(rows, scores) for name, scores in all_scores.items()
        },
        "paired_vs_bge": {
            name: paired_comparison(rows, scores, bge_scores, seed=args.seed)
            for name, scores in all_scores.items()
            if name != "bge_score"
        },
        "order_stability": {
            score_name: order_stability(rows, score_name)
            for score_name in (
                "attention_mass_all_layers",
                "attention_fraction_all_layers",
                "attention_mass_focus",
                "attention_fraction_focus",
            )
        },
        "conformal_pruning_pilot": three_way_conformal_trials(
            rows, alpha=args.alpha, trials=args.trials, seed=args.seed
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
