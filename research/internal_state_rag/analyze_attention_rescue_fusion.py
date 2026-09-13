"""Test whether answer-to-chunk attention improves the frozen internal fusion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from research.internal_state_rag.analyze_attention_only_clean_conformal import (
    load_attention,
    subset_attention,
)
from research.internal_state_rag.analyze_hidden_chunk_probe import query_metrics, query_z
from research.internal_state_rag.analyze_pairwise_conformal_clean_split import (
    conditional_metrics,
    conformal_threshold,
    keep_mask,
    probe_predictions,
)


WEIGHTS = (0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0)


def fit_attention_probe(train: dict[str, Any]) -> tuple[StandardScaler, LogisticRegression]:
    x = train["probe_features"].reshape(-1, 2)
    y = train["labels"].ravel().astype(int)
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(
        class_weight="balanced", random_state=0, max_iter=1_000
    ).fit(scaler.transform(x), y)
    return scaler, model


def attention_probe_score(
    data: dict[str, Any], scaler: StandardScaler, model: LogisticRegression
) -> np.ndarray:
    shape = data["labels"].shape
    x = data["probe_features"].reshape(-1, 2)
    return model.predict_proba(scaler.transform(x))[:, 1].reshape(shape)


def take_by_qid(data: Any, values: np.ndarray, qids: list[str]) -> np.ndarray:
    positions = {str(qid): index for index, qid in enumerate(data["qids"].tolist())}
    return values[np.asarray([positions[qid] for qid in qids], dtype=int)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attention-rows", type=Path, required=True)
    parser.add_argument("--attention-test-rows", type=Path, required=True)
    parser.add_argument("--probe-train", type=Path, required=True)
    parser.add_argument("--pairwise-calibration", type=Path, required=True)
    parser.add_argument("--pairwise-test", type=Path, required=True)
    parser.add_argument("--fusion-analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alphas", default="0.2,0.1,0.05")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    attention_train_all = load_attention(args.attention_rows, role="scorer_train")
    attention_cal_all = load_attention(
        args.attention_rows, role="conformal_calibration"
    )
    attention_test = load_attention(args.attention_test_rows, role="evaluation")
    probe_train = np.load(args.probe_train)
    pairwise_cal = np.load(args.pairwise_calibration)
    pairwise_test = np.load(args.pairwise_test)
    analysis = json.loads(args.fusion_analysis.read_text(encoding="utf-8"))
    weights = analysis["selected_three_signal_weights"]
    cal_probe, test_probe = probe_predictions(
        probe_train,
        [pairwise_cal, pairwise_test],
        layer=int(analysis["selected_layer"]),
        c=float(analysis["selected_c"]),
    )

    def internal(data: Any, hidden: np.ndarray) -> np.ndarray:
        return (
            query_z(data["bge_scores"].astype(np.float64))
            + float(weights["lm_head"])
            * query_z(data["lm_relevance_scores"].astype(np.float64))
            + float(weights["hidden_probe"]) * query_z(hidden)
        )

    internal_cal_all = internal(pairwise_cal, cal_probe)
    internal_test_all = internal(pairwise_test, test_probe)
    available_cal_qids = {str(qid) for qid in pairwise_cal["qids"].tolist()}
    train_qids = [
        str(qid)
        for qid in attention_train_all["qids"]
        if str(qid) in available_cal_qids
    ]
    cal_qids = [
        str(qid) for qid in attention_cal_all["qids"] if str(qid) in available_cal_qids
    ]
    test_qids = [str(qid) for qid in attention_test["qids"]]
    scaler, attention_model = fit_attention_probe(attention_train_all)
    attention_train = subset_attention(attention_train_all, train_qids)
    attention_cal = subset_attention(attention_cal_all, cal_qids)
    attention_scores = {
        "trained_probe": {
            "train": query_z(
                attention_probe_score(attention_train, scaler, attention_model)
            ),
            "calibration": query_z(
                attention_probe_score(attention_cal, scaler, attention_model)
            ),
            "test": query_z(
                attention_probe_score(attention_test, scaler, attention_model)
            ),
        },
        "position_controlled": {
            "train": attention_train["scores"]["attention_position_controlled"],
            "calibration": attention_cal["scores"][
                "attention_position_controlled"
            ],
            "test": attention_test["scores"]["attention_position_controlled"],
        },
    }
    internal_scores = {
        "train": take_by_qid(pairwise_cal, internal_cal_all, train_qids),
        "calibration": take_by_qid(pairwise_cal, internal_cal_all, cal_qids),
        "test": take_by_qid(pairwise_test, internal_test_all, test_qids),
    }
    pairwise_cal_labels = take_by_qid(
        pairwise_cal, pairwise_cal["labels"].astype(bool), cal_qids
    )
    pairwise_test_labels = take_by_qid(
        pairwise_test, pairwise_test["labels"].astype(bool), test_qids
    )
    if not np.array_equal(pairwise_cal_labels, attention_cal["labels"]):
        raise ValueError("Calibration labels disagree after qid matching")
    if not np.array_equal(pairwise_test_labels, attention_test["labels"]):
        raise ValueError("Test labels disagree after qid matching")

    tuning = []
    for signal_name, splits in attention_scores.items():
        for weight in WEIGHTS:
            score = internal_scores["train"] + weight * splits["train"]
            metrics = query_metrics(attention_train["labels"], score)
            tuning.append(
                {
                    "signal": signal_name,
                    "weight": weight,
                    **metrics,
                }
            )
    selected = max(
        tuning,
        key=lambda row: (
            row["mean_query_ap"],
            row["mrr"],
            -row["weight"],
        ),
    )
    selected_attention = attention_scores[str(selected["signal"])]
    fused_scores = {
        split: internal_scores[split]
        + float(selected["weight"]) * selected_attention[split]
        for split in ("train", "calibration", "test")
    }
    score_sets = {
        "three_signal_fusion": internal_scores,
        "four_signal_attention_fusion": fused_scores,
    }
    alphas = [float(value) for value in args.alphas.split(",")]
    retention = {}
    for alpha in alphas:
        retention[str(alpha)] = {}
        for name, split_scores in score_sets.items():
            threshold, order = conformal_threshold(
                split_scores["calibration"],
                attention_cal["labels"],
                alpha=alpha,
                coverage_target="all_support",
            )
            mask = keep_mask(split_scores["test"], threshold)
            retention[str(alpha)][name] = {
                "threshold": threshold,
                "finite_sample_order": order,
                "conditional_on_retrievable": conditional_metrics(
                    attention_test["labels"], mask
                ),
            }

    output = {
        "status": "complete",
        "method": "attention_rescue_added_to_frozen_three_signal_fusion",
        "splits": {
            "attention_probe_fit_queries": int(len(attention_train_all["labels"])),
            "attention_probe_train_queries": int(len(attention_train["labels"])),
            "conformal_calibration_queries": int(len(attention_cal["labels"])),
            "locked_test_queries": int(len(attention_test["labels"])),
        },
        "tuning": tuning,
        "selected_attention_signal": selected,
        "test_ranking": {
            name: query_metrics(attention_test["labels"], values["test"])
            for name, values in score_sets.items()
        },
        "conformal_retention": retention,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
