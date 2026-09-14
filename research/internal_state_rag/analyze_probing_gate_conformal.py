"""Fit a query-level Probing-RAG gate and evaluate gate-conditional conformal pruning.

The gate is trained on development query states produced from generated answers.
The calibration split is used to set the ordinary Jina-reranker conformal
threshold and a separate threshold inside each gate stratum (Mondrian split
conformal).  Official test labels are used only for the final audit.

``no_support_topK`` is an operational pruning-risk target: if the first K
retrieved candidates contain no labelled support, the gate should request a more
conservative context or another retrieval round.  It is not presented as a
ground-truth semantic uncertainty label.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from research.internal_state_rag.analyze_hidden_chunk_probe import query_metrics, query_z
from research.internal_state_rag.analyze_pairwise_conformal_clean_split import (
    conditional_metrics,
    conformal_threshold,
    end_to_end_metrics,
    keep_mask,
)


def load_gate(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = {str(row["qid"]): row for row in payload.get("queries", [])}
    if not rows:
        raise ValueError(f"No query states found in {path}")
    widths = {len(row["state_features"]) for row in rows.values()}
    if len(widths) != 1:
        raise ValueError(f"Inconsistent state feature widths in {path}: {widths}")
    return rows, payload


def align_features(
    gate_rows: dict[str, dict[str, Any]],
    npz_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    values = np.load(npz_path, allow_pickle=False)
    all_qids = values["qids"].astype(str)
    # Gate extraction may intentionally cover a bounded pilot subset while the
    # frozen feature artifact contains the complete split.  Restrict the
    # feature rows to the gate qids, retaining artifact order so labels and
    # reranker scores stay aligned.  A gate qid absent from the artifact is a
    # real alignment error and should still fail loudly.
    feature_qid_set = set(all_qids)
    missing_from_features = set(gate_rows).difference(feature_qid_set)
    if missing_from_features:
        raise ValueError(
            f"{len(missing_from_features)} gate qids are missing from feature artifact "
            f"({npz_path})"
        )
    indices = np.asarray(
        [index for index, qid in enumerate(all_qids) if qid in gate_rows], dtype=np.int64
    )
    if len(indices) == 0:
        raise ValueError(f"No gate qids found in feature artifact ({npz_path})")
    qids = all_qids[indices]
    ordered = [gate_rows[str(qid)] for qid in qids]
    x = np.asarray([row["state_features"] for row in ordered], dtype=np.float64)
    labels = values["labels"][indices].astype(bool)
    scores = values["reranker_scores"][indices].astype(np.float64)
    if not np.isfinite(x).all():
        raise ValueError("Gate features contain non-finite values")
    return x, labels, scores, qids, np.asarray([row["risk_labels"] for row in ordered], dtype=object)


def risk_vector(
    gate_rows: dict[str, dict[str, Any]], qids: np.ndarray, *, top_k: int
) -> np.ndarray:
    key = f"no_support_top{top_k}"
    values = []
    for qid in qids.astype(str):
        row = gate_rows[str(qid)]
        if key not in row["risk_labels"]:
            raise ValueError(f"Risk label {key} absent for {qid}")
        values.append(bool(row["risk_labels"][key]))
    return np.asarray(values, dtype=bool)


def safe_classifier(
    x: np.ndarray, y: np.ndarray, *, c: float
) -> tuple[Any, str]:
    """Fit a gate, falling back to a constant for a one-class split."""

    if len(np.unique(y)) < 2:
        return None, "constant"
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c,
            class_weight="balanced",
            solver="liblinear",
            max_iter=3000,
            random_state=0,
        ),
    )
    model.fit(x, y)
    return model, "logistic_standardized"


def gate_probability(model: Any, x: np.ndarray, fallback: float) -> np.ndarray:
    if model is None:
        return np.full(len(x), fallback, dtype=np.float64)
    return model.predict_proba(x)[:, 1].astype(np.float64)


def classification_metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {
        "queries": int(len(y)),
        "positive_rate": float(np.mean(y)) if len(y) else None,
    }
    if len(np.unique(y)) < 2:
        result.update({"roc_auc": None, "average_precision": None})
    else:
        result.update(
            {
                "roc_auc": float(roc_auc_score(y, probability)),
                "average_precision": float(average_precision_score(y, probability)),
            }
        )
    return result


def metrics(labels: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    retrievable = labels.any(axis=1)
    output: dict[str, Any] = {
        "retrieval_ceiling": float(np.mean(retrievable)),
        "all_test_queries": end_to_end_metrics(labels, mask),
    }
    if retrievable.any():
        output["conditional_retrievable"] = conditional_metrics(
            labels[retrievable], mask[retrievable]
        )
    else:
        output["conditional_retrievable"] = None
    return output


def baseline_threshold(
    calibration_scores: np.ndarray,
    calibration_labels: np.ndarray,
    *,
    alpha: float,
) -> tuple[float, int]:
    retrievable = calibration_labels.any(axis=1)
    if not retrievable.any():
        raise ValueError("Calibration has no retrievable queries")
    return conformal_threshold(
        calibration_scores[retrievable],
        calibration_labels[retrievable],
        alpha=alpha,
        coverage_target="all_support",
    )


def mondrian_mask(
    calibration_scores: np.ndarray,
    calibration_labels: np.ndarray,
    calibration_gate: np.ndarray,
    test_scores: np.ndarray,
    test_labels: np.ndarray,
    test_gate: np.ndarray,
    *,
    gate_threshold: float,
    alpha: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Calibrate an all-support threshold separately in easy/hard gate strata."""

    calibration_retrievable = calibration_labels.any(axis=1)
    test_mask = np.zeros_like(test_labels, dtype=bool)
    strata: dict[str, Any] = {}
    for name, hard in (("easy", False), ("hard", True)):
        calibration_group = calibration_retrievable & (
            (calibration_gate >= gate_threshold) == hard
        )
        test_group = (test_gate >= gate_threshold) == hard
        if calibration_group.any():
            threshold, order = conformal_threshold(
                calibration_scores[calibration_group],
                calibration_labels[calibration_group],
                alpha=alpha,
                coverage_target="all_support",
            )
            test_mask[test_group] = keep_mask(test_scores[test_group], threshold)
            strata[name] = {
                "calibration_retrievable_queries": int(calibration_group.sum()),
                "test_queries": int(test_group.sum()),
                "threshold": float(threshold),
                "finite_sample_order": int(order),
            }
        else:
            # A missing calibration stratum cannot certify a threshold. Keep all
            # candidates for that test stratum and expose the limitation.
            test_mask[test_group] = True
            strata[name] = {
                "calibration_retrievable_queries": 0,
                "test_queries": int(test_group.sum()),
                "threshold": None,
                "finite_sample_order": None,
                "fallback": "keep_all_candidates",
            }
    return test_mask, strata


def rescue_mask(
    baseline: np.ndarray,
    scores: np.ndarray,
    gate_probability_values: np.ndarray,
    *,
    gate_threshold: float,
    rescue_k: int,
) -> np.ndarray:
    """Heuristic union of conformal output and Top-k for gate-hard queries."""

    output = baseline.copy()
    hard = gate_probability_values >= gate_threshold
    if hard.any():
        order = np.argsort(-scores[hard], axis=1, kind="stable")[:, :rescue_k]
        rows = np.flatnonzero(hard)
        for row, indices in zip(rows, order, strict=True):
            output[row, indices] = True
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-gate", type=Path, required=True)
    parser.add_argument("--calibration-gate", type=Path, required=True)
    parser.add_argument("--test-gate", type=Path, required=True)
    parser.add_argument("--probe-features", type=Path, required=True)
    parser.add_argument("--calibration-features", type=Path, required=True)
    parser.add_argument("--test-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--risk-top-k", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gate-threshold", type=float, default=0.5)
    parser.add_argument("--rescue-k", type=int, default=10)
    parser.add_argument("--probe-c", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.risk_top_k < 1 or args.risk_top_k > 30:
        raise ValueError("risk top-k must lie in [1, 30]")
    if not 0 < args.alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    if not 0 <= args.gate_threshold <= 1:
        raise ValueError("gate threshold must lie in [0, 1]")
    if args.rescue_k < 1 or args.rescue_k > 30:
        raise ValueError("rescue-k must lie in [1, 30]")

    probe_rows, probe_payload = load_gate(args.probe_gate)
    calibration_rows, calibration_payload = load_gate(args.calibration_gate)
    test_rows, test_payload = load_gate(args.test_gate)
    probe_x, probe_labels, probe_scores, probe_qids, _ = align_features(
        probe_rows, args.probe_features
    )
    calibration_x, calibration_labels, calibration_scores, calibration_qids, _ = align_features(
        calibration_rows, args.calibration_features
    )
    test_x, test_labels, test_scores, test_qids, _ = align_features(
        test_rows, args.test_features
    )
    qid_sets = [set(probe_qids), set(calibration_qids), set(test_qids)]
    overlaps = {
        "probe_calibration": len(qid_sets[0] & qid_sets[1]),
        "probe_test": len(qid_sets[0] & qid_sets[2]),
        "calibration_test": len(qid_sets[1] & qid_sets[2]),
    }
    if any(overlaps.values()):
        raise ValueError(f"Gate query splits overlap: {overlaps}")

    probe_risk = risk_vector(probe_rows, probe_qids, top_k=args.risk_top_k)
    calibration_risk = risk_vector(
        calibration_rows, calibration_qids, top_k=args.risk_top_k
    )
    test_risk = risk_vector(test_rows, test_qids, top_k=args.risk_top_k)
    model, model_type = safe_classifier(probe_x, probe_risk, c=args.probe_c)
    fallback = float(np.mean(probe_risk))
    probe_probability = gate_probability(model, probe_x, fallback)
    calibration_probability = gate_probability(model, calibration_x, fallback)
    test_probability = gate_probability(model, test_x, fallback)

    # Candidate score is the already-frozen Jina reranker, normalized within
    # each query in the same way as the existing query-level conformal baseline.
    probe_rank_scores = query_z(probe_scores)
    calibration_rank_scores = query_z(calibration_scores)
    test_rank_scores = query_z(test_scores)
    base_threshold, base_order = baseline_threshold(
        calibration_rank_scores, calibration_labels, alpha=args.alpha
    )
    base_mask = keep_mask(test_rank_scores, base_threshold)
    gate_mask, strata = mondrian_mask(
        calibration_rank_scores,
        calibration_labels,
        calibration_probability,
        test_rank_scores,
        test_labels,
        test_probability,
        gate_threshold=args.gate_threshold,
        alpha=args.alpha,
    )
    rescue = rescue_mask(
        base_mask,
        test_rank_scores,
        test_probability,
        gate_threshold=args.gate_threshold,
        rescue_k=args.rescue_k,
    )

    output = {
        "status": "complete",
        "method": "probing_rag_query_gate_mondrian_conformal_pruning",
        "gate": {
            "model_type": model_type,
            "risk_target": f"no_support_top{args.risk_top_k}",
            "probe_c": args.probe_c,
            "gate_threshold": args.gate_threshold,
            "probe": classification_metrics(probe_risk, probe_probability),
            "calibration": classification_metrics(calibration_risk, calibration_probability),
            "test": classification_metrics(test_risk, test_probability),
            "hard_rate_calibration": float(np.mean(calibration_probability >= args.gate_threshold)),
            "hard_rate_test": float(np.mean(test_probability >= args.gate_threshold)),
        },
        "conformal": {
            "alpha": args.alpha,
            "score": "within-query z-normalized Jina reranker score",
            "baseline": {
                "threshold": float(base_threshold),
                "finite_sample_order": int(base_order),
                "metrics": metrics(test_labels, base_mask),
            },
            "probing_gate_mondrian": {
                "strata": strata,
                "metrics": metrics(test_labels, gate_mask),
            },
            "probing_gate_rescue_heuristic": {
                "rescue_k": args.rescue_k,
                "metrics": metrics(test_labels, rescue),
                "guarantee": "heuristic union; report separately from conformal Mondrian result",
            },
        },
        "splits": {
            "probe_queries": int(len(probe_qids)),
            "calibration_queries": int(len(calibration_qids)),
            "test_queries": int(len(test_qids)),
            "probe_retrievable": int(probe_labels.any(axis=1).sum()),
            "calibration_retrievable": int(calibration_labels.any(axis=1).sum()),
            "test_retrievable": int(test_labels.any(axis=1).sum()),
            "query_id_overlaps": overlaps,
        },
        "state_artifacts": {
            "probe": str(args.probe_gate),
            "calibration": str(args.calibration_gate),
            "test": str(args.test_gate),
            "probe_model": probe_payload.get("model"),
            "calibration_model": calibration_payload.get("model"),
            "test_model": test_payload.get("model"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
