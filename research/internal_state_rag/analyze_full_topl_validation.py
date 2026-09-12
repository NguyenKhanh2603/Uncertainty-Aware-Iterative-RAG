"""Fit, conformally calibrate, and evaluate full-Top-L attention fusion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from research.internal_state_rag.analyze_position_control import conformal_threshold
from research.internal_state_rag.ranking import grouped_z_scores, ranking_metrics

FEATURES = ["bge_score", "mean_attention_mass", "mean_attention_fraction"]


def load_observations(path: Path) -> list[dict[str, Any]]:
    observations = []
    for line in path.read_text(encoding="utf-8").splitlines():
        query = json.loads(line)
        if query.get("status") != "complete":
            continue
        for candidate in query["candidates"]:
            observations.append(
                {
                    **candidate,
                    "qid": str(query["qid"]),
                    "role": str(query["role"]),
                    "stratum": str(query["stratum"]),
                }
            )
    return observations


def subset_ranking(
    rows: list[dict[str, Any]], scores: np.ndarray, *, role: str
) -> dict[str, dict[str, float]]:
    output = {}
    for stratum in ("all", "easy", "hard"):
        indices = [
            index
            for index, row in enumerate(rows)
            if row["role"] == role
            and (stratum == "all" or row["stratum"] == stratum)
        ]
        selected = [rows[index] for index in indices]
        output[stratum] = {
            "n_queries": len({row["qid"] for row in selected}),
            **ranking_metrics(selected, scores[indices]),
        }
    return output


def retention_metrics(
    rows: list[dict[str, Any]], scores: np.ndarray, *, role: str, threshold: float
) -> dict[str, float]:
    indices = [index for index, row in enumerate(rows) if row["role"] == role]
    kept = scores[indices] >= threshold
    labels = np.asarray([bool(rows[index]["is_support"]) for index in indices])
    token_counts = np.asarray([int(rows[index]["chunk_token_count"]) for index in indices])
    qids = sorted({rows[index]["qid"] for index in indices})
    all_support_coverage = []
    kept_per_query = []
    for qid in qids:
        query_indices = [
            index
            for index in indices
            if rows[index]["qid"] == qid
        ]
        support_indices = [index for index in query_indices if rows[index]["is_support"]]
        all_support_coverage.append(bool(np.all(scores[support_indices] >= threshold)))
        kept_per_query.append(int(np.sum(scores[query_indices] >= threshold)))
    return {
        "n_queries": len(qids),
        "query_all_support_coverage": float(np.mean(all_support_coverage)),
        "harmful_prune_rate": float(1 - np.mean(all_support_coverage)),
        "support_chunk_recall": float(np.mean(kept[labels])),
        "false_chunk_pruning_rate": float(np.mean(~kept[~labels])),
        "candidate_keep_rate": float(np.mean(kept)),
        "context_token_keep_rate": float(token_counts[kept].sum() / token_counts.sum()),
        "mean_chunks_kept": float(np.mean(kept_per_query)),
        "median_chunks_kept": float(np.median(kept_per_query)),
        "empty_context_rate": float(np.mean(np.asarray(kept_per_query) == 0)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_observations(args.rows)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    features = np.column_stack([grouped_z_scores(rows, name) for name in FEATURES])
    labels = np.asarray([int(row["is_support"]) for row in rows])
    roles = np.asarray([str(row["role"]) for row in rows])
    train = roles == "scorer_train"
    scaler = StandardScaler().fit(features[train])
    model = LogisticRegression(
        class_weight="balanced", random_state=0, max_iter=1_000
    ).fit(scaler.transform(features[train]), labels[train])
    score_values = {
        "cosine": grouped_z_scores(rows, "cosine_score"),
        "bge": features[:, 0],
        "original_attention": grouped_z_scores(rows, "original_attention_mass"),
        "reversed_attention": grouped_z_scores(rows, "reversed_attention_mass"),
        "position_controlled_attention": features[:, 1],
        "fusion": model.predict_proba(scaler.transform(features))[:, 1],
    }
    calibration_qids = {
        str(row["qid"]) for row in rows if row["role"] == "conformal_calibration"
    }
    calibration = {}
    for alpha in (0.05, 0.1, 0.2):
        calibration[str(alpha)] = {}
        for name, scores in score_values.items():
            threshold = conformal_threshold(rows, scores, calibration_qids, alpha)
            calibration[str(alpha)][name] = {
                "threshold": threshold,
                "calibration": retention_metrics(
                    rows,
                    scores,
                    role="conformal_calibration",
                    threshold=threshold,
                ),
                "evaluation": retention_metrics(
                    rows, scores, role="evaluation", threshold=threshold
                ),
            }

    output = {
        "protocol": {
            "dataset": manifest["dataset"],
            "input_role": manifest["input_role"],
            "top_l": manifest["top_l"],
            "conditional_retrieval_ceiling": manifest["conditional_retrieval_ceiling"],
            "role_query_counts": {
                role: len({row["qid"] for row in rows if row["role"] == role})
                for role in (
                    "scorer_train",
                    "conformal_calibration",
                    "evaluation",
                )
            },
            "n_candidates": len(rows),
        },
        "fusion": {
            "features": FEATURES,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "coefficients": model.coef_[0].tolist(),
            "intercept": float(model.intercept_[0]),
        },
        "evaluation_ranking": {
            name: subset_ranking(rows, scores, role="evaluation")
            for name, scores in score_values.items()
        },
        "conformal_retention": calibration,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
