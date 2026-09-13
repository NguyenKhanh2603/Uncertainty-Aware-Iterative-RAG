"""Compare cosine BY, attention-only, BGE, and internal-fusion pruning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from research.internal_state_rag.analyze_hidden_chunk_probe import query_z
from research.internal_state_rag.analyze_pairwise_conformal_clean_split import (
    conditional_metrics,
    conformal_threshold,
    end_to_end_metrics,
    fixed_k_metrics,
    keep_mask,
)


ATTENTION_SIGNALS = {
    "attention_raw": "original_attention_mass",
    "attention_position_controlled": "mean_attention_mass",
}


def load_attention(path: Path, *, role: str = "") -> dict[str, Any]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") != "complete":
            continue
        if role and row.get("role") != role:
            continue
        rows.append(row)
    if not rows:
        raise ValueError(f"No complete attention rows in {path}")
    top_l = len(rows[0]["candidates"])
    if any(len(row["candidates"]) != top_l for row in rows):
        raise ValueError(f"Inconsistent candidate counts in {path}")
    return {
        "qids": np.asarray([str(row["qid"]) for row in rows]),
        "labels": np.asarray(
            [[bool(c["is_support"]) for c in row["candidates"]] for row in rows]
        ),
        "scores": {
            name: query_z(
                np.asarray(
                    [[float(c[field]) for c in row["candidates"]] for row in rows]
                )
            )
            for name, field in ATTENTION_SIGNALS.items()
        },
        "probe_features": np.stack(
            [
                query_z(
                    np.asarray(
                        [
                            [float(c[field]) for c in row["candidates"]]
                            for row in rows
                        ]
                    )
                )
                for field in ("mean_attention_mass", "mean_attention_fraction")
            ],
            axis=-1,
        ),
    }


def by_rows(path: Path, alphas: list[float]) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    output = []
    for strategy in ("dataset_pooled", "modality_aware"):
        for alpha in alphas:
            metrics = payload["strategies"][strategy][str(alpha)]["overall"]
            output.append(
                {
                    "method": f"cosine_BY_{strategy}",
                    "selection_rule": "candidate-wise BY-FDR",
                    "alpha": alpha,
                    "evaluation_scope": "all 1000 test queries",
                    "n_queries": int(metrics["queries"]),
                    "mean_chunks_kept": float(metrics["average_selected_chunks"]),
                    "chunk_precision": float(metrics["micro_evidence_precision"]),
                    "micro_support_recall": float(
                        metrics["conditional_reserve_support_recall"]
                    ),
                    "query_all_support_coverage": None,
                    "empty_context_rate": float(metrics["empty_context_rate"]),
                }
            )
    return output


def query_conformal_rows(
    calibration: dict[str, Any],
    test: dict[str, Any],
    *,
    alphas: list[float],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cal_retrievable = calibration["labels"].any(axis=1)
    test_retrievable = test["labels"].any(axis=1)
    output, details = [], {}
    for name in ATTENTION_SIGNALS:
        details[name] = {
            "ranking_on_retrievable_test_queries": fixed_k_metrics(
                test["labels"], test["scores"][name]
            )
        }
        for alpha in alphas:
            threshold, order = conformal_threshold(
                calibration["scores"][name][cal_retrievable],
                calibration["labels"][cal_retrievable],
                alpha=alpha,
                coverage_target="all_support",
            )
            mask = keep_mask(test["scores"][name], threshold)
            conditional = conditional_metrics(
                test["labels"][test_retrievable], mask[test_retrievable]
            )
            end_to_end = end_to_end_metrics(test["labels"], mask)
            output.append(
                {
                    "method": name,
                    "selection_rule": "query-level conformal all-support",
                    "alpha": alpha,
                    "evaluation_scope": "retrievable test queries",
                    "n_queries": conditional["queries"],
                    "mean_chunks_kept": conditional["mean_chunks_kept"],
                    "chunk_precision": conditional["chunk_precision_among_kept"],
                    "micro_support_recall": conditional["micro_support_recall"],
                    "mean_support_recall": conditional["mean_support_recall"],
                    "query_all_support_coverage": conditional[
                        "query_all_support_coverage"
                    ],
                    "empty_context_rate": 0.0,
                    "end_to_end_query_all_support_coverage": end_to_end[
                        "query_all_support_coverage"
                    ],
                    "calibrated_threshold": threshold,
                    "finite_sample_order": order,
                }
            )
            details[name][str(alpha)] = {
                "calibrated_threshold": threshold,
                "finite_sample_order": order,
                "conditional_on_retrievable": conditional,
                "end_to_end_all_test_queries": end_to_end,
            }
    return output, details


def learned_attention_scores(
    train: dict[str, Any], calibration: dict[str, Any], test: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit the old attention-only logistic probe without BGE features."""

    train_x = train["probe_features"].reshape(-1, 2)
    train_y = train["labels"].ravel().astype(int)
    scaler = StandardScaler().fit(train_x)
    model = LogisticRegression(
        class_weight="balanced", random_state=0, max_iter=1_000
    ).fit(scaler.transform(train_x), train_y)

    def predict(data: dict[str, Any]) -> np.ndarray:
        shape = data["labels"].shape
        values = data["probe_features"].reshape(-1, 2)
        return model.predict_proba(scaler.transform(values))[:, 1].reshape(shape)

    metadata = {
        "features": ["mean_attention_mass", "mean_attention_fraction"],
        "training_queries": int(len(train["labels"])),
        "coefficients": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
    }
    return predict(calibration), predict(test), metadata


def learned_attention_rows(
    train: dict[str, Any],
    calibration: dict[str, Any],
    test: dict[str, Any],
    *,
    alphas: list[float],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    calibration_scores, test_scores, metadata = learned_attention_scores(
        train, calibration, test
    )
    cal_retrievable = calibration["labels"].any(axis=1)
    test_retrievable = test["labels"].any(axis=1)
    details = {
        **metadata,
        "ranking_on_retrievable_test_queries": fixed_k_metrics(
            test["labels"], test_scores
        ),
    }
    output = []
    for alpha in alphas:
        threshold, order = conformal_threshold(
            calibration_scores[cal_retrievable],
            calibration["labels"][cal_retrievable],
            alpha=alpha,
            coverage_target="all_support",
        )
        mask = keep_mask(test_scores, threshold)
        conditional = conditional_metrics(
            test["labels"][test_retrievable], mask[test_retrievable]
        )
        end_to_end = end_to_end_metrics(test["labels"], mask)
        output.append(
            {
                "method": "attention_learned_probe",
                "selection_rule": "query-level conformal all-support",
                "alpha": alpha,
                "evaluation_scope": "retrievable test queries",
                "n_queries": conditional["queries"],
                "mean_chunks_kept": conditional["mean_chunks_kept"],
                "chunk_precision": conditional["chunk_precision_among_kept"],
                "micro_support_recall": conditional["micro_support_recall"],
                "mean_support_recall": conditional["mean_support_recall"],
                "query_all_support_coverage": conditional[
                    "query_all_support_coverage"
                ],
                "empty_context_rate": 0.0,
                "end_to_end_query_all_support_coverage": end_to_end[
                    "query_all_support_coverage"
                ],
                "calibrated_threshold": threshold,
                "finite_sample_order": order,
            }
        )
        details[str(alpha)] = {
            "calibrated_threshold": threshold,
            "finite_sample_order": order,
            "conditional_on_retrievable": conditional,
            "end_to_end_all_test_queries": end_to_end,
        }
    return output, details


def internal_rows(path: Path, alphas: list[float]) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    output = []
    for method in ("bge", "three_signal_fusion"):
        for alpha in alphas:
            result = payload["conformal"]["all_support"][str(alpha)][method]
            metrics = result["conditional_on_retrievable"]
            end_to_end = result["end_to_end_all_test_queries"]
            output.append(
                {
                    "method": method,
                    "selection_rule": "query-level conformal all-support",
                    "alpha": alpha,
                    "evaluation_scope": "retrievable test queries",
                    "n_queries": metrics["queries"],
                    "mean_chunks_kept": metrics["mean_chunks_kept"],
                    "chunk_precision": metrics["chunk_precision_among_kept"],
                    "micro_support_recall": metrics["micro_support_recall"],
                    "mean_support_recall": metrics["mean_support_recall"],
                    "query_all_support_coverage": metrics[
                        "query_all_support_coverage"
                    ],
                    "empty_context_rate": 0.0,
                    "end_to_end_query_all_support_coverage": end_to_end[
                        "query_all_support_coverage"
                    ],
                    "calibrated_threshold": result["calibrated_threshold"],
                    "finite_sample_order": result["finite_sample_order"],
                }
            )
    return output


def markdown_table(rows: list[dict[str, Any]]) -> str:
    names = {
        "cosine_BY_dataset_pooled": "Cosine + BY (pooled)",
        "cosine_BY_modality_aware": "Cosine + BY (modality)",
        "attention_raw": "Attention only (raw)",
        "attention_position_controlled": "Attention only (position-controlled)",
        "attention_learned_probe": "Attention-only trained probe",
        "bge": "BGE",
        "three_signal_fusion": "BGE + LM-head + hidden probe",
    }
    lines = [
        "| Method | Rule | α | Scope (n) | Chunks | Precision | Micro recall | Mean query recall | All-support coverage | Empty |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda x: (-x["alpha"], x["method"])):
        def pct(key: str) -> str:
            value = row.get(key)
            return "—" if value is None else f"{100 * value:.1f}%"

        rule = "BY-FDR" if "BY" in row["method"] else "query conformal"
        lines.append(
            "| {name} | {rule} | {alpha:.2f} | {n} | {chunks:.2f} | {precision} | "
            "{recall} | {mean_recall} | {coverage} | {empty} |".format(
                name=names[row["method"]],
                rule=rule,
                alpha=row["alpha"],
                n=row["n_queries"],
                chunks=row["mean_chunks_kept"],
                precision=pct("chunk_precision"),
                recall=pct("micro_support_recall"),
                mean_recall=pct("mean_support_recall"),
                coverage=pct("query_all_support_coverage"),
                empty=pct("empty_context_rate"),
            )
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-rows", type=Path, required=True)
    parser.add_argument("--attention-train-rows", type=Path)
    parser.add_argument("--test-rows", type=Path, required=True)
    parser.add_argument("--internal-results", type=Path, required=True)
    parser.add_argument("--by-results", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--alphas", default="0.2,0.1,0.05")
    parser.add_argument("--calibration-role", default="")
    parser.add_argument("--attention-train-role", default="scorer_train")
    parser.add_argument("--test-role", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    alphas = [float(value) for value in args.alphas.split(",")]
    calibration = load_attention(args.calibration_rows, role=args.calibration_role)
    test = load_attention(args.test_rows, role=args.test_role)
    overlap = len(set(calibration["qids"]) & set(test["qids"]))
    if overlap:
        raise ValueError(f"Calibration/test query overlap: {overlap}")
    attention_table, attention_details = query_conformal_rows(
        calibration, test, alphas=alphas
    )
    learned_table: list[dict[str, Any]] = []
    if args.attention_train_rows:
        attention_train = load_attention(
            args.attention_train_rows, role=args.attention_train_role
        )
        learned_table, learned_details = learned_attention_rows(
            attention_train, calibration, test, alphas=alphas
        )
        attention_details["attention_learned_probe"] = learned_details
    rows = [
        *by_rows(args.by_results, alphas),
        *attention_table,
        *learned_table,
        *internal_rows(args.internal_results, alphas),
    ]
    output = {
        "status": "complete",
        "comparison_note": (
            "BY controls candidate-level false discovery and can return an empty set. "
            "Query-level conformal targets retention of every support chunk conditional "
            "on support existing in Top-30 and uses deterministic Top-1 fallback."
        ),
        "splits": {
            "attention_calibration_total": int(len(calibration["labels"])),
            "attention_calibration_retrievable": int(
                calibration["labels"].any(axis=1).sum()
            ),
            "attention_test_total": int(len(test["labels"])),
            "attention_test_retrievable": int(test["labels"].any(axis=1).sum()),
            "overlap": overlap,
        },
        "table": rows,
        "attention_details": attention_details,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    markdown = (
        "# Full pruning comparison\n\n"
        + output["comparison_note"]
        + "\n\n"
        + markdown_table(rows)
        + "\n"
    )
    args.output_markdown.write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
