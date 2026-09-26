"""Evaluate alpha-free query-level cosine operating points on frozen splits.

This is deliberately separate from conformal prediction.  It selects a
query-normalised cosine threshold *only on the calibration qids*, then freezes
that threshold for the disjoint test qids.  It therefore has no ``alpha``
input at inference time, but it also has no conformal coverage claim.

Two pre-registered operating points are reported:

``f1_calibrated``
    maximises micro evidence F1 on calibration candidates;
``budget10_all_support``
    maximises calibration query all-support rate subject to retaining at most
    ten candidates per query on average.

Both selectors use the existing per-query z-normalised Jina cosine score and
the existing deterministic Top-1 fallback.  The report also includes the
alpha=.10 query-level conformal reference, so this is an operating-point
comparison rather than a replacement of the conformal method.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from research.internal_state_rag.run_all_datasets_cosine_six_methods import (
    CONFIGS,
    load_dataset,
    query_level_cosine_mask,
    query_z,
)


DATASETS = ("hotpotqa", "mmqa", "tatqa", "webqa")
GRID_SIZE = 1001
CONTEXT_BUDGET = 10.0


@dataclass(frozen=True)
class ScoreSummary:
    mean_chunks: float
    precision: float
    recall: float
    f1: float
    empty_rate: float
    any_support: float
    all_support: float

    def to_dict(self) -> dict[str, float]:
        return {
            "mean_chunks": self.mean_chunks,
            "precision": self.precision,
            "support_recall": self.recall,
            "evidence_f1": self.f1,
            "empty_rate": self.empty_rate,
            "query_any_support_conditional": self.any_support,
            "query_all_support_conditional": self.all_support,
        }


def score_arrays(rows: Sequence[Sequence[dict[str, Any]]]) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(
        [[float(item["cosine_score"]) for item in query] for query in rows], dtype=float
    )
    labels = np.asarray(
        [[item["support_label"] == "support" for item in query] for query in rows], dtype=bool
    )
    if scores.ndim != 2 or not scores.size or scores.shape != labels.shape:
        raise ValueError("Expected a non-empty rectangular candidate matrix")
    return query_z(scores), labels


def mask_from_threshold(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Threshold query-normalised scores, retaining Top-1 for empty contexts."""

    mask = scores >= threshold
    empty = ~mask.any(axis=1)
    if empty.any():
        mask[empty, np.argmax(scores[empty], axis=1)] = True
    return mask


def summarise(labels: np.ndarray, mask: np.ndarray) -> ScoreSummary:
    selected = int(mask.sum())
    retained_support = int((labels & mask).sum())
    supports = int(labels.sum())
    precision = retained_support / selected if selected else 0.0
    recall = retained_support / supports if supports else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    retrievable = labels.any(axis=1)
    retained_per_query = labels & mask
    any_support = (
        float(retained_per_query[retrievable].any(axis=1).mean()) if retrievable.any() else 0.0
    )
    all_support = (
        float((retained_per_query[retrievable].sum(axis=1) == labels[retrievable].sum(axis=1)).mean())
        if retrievable.any()
        else 0.0
    )
    return ScoreSummary(
        mean_chunks=float(mask.sum(axis=1).mean()),
        precision=precision,
        recall=recall,
        f1=f1,
        # The Top-1 fallback makes this exactly zero. Retain the field so the
        # report is directly comparable with the existing query-level row.
        empty_rate=float((~mask.any(axis=1)).mean()),
        any_support=any_support,
        all_support=all_support,
    )


def threshold_grid(scores: np.ndarray) -> np.ndarray:
    """A deterministic dense grid whose construction does not inspect labels."""

    return np.unique(np.quantile(scores.ravel(), np.linspace(0.0, 1.0, GRID_SIZE)))


def choose_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    objective: str,
) -> tuple[float, ScoreSummary]:
    """Select one threshold using calibration labels and a pre-set objective."""

    best: tuple[tuple[float, ...], float, ScoreSummary] | None = None
    for threshold in threshold_grid(scores):
        summary = summarise(labels, mask_from_threshold(scores, float(threshold)))
        if objective == "f1_calibrated":
            # Deterministic tie breaks avoid silently favouring broad contexts.
            key = (summary.f1, summary.recall, summary.precision, -summary.mean_chunks, threshold)
        elif objective == "budget10_all_support":
            if summary.mean_chunks > CONTEXT_BUDGET + 1e-12:
                continue
            key = (
                summary.all_support,
                summary.recall,
                summary.precision,
                -summary.mean_chunks,
                threshold,
            )
        else:
            raise ValueError(f"Unknown objective: {objective}")
        if best is None or key > best[0]:
            best = (key, float(threshold), summary)
    if best is None:
        raise RuntimeError(f"No threshold met objective {objective}")
    return best[1], best[2]


def cross_validated_f1(
    scores: np.ndarray, labels: np.ndarray, objective: str, folds: int = 5
) -> dict[str, float]:
    """Estimate calibration-threshold stability without exposing held-out test labels."""

    indices = np.arange(len(scores))
    values: list[float] = []
    chunks: list[float] = []
    for heldout in np.array_split(indices, folds):
        train = np.setdiff1d(indices, heldout, assume_unique=True)
        threshold, _ = choose_threshold(scores[train], labels[train], objective)
        result = summarise(labels[heldout], mask_from_threshold(scores[heldout], threshold))
        values.append(result.f1)
        chunks.append(result.mean_chunks)
    return {
        "folds": folds,
        "held_out_evidence_f1_mean": float(np.mean(values)),
        "held_out_evidence_f1_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "held_out_mean_chunks_mean": float(np.mean(chunks)),
    }


def evaluate_dataset(
    dataset: str, calibration: list[list[dict[str, Any]]], test: list[list[dict[str, Any]]]
) -> dict[str, Any]:
    cal_scores, cal_labels = score_arrays(calibration)
    test_scores, test_labels = score_arrays(test)
    methods: dict[str, dict[str, Any]] = {}
    conformal_masks, conformal_audit = query_level_cosine_mask(calibration, test, 0.10)
    methods["query_level_conformal_alpha_0.10_reference"] = {
        "definition": "Existing query-level all-support conformal reference; alpha=0.10.",
        "test_selection": summarise(test_labels, np.asarray(conformal_masks, dtype=bool)).to_dict(),
        "calibration": conformal_audit,
    }
    for objective, display in (
        ("f1_calibrated", "Query-level cosine, calibration-selected evidence F1 (alpha-free at test)"),
        ("budget10_all_support", "Query-level cosine, calibration-selected all-support under 10-chunk budget"),
    ):
        threshold, calibration_selection = choose_threshold(cal_scores, cal_labels, objective)
        test_selection = summarise(test_labels, mask_from_threshold(test_scores, threshold))
        methods[f"query_level_{objective}"] = {
            "definition": display,
            "threshold": threshold,
            "calibration_selection": calibration_selection.to_dict(),
            "calibration_cross_validation": cross_validated_f1(cal_scores, cal_labels, objective),
            "test_selection": test_selection.to_dict(),
        }
    return {
        "dataset": dataset,
        "calibration_queries": len(calibration),
        "test_queries": len(test),
        "top_l": int(cal_scores.shape[1]),
        "methods": methods,
    }


def report(results: dict[str, dict[str, Any]], output: Path, plan_root: Path) -> None:
    lines = [
        "# Alpha-free query-level cosine operating points: 1,000 calibration / 100 test",
        "",
        "This is **not a conformal method**. The two new selectors choose a Jina cosine threshold from calibration labels, freeze it, and apply it to the disjoint test qids without receiving `alpha` at inference. The price is that neither selector has the query-level coverage guarantee of the alpha=.10 conformal reference.",
        "",
        f"- Frozen split manifests: [`{plan_root}`](../all_datasets_cosine_six_methods_1000cal_100test_2026_09_26/splits/)",
        "- Candidate pool: frozen Jina cosine Top-30; per-query z-normalisation; deterministic Top-1 fallback.",
        "- `F1-selected`: maximises micro evidence F1 on the 1,000 calibration qids. `Budget-10`: maximises conditional all-support coverage while calibration mean context is at most ten chunks.",
        "",
    ]
    order = (
        "query_level_conformal_alpha_0.10_reference",
        "query_level_f1_calibrated",
        "query_level_budget10_all_support",
    )
    names = {
        "query_level_conformal_alpha_0.10_reference": "Query-level conformal cosine (α=.10 reference)",
        "query_level_f1_calibrated": "Query-level cosine F1-selected (no α at test)",
        "query_level_budget10_all_support": "Query-level cosine budget-10 all-support (no α at test)",
    }
    for dataset in DATASETS:
        result = results[dataset]
        lines.extend(
            [
                f"## {dataset} (calibration={result['calibration_queries']}, test={result['test_queries']})",
                "",
                "| Method | Threshold | Chunks | Precision | Recall | Evidence F1 | Empty | Any support* | All support* |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for method in order:
            row = result["methods"][method]
            selection = row["test_selection"]
            threshold = row.get("threshold")
            display_threshold = f"{threshold:.4f}" if threshold is not None else "conformal"
            lines.append(
                f"| {names[method]} | {display_threshold} | {selection['mean_chunks']:.2f} | "
                f"{selection['precision']:.1%} | {selection['support_recall']:.1%} | "
                f"{selection['evidence_f1']:.3f} | {selection['empty_rate']:.1%} | "
                f"{selection['query_any_support_conditional']:.1%} | "
                f"{selection['query_all_support_conditional']:.1%} |"
            )
        lines.extend(
            [
                "",
                "*Any/all-support are conditional on at least one labelled support in Top-30; the test labels are used only for this evaluation.*",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation",
            "",
            "An alpha-free row is an empirically selected utility operating point, not a confidence procedure. It is appropriate when the product objective is fixed (for example, maximum retrieval F1 or a ten-chunk context budget). Use the conformal row when the stated objective is a pre-declared all-support error level.",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan-root",
        type=Path,
        default=Path(
            "research/internal_state_rag/results/"
            "all_datasets_cosine_six_methods_1000cal_100test_2026_09_26/splits"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "research/internal_state_rag/results/"
            "alpha_free_query_level_1000cal_100test_2026_09_26"
        ),
    )
    parser.add_argument("--datasets", default=",".join(DATASETS))
    args = parser.parse_args()
    datasets = tuple(item.strip() for item in args.datasets.split(",") if item.strip())
    if not datasets or any(dataset not in DATASETS for dataset in datasets):
        raise ValueError(f"datasets must be a non-empty subset of {DATASETS}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = {}
    for dataset in datasets:
        calibration_plan = args.plan_root / dataset / "calibration_manifest.json"
        test_plan = args.plan_root / dataset / "test_manifest.json"
        calibration_qids, calibration, test_qids, test = load_dataset(
            CONFIGS[dataset], calibration_plan=calibration_plan, test_plan=test_plan
        )
        if len(calibration_qids) != 1000 or len(test_qids) != 100:
            raise ValueError(f"{dataset}: expected exactly 1,000 calibration and 100 test qids")
        results[dataset] = evaluate_dataset(dataset, calibration, test)
    # Keep a stable ordered JSON result for downstream inspection and queued QA.
    (args.output_dir / "selection_summary.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report(results, args.output_dir / "REPORT.md", args.plan_root)
    print(json.dumps({dataset: results[dataset]["methods"] for dataset in datasets}, indent=2))


if __name__ == "__main__":
    main()
