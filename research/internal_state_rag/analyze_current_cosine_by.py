"""Evaluate candidate-wise BY with a configurable retrieval score.

The original experiment used ``cosine_score``.  Reranker outputs can be
evaluated on the same fixed Top-L candidate pool by selecting a different
``--score-field`` (for example ``jina_reranker_score`` or
``selection_score``).  The score's own rank is used for the context cap unless
``--order-field`` is supplied explicitly.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from uncertainty_rag.core.conformal_selection import benjamini_yekutieli


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alphas", default="0.2,0.1,0.05")
    parser.add_argument("--max-context", type=int, default=10)
    parser.add_argument(
        "--score-field",
        default="cosine_score",
        help="Candidate field used for the false-score bank and BY p-values.",
    )
    parser.add_argument(
        "--order-field",
        default="",
        help=(
            "Candidate rank field used for the context cap. Empty selects "
            "cosine_rank for cosine_score and rank otherwise."
        ),
    )
    return parser.parse_args()


def load_rows(path: Path) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                grouped[str(row["qid"])].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: (int(row["rank"]), str(row["chunk_id"])))
    top_l = {len(rows) for rows in grouped.values()}
    if len(top_l) != 1:
        raise ValueError(f"Inconsistent candidate counts: {sorted(top_l)}")
    return grouped


def false_banks(
    calibration: list[list[dict]], conditioning: str, score_field: str
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    values: dict[str, list[float]] = defaultdict(list)
    for rows in calibration:
        for row in rows:
            if row["support_label"] == "support":
                continue
            key = str(row["modality"]) if conditioning == "modality" else "pooled"
            values[key].append(float(row[score_field]))
    banks = {key: np.sort(score) for key, score in values.items()}
    return banks, {key: len(score) for key, score in values.items()}


def select(
    test: list[list[dict]],
    banks: dict[str, np.ndarray],
    *,
    conditioning: str,
    alpha: float,
    max_context: int,
    score_field: str,
    order_field: str,
) -> np.ndarray:
    mask = np.zeros((len(test), len(test[0])), dtype=bool)
    for query_index, rows in enumerate(test):
        p_values = []
        for row in rows:
            key = str(row["modality"]) if conditioning == "modality" else "pooled"
            bank = banks.get(key)
            if bank is None:
                raise ValueError(f"No false-score bank for {key!r}")
            score = float(row[score_field])
            greater_equal = len(bank) - int(np.searchsorted(bank, score, side="left"))
            p_values.append((1.0 + greater_equal) / (len(bank) + 1.0))
        rejected = benjamini_yekutieli(p_values, alpha).rejected_indices
        chosen = sorted(
            rejected,
            key=lambda index: (
                int(rows[index][order_field]),
                str(rows[index]["chunk_id"]),
            ),
        )[:max_context]
        mask[query_index, chosen] = True
    return mask


def metrics(labels: np.ndarray, mask: np.ndarray) -> dict[str, object]:
    retrievable = labels.any(axis=1)
    support_total = labels.sum(axis=1)
    retained = (labels & mask).sum(axis=1)
    kept = mask.sum(axis=1)
    selected = int(mask.sum())
    conditional_support = int(support_total[retrievable].sum())
    conditional_retained = int(retained[retrievable].sum())
    return {
        "all_test_queries": int(len(labels)),
        "retrievable_queries": int(retrievable.sum()),
        "retrieval_query_coverage_ceiling": float(retrievable.mean()),
        "mean_chunks_kept_all_test": float(kept.mean()),
        "mean_chunks_kept_retrievable": float(kept[retrievable].mean()),
        "empty_context_rate_all_test": float(np.mean(kept == 0)),
        "chunk_precision": float(retained.sum() / selected) if selected else 0.0,
        "micro_support_recall_conditional": (
            float(conditional_retained / conditional_support)
            if conditional_support
            else 0.0
        ),
        "query_any_support_conditional": float(np.mean(retained[retrievable] > 0)),
        "query_all_support_conditional": float(
            np.mean(retained[retrievable] == support_total[retrievable])
        ),
        "query_any_support_end_to_end": float(np.mean(retained > 0)),
        "query_all_support_end_to_end": float(
            np.mean(retrievable & (retained == support_total))
        ),
        "support_chunks_conditional": conditional_support,
        "support_chunks_retained_conditional": conditional_retained,
        "selected_chunks": selected,
    }


def main() -> None:
    args = parse_args()
    grouped = load_rows(args.retrieval)
    sample_row = next(iter(grouped.values()))[0]
    if args.order_field:
        order_field = args.order_field
    elif args.score_field == "cosine_score" and "cosine_rank" in sample_row:
        order_field = "cosine_rank"
    else:
        order_field = "rank"
    for field in (args.score_field, order_field):
        if field not in sample_row:
            raise ValueError(
                f"Field {field!r} is absent from retrieval rows; available fields: "
                f"{sorted(sample_row)}"
            )
    by_role: dict[str, list[list[dict]]] = defaultdict(list)
    for qid in sorted(grouped):
        rows = grouped[qid]
        role = str(rows[0]["split_role"])
        if any(str(row["split_role"]) != role for row in rows):
            raise ValueError(f"Mixed roles within query {qid}")
        by_role[role].append(rows)
    calibration = by_role["calibration"]
    test = by_role["test"]
    labels = np.asarray(
        [[row["support_label"] == "support" for row in rows] for rows in test],
        dtype=bool,
    )
    results = {}
    bank_audit = {}
    for conditioning in ("pooled", "modality"):
        banks, counts = false_banks(calibration, conditioning, args.score_field)
        bank_audit[conditioning] = counts
        results[conditioning] = {}
        for alpha in (float(value) for value in args.alphas.split(",")):
            mask = select(
                test,
                banks,
                conditioning=conditioning,
                alpha=alpha,
                max_context=args.max_context,
                score_field=args.score_field,
                order_field=order_field,
            )
            results[conditioning][str(alpha)] = metrics(labels, mask)
    output = {
        "status": "complete",
        "dataset": args.dataset,
        "method": (
            f"{args.score_field} false-score p-values + candidate-wise BY + Top-K cap"
        ),
        "retrieval": str(args.retrieval),
        "score_field": args.score_field,
        "order_field": order_field,
        "top_l": len(test[0]),
        "max_context": args.max_context,
        "role_counts": {role: len(rows) for role, rows in by_role.items()},
        "false_bank_counts": bank_audit,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
