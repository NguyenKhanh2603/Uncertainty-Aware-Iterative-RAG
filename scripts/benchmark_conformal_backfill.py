"""Benchmark whether conformal backfill improves evidence selection.

This is an evidence-selection benchmark.  It compares the same saved candidate
p-values under three policies: fixed top-K, conformal filtering restricted to
the original top-K, and conformal filtering followed by verified backfill from
ranks K+1..L.  It does not run a generator or report answer EM/F1.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from tqdm import tqdm

from uncertainty_rag.core.conformal_retrieval import ConformalDataError
from uncertainty_rag.core.conformal_selection import (
    backfill_rejected_indices,
    benjamini_yekutieli,
)

METHODS = ("fixed_top_k", "conformal_no_backfill", "conformal_backfill")


def parse_alpha_grid(specification: str) -> tuple[float, ...]:
    values = sorted({float(part.strip()) for part in specification.split(",") if part.strip()})
    if not values or any(not 0 < value < 1 for value in values):
        raise ConformalDataError("alphas must be comma-separated values strictly between 0 and 1")
    return tuple(values)


def policy_indices(
    decisions: list[Mapping[str, Any]], *, alpha: float, max_context: int
) -> dict[str, set[int]]:
    """Return all three benchmark contexts for one query."""

    retrieval_order = sorted(
        range(len(decisions)),
        key=lambda index: (int(decisions[index]["rank"]), str(decisions[index]["chunk_id"])),
    )
    top_k = set(retrieval_order[:max_context])
    by = benjamini_yekutieli([float(item["p_value"]) for item in decisions], alpha)
    with_backfill, without_backfill = backfill_rejected_indices(
        [int(item["rank"]) for item in decisions],
        [str(item["chunk_id"]) for item in decisions],
        by.rejected_indices,
        max_context,
    )
    return {
        "fixed_top_k": top_k,
        "conformal_no_backfill": set(without_backfill),
        "conformal_backfill": set(with_backfill),
    }


@dataclass
class EvidenceMetrics:
    queries: int = 0
    reserve_supports: int = 0
    selected: int = 0
    supports: int = 0
    false: int = 0
    unknown: int = 0
    empty: int = 0
    queries_with_support: int = 0
    query_fdp_sum: float = 0.0
    nonempty_query_fdp_sum: float = 0.0

    def update(self, decisions: list[Mapping[str, Any]], selected: set[int]) -> None:
        supports = sum(decisions[index]["support_label"] == "support" for index in selected)
        false = sum(decisions[index]["support_label"] == "false" for index in selected)
        unknown = len(selected) - supports - false
        reserve_supports = sum(item["support_label"] == "support" for item in decisions)
        fdp = false / len(selected) if selected else 0.0

        self.queries += 1
        self.reserve_supports += reserve_supports
        self.selected += len(selected)
        self.supports += supports
        self.false += false
        self.unknown += unknown
        self.empty += int(not selected)
        self.queries_with_support += int(supports > 0)
        self.query_fdp_sum += fdp
        if selected:
            self.nonempty_query_fdp_sum += fdp

    @staticmethod
    def ratio(numerator: int | float, denominator: int | float) -> float | None:
        return numerator / denominator if denominator else None

    def finish(self) -> dict[str, Any]:
        precision = self.ratio(self.supports, self.supports + self.false)
        recall = self.ratio(self.supports, self.reserve_supports)
        support_f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall
            else 0.0
        )
        nonempty = self.queries - self.empty
        return {
            "queries": self.queries,
            "average_selected_chunks": self.ratio(self.selected, self.queries),
            "empty_context_rate": self.ratio(self.empty, self.queries),
            "queries_with_support_rate": self.ratio(self.queries_with_support, self.queries),
            "selected_supports": self.supports,
            "selected_false": self.false,
            "selected_unknown": self.unknown,
            "micro_evidence_precision": precision,
            "conditional_reserve_support_recall": recall,
            "support_f1": support_f1,
            "micro_false_share": self.ratio(self.false, self.supports + self.false),
            "mean_query_fdp": self.ratio(self.query_fdp_sum, self.queries),
            "mean_query_fdp_given_nonempty": self.ratio(self.nonempty_query_fdp_sum, nonempty),
        }


@dataclass
class BackfillDelta:
    queries: int = 0
    queries_using_backfill: int = 0
    chunks_added: int = 0
    supports_added: int = 0
    false_added: int = 0
    unknown_added: int = 0
    support_rescue_queries: int = 0

    def update(
        self,
        decisions: list[Mapping[str, Any]],
        without_backfill: set[int],
        with_backfill: set[int],
    ) -> None:
        added = with_backfill - without_backfill
        before_supports = sum(
            decisions[index]["support_label"] == "support" for index in without_backfill
        )
        added_supports = sum(decisions[index]["support_label"] == "support" for index in added)
        added_false = sum(decisions[index]["support_label"] == "false" for index in added)
        self.queries += 1
        self.queries_using_backfill += int(bool(added))
        self.chunks_added += len(added)
        self.supports_added += added_supports
        self.false_added += added_false
        self.unknown_added += len(added) - added_supports - added_false
        self.support_rescue_queries += int(before_supports == 0 and added_supports > 0)

    @staticmethod
    def ratio(numerator: int | float, denominator: int | float) -> float | None:
        return numerator / denominator if denominator else None

    def finish(self) -> dict[str, Any]:
        return {
            "queries": self.queries,
            "backfill_query_rate": self.ratio(self.queries_using_backfill, self.queries),
            "backfill_chunks_added": self.chunks_added,
            "backfill_supports_added": self.supports_added,
            "backfill_false_added": self.false_added,
            "backfill_unknown_added": self.unknown_added,
            "backfill_added_precision": self.ratio(
                self.supports_added, self.supports_added + self.false_added
            ),
            "support_rescue_queries": self.support_rescue_queries,
            "support_rescue_query_rate": self.ratio(self.support_rescue_queries, self.queries),
        }


def iter_rows(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark fixed top-K, conformal filtering, and conformal backfill."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alphas", default="0.01,0.025,0.05,0.10,0.20,0.30,0.50")
    parser.add_argument("--max-context", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_context < 1:
        raise ConformalDataError("max_context must be positive")
    alphas = parse_alpha_grid(args.alphas)
    metrics: dict[tuple[str, float, str, str], EvidenceMetrics] = defaultdict(EvidenceMetrics)
    deltas: dict[tuple[str, float, str], BackfillDelta] = defaultdict(BackfillDelta)
    strategies: set[str] = set()
    query_output = args.output_dir / "backfill_benchmark_queries.jsonl.gz"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with gzip.open(query_output, "wt", encoding="utf-8") as output_handle:
        for wrapper in tqdm(iter_rows(args.input), desc="Benchmark backfill", unit="query"):
            for strategy, base in wrapper["strategies"].items():
                strategies.add(strategy)
                decisions = base["decisions"]
                dataset = str(base["dataset"])
                for alpha in alphas:
                    policies = policy_indices(decisions, alpha=alpha, max_context=args.max_context)
                    for method, selected in policies.items():
                        metrics[(strategy, alpha, "ALL", method)].update(decisions, selected)
                        metrics[(strategy, alpha, dataset, method)].update(decisions, selected)
                    for scope in ("ALL", dataset):
                        deltas[(strategy, alpha, scope)].update(
                            decisions,
                            policies["conformal_no_backfill"],
                            policies["conformal_backfill"],
                        )
                    output_handle.write(
                        json.dumps(
                            {
                                "strategy": strategy,
                                "alpha": alpha,
                                "dataset": dataset,
                                "qid": base["qid"],
                                "selected_ranks": {
                                    method: sorted(
                                        int(decisions[index]["rank"]) for index in selected
                                    )
                                    for method, selected in policies.items()
                                },
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

    rows: list[dict[str, Any]] = []
    nested: dict[str, Any] = {}
    scopes = sorted({key[2] for key in metrics}, key=lambda value: (value != "ALL", value))
    for strategy in sorted(strategies):
        nested[strategy] = {}
        for alpha in alphas:
            alpha_result: dict[str, Any] = {}
            for scope in scopes:
                method_results = {
                    method: metrics[(strategy, alpha, scope, method)].finish()
                    for method in METHODS
                    if metrics[(strategy, alpha, scope, method)].queries
                }
                if not method_results:
                    continue
                delta_result = deltas[(strategy, alpha, scope)].finish()
                alpha_result[scope] = {
                    "methods": method_results,
                    "backfill_delta": delta_result,
                }
                for method, result in method_results.items():
                    rows.append(
                        {
                            "strategy": strategy,
                            "alpha": alpha,
                            "dataset": scope,
                            "method": method,
                            **result,
                            **delta_result,
                        }
                    )
            nested[strategy][f"{alpha:g}"] = alpha_result

    artifact = {
        "is_paper_ready": False,
        "benchmark_scope": "evidence_selection_only",
        "formal_capped_risk_guarantee": False,
        "warnings": [
            "This benchmark does not run answer generation and cannot report EM/F1.",
            "Recall is conditional on labelled supports present in the frozen top-L reserve.",
            "BY plus the at-most-K backfill rule remains an experimental procedure.",
        ],
        "configuration": {"alphas": alphas, "max_context": args.max_context},
        "strategies": nested,
        "outputs": {"per_query": query_output.name},
    }
    json_path = args.output_dir / "backfill_benchmark_summary.json"
    csv_path = args.output_dir / "backfill_benchmark_summary.csv"
    json_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    write_csv(csv_path, rows)
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {query_output}")
    for strategy in sorted(strategies):
        for alpha in alphas:
            overall = nested[strategy][f"{alpha:g}"].get("ALL")
            if not overall:
                continue
            before = overall["methods"]["conformal_no_backfill"]
            after = overall["methods"]["conformal_backfill"]
            delta = overall["backfill_delta"]
            print(
                f"{strategy} alpha={alpha:g}: "
                f"no-backfill R={before['conditional_reserve_support_recall']:.3f} -> "
                f"backfill R={after['conditional_reserve_support_recall']:.3f}; "
                f"support+={delta['backfill_supports_added']} "
                f"false+={delta['backfill_false_added']}"
            )


if __name__ == "__main__":
    main()
