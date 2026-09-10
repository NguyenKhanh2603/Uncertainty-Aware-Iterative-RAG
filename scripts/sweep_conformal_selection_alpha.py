"""Sweep BY risk levels using p-values already written by the selection runner."""

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
from uncertainty_rag.core.conformal_selection import benjamini_yekutieli


def parse_alpha_grid(specification: str) -> tuple[float, ...]:
    values = sorted({float(part.strip()) for part in specification.split(",") if part.strip()})
    if not values:
        raise ConformalDataError("At least one alpha value is required")
    if any(not 0 < value < 1 for value in values):
        raise ConformalDataError("Every alpha must be strictly between 0 and 1")
    return tuple(values)


def selected_indices(
    decisions: list[Mapping[str, Any]], *, alpha: float, max_context: int
) -> tuple[set[int], int]:
    """Reapply BY and the experimental p-value cap without recomputing p-values."""

    by = benjamini_yekutieli([float(item["p_value"]) for item in decisions], alpha)
    rejected = set(by.rejected_indices)
    capped = sorted(
        rejected,
        key=lambda index: (
            decisions[index]["p_value"],
            decisions[index]["rank"],
            decisions[index]["chunk_id"],
        ),
    )[:max_context]
    return set(capped), len(rejected)


@dataclass
class SweepMetrics:
    queries: int = 0
    reserve_candidates: int = 0
    reserve_supports: int = 0
    by_rejections: int = 0
    selected: int = 0
    selected_supports: int = 0
    selected_false: int = 0
    selected_unknown: int = 0
    empty: int = 0
    backfill_chunks: int = 0
    backfill_queries: int = 0
    capped_queries: int = 0
    query_fdp_sum: float = 0.0
    nonempty_query_fdp_sum: float = 0.0
    baseline_chunks: int = 0
    baseline_supports: int = 0
    baseline_false: int = 0

    def update(
        self,
        base: Mapping[str, Any],
        decisions: list[Mapping[str, Any]],
        selected: set[int],
        by_rejections: int,
        max_context: int,
    ) -> None:
        supports = sum(decisions[index]["support_label"] == "support" for index in selected)
        false = sum(decisions[index]["support_label"] == "false" for index in selected)
        unknown = len(selected) - supports - false
        backfills = sum(int(decisions[index]["rank"]) > max_context for index in selected)
        fdp = false / len(selected) if selected else 0.0

        self.queries += 1
        self.reserve_candidates += len(decisions)
        self.reserve_supports += int(base["reserve_supports"])
        self.by_rejections += by_rejections
        self.selected += len(selected)
        self.selected_supports += supports
        self.selected_false += false
        self.selected_unknown += unknown
        self.empty += int(not selected)
        self.backfill_chunks += backfills
        self.backfill_queries += int(backfills > 0)
        self.capped_queries += int(by_rejections > len(selected))
        self.query_fdp_sum += fdp
        if selected:
            self.nonempty_query_fdp_sum += fdp
        self.baseline_chunks += int(base["baseline_count"])
        self.baseline_supports += int(base["baseline_supports"])
        self.baseline_false += int(base["baseline_false"])

    @staticmethod
    def ratio(numerator: int | float, denominator: int | float) -> float | None:
        return numerator / denominator if denominator else None

    def finish(self) -> dict[str, Any]:
        precision = self.ratio(self.selected_supports, self.selected_supports + self.selected_false)
        recall = self.ratio(self.selected_supports, self.reserve_supports)
        support_f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall
            else 0.0
        )
        nonempty = self.queries - self.empty
        return {
            "queries": self.queries,
            "average_reserve_size": self.ratio(self.reserve_candidates, self.queries),
            "by_rejections": self.by_rejections,
            "selected_chunks": self.selected,
            "average_selected_chunks": self.ratio(self.selected, self.queries),
            "empty_context_rate": self.ratio(self.empty, self.queries),
            "micro_evidence_precision": precision,
            "conditional_reserve_support_recall": recall,
            "support_f1": support_f1,
            "micro_false_share": self.ratio(
                self.selected_false, self.selected_supports + self.selected_false
            ),
            "mean_query_fdp": self.ratio(self.query_fdp_sum, self.queries),
            "mean_query_fdp_given_nonempty": self.ratio(self.nonempty_query_fdp_sum, nonempty),
            "backfill_query_rate": self.ratio(self.backfill_queries, self.queries),
            "backfill_chunks": self.backfill_chunks,
            "queries_hitting_context_cap": self.capped_queries,
            "baseline_top_k_precision": self.ratio(
                self.baseline_supports, self.baseline_supports + self.baseline_false
            ),
            "baseline_conditional_reserve_support_recall": self.ratio(
                self.baseline_supports, self.reserve_supports
            ),
        }


def iter_decision_rows(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep alpha over saved conformal p-values.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alphas", default="0.01,0.025,0.05,0.10,0.20,0.30,0.50,0.75,0.90")
    parser.add_argument("--max-context", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_context < 1:
        raise ConformalDataError("max_context must be positive")
    alphas = parse_alpha_grid(args.alphas)
    overall: dict[tuple[str, float], SweepMetrics] = defaultdict(SweepMetrics)
    by_dataset: dict[tuple[str, float, str], SweepMetrics] = defaultdict(SweepMetrics)
    strategies: set[str] = set()

    for wrapper in tqdm(iter_decision_rows(args.input), desc="Sweep saved p-values", unit="query"):
        for strategy, base in wrapper["strategies"].items():
            strategies.add(strategy)
            decisions = base["decisions"]
            for alpha in alphas:
                selected, by_rejections = selected_indices(
                    decisions, alpha=alpha, max_context=args.max_context
                )
                overall[(strategy, alpha)].update(
                    base, decisions, selected, by_rejections, args.max_context
                )
                by_dataset[(strategy, alpha, base["dataset"])].update(
                    base, decisions, selected, by_rejections, args.max_context
                )

    json_results: dict[str, Any] = {}
    csv_rows = []
    for strategy in sorted(strategies):
        strategy_results = {}
        for alpha in alphas:
            aggregate = overall[(strategy, alpha)].finish()
            datasets = {
                dataset: metrics.finish()
                for (candidate_strategy, candidate_alpha, dataset), metrics in sorted(
                    by_dataset.items()
                )
                if candidate_strategy == strategy and candidate_alpha == alpha
            }
            strategy_results[f"{alpha:g}"] = {"overall": aggregate, "by_dataset": datasets}
            csv_rows.append({"strategy": strategy, "alpha": alpha, "dataset": "ALL", **aggregate})
            for dataset, metrics in datasets.items():
                csv_rows.append(
                    {"strategy": strategy, "alpha": alpha, "dataset": dataset, **metrics}
                )
        json_results[strategy] = strategy_results

    artifact = {
        "is_paper_ready": False,
        "formal_capped_risk_guarantee": False,
        "warning": (
            "Alpha sweep reuses development-smoke p-values. High-alpha rows are diagnostics, "
            "not recommended risk settings."
        ),
        "configuration": {"alphas": alphas, "max_context": args.max_context},
        "strategies": json_results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "alpha_sweep_summary.json"
    csv_path = args.output_dir / "alpha_sweep_summary.csv"
    json_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    write_csv(csv_path, csv_rows)
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    for strategy in sorted(strategies):
        print(f"\n{strategy}")
        for alpha in alphas:
            result = json_results[strategy][f"{alpha:g}"]["overall"]
            print(
                f"alpha={alpha:g} selected={result['average_selected_chunks']:.3f} "
                f"empty={result['empty_context_rate']:.3f} "
                f"precision={result['micro_evidence_precision']:.3f} "
                f"recall={result['conditional_reserve_support_recall']:.3f} "
                f"support_f1={result['support_f1']:.3f} "
                f"mean_fdp={result['mean_query_fdp']:.3f}"
            )


if __name__ == "__main__":
    main()
