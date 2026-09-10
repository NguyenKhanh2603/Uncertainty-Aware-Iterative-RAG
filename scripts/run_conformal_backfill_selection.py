"""Run conformal p-values, experimental BY selection, and backfill evaluation."""

from __future__ import annotations

import argparse
import gzip
import io
import json
from collections import defaultdict
from contextlib import contextmanager
from itertools import groupby
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, TextIO
from zipfile import ZipFile

from tqdm import tqdm

from uncertainty_rag.core.conformal_selection import (
    ReferenceBankIndex,
    select_query_context,
)


def open_json(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    kwargs = {"encoding": "utf-8"}
    with opener(path, "rt", **kwargs) as handle:
        return json.load(handle)


@contextmanager
def open_output(path: Path) -> Iterator[TextIO]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    opener = gzip.open if path.suffix == ".gz" else Path.open
    kwargs = {"encoding": "utf-8"}
    with opener(temporary, "wt", **kwargs) as handle:
        yield handle
    temporary.replace(path)


class ResultsReader:
    """Replay retrieval rows from either normal files or a downloaded result ZIP."""

    def __init__(
        self,
        *,
        results_zip: Path | None,
        inputs: list[Path] | None,
        bank: Path | None,
    ) -> None:
        self.results_zip = results_zip
        self.inputs = inputs or []
        self.bank = bank
        if results_zip is not None:
            with ZipFile(results_zip) as archive:
                names = archive.namelist()
            matches = [name for name in names if name.endswith("combined_reference_banks.json.gz")]
            if len(matches) != 1:
                raise ValueError("Result ZIP must contain one combined_reference_banks.json.gz")
            self.bank_entry = matches[0]
            self.log_entries = sorted(
                name for name in names if name.endswith("_retrieval.jsonl.gz")
            )
            if not self.log_entries:
                raise ValueError("Result ZIP contains no retrieval JSONL.GZ logs")
        else:
            if not self.inputs or bank is None:
                raise ValueError("--input and --bank are required without --results-zip")
            self.bank_entry = None
            self.log_entries = []

    def read_bank(self) -> dict[str, Any]:
        if self.results_zip is None:
            assert self.bank is not None
            return open_json(self.bank)
        assert self.bank_entry is not None
        with ZipFile(self.results_zip) as archive:
            compressed = archive.read(self.bank_entry)
        return json.loads(gzip.decompress(compressed))

    def iter_rows(self) -> Iterator[dict[str, Any]]:
        if self.results_zip is not None:
            for entry in self.log_entries:
                with ZipFile(self.results_zip) as archive:
                    with archive.open(entry) as compressed_handle:
                        with gzip.GzipFile(fileobj=compressed_handle) as binary_handle:
                            with io.TextIOWrapper(binary_handle, encoding="utf-8") as handle:
                                for line in handle:
                                    if line.strip():
                                        yield json.loads(line)
            return
        for path in self.inputs:
            opener = gzip.open if path.suffix == ".gz" else Path.open
            with opener(path, "rt", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        yield json.loads(line)

    def query_count(self, split_role: str) -> int | None:
        manifests: list[Mapping[str, Any]] = []
        if self.results_zip is not None:
            with ZipFile(self.results_zip) as archive:
                for name in archive.namelist():
                    if name.endswith("_retrieval.jsonl.gz.manifest.json"):
                        manifests.append(json.loads(archive.read(name)))
        else:
            for path in self.inputs:
                manifest_path = Path(f"{path}.manifest.json")
                if manifest_path.is_file():
                    manifests.append(json.loads(manifest_path.read_text(encoding="utf-8")))
        if not manifests:
            return None
        return sum(int(item.get("role_counts", {}).get(split_role, 0)) for item in manifests)


def build_dataset_pooled_artifact(
    rows: Iterable[Mapping[str, Any]],
    template: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a dataset-only false-score bank for the calibration ablation."""

    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["split_role"] == "calibration" and row["support_label"] == "false":
            grouped[str(row["dataset"])].append(float(row["cosine_score"]))
    banks = []
    for dataset, scores in sorted(grouped.items()):
        scores.sort()
        banks.append(
            {
                "condition": {"dataset": dataset},
                "n_false_scores": len(scores),
                "scores": scores,
            }
        )
    return {
        "top_l": template["top_l"],
        "conditioning": ["dataset"],
        "rank_bins": [],
        "min_bank_size": template.get("min_bank_size", 1),
        "pipeline_fingerprints": template.get("pipeline_fingerprints", {}),
        "banks": banks,
    }


class SummaryAccumulator:
    def __init__(self) -> None:
        self.queries = 0
        self.reserve_candidates = 0
        self.reserve_supports = 0
        self.by_rejections = 0
        self.selected = 0
        self.selected_supports = 0
        self.selected_false = 0
        self.selected_unknown = 0
        self.empty_contexts = 0
        self.backfill_chunks = 0
        self.backfill_queries = 0
        self.capped_queries = 0
        self.baseline = 0
        self.baseline_supports = 0
        self.baseline_false = 0
        self.query_fdp_sum = 0.0
        self.query_recall_sum = 0.0
        self.calibration: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(
            lambda: {
                "false": 0,
                "support": 0,
                "unknown": 0,
                "missing_bank": 0,
                "underpowered": 0,
                "false_p_le": {"0.01": 0, "0.05": 0, "0.10": 0},
                "support_p_le": {"0.01": 0, "0.05": 0, "0.10": 0},
            }
        )

    def update(self, result: Mapping[str, Any]) -> None:
        self.queries += 1
        self.reserve_candidates += int(result["reserve_size"])
        self.reserve_supports += int(result["reserve_supports"])
        self.by_rejections += int(result["by_rejections"])
        self.selected += int(result["selected_count"])
        self.selected_supports += int(result["selected_supports"])
        self.selected_false += int(result["selected_false"])
        self.selected_unknown += int(result["selected_unknown"])
        self.empty_contexts += int(result["selected_count"] == 0)
        self.backfill_chunks += int(result["backfill_count"])
        self.backfill_queries += int(result["backfill_count"] > 0)
        self.capped_queries += int(result["by_rejections"] > result["selected_count"])
        self.baseline += int(result["baseline_count"])
        self.baseline_supports += int(result["baseline_supports"])
        self.baseline_false += int(result["baseline_false"])
        if result["selected_count"]:
            self.query_fdp_sum += result["selected_false"] / result["selected_count"]
        if result["reserve_supports"]:
            self.query_recall_sum += result["selected_supports"] / result["reserve_supports"]

        for decision in result["decisions"]:
            key = (result["dataset"], decision["modality"], decision["bank_status"])
            bucket = self.calibration[key]
            label = decision["support_label"]
            bucket[label] += 1
            bucket["missing_bank"] += int(decision["bank_status"] == "missing")
            bucket["underpowered"] += int("underpowered" in decision["bank_status"])
            if label in {"false", "support"}:
                p_value = decision["p_value"]
                for threshold in (0.01, 0.05, 0.10):
                    if p_value <= threshold:
                        bucket[f"{label}_p_le"][f"{threshold:.2f}"] += 1

    @staticmethod
    def _ratio(numerator: int | float, denominator: int | float) -> float | None:
        return numerator / denominator if denominator else None

    def finish(self) -> dict[str, Any]:
        diagnostics = []
        for (dataset, modality, bank_status), bucket in sorted(self.calibration.items()):
            item = {
                "dataset": dataset,
                "modality": modality,
                "bank_status": bank_status,
                **bucket,
                "false_empirical_cdf": {},
                "support_power": {},
            }
            for threshold in ("0.01", "0.05", "0.10"):
                item["false_empirical_cdf"][threshold] = self._ratio(
                    bucket["false_p_le"][threshold], bucket["false"]
                )
                item["support_power"][threshold] = self._ratio(
                    bucket["support_p_le"][threshold], bucket["support"]
                )
            diagnostics.append(item)

        labelled_selected = self.selected_supports + self.selected_false
        labelled_baseline = self.baseline_supports + self.baseline_false
        return {
            "queries": self.queries,
            "reserve_candidates": self.reserve_candidates,
            "average_reserve_size": self._ratio(self.reserve_candidates, self.queries),
            "reserve_supports": self.reserve_supports,
            "by_rejections": self.by_rejections,
            "selected_chunks": self.selected,
            "average_selected_chunks": self._ratio(self.selected, self.queries),
            "empty_contexts": self.empty_contexts,
            "empty_context_rate": self._ratio(self.empty_contexts, self.queries),
            "selected_supports": self.selected_supports,
            "selected_false": self.selected_false,
            "selected_unknown": self.selected_unknown,
            "micro_evidence_precision": self._ratio(self.selected_supports, labelled_selected),
            "conditional_reserve_support_recall": self._ratio(
                self.selected_supports, self.reserve_supports
            ),
            "mean_query_false_discovery_proportion": self._ratio(self.query_fdp_sum, self.queries),
            "mean_query_conditional_support_recall": self._ratio(
                self.query_recall_sum, self.queries
            ),
            "backfill_chunks": self.backfill_chunks,
            "backfill_queries": self.backfill_queries,
            "backfill_query_rate": self._ratio(self.backfill_queries, self.queries),
            "queries_hitting_context_cap": self.capped_queries,
            "baseline_top_k_chunks": self.baseline,
            "baseline_top_k_supports": self.baseline_supports,
            "baseline_top_k_false": self.baseline_false,
            "baseline_micro_evidence_precision": self._ratio(
                self.baseline_supports, labelled_baseline
            ),
            "baseline_conditional_reserve_support_recall": self._ratio(
                self.baseline_supports, self.reserve_supports
            ),
            "calibration_diagnostics": diagnostics,
        }


def iter_query_groups(
    rows: Iterable[Mapping[str, Any]], split_role: str
) -> Iterator[list[Mapping[str, Any]]]:
    filtered = (row for row in rows if row["split_role"] == split_role)
    for _, group in groupby(filtered, key=lambda row: (row["dataset"], row["qid"])):
        yield list(group)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply conformal p-values and experimental BY/backfill to retrieval logs."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--results-zip", type=Path)
    source.add_argument("--input", type=Path, nargs="+")
    parser.add_argument("--bank", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-role", choices=("development", "test"), default="development")
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--max-context", type=int, default=10)
    parser.add_argument("--allow-underpowered-banks", action="store_true")
    parser.add_argument("--compare-pooled", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reader = ResultsReader(results_zip=args.results_zip, inputs=args.input, bank=args.bank)
    bank_artifact = reader.read_bank()
    strategies = {"modality_aware": ReferenceBankIndex(bank_artifact)}
    if args.compare_pooled:
        print("Building dataset-pooled calibration ablation...", flush=True)
        pooled_artifact = build_dataset_pooled_artifact(reader.iter_rows(), bank_artifact)
        strategies["dataset_pooled"] = ReferenceBankIndex(pooled_artifact)

    output_path = args.output_dir / "selection_decisions.jsonl.gz"
    accumulators = {name: SummaryAccumulator() for name in strategies}
    dataset_accumulators: dict[str, dict[str, SummaryAccumulator]] = {
        name: defaultdict(SummaryAccumulator) for name in strategies
    }
    expected_queries = reader.query_count(args.split_role)
    seen_queries = 0
    with open_output(output_path) as handle:
        progress = tqdm(
            iter_query_groups(reader.iter_rows(), args.split_role),
            total=expected_queries,
            desc=f"Conformal selection ({args.split_role})",
            unit="query",
        )
        for query_rows in progress:
            results = {}
            for name, bank_index in strategies.items():
                result = select_query_context(
                    query_rows,
                    bank_index,
                    alpha=args.alpha,
                    max_context=args.max_context,
                    allow_underpowered=args.allow_underpowered_banks,
                )
                accumulators[name].update(result)
                dataset_accumulators[name][result["dataset"]].update(result)
                results[name] = result
            handle.write(json.dumps({"strategies": results}, ensure_ascii=False) + "\n")
            seen_queries += 1

    if seen_queries == 0:
        raise RuntimeError(f"No queries with split_role={args.split_role!r} were found")
    strategy_summaries = {}
    for name, accumulator in accumulators.items():
        strategy_summary = accumulator.finish()
        strategy_summary["by_dataset"] = {
            dataset: dataset_accumulator.finish()
            for dataset, dataset_accumulator in sorted(dataset_accumulators[name].items())
        }
        strategy_summaries[name] = strategy_summary

    summary = {
        "is_paper_ready": False,
        "formal_capped_risk_guarantee": False,
        "warnings": [
            "This is an experimental BY-then-cap implementation; the at-most-K proof remains open.",
            "Evidence recall is conditional on support present in the frozen reserve pool.",
            "Development-split results are for debugging/model selection, not final evaluation.",
        ],
        "configuration": {
            "split_role": args.split_role,
            "alpha": args.alpha,
            "max_context": args.max_context,
            "allow_underpowered_banks": args.allow_underpowered_banks,
            "strategies": list(strategies),
            "bank_conditioning": bank_artifact["conditioning"],
            "bank_is_paper_ready": bank_artifact.get("is_paper_ready"),
        },
        "strategies": strategy_summaries,
        "outputs": {"decisions": output_path.name},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "selection_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Decisions: {output_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
