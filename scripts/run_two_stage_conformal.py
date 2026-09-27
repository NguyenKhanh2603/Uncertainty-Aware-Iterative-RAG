"""Two-Stage Conformal Selection:
Stage 1: BH with false-null p-values at alpha1 (generous) → high recall
Stage 2: Support-null post-filter at alpha2 → boost precision by removing
         chunks that look unlike support chunks.

The support-null p-value uses the LOWER tail:
  p = (1 + #{support scores <= observed}) / (n + 1)
A chunk with unusually LOW cosine (compared to support distribution) gets
a small p-value → rejected by BH → removed from context.
"""

from __future__ import annotations

import argparse
import gzip
import json
from bisect import bisect_right
from collections import defaultdict
from itertools import groupby
from pathlib import Path
from typing import Any, Iterable, Mapping, Iterator

from tqdm import tqdm

from uncertainty_rag.core.conformal_retrieval import (
    RetrievalCandidate,
    conformal_p_value,
)
from uncertainty_rag.core.conformal_selection import (
    ReferenceBankIndex,
    backfill_rejected_indices,
    benjamini_hochberg,
)


def iter_rows(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def iter_query_groups(
    rows: Iterable[Mapping[str, Any]], split_role: str
) -> Iterator[list[Mapping[str, Any]]]:
    filtered = (row for row in rows if row["split_role"] == split_role)
    for _, group in groupby(filtered, key=lambda row: (row["dataset"], row["qid"])):
        yield list(group)


def build_dataset_pooled_artifact(
    rows: Iterable[Mapping[str, Any]], template: Mapping[str, Any]
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["split_role"] == "calibration" and row["support_label"] == "false":
            grouped[str(row["dataset"])].append(float(row["cosine_score"]))
    banks = []
    for dataset, scores in sorted(grouped.items()):
        scores.sort()
        banks.append({
            "condition": {"dataset": dataset},
            "n_false_scores": len(scores),
            "scores": scores,
        })
    return {
        "top_l": template["top_l"],
        "conditioning": ["dataset"],
        "rank_bins": [],
        "min_bank_size": template.get("min_bank_size", 1),
        "pipeline_fingerprints": template.get("pipeline_fingerprints", {}),
        "banks": banks,
    }


# ===== SUPPORT BANK =====

class SupportBankIndex:
    """Reference bank of SUPPORT chunk cosine scores for the support-null test."""

    def __init__(self, conditioning: str = "dataset"):
        self.conditioning = conditioning
        self._scores: dict[str, list[float]] = defaultdict(list)
        self._sorted: dict[str, tuple[float, ...]] = {}

    def add(self, row: Mapping[str, Any]) -> None:
        if row["split_role"] == "calibration" and row["support_label"] == "support":
            if self.conditioning == "dataset":
                key = str(row["dataset"])
            else:
                key = f"{row['dataset']}_{row['modality']}"
            self._scores[key].append(float(row["cosine_score"]))

    def finalize(self) -> None:
        for key, scores in self._scores.items():
            scores.sort()
            self._sorted[key] = tuple(scores)
        print(f"Support banks ({self.conditioning}):")
        for key, scores in sorted(self._sorted.items()):
            print(f"  {key}: {len(scores)} support scores")

    def p_value(self, cosine_score: float, dataset: str, modality: str = "") -> float:
        """Lower-tail conformal p-value against support distribution.

        Small p → chunk has unusually low cosine for support → likely false.
        """
        if self.conditioning == "dataset":
            key = dataset
        else:
            key = f"{dataset}_{modality}"
        scores = self._sorted.get(key)
        if not scores:
            return 1.0  # No support bank → conservative, keep chunk
        count_le = bisect_right(scores, cosine_score)
        return (1.0 + count_le) / (len(scores) + 1.0)


# ===== METRICS =====

class MetricsAccumulator:
    def __init__(self):
        self.queries = 0
        self.reserve_supports = 0
        self.selected = 0
        self.selected_supports = 0
        self.selected_false = 0
        self.empty = 0

    def update(self, candidates, selected_indices):
        supports = sum(candidates[i].support_label == "support" for i in selected_indices)
        false = sum(candidates[i].support_label == "false" for i in selected_indices)
        total_supports = sum(c.support_label == "support" for c in candidates)
        self.queries += 1
        self.reserve_supports += total_supports
        self.selected += len(selected_indices)
        self.selected_supports += supports
        self.selected_false += false
        self.empty += int(not selected_indices)

    def finish(self):
        labelled = self.selected_supports + self.selected_false
        return {
            "queries": self.queries,
            "average_selected_chunks": self.selected / self.queries if self.queries else 0,
            "micro_evidence_precision": self.selected_supports / labelled if labelled else None,
            "conditional_reserve_support_recall": self.selected_supports / self.reserve_supports if self.reserve_supports else None,
            "empty_context_rate": self.empty / self.queries if self.queries else 0,
        }


def format_pct(val):
    if val is None:
        return "N/A"
    return f"{val * 100:.1f}%"


def two_stage_select(
    query_rows: list[Mapping[str, Any]],
    bank_false: ReferenceBankIndex,
    support_bank: SupportBankIndex,
    *,
    alpha1: float,
    alpha2: float,
    max_context: int,
) -> set[int]:
    """Two-stage conformal selection.

    Stage 1: BH on false-null p-values at alpha1 → initial selection (high recall)
    Stage 2: BH on support-null p-values of selected chunks at alpha2 → remove
             chunks that look unlike support (boost precision)
    """
    candidates = [RetrievalCandidate.from_mapping(row) for row in query_rows]

    # Stage 1: Standard false-null selection
    scored = [bank_false.score(row, allow_underpowered=True) for row in query_rows]
    p_false = [item["p_value"] for item in scored]
    bh1 = benjamini_hochberg(p_false, alpha1)
    stage1_selected, _ = backfill_rejected_indices(
        [c.rank for c in candidates],
        [c.chunk_id for c in candidates],
        bh1.rejected_indices,
        max_context,
    )
    stage1 = set(stage1_selected)

    if not stage1 or alpha2 <= 0:
        return stage1

    # Stage 2: Support-null filter on stage1 selection
    stage1_list = sorted(stage1)
    p_support = []
    for idx in stage1_list:
        p = support_bank.p_value(
            candidates[idx].cosine_score,
            candidates[idx].dataset,
            candidates[idx].modality,
        )
        p_support.append(p)

    # BH on support p-values: reject (remove) chunks that look unlike support
    bh2 = benjamini_hochberg(p_support, alpha2)
    rejected_in_stage1 = {stage1_list[i] for i in bh2.rejected_indices}

    # Final: keep stage1 minus rejected by stage2
    final = stage1 - rejected_in_stage1
    return final


def main():
    parser = argparse.ArgumentParser(description="Two-Stage Conformal Selection")
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--bank", type=Path, required=True, help="False-null modality-aware bank")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alpha1", type=str, default="0.9,0.99",
                        help="Stage 1 alphas (comma-separated)")
    parser.add_argument("--alpha2", type=str, default="0.05,0.10,0.15,0.20,0.30",
                        help="Stage 2 alphas (comma-separated)")
    parser.add_argument("--max-context", type=int, default=10)
    parser.add_argument("--support-conditioning", choices=["dataset", "modality"],
                        default="dataset")
    args = parser.parse_args()

    # Load false-null bank (modality-aware)
    opener = gzip.open if args.bank.suffix == ".gz" else Path.open
    with opener(args.bank, "rt", encoding="utf-8") as f:
        bank_artifact = json.load(f)
    bank_modality = ReferenceBankIndex(bank_artifact)

    # Build pooled false-null bank
    def all_rows():
        for path in args.input:
            yield from iter_rows(path)
    pooled_artifact = build_dataset_pooled_artifact(all_rows(), bank_artifact)
    bank_pooled = ReferenceBankIndex(pooled_artifact)

    # Build support banks
    support_bank = SupportBankIndex(conditioning=args.support_conditioning)
    for row in all_rows():
        support_bank.add(row)
    support_bank.finalize()

    alpha1_values = sorted({float(a.strip()) for a in args.alpha1.split(",")})
    alpha2_values = sorted({float(a.strip()) for a in args.alpha2.split(",")})

    # Strategies to evaluate
    false_banks = {
        "dataset_pooled": bank_pooled,
        "modality_aware": bank_modality,
    }

    # Results storage
    results: dict[str, dict] = {}
    datasets_order = ["hotpotqa", "mmqa", "tatqa", "webqa"]

    # Fixed top-K baselines
    for label, k in [("fixed_top_10", 10), ("fixed_top_20", 20)]:
        results[label] = {ds: MetricsAccumulator() for ds in datasets_order + ["ALL"]}

    # Single-stage (alpha2=0, no support filter)
    for strategy in false_banks:
        for a1 in alpha1_values:
            key = f"{strategy}_a1={a1:.2f}_no_filter"
            results[key] = {ds: MetricsAccumulator() for ds in datasets_order + ["ALL"]}

    # Two-stage
    for strategy in false_banks:
        for a1 in alpha1_values:
            for a2 in alpha2_values:
                key = f"{strategy}_a1={a1:.2f}_a2={a2:.2f}"
                results[key] = {ds: MetricsAccumulator() for ds in datasets_order + ["ALL"]}

    print("Processing test queries...", flush=True)
    query_count = 0
    for query_rows in tqdm(iter_query_groups(all_rows(), "test"), desc="Queries", unit="q"):
        query_count += 1
        candidates = [RetrievalCandidate.from_mapping(row) for row in query_rows]
        dataset = candidates[0].dataset

        # Fixed Top-K
        retrieval_order = sorted(range(len(candidates)), key=lambda i: (candidates[i].rank, candidates[i].chunk_id))
        for label, k in [("fixed_top_10", 10), ("fixed_top_20", 20)]:
            top_k = set(retrieval_order[:k])
            results[label][dataset].update(candidates, top_k)
            results[label]["ALL"].update(candidates, top_k)

        for strategy_name, bank_false in false_banks.items():
            for a1 in alpha1_values:
                # Single-stage (no support filter)
                sel_single = two_stage_select(
                    query_rows, bank_false, support_bank,
                    alpha1=a1, alpha2=0, max_context=args.max_context,
                )
                key_single = f"{strategy_name}_a1={a1:.2f}_no_filter"
                results[key_single][dataset].update(candidates, sel_single)
                results[key_single]["ALL"].update(candidates, sel_single)

                # Two-stage with varying alpha2
                for a2 in alpha2_values:
                    sel = two_stage_select(
                        query_rows, bank_false, support_bank,
                        alpha1=a1, alpha2=a2, max_context=args.max_context,
                    )
                    key = f"{strategy_name}_a1={a1:.2f}_a2={a2:.2f}"
                    results[key][dataset].update(candidates, sel)
                    results[key]["ALL"].update(candidates, sel)

    print(f"\nProcessed {query_count} queries.\n")

    # Generate report
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "two_stage_report.md"

    with open(report_path, "w", encoding="utf-8") as out:
        out.write("# Two-Stage Conformal Selection Report (1000 Cali / 100 Test)\n\n")
        out.write("**Stage 1**: BH on false-null (high recall)\n")
        out.write("**Stage 2**: BH on support-null (remove low-cosine chunks to boost precision)\n\n")
        out.write(f"Support bank conditioning: `{args.support_conditioning}`\n\n")

        for ds in datasets_order:
            out.write(f"### Dataset: {ds.upper()}\n")
            out.write("| Method | Chunks | Precision | Recall | Empty |\n")
            out.write("| :--- | :--- | :--- | :--- | :--- |\n")

            # Fixed baselines
            for label in ["fixed_top_10", "fixed_top_20"]:
                m = results[label][ds].finish()
                out.write(f"| {label} | {m['average_selected_chunks']:.2f} | {format_pct(m['micro_evidence_precision'])} | {format_pct(m['conditional_reserve_support_recall'])} | {format_pct(m['empty_context_rate'])} |\n")

            # Single-stage then two-stage
            for strategy in false_banks:
                for a1 in alpha1_values:
                    # Single stage
                    key = f"{strategy}_a1={a1:.2f}_no_filter"
                    m = results[key][ds].finish()
                    out.write(f"| {strategy} a1={a1} (no filter) | {m['average_selected_chunks']:.2f} | {format_pct(m['micro_evidence_precision'])} | {format_pct(m['conditional_reserve_support_recall'])} | {format_pct(m['empty_context_rate'])} |\n")

                    # Two stage
                    for a2 in alpha2_values:
                        key = f"{strategy}_a1={a1:.2f}_a2={a2:.2f}"
                        m = results[key][ds].finish()
                        out.write(f"| **{strategy} a1={a1} +filter a2={a2}** | {m['average_selected_chunks']:.2f} | {format_pct(m['micro_evidence_precision'])} | {format_pct(m['conditional_reserve_support_recall'])} | {format_pct(m['empty_context_rate'])} |\n")

            out.write("\n")

    # Also save JSON
    json_output = {}
    for key, ds_metrics in results.items():
        json_output[key] = {ds: ds_metrics[ds].finish() for ds in datasets_order + ["ALL"]}
    json_path = args.output_dir / "two_stage_summary.json"
    json_path.write_text(json.dumps(json_output, indent=2) + "\n", encoding="utf-8")

    print(f"Report: {report_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
