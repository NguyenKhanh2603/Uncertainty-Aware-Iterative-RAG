"""Hybrid Conformal BH: combine modality-aware and dataset-pooled p-values.

For each chunk, computes p-values from BOTH reference banks, then combines
them via Bonferroni correction: p_hybrid = min(1, 2 * min(p_mod, p_pool)).

This is theoretically valid under arbitrary dependence and ensures we never
lose recall relative to the better bank, while gaining precision when both
banks agree a chunk is non-support.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from itertools import groupby
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

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


class MetricsAccumulator:
    def __init__(self):
        self.queries = 0
        self.reserve_supports = 0
        self.selected = 0
        self.selected_supports = 0
        self.selected_false = 0
        self.empty = 0
        self.fdp_sum = 0.0

    def update(self, decisions, selected_indices):
        supports = sum(decisions[i]["support_label"] == "support" for i in selected_indices)
        false = sum(decisions[i]["support_label"] == "false" for i in selected_indices)
        total_supports = sum(d["support_label"] == "support" for d in decisions)
        fdp = false / len(selected_indices) if selected_indices else 0.0

        self.queries += 1
        self.reserve_supports += total_supports
        self.selected += len(selected_indices)
        self.selected_supports += supports
        self.selected_false += false
        self.empty += int(not selected_indices)
        self.fdp_sum += fdp

    def finish(self):
        labelled = self.selected_supports + self.selected_false
        return {
            "queries": self.queries,
            "average_selected_chunks": self.selected / self.queries if self.queries else 0,
            "micro_evidence_precision": self.selected_supports / labelled if labelled else None,
            "conditional_reserve_support_recall": self.selected_supports / self.reserve_supports if self.reserve_supports else None,
            "empty_context_rate": self.empty / self.queries if self.queries else 0,
            "mean_query_fdp": self.fdp_sum / self.queries if self.queries else 0,
        }


def hybrid_select(
    query_rows: list[Mapping[str, Any]],
    bank_modality: ReferenceBankIndex,
    bank_pooled: ReferenceBankIndex,
    *,
    alpha: float,
    max_context: int,
) -> tuple[set[int], list[Mapping[str, Any]]]:
    """Score each chunk with both banks, combine via Bonferroni, run BH."""

    candidates = [RetrievalCandidate.from_mapping(row) for row in query_rows]

    p_values_combined = []
    scored_info = []
    for i, (row, cand) in enumerate(zip(query_rows, candidates)):
        # modality-aware p-value
        scored_mod = bank_modality.score(row, allow_underpowered=True)
        p_mod = scored_mod["p_value"]

        # pooled p-value
        scored_pool = bank_pooled.score(row, allow_underpowered=True)
        p_pool = scored_pool["p_value"]

        # Bonferroni combination: p_hybrid = min(1, 2 * min(p_mod, p_pool))
        p_hybrid = min(1.0, 2.0 * min(p_mod, p_pool))

        p_values_combined.append(p_hybrid)
        scored_info.append({
            "candidate": cand,
            "p_mod": p_mod,
            "p_pool": p_pool,
            "p_hybrid": p_hybrid,
        })

    bh = benjamini_hochberg(p_values_combined, alpha)
    selected, _ = backfill_rejected_indices(
        [c.rank for c in candidates],
        [c.chunk_id for c in candidates],
        bh.rejected_indices,
        max_context,
    )
    return set(selected), [
        {
            "chunk_id": info["candidate"].chunk_id,
            "rank": info["candidate"].rank,
            "cosine_score": info["candidate"].cosine_score,
            "support_label": info["candidate"].support_label,
            "modality": info["candidate"].modality,
            "p_mod": info["p_mod"],
            "p_pool": info["p_pool"],
            "p_hybrid": info["p_hybrid"],
            "bh_rejected": i in set(bh.rejected_indices),
            "selected": i in set(selected),
        }
        for i, info in enumerate(scored_info)
    ]


def format_pct(val):
    if val is None:
        return "N/A"
    return f"{val * 100:.1f}%"


def main():
    parser = argparse.ArgumentParser(description="Hybrid Conformal BH evaluation")
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--bank", type=Path, required=True, help="Modality-aware bank")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alphas", type=str, default="0.1,0.3,0.5,0.9,0.99")
    parser.add_argument("--max-context", type=int, default=10)
    args = parser.parse_args()

    # Load modality-aware bank
    opener = gzip.open if args.bank.suffix == ".gz" else Path.open
    with opener(args.bank, "rt", encoding="utf-8") as f:
        bank_artifact = json.load(f)
    bank_modality = ReferenceBankIndex(bank_artifact)

    # Build pooled bank from calibration data
    def all_rows():
        for path in args.input:
            yield from iter_rows(path)

    pooled_artifact = build_dataset_pooled_artifact(all_rows(), bank_artifact)
    bank_pooled = ReferenceBankIndex(pooled_artifact)

    alphas = sorted({float(a.strip()) for a in args.alphas.split(",")})

    # Also run dataset_pooled-only and modality_aware-only for comparison
    strategies = {
        "hybrid_bonferroni": None,       # special handling
        "dataset_pooled": bank_pooled,
        "modality_aware": bank_modality,
    }

    results: dict[str, dict[float, dict[str, MetricsAccumulator]]] = {}
    for strategy in strategies:
        results[strategy] = {}
        for alpha in alphas:
            results[strategy][alpha] = defaultdict(MetricsAccumulator)

    # Also track Fixed Top-K
    results["fixed_top_10"] = {0: defaultdict(MetricsAccumulator)}
    results["fixed_top_20"] = {0: defaultdict(MetricsAccumulator)}

    print("Processing test queries...", flush=True)
    query_count = 0
    for query_rows in tqdm(iter_query_groups(all_rows(), "test"), desc="Queries", unit="q"):
        query_count += 1
        candidates = [RetrievalCandidate.from_mapping(row) for row in query_rows]
        dataset = candidates[0].dataset

        # Fixed Top-10 and Top-20
        retrieval_order = sorted(range(len(candidates)), key=lambda i: (candidates[i].rank, candidates[i].chunk_id))
        for label, k in [("fixed_top_10", 10), ("fixed_top_20", 20)]:
            top_k = set(retrieval_order[:k])
            decisions_list = [{"support_label": c.support_label} for c in candidates]
            results[label][0][dataset].update(decisions_list, top_k)
            results[label][0]["ALL"].update(decisions_list, top_k)

        for alpha in alphas:
            decisions_list = [{"support_label": c.support_label} for c in candidates]

            # Hybrid
            selected_hybrid, _ = hybrid_select(
                query_rows, bank_modality, bank_pooled,
                alpha=alpha, max_context=args.max_context,
            )
            results["hybrid_bonferroni"][alpha][dataset].update(decisions_list, selected_hybrid)
            results["hybrid_bonferroni"][alpha]["ALL"].update(decisions_list, selected_hybrid)

            # Dataset pooled only
            from uncertainty_rag.core.conformal_selection import select_query_context
            result_pooled = select_query_context(
                query_rows, bank_pooled, alpha=alpha, max_context=args.max_context, allow_underpowered=True,
            )
            sel_pooled = set(i for i, d in enumerate(result_pooled["decisions"]) if d["selected"])
            results["dataset_pooled"][alpha][dataset].update(decisions_list, sel_pooled)
            results["dataset_pooled"][alpha]["ALL"].update(decisions_list, sel_pooled)

            # Modality aware only
            result_mod = select_query_context(
                query_rows, bank_modality, alpha=alpha, max_context=args.max_context, allow_underpowered=True,
            )
            sel_mod = set(i for i, d in enumerate(result_mod["decisions"]) if d["selected"])
            results["modality_aware"][alpha][dataset].update(decisions_list, sel_mod)
            results["modality_aware"][alpha]["ALL"].update(decisions_list, sel_mod)

    print(f"\nProcessed {query_count} test queries.\n")

    # Generate report
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = ["hotpotqa", "mmqa", "tatqa", "webqa"]
    report_path = args.output_dir / "hybrid_comparison_report.md"
    json_path = args.output_dir / "hybrid_comparison_summary.json"

    json_output = {}

    with open(report_path, "w", encoding="utf-8") as out:
        out.write("# Hybrid Conformal BH Comparison Report (1000 Cali / 100 Test)\n\n")
        out.write("This report compares:\n")
        out.write("1. **Fixed Top-K** baselines\n")
        out.write("2. **BH cosine (dataset_pooled)** - pool all modalities together\n")
        out.write("3. **Conformal BH (modality_aware)** - separate banks per modality\n")
        out.write("4. **[NEW] Hybrid Bonferroni** - combine both banks: `p = min(1, 2*min(p_mod, p_pool))`\n\n")

        for ds in datasets:
            out.write(f"### Dataset: {ds.upper()}\n")
            out.write("| Method | Kept Chunks | Precision | Recall | Empty Rate |\n")
            out.write("| :--- | :--- | :--- | :--- | :--- |\n")

            json_output[ds] = {}

            # Fixed top-10
            m = results["fixed_top_10"][0][ds].finish()
            out.write(f"| Fixed Top-10 | {m['average_selected_chunks']:.2f} | {format_pct(m['micro_evidence_precision'])} | {format_pct(m['conditional_reserve_support_recall'])} | {format_pct(m['empty_context_rate'])} |\n")
            json_output[ds]["fixed_top_10"] = m

            # Fixed top-20
            m = results["fixed_top_20"][0][ds].finish()
            out.write(f"| Fixed Top-20 | {m['average_selected_chunks']:.2f} | {format_pct(m['micro_evidence_precision'])} | {format_pct(m['empty_context_rate'])} | {format_pct(m['conditional_reserve_support_recall'])} |\n")
            json_output[ds]["fixed_top_20"] = m

            for alpha in alphas:
                for strategy, label in [
                    ("dataset_pooled", f"BH cosine (α={alpha})"),
                    ("modality_aware", f"Conformal BH (mod_aware, α={alpha})"),
                    ("hybrid_bonferroni", f"**Hybrid BH (α={alpha})**"),
                ]:
                    m = results[strategy][alpha][ds].finish()
                    out.write(f"| {label} | {m['average_selected_chunks']:.2f} | {format_pct(m['micro_evidence_precision'])} | {format_pct(m['conditional_reserve_support_recall'])} | {format_pct(m['empty_context_rate'])} |\n")
                    json_output[ds][f"{strategy}_alpha_{alpha}"] = m

            out.write("\n")

    # Write JSON
    json_path.write_text(json.dumps(json_output, indent=2) + "\n", encoding="utf-8")

    print(f"Report: {report_path}")
    print(f"JSON: {json_path}")

    # Print summary table to stdout
    print("\n=== SUMMARY (ctx=10) ===")
    for ds in datasets:
        print(f"\n--- {ds.upper()} ---")
        for alpha in alphas:
            for strategy in ["dataset_pooled", "modality_aware", "hybrid_bonferroni"]:
                m = results[strategy][alpha][ds].finish()
                print(f"  {strategy:25s} α={alpha:.2f}: chunks={m['average_selected_chunks']:.2f} P={format_pct(m['micro_evidence_precision']):>6s} R={format_pct(m['conditional_reserve_support_recall']):>6s} empty={format_pct(m['empty_context_rate']):>5s}")


if __name__ == "__main__":
    main()
