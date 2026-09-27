"""Build one provenance-preserving table for all completed 1,000/100 runs.

The repository contains separate reports for the original cosine/BY/BH sweep,
the protocol-faithful CCE/CONFLARE/TRAQ adaptations, Qwen-7B internal-signal
ablations, and alpha-free operating points.  This program reads their frozen
JSON summaries and renders them in one table per dataset without recomputing
or relabelling any result.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DATASETS = ("hotpotqa", "mmqa", "tatqa", "webqa")


@dataclass(frozen=True)
class RowSpec:
    family: str
    label: str
    source: str
    method: str


ROWS = (
    RowSpec("Fixed", "Fixed Top-10", "original", "fixed_top10"),
    RowSpec("Fixed", "Fixed Top-20", "original", "fixed_top20"),
    RowSpec("Literature", "CCE Conformal-Embedding, Jina adaptation (α=.10)", "literature", "cce_conformal_embedding_jina_alpha_0.10"),
    RowSpec("Literature", "CONFLARE source-question, Jina adaptation (α=.10)", "literature", "conflare_source_question_jina_alpha_0.10"),
    RowSpec("Literature", "TRAQ retrieval Bonferroni, Jina adaptation (α=.10; α_R=.05)", "literature", "traq_retrieval_bonferroni_jina_alpha_0.10"),
    RowSpec("Original cosine", "Query-level cosine all-support (α=.10; 1,000 cal)", "original", "query_level_cosine_alpha_0.10"),
    RowSpec("Original cosine", "BY cosine (α=.10)", "original", "by_cosine_alpha_0.10"),
    RowSpec("Original cosine", "BY cosine (α=.30)", "original", "by_cosine_alpha_0.30"),
    RowSpec("Original cosine", "BY cosine (α=.50)", "original", "by_cosine_alpha_0.50"),
    RowSpec("Original cosine", "BH cosine, ctx=10 (α=.10)", "original", "bh_cosine_alpha_0.10_ctx10"),
    RowSpec("Original cosine", "BH cosine, ctx=10 (α=.30)", "original", "bh_cosine_alpha_0.30_ctx10"),
    RowSpec("Original cosine", "BH cosine, ctx=10 (α=.50)", "original", "bh_cosine_alpha_0.50_ctx10"),
    RowSpec("Original cosine", "BH cosine, ctx=10 (α=.90)", "original", "bh_cosine_alpha_0.90_ctx10"),
    RowSpec("Original cosine", "BH cosine, ctx=10 (α=.99)", "original", "bh_cosine_alpha_0.99_ctx10"),
    RowSpec("Original cosine", "BH cosine, ctx=20 (α=.99)", "original", "bh_cosine_alpha_0.99_ctx20"),
    RowSpec("Internal Qwen-7B", "Cosine baseline (500 probe / 500 conformal cal; α=.10)", "internal", "query_level_cosine_matched_alpha_0.10"),
    RowSpec("Internal Qwen-7B", "LM-head only (500 / 500; α=.10)", "internal", "query_level_lm_head_alpha_0.10"),
    RowSpec("Internal Qwen-7B", "Hidden-state relevance probe only (500 / 500; α=.10)", "internal", "query_level_hidden_probe_alpha_0.10"),
    RowSpec("Internal Qwen-7B", "LM-head + hidden-state probe (500 / 500; α=.10)", "internal", "query_level_lm_head_hidden_alpha_0.10"),
    RowSpec("Internal Qwen-7B", "Cosine + LM-head + hidden-state probe (500 / 500; α=.10)", "internal", "query_level_cosine_lm_head_hidden_alpha_0.10"),
    RowSpec("Alpha-free", "Query-level cosine F1-selected (1,000 cal; no α at test)", "alpha_free", "query_level_f1_calibrated"),
    RowSpec("Alpha-free", "Query-level cosine budget-10 all-support (1,000 cal; no α at test)", "alpha_free", "query_level_budget10_all_support"),
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric(value: dict[str, Any], *keys: str) -> float:
    for key in keys:
        if key in value:
            return float(value[key])
    raise KeyError(f"None of {keys} appears in result keys: {tuple(value)}")


def validate(sources: dict[str, dict[str, Any]]) -> None:
    for source in ("original", "literature", "alpha_free"):
        for dataset in DATASETS:
            state = sources[source]["datasets"][dataset]
            if state.get("test_queries") != 100 or state.get("calibration_queries") != 1000:
                raise RuntimeError(f"{source}/{dataset}: expected 1,000 calibration / 100 test")
    for dataset in DATASETS:
        state = sources["internal"]["datasets"][dataset]
        split = state.get("split", {})
        if state.get("status") != "complete" or split.get("heldout_test_queries") != 100:
            raise RuntimeError(f"internal/{dataset}: incomplete or wrong test split")
        if split.get("probe_train_queries", 0) + split.get("conformal_calibration_queries", 0) != 1000:
            raise RuntimeError(f"internal/{dataset}: expected 1,000 parent calibration qids")


def row(spec: RowSpec, state: dict[str, Any]) -> str:
    selection = state["selection"][spec.method]
    downstream = state["downstream"][spec.method]
    any_support = metric(selection, "query_any_support", "query_any_support_conditional")
    all_support = metric(selection, "query_all_support", "query_all_support_conditional")
    return (
        f"| {spec.family} | {spec.label} | {metric(selection, 'mean_chunks'):.2f} | "
        f"{metric(selection, 'precision'):.1%} | {metric(selection, 'support_recall'):.1%} | "
        f"{metric(selection, 'empty_rate'):.1%} | {any_support:.1%} | {all_support:.1%} | "
        f"{metric(downstream, 'em'):.3f} | {metric(downstream, 'f1'):.3f} | "
        f"{metric(downstream, 'numerical_accuracy'):.3f} |"
    )


def write_report(output: Path, sources: dict[str, dict[str, Any]], source_paths: dict[str, Path]) -> None:
    lines = [
        "# Full context-selection comparison: original cosine, literature baselines, and internal signals",
        "",
        "Every row evaluates the same frozen Jina Top-30 candidates on the same 100 held-out test qids per dataset. This consolidates completed artifacts; it does not rerun, average, or overwrite any experiment.",
        "",
        "## Protocol distinctions that must remain visible",
        "",
        "- **Original cosine / CCE / CONFLARE / TRAQ / alpha-free:** 1,000 calibration qids and 100 held-out test qids. CCE, CONFLARE, and TRAQ are Jina adaptations with their distinct retrieval calibration units. TRAQ is its retrieval component only, not the full semantic answer prediction-set procedure.",
        "- **Internal Qwen-7B rows:** the same 1,000 parent calibration qids are split into disjoint probe-train and conformal-calibration roles (about 500 / 500, exact counts shown in the internal report), followed by the same 100 test qids. Their use of 500 calibration qids means they are an ablation, not a strictly matched 1,000-calibration head-to-head result.",
        "- **Alpha-free rows:** threshold is chosen only on the 1,000 calibration qids for the stated empirical utility. It is frozen for test inference, but it has no conformal risk guarantee.",
        "- **BY/BH:** `alpha` is the multiple-testing level. BH rows use the reported experimental context cap; no capped-procedure FDR guarantee is claimed.",
        "",
        "## Metrics",
        "",
        "Chunks is mean retained chunks/query; Precision and Recall are micro support metrics inside frozen Top-30; Empty is the fraction retaining no chunk. Any/All are query support-retention rates recorded by the source artifact. EM, token F1, and Numeric are shared deterministic Qwen direct-answer diagnostics on the held-out 100 qids, not conformal guarantees.",
        "",
    ]
    for dataset in DATASETS:
        lines.extend(
            [
                f"## {dataset}",
                "",
                "| Family | Method | Chunks | Precision | Recall | Empty | Any support | All support | EM | F1 | Numeric |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for spec in ROWS:
            lines.append(row(spec, sources[spec.source]["datasets"][dataset]))
        lines.append("")
    lines.extend(
        [
            "## Exact source artifacts",
            "",
            f"- [Original cosine, BY/BH α sweep]({source_paths['original'].as_posix()})",
            f"- [CCE / CONFLARE / TRAQ adapted protocol]({source_paths['literature'].as_posix()})",
            f"- [Qwen-7B internal-signal ablation]({source_paths['internal'].as_posix()})",
            f"- [Alpha-free query-level operating points]({source_paths['alpha_free'].as_posix()})",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "research/internal_state_rag/results/"
            "full_context_selection_1000cal_100test_2026_09_27"
        ),
    )
    args = parser.parse_args()
    root = Path("research/internal_state_rag/results")
    paths = {
        "original": root / "all_datasets_cosine_six_methods_1000cal_100test_2026_09_26/summary.json",
        "literature": root / "literature_protocol_1000cal_100test_2026_09_26/summary.json",
        "internal": root / "all_datasets_cosine_six_methods_1000cal_100test_2026_09_26/internal_signal_ablation_7b/summary.json",
        "alpha_free": root / "alpha_free_query_level_1000cal_100test_2026_09_26/downstream_summary.json",
    }
    sources = {name: read_json(path) for name, path in paths.items()}
    validate(sources)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Relative paths keep the provenance links valid on GitHub from this result directory.
    source_paths = {
        name: Path("..") / path.parent.relative_to(root) / "REPORT.md" for name, path in paths.items()
    }
    source_paths["alpha_free"] = Path("..") / paths["alpha_free"].parent.relative_to(root) / "DOWNSTREAM_REPORT.md"
    write_report(args.output_dir / "REPORT.md", sources, source_paths)
    (args.output_dir / "SOURCE_ARTIFACTS.json").write_text(
        json.dumps(
            {
                "sources": {name: str(path) for name, path in paths.items()},
                "rows_per_dataset": len(ROWS),
                "datasets": list(DATASETS),
                "test_queries_per_dataset": 100,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output_dir / 'REPORT.md'} with {len(ROWS)} rows per dataset")


if __name__ == "__main__":
    main()
