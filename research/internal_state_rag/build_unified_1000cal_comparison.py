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
    RowSpec("Legacy cosine proxy", "CCE-style global positive-cosine proxy (α=.10)", "original", "cce_alpha_0.10"),
    RowSpec("Legacy cosine proxy", "CONFLARE-style global positive-cosine proxy (α=.10)", "original", "conflare_alpha_0.10"),
    RowSpec("Legacy cosine proxy", "TRAQ-style loose positive-cosine proxy (α=.10)", "original", "traq_alpha_0.10"),
    RowSpec("Literature", "CCE Conformal-Embedding, Jina adaptation (α=.10; identical to CCE-style proxy)", "literature", "cce_conformal_embedding_jina_alpha_0.10"),
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
        "> **Fidelity status:** CCE, CONFLARE, and TRAQ rows are **original-formula / calibration-unit adaptations**, not end-to-end executions of their released pipelines. The table intentionally includes both the earlier *legacy cosine proxies* and later Jina adaptations. The CCE adaptation is mathematically identical to its legacy CCE-style proxy under this shared Jina setup; CONFLARE and TRAQ use different calibration units. See [FIDELITY_AUDIT.md](FIDELITY_AUDIT.md); do not claim these values reproduce or outperform the published systems.",
        "",
        "## Protocol distinctions that must remain visible",
        "",
        "- **Original cosine / CCE / CONFLARE / TRAQ / alpha-free:** 1,000 calibration qids and 100 held-out test qids. CCE, CONFLARE, and TRAQ are Jina adaptations with their distinct retrieval calibration units. TRAQ is its retrieval component only, not the full semantic answer prediction-set procedure.",
        "- **Internal Qwen-7B rows:** the same 1,000 parent calibration qids are split into disjoint probe-train and conformal-calibration roles (about 500 / 500, exact counts shown in the internal report), followed by the same 100 test qids. Their use of 500 calibration qids means they are an ablation, not a strictly matched 1,000-calibration head-to-head result.",
        "- **Alpha-free rows:** threshold is chosen only on the 1,000 calibration qids for the stated empirical utility. It is frozen for test inference, but it has no conformal risk guarantee.",
        "- **BY/BH:** `alpha` is the multiple-testing level. BH rows use the reported experimental context cap; no capped-procedure FDR guarantee is claimed.",
        "- **Legacy cosine proxy vs Jina adaptation:** neither is a full published-pipeline reproduction. CCE's two rows are exactly the same selection because both use all positive Jina scores and the same finite lower-alpha quantile. CONFLARE and TRAQ adaptation rows preserve their per-question calibration units, unlike the old global positive-score proxies.",
        "",
        "## How each chunk-pruning algorithm works",
        "",
        "**CCE Conformal-Embedding.** For every labelled relevant calibration question–chunk pair, it forms the nonconformity score `A = 1 − cosine(q, chunk)`, takes the finite `1 − alpha` quantile of those positive-pair scores, and retains a test chunk when `A <= tau` (equivalently when cosine clears a global lower-alpha threshold). In this shared Jina setup it is exactly the same selector as the older `CCE-style` positive-cosine proxy, because both pool precisely the same positive scores; it is a pointwise chunk rule rather than a per-query all-support rule.",
        "",
        "**CONFLARE source-question adaptation.** The released method creates one calibration record per generated source-question pair and filters retrieval results by a strict cosine-distance percentile. The matched adapter instead takes one highest-scoring labelled support for each benchmark calibration question, computes its cosine distance, sets the `1 − alpha` distance percentile, and keeps test chunks with distance strictly below it. Thus it keeps CONFLARE's one-record-per-question filter but does not run its question-generation, Chroma, or answer-generation pipeline.",
        "",
        "**TRAQ retrieval Bonferroni adaptation.** TRAQ assigns part of a total error budget to retrieval and another part to answer-set uncertainty. The row here sets total `alpha=.10`, assigns `alpha_R=.05` to retrieval, takes one best true-retrieval Jina score per calibration question, and keeps every test candidate whose score is at least the lower `alpha_R` quantile. It measures only TRAQ's retrieval component; generated answer samples, semantic clustering, answer conformity, and end-to-end TRAQ answer-set coverage are not part of this table.",
        "",
        "**Query-level cosine conformal.** Each query's 30 cosine scores are z-normalised within that query. For every retrievable calibration query, the critical score is the minimum z-score among all of its labelled supports, so retaining above the calibrated threshold targets the event that every available support survives. The finite `1 − alpha` quantile of these query-level critical scores becomes one global z-threshold; a deterministic Top-1 fallback prevents an empty context. This is the main all-support query-level conformal baseline at `alpha=.10`.",
        "",
        "**Query-level cosine + internal Qwen signals.** Every candidate is also scored by Qwen-7B's direct Yes-versus-No LM-head logit and by a logistic relevance probe on a selected hidden-state layer. The probe is fit only on the disjoint probe-training portion of the 1,000 parent calibration qids; the cosine, LM-head, and hidden-probe scores are fused with weights selected by grouped out-of-fold ranking quality, then a separate disjoint calibration portion supplies the query-level all-support conformal threshold at `alpha=.10`. This is why the internal rows use roughly 500 probe-training and 500 conformal-calibration qids rather than the full 1,000 solely for calibration.",
        "",
        "**Benjamini–Hochberg (BH) cosine.** Every candidate receives a conformal p-value by comparing its cosine score with a false-evidence reference bank. BH orders the p-values and accepts the largest prefix satisfying `p_(i) <= i alpha / m`, where `m` is the query's candidate count. The reported rows then keep at most the configured context cap in retrieval-rank order; that additional cap is experimentally useful but is not covered by the ordinary BH guarantee.",
        "",
        "**Benjamini–Yekutieli (BY) cosine.** BY uses the same candidate p-values but replaces BH's threshold with the more conservative `p_(i) <= i alpha / (m H_m)`, where `H_m` is the harmonic number. It is designed for arbitrary p-value dependence, so it often rejects far fewer chunks; unlike query-level cosine, it has no Top-1 fallback and can return an empty context. The alpha sweep shows the resulting precision–recall and empty-context trade-off.",
        "",
        "**Alpha-free query-level operating points.** These non-conformal ablations select a z-score threshold from the 1,000 calibration qids to maximise either micro evidence F1 or all-support retention subject to a ten-chunk mean budget, then freeze it for test inference. They contain no alpha at test time, but consequently make no finite-sample conformal coverage claim.",
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
            "- [Fidelity audit for CCE / CONFLARE / TRAQ](FIDELITY_AUDIT.md)",
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
