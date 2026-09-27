"""Render the full comparison table for the ``splits_khanh_27_09`` rerun.

Only three literature retrieval selectors are intentionally evaluated in this
artifact: CCE Conformal-Embedding, CONFLARE source-question, and the TRAQ
retrieval Bonferroni component.  All other full-table rows remain explicitly
``pending`` so a reader cannot mistake results copied from a different split
for measurements on this one.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from research.internal_state_rag.build_unified_1000cal_comparison import DATASETS, ROWS


EVALUATED_METHODS = frozenset(
    {
        "cce_conformal_embedding_jina_alpha_0.10",
        "conflare_source_question_jina_alpha_0.10",
        "traq_retrieval_bonferroni_jina_alpha_0.10",
    }
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def split_audit(split_root: Path) -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    for dataset in DATASETS:
        calibration = read_json(split_root / dataset / "calibration_manifest.json")["plan"]
        test = read_json(split_root / dataset / "test_manifest.json")["plan"]
        overlap = sorted(set(calibration).intersection(test))
        if len(calibration) != 1000 or len(test) != 100 or overlap:
            raise RuntimeError(
                f"{dataset}: require 1,000 calibration, 100 test, and no overlap; "
                f"got {len(calibration)}, {len(test)}, overlap={len(overlap)}"
            )
        datasets[dataset] = {
            "calibration_queries": len(calibration),
            "test_queries": len(test),
            "overlap": len(overlap),
            "calibration_manifest": str(
                Path("splits") / dataset / "calibration_manifest.json"
            ),
            "test_manifest": str(Path("splits") / dataset / "test_manifest.json"),
        }
    return datasets


def rendered_metrics(result: dict[str, Any]) -> str:
    selection = result["selection"]
    downstream = result["downstream"]
    return (
        f"{float(selection['mean_chunks']):.2f} | "
        f"{float(selection['precision']):.1%} | "
        f"{float(selection['support_recall']):.1%} | "
        f"{float(selection['empty_rate']):.1%} | "
        f"{float(selection['query_any_support_conditional']):.1%} | "
        f"{float(selection['query_all_support_conditional']):.1%} | "
        f"{float(downstream['em']):.3f} | {float(downstream['f1']):.3f} | "
        f"{float(downstream['numerical_accuracy']):.3f}"
    )


def write_report(
    output: Path,
    *,
    results: dict[str, Any],
    archive: Path,
    split_datasets: dict[str, Any],
) -> None:
    pool_summary = "; ".join(
        f"{dataset}: {results['datasets'][dataset]['candidate_pool']['test_min']}–"
        f"{results['datasets'][dataset]['candidate_pool']['test_max']} candidates/query "
        f"(mean {results['datasets'][dataset]['candidate_pool']['test_mean']:.2f})"
        for dataset in DATASETS
    )
    lines = [
        "# Full context-selection comparison — `splits_khanh_27_09`",
        "",
        "This is a new, split-specific rerun. It uses the qid manifests from "
        "`splits_khanh_27_09.zip` (SHA-256 "
        f"`{sha256(archive)}`): 1,000 calibration qids and 100 held-out test "
        "qids per dataset, with zero qid overlap. Every evaluated selector "
        "receives the frozen Jina-v4 dataset-provided candidate pool, capped at "
        f"Top-L=30, and is scored by the shared greedy Qwen2-VL-7B direct-answer "
        f"diagnostic. Test-pool sizes are ragged: {pool_summary}.",
        "",
        "Only the three literature retrieval adaptations below were run on this "
        "split: CCE Conformal-Embedding, CONFLARE source-question, and TRAQ "
        "retrieval Bonferroni. Every other row is deliberately `pending`; no "
        "value from an earlier split is copied into this report.",
        "",
        "## Metrics",
        "",
        "Chunks is mean retained chunks/query. Precision and Recall are micro "
        "support metrics in each frozen candidate pool. Empty is the fraction of queries "
        "that retained no chunk. Any/All are conditional support-retention rates "
        "among queries with at least one labelled support in Top-30. EM, token F1, "
        "and Numeric are Qwen direct-answer measurements on the 100 held-out "
        "queries; they are not conformal guarantees.",
        "",
    ]
    for dataset in DATASETS:
        state = results["datasets"].get(dataset)
        if state is None or state.get("status") != "complete":
            raise RuntimeError(f"{dataset}: literature run is incomplete")
        if state.get("calibration_queries") != 1000 or state.get("test_queries") != 100:
            raise RuntimeError(f"{dataset}: unexpected calibration/test counts")
        lines.extend(
            [
                f"## {dataset} (calibration=1,000; test=100; split=`splits_khanh_27_09`)",
                "",
                "| Family | Method | Chunks | Precision | Recall | Empty | Any support | All support | EM | F1 | Numeric |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for spec in ROWS:
            if spec.method in EVALUATED_METHODS:
                if spec.source != "literature":
                    raise RuntimeError(f"Unexpected non-literature row: {spec.method}")
                method_result = state["selection"].get(spec.method)
                qa_result = state.get("downstream", {}).get(spec.method)
                if method_result is None or qa_result is None:
                    raise RuntimeError(f"{dataset}: missing result for {spec.method}")
                values = rendered_metrics({"selection": method_result, "downstream": qa_result})
            else:
                values = "pending | pending | pending | pending | pending | pending | pending | pending | pending"
            lines.append(f"| {spec.family} | {spec.label} | {values} |")
        lines.append("")
    lines.extend(
        [
            "## Split and code provenance",
            "",
            "- Archive supplied for this rerun: `splits_khanh_27_09.zip`; its SHA-256 is recorded above.",
            "- Materialized, exact manifests: [splits](splits), including one calibration and one test manifest per dataset.",
            "- Split audit: [SPLIT_INTEGRITY.json](splits/SPLIT_INTEGRITY.json).",
            "- Literature selector/downstream runner: [run_literature_protocol_1000cal.py](../../run_literature_protocol_1000cal.py).",
            "- Full-table renderer: [build_splits_khanh_27_09_literature_report.py](../../build_splits_khanh_27_09_literature_report.py).",
            "- Shared frozen-candidate configuration and data mapping: [run_all_datasets_cosine_six_methods.py](../../run_all_datasets_cosine_six_methods.py).",
            "- Raw completed selection and QA summary: [summary.json](summary.json).",
            "",
            "The three literature rows are matched Jina adaptations, not full end-to-end replications of the original CCE, CONFLARE, or TRAQ systems. TRAQ reports its retrieval component only; it does not report TRAQ's semantic answer-set coverage procedure.",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path, default=Path("splits_khanh_27_09.zip"))
    args = parser.parse_args()

    summary_path = args.output_dir / "summary.json"
    split_root = args.output_dir / "splits"
    if not args.archive.is_file():
        raise FileNotFoundError(args.archive)
    if not summary_path.is_file() or not split_root.is_dir():
        raise FileNotFoundError("Run the literature selector before rendering this report")
    result = read_json(summary_path)
    if result.get("status") != "complete":
        raise RuntimeError("The literature run is not complete")
    split_datasets = split_audit(split_root)
    (split_root / "SPLIT_INTEGRITY.json").write_text(
        json.dumps(
            {
                "source_archive": str(args.archive),
                "source_archive_sha256": sha256(args.archive),
                "datasets": split_datasets,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_report(
        args.output_dir / "REPORT.md",
        results=result,
        archive=args.archive,
        split_datasets=split_datasets,
    )
    print(f"wrote {args.output_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
