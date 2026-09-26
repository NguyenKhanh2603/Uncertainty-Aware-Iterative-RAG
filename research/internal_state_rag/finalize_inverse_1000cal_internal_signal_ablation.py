"""Verify and index a completed 1,000-calibration internal-signal ablation.

This deliberately validates only compact, reviewable output artifacts.  The
large extracted Qwen feature caches are reproducible from the manifests and
the pinned model, so they are not needed to review the result or commit it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


DATASETS = ("hotpotqa", "mmqa", "tatqa", "webqa")
METHODS = {
    "query_level_cosine_matched_alpha_0.10",
    "query_level_lm_head_alpha_0.10",
    "query_level_hidden_probe_alpha_0.10",
    "query_level_lm_head_hidden_alpha_0.10",
    "query_level_cosine_lm_head_hidden_alpha_0.10",
}
MASKS = {
    "mask_jina_v4_cosine_alpha_0.1",
    "mask_lm_head_only_alpha_0.1",
    "mask_hidden_probe_only_alpha_0.1",
    "mask_internal_lm_hidden_alpha_0.1",
    "mask_cosine_internal_alpha_0.1",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify(output_dir: Path) -> dict[str, Any]:
    summary = read_json(output_dir / "summary.json")
    require(summary.get("status") == "complete", "summary is not complete")
    reported = summary.get("datasets", {})
    require(set(reported) == set(DATASETS), "summary does not contain exactly four datasets")
    audit = read_json(output_dir / "splits" / "SPLIT_INTEGRITY.json")
    checks: dict[str, Any] = {}
    for dataset in DATASETS:
        split_dir = output_dir / "splits" / dataset
        probe = read_json(split_dir / "probe_train_manifest.json")
        calibration = read_json(split_dir / "conformal_calibration_manifest.json")
        test = read_json(split_dir / "test_manifest.json")
        probe_qids = [str(qid) for qid in probe["plan"]]
        calibration_qids = [str(qid) for qid in calibration["plan"]]
        test_qids = [str(qid) for qid in test["plan"]]
        require(len(test_qids) == 100, f"{dataset}: test set is not 100 qids")
        require(not (set(probe_qids) & set(calibration_qids)), f"{dataset}: probe/calibration overlap")
        require(not (set(probe_qids) & set(test_qids)), f"{dataset}: probe/test overlap")
        require(not (set(calibration_qids) & set(test_qids)), f"{dataset}: calibration/test overlap")
        require(
            all(value == 0 for value in audit["datasets"][dataset]["overlaps"].values()),
            f"{dataset}: split-integrity report has overlap",
        )
        result = reported[dataset]
        require(result.get("status") == "complete", f"{dataset}: downstream status is not complete")
        require(set(result.get("selection", {})) == METHODS, f"{dataset}: missing selection method")
        require(set(result.get("downstream", {})) == METHODS, f"{dataset}: missing downstream method")
        analysis = read_json(output_dir / "analysis" / dataset / "fusion_analysis.json")
        require(analysis.get("status") == "complete", f"{dataset}: analysis is not complete")
        artifact = output_dir / "analysis" / dataset / "fusion_predictions.npz"
        with np.load(artifact) as predictions:
            require(
                predictions["qids"].astype(str).tolist() == test_qids,
                f"{dataset}: mask qids differ from held-out plan",
            )
            require(MASKS.issubset(predictions.files), f"{dataset}: missing required masks")
        records = read_jsonl(output_dir / f"{dataset}_downstream_predictions.jsonl")
        record_qids = [str(row["qid"]) for row in records]
        require(record_qids == test_qids, f"{dataset}: downstream qids differ from held-out plan")
        for row in records:
            require(set(row.get("predictions", {})) == METHODS, f"{dataset}: incomplete predictions")
            require(set(row.get("metrics", {})) == METHODS, f"{dataset}: incomplete metrics")
        checks[dataset] = {
            "probe_train_queries": len(probe_qids),
            "conformal_calibration_queries": len(calibration_qids),
            "heldout_test_queries": len(test_qids),
            "downstream_records": len(records),
            "selected_layer": analysis["probe"]["selected_layer"],
            "selected_c": analysis["probe"]["selected_c"],
        }
    return {"status": "complete", "datasets": checks}


def write_readme(output_dir: Path, audit: dict[str, Any]) -> None:
    lines = [
        "# Audited Qwen-7B internal-signal ablation",
        "",
        "This artifact evaluates a frozen Qwen2-VL-7B relevance signal against query-level cosine pruning. Each dataset begins with 1,000 parent calibration qids, divided into a disjoint probe-training role and conformal-calibration role; the same 100 held-out test qids are used by all five methods.",
        "",
        "The full selection and Qwen downstream table is in [REPORT.md](REPORT.md).",
        "",
        "## Reproduction and audit",
        "",
        "- [Pipeline runner](../../../run_inverse_1000cal_internal_signal_ablation.sh)",
        "- [Split/materialization code](../../../prepare_inverse_1000cal_internal_signal_plans.py)",
        "- [Internal-score analysis](../../../analyze_qwen2vl_jina_ablation.py)",
        "- [Selection and downstream evaluator](../../../run_inverse_1000cal_internal_signal_ablation.py)",
        "- [Feature extractor](../../../run_qwen2vl_pairwise_features.py)",
        "- [Method-positioning note](../../../QUERY_LEVEL_CONFORMAL_POSITIONING_CCE_TRAQ_CONFLARE.md)",
        "- [Programmatic completion audit](FINALIZATION_AUDIT.json)",
        "",
        "| Dataset | Probe train | Conformal calibration | Held-out test | Downstream records |",
        "|---|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        row = audit["datasets"][dataset]
        lines.append(
            f"| {dataset} | {row['probe_train_queries']} | {row['conformal_calibration_queries']} | "
            f"{row['heldout_test_queries']} | {row['downstream_records']} |"
        )
    lines.extend(["", "## Exact inputs and per-query artifacts", ""])
    for dataset in DATASETS:
        lines.append(
            f"- **{dataset}:** "
            f"[probe qids](splits/{dataset}/probe_train_manifest.json), "
            f"[conformal-calibration qids](splits/{dataset}/conformal_calibration_manifest.json), "
            f"[test qids](splits/{dataset}/test_manifest.json), "
            f"[analysis](analysis/{dataset}/fusion_analysis.json), and "
            f"[downstream predictions]({dataset}_downstream_predictions.jsonl)."
        )
    lines.extend(
        [
            "",
            "The large Qwen feature caches are intentionally not versioned here. Their exact qid manifests, frozen Top-30 candidate ordering, extractor code, model revision, and analysis artifacts above are sufficient to reproduce the masks.",
            "",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    audit = verify(args.output_dir)
    (args.output_dir / "FINALIZATION_AUDIT.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    write_readme(args.output_dir, audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
