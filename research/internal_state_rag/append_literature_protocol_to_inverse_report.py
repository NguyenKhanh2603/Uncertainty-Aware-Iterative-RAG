"""Append literature-protocol rows to the legacy inverse 1,000/100 report.

The report keeps its original table and rows.  This program only owns the
marker-bounded rows labelled ``literature protocol`` and the metric-definition
section, making it safe to re-run after resumable downstream evaluation.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from research.internal_state_rag.run_all_datasets_cosine_six_methods import CONFIGS, load_dataset
from research.internal_state_rag.run_hotpotqa_cosine_six_methods import selection_summary
from research.internal_state_rag.run_literature_protocol_1000cal import (
    DISPLAY,
    METHODS,
    build_masks,
)


DATASETS = ("hotpotqa", "mmqa", "tatqa", "webqa")
START = "<!-- literature-protocol-1000cal-start -->"
END = "<!-- literature-protocol-1000cal-end -->"
METRIC_START = "<!-- metric-definitions-start -->"
METRIC_END = "<!-- metric-definitions-end -->"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def downstream_for(source: Path, dataset: str) -> dict[str, dict[str, float]]:
    summary = source / f"{dataset}_summary.json"
    if not summary.is_file():
        return {}
    return read_json(summary).get("downstream", {})


def format_row(name: str, selection: dict[str, float], downstream: dict[str, float] | None) -> str:
    tail = (
        f"{downstream['em']:.3f} | {downstream['f1']:.3f} | "
        f"{downstream['numerical_accuracy']:.3f}"
        if downstream
        else "pending | pending | pending"
    )
    return (
        f"| {name}† | {selection['mean_chunks']:.2f} | {selection['precision']:.1%} | "
        f"{selection['support_recall']:.1%} | {selection['empty_rate']:.1%} | "
        f"{selection['query_any_support']:.1%} | {selection['query_all_support']:.1%} | {tail} |"
    )


def metric_section() -> str:
    return "\n".join(
        [
            METRIC_START,
            "## Metric definitions",
            "",
            "All selection metrics use the frozen Jina Top-30 candidate pool. A `support` is a dataset-labelled evidence chunk inside that pool.",
            "",
            "- **Chunks:** mean number of retained candidates per test query.",
            "- **Precision:** micro precision, `retained support chunks / all retained chunks`.",
            "- **Recall:** micro support recall, `retained support chunks / all labelled support chunks in Top-30`. It cannot recover evidence absent from Top-30.",
            "- **Empty:** fraction of queries for which a selector retains no chunk. Fixed Top-K and query-level cosine use their documented non-empty policies; pointwise selectors may be empty.",
            "- **Any support:** legacy retrieval-pool coverage: at least one support is retained for a query with labelled Top-30 support. A query with no labelled support in Top-30 is counted as covered because no available support can be lost. The retrieval ceiling for each dataset is available in its summary JSON.",
            "- **All support:** fraction of all test queries whose labelled Top-30 support set is a subset of the retained context; a query with no labelled Top-30 support is likewise counted as covered. This is why all-support should be read alongside Recall and the retrieval ceiling.",
            "- **EM / F1 / Numeric:** mean Exact Match, token-level F1, and numerical accuracy of the same deterministic Qwen direct-answer evaluation over the 100 held-out qids. These are downstream diagnostics, not conformal coverage guarantees.",
            "",
            "Rows marked **†** are the separately run literature-protocol records: they retain the same qids and Top-30 candidates but use CCE's positive-pair unit, CONFLARE's one-relevant-record unit, and TRAQ's Bonferroni retrieval unit. TRAQ's row is retrieval-only, not its full semantic answer-prediction-set system.",
            METRIC_END,
        ]
    )


def replace_bounded(text: str, start: str, end: str, replacement: str) -> str:
    pattern = re.escape(start) + r".*?" + re.escape(end) + r"\n?"
    return re.sub(pattern, replacement + "\n", text, flags=re.DOTALL)


def insert_metric_section(text: str) -> str:
    section = metric_section()
    if METRIC_START in text:
        return replace_bounded(text, METRIC_START, METRIC_END, section)
    anchor = "## hotpotqa"
    position = text.find(anchor)
    if position < 0:
        raise RuntimeError("legacy report lacks the HotpotQA table heading")
    return text[:position] + section + "\n\n" + text[position:]


def insert_rows(text: str, dataset: str, rows: list[str]) -> str:
    heading = f"## {dataset}"
    start = text.find(heading)
    if start < 0:
        raise RuntimeError(f"legacy report lacks {dataset} table")
    end = text.find("\n## ", start + len(heading))
    if end < 0:
        end = len(text)
    section = text[start:end]
    block = "\n".join([START, *rows, END])
    if START in section:
        section = replace_bounded(section, START, END, block)
    else:
        # The existing query-level cosine row is the last non-BY row.  Insert
        # after it so the new entries remain in the same selection table.
        anchor = "| query_level_cosine_alpha_0.10 |"
        row_end = section.find("\n", section.find(anchor))
        if row_end < 0 or anchor not in section:
            raise RuntimeError(f"{dataset}: cannot find query-level row anchor")
        section = section[: row_end + 1] + block + "\n" + section[row_end + 1 :]
    return text[:start] + section + text[end:]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--legacy-report",
        type=Path,
        default=Path(
            "research/internal_state_rag/results/"
            "all_datasets_cosine_six_methods_1000cal_100test_2026_09_26/REPORT.md"
        ),
    )
    parser.add_argument(
        "--plan-root",
        type=Path,
        default=Path(
            "research/internal_state_rag/results/"
            "all_datasets_cosine_six_methods_1000cal_100test_2026_09_26/splits"
        ),
    )
    parser.add_argument(
        "--literature-output",
        type=Path,
        default=Path(
            "research/internal_state_rag/results/"
            "literature_protocol_1000cal_100test_2026_09_26"
        ),
    )
    args = parser.parse_args()

    text = insert_metric_section(args.legacy_report.read_text(encoding="utf-8"))
    audit: dict[str, Any] = {"datasets": {}}
    for dataset in DATASETS:
        calibration_plan = args.plan_root / dataset / "calibration_manifest.json"
        test_plan = args.plan_root / dataset / "test_manifest.json"
        calibration_qids, calibration, test_qids, test = load_dataset(
            CONFIGS[dataset], calibration_plan=calibration_plan, test_plan=test_plan
        )
        if len(calibration_qids) != 1000 or len(test_qids) != 100:
            raise RuntimeError(f"{dataset}: expected inverse 1000/100 plans")
        masks, _thresholds = build_masks(calibration, test, 0.10)
        selection = selection_summary(test, masks)
        downstream = downstream_for(args.literature_output, dataset)
        rows = [format_row(DISPLAY[method], selection[method], downstream.get(method)) for method in METHODS]
        text = insert_rows(text, dataset, rows)
        audit["datasets"][dataset] = {
            "calibration_qids": len(calibration_qids),
            "test_qids": len(test_qids),
            "downstream_complete": len(downstream) == len(METHODS),
        }
    args.legacy_report.write_text(text, encoding="utf-8")
    audit_path = args.literature_output / "LEGACY_TABLE_MERGE_AUDIT.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
