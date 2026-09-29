#!/usr/bin/env python3
"""Append Colab latency and approximate-TFLOPs columns to REPORT_TEMPLATE.md."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DATASETS = ("hotpotqa", "mmqa", "tatqa", "webqa")
METHOD_MARKERS = {
    "CCE Conformal-Embedding, Jina adaptation": "cce_conformal_embedding_jina_alpha_0.10",
    "CONFLARE source-question, Jina adaptation": "conflare_source_question_jina_alpha_0.10",
    "TRAQ retrieval Bonferroni, Jina adaptation": "traq_retrieval_bonferroni_jina_alpha_0.10",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--template", type=Path, default=HERE / "REPORT_TEMPLATE.md")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_metrics(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    metrics: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        dataset = str(row["Dataset"]).lower()
        if dataset not in DATASETS:
            continue
        key = (dataset, str(row["Method"]))
        if key in metrics:
            raise ValueError(f"Duplicate CSV row: {key}")
        if int(row["Queries"]) != 100:
            raise ValueError(f"Incomplete metric row {key}: {row['Queries']} queries")
        metrics[key] = row
    expected = {(dataset, method) for dataset in DATASETS for method in METHOD_MARKERS.values()}
    missing = expected.difference(metrics)
    if missing:
        raise ValueError(
            f"CSV is missing {len(missing)} completed rows; first: {sorted(missing)[0]}"
        )
    return metrics


def main() -> None:
    args = parse_args()
    metrics = load_metrics(args.csv)
    source = args.template.read_text(encoding="utf-8")
    if "| Latency (s) | Approx. TFLOPs |" in source:
        raise ValueError("Template already contains resource columns")

    output: list[str] = []
    dataset = ""
    in_result_table = False
    metric_note_added = False
    for line in source.splitlines():
        if line.startswith("## "):
            dataset = next((value for value in DATASETS if line.startswith(f"## {value} ")), "")
            in_result_table = False
        if line.startswith("Chunks is mean retained chunks/query.") and not metric_note_added:
            line += (
                " Latency is CUDA-synchronized `model.generate` wall time per query. "
                "Approx. TFLOPs uses `2 × parameter_count × returned_sequence_length`; "
                "it is a reporting proxy rather than profiler-measured FLOPs."
            )
            metric_note_added = True
        if dataset and line.startswith("| Family | Method |"):
            output.append(line + " Latency (s) | Approx. TFLOPs |")
            in_result_table = True
            continue
        if in_result_table and line.startswith("|---|---|"):
            output.append(line + "---:|---:|")
            continue
        if in_result_table and line.startswith("|"):
            method = next(
                (value for marker, value in METHOD_MARKERS.items() if marker in line), None
            )
            if method is None:
                output.append(line + " pending | pending |")
            else:
                row = metrics[(dataset, method)]
                output.append(
                    line
                    + f" {float(row['Latency (s)']):.3f} | "
                    + f"{float(row['Approx. TFLOPs']):.3f} |"
                )
            continue
        if in_result_table and not line.startswith("|"):
            in_result_table = False
        output.append(line)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(output) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
