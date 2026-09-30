#!/usr/bin/env python3
"""Export portable per-query cosine scores and package the BY-ready artifacts."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any, Iterable

DATASETS = ("hotpotqa", "mmqa", "tatqa", "webqa")
KEEP_FIELDS = (
    "chunk_id",
    "source_doc_id",
    "modality",
    "rank",
    "cosine_score",
    "support_label",
)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_querywise(source: Path, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    query_count = candidate_count = 0
    role_counts: dict[str, int] = {}
    current_key: tuple[str, str, str] | None = None
    candidates: list[dict[str, Any]] = []
    header: dict[str, Any] = {}

    def flush(handle) -> None:
        nonlocal query_count, candidate_count, candidates
        if current_key is None:
            return
        handle.write(
            json.dumps(
                {**header, "candidates": candidates},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        )
        query_count += 1
        candidate_count += len(candidates)
        role = str(header["split_role"])
        role_counts[role] = role_counts.get(role, 0) + 1
        candidates = []

    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as handle:
        for row in iter_jsonl(source):
            key = (str(row["dataset"]), str(row["qid"]), str(row["split_role"]))
            if current_key is not None and key != current_key:
                flush(handle)
            if key != current_key:
                current_key = key
                header = {
                    "dataset": key[0],
                    "qid": key[1],
                    "split_role": key[2],
                    "source_split": row.get("source_split"),
                    "query_type": row.get("query_type"),
                    "top_l": int(row["top_l"]),
                    "retriever_id": row["retriever_id"],
                    "corpus_revision": row["corpus_revision"],
                    "preprocess_hash": row["preprocess_hash"],
                }
            expected_rank = len(candidates) + 1
            if int(row["rank"]) != expected_rank:
                raise ValueError(f"Non-contiguous ranks for {key}: expected {expected_rank}")
            candidates.append({field: row[field] for field in KEEP_FIELDS})
        flush(handle)
    temporary.replace(destination)
    return {
        "queries": query_count,
        "candidates": candidate_count,
        "queries_by_role": role_counts,
        "bytes": destination.stat().st_size,
        "sha256": sha256(destination),
    }


def copy_file(source: Path, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return {
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": sha256(destination),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-dir", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--selection-dir", type=Path)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--zip", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets: dict[str, Any] = {}
    for dataset in DATASETS:
        source = args.retrieval_dir / f"{dataset}_top30_retrieval.jsonl.gz"
        source_manifest = source.with_name(source.name + ".manifest.json")
        if not source.is_file() or not source_manifest.is_file():
            raise FileNotFoundError(f"Missing completed retrieval log for {dataset}: {source}")
        score_target = args.output_dir / "rowwise" / source.name
        manifest_target = score_target.with_name(score_target.name + ".manifest.json")
        rowwise = copy_file(source, score_target)
        copy_file(source_manifest, manifest_target)
        querywise_path = args.output_dir / "querywise" / f"{dataset}_cosine_by_query.jsonl.gz"
        querywise = write_querywise(source, querywise_path)
        split_dir = args.output_dir / "splits" / dataset
        calibration_manifest = args.split_root / dataset / "calibration_manifest.json"
        test_manifest = args.split_root / dataset / "test_manifest.json"
        copy_file(calibration_manifest, split_dir / calibration_manifest.name)
        copy_file(test_manifest, split_dir / test_manifest.name)
        datasets[dataset] = {
            "rowwise": rowwise,
            "querywise": {"path": str(querywise_path), **querywise},
        }

    bank_target = args.output_dir / "combined_reference_banks.json.gz"
    bank_info = copy_file(args.bank, bank_target)
    if args.selection_dir and args.selection_dir.is_dir():
        for name in (
            "selection_summary.json",
            "selection_query_metrics.csv",
            "selection_decisions.jsonl.gz",
        ):
            source = args.selection_dir / name
            if source.is_file():
                copy_file(source, args.output_dir / "by_selection" / name)

    manifest = {
        "schema_version": 1,
        "protocol": "jina_v4_cosine_cal1000_official_test_full_top30",
        "purpose": "precomputed cosine input for BY, CCE, CONFLARE, and TRAQ",
        "datasets": datasets,
        "reference_bank": bank_info,
    }
    (args.output_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    readme = """# Precomputed Jina-v4 cosine scores — calibration 1,000 + official full test

This portable bundle contains no model weights, corpus text, or images.  It stores the
exact Top-L candidate rows used by the experiment, including each candidate's cosine
score, modality, rank, support label, and pipeline fingerprint.

- `rowwise/`: original retrieval-log schema accepted directly by the conformal scripts;
- `querywise/`: one gzip JSONL record per query with a nested `candidates` list;
- `combined_reference_banks.json.gz`: false-score banks built only from calibration rows;
- `by_selection/`: candidate-wise BY decisions and summary on the official test rows;
- `splits/`: frozen calibration and test qid manifests;
- `MANIFEST.json`: file hashes, sizes, and query counts.

The retriever is `jinaai/jina-embeddings-v4` at its pinned repository revision.  All
methods consume the same cosine rows; running BY from this folder requires no GPU encode.
"""
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")

    args.zip.parent.mkdir(parents=True, exist_ok=True)
    temporary_zip = args.zip.with_suffix(args.zip.suffix + ".tmp")
    with zipfile.ZipFile(temporary_zip, "w", compression=zipfile.ZIP_STORED) as archive:
        for path in sorted(args.output_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(args.output_dir.parent))
    temporary_zip.replace(args.zip)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "zip": str(args.zip),
                "zip_bytes": args.zip.stat().st_size,
                "zip_sha256": sha256(args.zip),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
