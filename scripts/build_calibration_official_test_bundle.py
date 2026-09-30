#!/usr/bin/env python3
"""Merge frozen 1k calibration queries with complete official labelled test splits.

The merged corpus gives calibration and test rows one identical corpus revision,
retriever fingerprint, and preprocessing fingerprint.  This is required by the
candidate-wise conformal bank validator and makes the resulting cosine logs
portable to the Colab BY runner without recomputing embeddings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

DATASETS = ("hotpotqa", "mmqa", "tatqa", "webqa")


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_manifest(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    qids = [str(value) for value in payload["plan"]]
    if len(qids) != int(payload.get("count", len(qids))) or len(qids) != len(set(qids)):
        raise ValueError(f"Invalid manifest: {path}")
    return qids


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    temporary.replace(path)
    return count


def source_root(dataset: str, calibration_root: Path, webqa_calibration_root: Path) -> Path:
    return webqa_calibration_root if dataset == "webqa" else calibration_root


def canonical_corpus_row(row: dict[str, Any], root: Path) -> dict[str, Any]:
    result = dict(row)
    if str(result.get("modality", "text")).lower() == "image":
        content = str(result["content"])
        if not content.startswith(("http://", "https://")):
            # ``absolute`` avoids one metadata lookup per image.  Resolving
            # 100k WebQA paths on the NFS workspace otherwise takes minutes.
            result["content"] = str((root / content).absolute())
    return result


def add_corpus_rows(
    destination: dict[str, dict[str, Any]],
    path: Path,
    root: Path,
    needed: set[str],
) -> None:
    for raw in iter_jsonl(path):
        chunk_id = str(raw["id"])
        if chunk_id not in needed:
            continue
        row = canonical_corpus_row(raw, root)
        previous = destination.get(chunk_id)
        if previous is None:
            destination[chunk_id] = row
            continue
        # The same official image can occur in train and dev archives at two
        # local paths.  Its ID, modality, source ID, and caption must still agree.
        comparable = ("id", "source_doc_id", "modality", "caption")
        if any(str(previous.get(key, "")) != str(row.get(key, "")) for key in comparable):
            raise ValueError(f"Conflicting corpus metadata for {chunk_id}")
        if str(row.get("modality", "text")) != "image" and previous != row:
            raise ValueError(f"Conflicting corpus content for {chunk_id}")


def selected_questions(path: Path, qids: list[str], role: str) -> list[dict[str, Any]]:
    wanted = set(qids)
    rows: dict[str, dict[str, Any]] = {}
    for raw in iter_jsonl(path):
        qid = str(raw["qid"])
        if qid not in wanted:
            continue
        row = dict(raw)
        metadata = dict(row.get("metadata") or {})
        metadata["split_role"] = role
        metadata["precomputed_cosine_role"] = role
        row["metadata"] = metadata
        rows[qid] = row
    missing = [qid for qid in qids if qid not in rows]
    if missing:
        raise ValueError(f"{path} misses {len(missing)} selected qids; first={missing[0]}")
    return [rows[qid] for qid in qids]


def build_dataset(args: argparse.Namespace, dataset: str) -> dict[str, Any]:
    calibration_source = source_root(
        dataset, args.calibration_bundle, args.webqa_calibration_bundle
    )
    calibration_manifest = args.calibration_manifests / dataset / "calibration_manifest.json"
    test_manifest = args.test_manifests / dataset / "test_manifest.json"
    calibration_qids = load_manifest(calibration_manifest)
    test_qids = load_manifest(test_manifest)
    overlap = set(calibration_qids).intersection(test_qids)
    if overlap:
        raise ValueError(f"{dataset}: calibration/test overlap: {sorted(overlap)[:3]}")

    calibration_questions = selected_questions(
        calibration_source / dataset / "questions.jsonl", calibration_qids, "calibration"
    )
    test_questions = selected_questions(
        args.test_bundle / dataset / "questions.jsonl", test_qids, "test"
    )
    questions = calibration_questions + test_questions
    candidate_ids = {
        str(chunk_id)
        for row in questions
        for chunk_id in row.get("candidate_ids", [])
    }
    corpus: dict[str, dict[str, Any]] = {}
    add_corpus_rows(
        corpus,
        calibration_source / dataset / "corpus.jsonl",
        calibration_source,
        candidate_ids,
    )
    add_corpus_rows(
        corpus,
        args.test_bundle / dataset / "corpus.jsonl",
        args.test_bundle,
        candidate_ids,
    )
    missing_chunks = candidate_ids.difference(corpus)
    if missing_chunks:
        raise ValueError(f"{dataset}: missing {len(missing_chunks)} candidate chunks")

    destination = args.output_dir / dataset
    question_path = destination / "questions.jsonl"
    corpus_path = destination / "corpus.jsonl"
    atomic_jsonl(question_path, questions)
    atomic_jsonl(corpus_path, (corpus[key] for key in sorted(corpus)))
    return {
        "calibration_queries": len(calibration_qids),
        "test_queries": len(test_qids),
        "total_queries": len(questions),
        "corpus_chunks": len(corpus),
        "calibration_test_overlap": 0,
        "questions_sha256": sha256(question_path),
        "corpus_sha256": sha256(corpus_path),
        "calibration_manifest": str(calibration_manifest),
        "test_manifest": str(test_manifest),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibration-bundle", type=Path, default=Path("data/zip_calibration_split_20_09")
    )
    parser.add_argument(
        "--webqa-calibration-bundle",
        type=Path,
        default=Path("data/zip_calibration_split_20_09_webqa"),
    )
    parser.add_argument(
        "--calibration-manifests",
        type=Path,
        default=Path(
            "research/internal_state_rag/results/"
            "full_context_selection_splits_khanh_27_09_2026_09_27/splits"
        ),
    )
    parser.add_argument(
        "--test-bundle", type=Path, default=Path("data/official_evaluable_test_2026_09_30")
    )
    parser.add_argument(
        "--test-manifests",
        type=Path,
        default=Path("research/internal_state_rag/splits/official_evaluable_test_2026_09_30"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/precomputed_cosine_cal1000_official_test_2026_09_30"),
    )
    parser.add_argument("--datasets", default=",".join(DATASETS))
    args = parser.parse_args()
    datasets = [value.strip() for value in args.datasets.split(",") if value.strip()]
    if not datasets or set(datasets).difference(DATASETS):
        parser.error(f"--datasets must be a subset of {DATASETS}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    results = dict(previous.get("datasets", {}))
    for dataset in datasets:
        results[dataset] = build_dataset(args, dataset)
    payload = {
        "schema_version": 1,
        "protocol": "calibration_1000_plus_official_evaluable_full_test",
        "datasets": results,
    }
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
