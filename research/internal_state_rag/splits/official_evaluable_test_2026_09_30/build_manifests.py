#!/usr/bin/env python3
"""Build full public-label evaluation qid manifests from pinned official sources."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq
import requests
from huggingface_hub import hf_hub_download

HOTPOT_REPO = "hotpotqa/hotpot_qa"
HOTPOT_REVISION = "1908d6afbbead072334abe2965f91bd2709910ab"
HOTPOT_FILE = "distractor/validation-00000-of-00001.parquet"

MMQA_REVISION = "4dd14328c6d02a4daa357cc6032915a0b14602e3"
MMQA_URL = (
    "https://raw.githubusercontent.com/allenai/multimodalqa/"
    f"{MMQA_REVISION}/dataset/MMQA_dev.jsonl.gz"
)

TATQA_REVISION = "870accc41953dcde885aabeb963d94aabdc0fbc3"
TATQA_URL = (
    "https://raw.githubusercontent.com/NExTplusplus/TAT-QA/"
    f"{TATQA_REVISION}/dataset_raw/tatqa_dataset_dev.json"
)

WEBQA_REPO = "Ancci/webqa_large"
WEBQA_REVISION = "5d2222ebafe1443d2e31f396a5eafb8803394146"
WEBQA_FILE = "test.json"

EXPECTED_COUNTS = {
    "hotpotqa": 7405,
    "mmqa": 2441,
    "tatqa": 1668,
    "webqa": 4966,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_qids(qids: Iterable[str]) -> str:
    payload = "".join(f"{qid}\n" for qid in qids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def download(url: str, destination: Path) -> Path:
    if destination.is_file() and destination.stat().st_size:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with temporary.open("wb") as handle:
            for block in response.iter_content(chunk_size=1024 * 1024):
                if block:
                    handle.write(block)
    temporary.replace(destination)
    return destination


def jsonl(path: Path, *, compressed: bool = False) -> Iterable[dict[str, Any]]:
    opener = gzip.open if compressed else Path.open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def validate(dataset: str, qids: list[str]) -> None:
    expected = EXPECTED_COUNTS[dataset]
    if len(qids) != expected:
        raise RuntimeError(f"{dataset}: expected {expected} qids, found {len(qids)}")
    if len(set(qids)) != len(qids):
        raise RuntimeError(f"{dataset}: duplicate qids detected")
    if any(not qid for qid in qids):
        raise RuntimeError(f"{dataset}: empty qid detected")


def write_manifest(
    output_root: Path,
    dataset: str,
    qids: list[str],
    *,
    source_split: str,
    source: dict[str, Any],
) -> dict[str, Any]:
    validate(dataset, qids)
    dataset_dir = output_root / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": dataset,
        "role": "test",
        "protocol_meaning": "full_official_split_with_public_answer_and_support_labels",
        "source_split": source_split,
        "order": "official_source_order",
        "count": len(qids),
        "qids_sha256": sha256_qids(qids),
        "source": source,
        "plan": qids,
    }
    manifest_path = dataset_dir / "test_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    (dataset_dir / "test_qids.txt").write_text("".join(f"{qid}\n" for qid in qids))
    return {
        "count": len(qids),
        "source_split": source_split,
        "qids_sha256": manifest["qids_sha256"],
        "test_manifest": str(manifest_path.relative_to(output_root)),
        "test_qids": str((dataset_dir / "test_qids.txt").relative_to(output_root)),
    }


def build(output_root: Path, cache_dir: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    hotpot_path = Path(
        hf_hub_download(
            repo_id=HOTPOT_REPO,
            repo_type="dataset",
            revision=HOTPOT_REVISION,
            filename=HOTPOT_FILE,
            cache_dir=cache_dir / "huggingface",
        )
    )
    hotpot_table = pq.read_table(hotpot_path, columns=["id"])
    hotpot_qids = [str(value) for value in hotpot_table.column("id").to_pylist()]

    mmqa_path = download(MMQA_URL, cache_dir / "MMQA_dev.jsonl.gz")
    mmqa_qids = [str(row["qid"]) for row in jsonl(mmqa_path, compressed=True)]

    tatqa_path = download(TATQA_URL, cache_dir / "tatqa_dataset_dev.json")
    tatqa_documents = json.loads(tatqa_path.read_text(encoding="utf-8"))
    tatqa_qids = [
        str(question["uid"])
        for document in tatqa_documents
        for question in document.get("questions", [])
    ]

    webqa_path = Path(
        hf_hub_download(
            repo_id=WEBQA_REPO,
            repo_type="dataset",
            revision=WEBQA_REVISION,
            filename=WEBQA_FILE,
            cache_dir=cache_dir / "huggingface",
        )
    )
    webqa_qids = [str(row["qid"]) for row in jsonl(webqa_path)]

    datasets = {
        "hotpotqa": write_manifest(
            output_root,
            "hotpotqa",
            hotpot_qids,
            source_split="distractor/validation",
            source={
                "repository": HOTPOT_REPO,
                "revision": HOTPOT_REVISION,
                "file": HOTPOT_FILE,
                "file_sha256": sha256_file(hotpot_path),
            },
        ),
        "mmqa": write_manifest(
            output_root,
            "mmqa",
            mmqa_qids,
            source_split="dev",
            source={
                "repository": "allenai/multimodalqa",
                "revision": MMQA_REVISION,
                "file": "dataset/MMQA_dev.jsonl.gz",
                "file_sha256": sha256_file(mmqa_path),
            },
        ),
        "tatqa": write_manifest(
            output_root,
            "tatqa",
            tatqa_qids,
            source_split="dev",
            source={
                "repository": "NExTplusplus/TAT-QA",
                "revision": TATQA_REVISION,
                "file": "dataset_raw/tatqa_dataset_dev.json",
                "file_sha256": sha256_file(tatqa_path),
            },
        ),
        "webqa": write_manifest(
            output_root,
            "webqa",
            webqa_qids,
            source_split="validation",
            source={
                "repository": WEBQA_REPO,
                "revision": WEBQA_REVISION,
                "file": WEBQA_FILE,
                "note": (
                    "Pinned redistribution names the official WebQA validation split test.json."
                ),
                "file_sha256": sha256_file(webqa_path),
            },
        ),
    }
    summary = {
        "protocol": "full_official_splits_with_public_answer_and_support_labels",
        "created_date": "2026-09-30",
        "total_queries": sum(row["count"] for row in datasets.values()),
        "datasets": datasets,
    }
    (output_root / "SPLIT_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache/official_evaluable_test_sources"),
    )
    args = parser.parse_args()
    summary = build(args.output_dir, args.cache_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
