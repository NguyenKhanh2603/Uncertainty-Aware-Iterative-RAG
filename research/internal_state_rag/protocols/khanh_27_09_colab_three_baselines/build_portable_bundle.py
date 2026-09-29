#!/usr/bin/env python3
"""Build the minimal 400-query data bundle consumed by run_colab.py."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Iterable

HERE = Path(__file__).resolve().parent
DATASETS = ("hotpotqa", "mmqa", "tatqa", "webqa")
METHODS = (
    "cce_conformal_embedding_jina_alpha_0.10",
    "conflare_source_question_jina_alpha_0.10",
    "traq_retrieval_bonferroni_jina_alpha_0.10",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--webqa-root", type=Path, required=True)
    parser.add_argument("--selection-root", type=Path, default=HERE / "selections")
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def keyed(path: Path, key: str) -> dict[str, dict[str, Any]]:
    return {str(row[key]): row for row in iter_jsonl(path)}


def locate_image(content: str, root: Path) -> Path:
    raw = Path(content)
    for candidate in (raw, root / raw, root.parent / "official_bundle_1000_more" / raw):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(content)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"Output must be absent or empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "name": "khanh_27_09_three_baselines_test_only_v1",
        "datasets": {},
    }

    for dataset in DATASETS:
        source_root = args.webqa_root if dataset == "webqa" else args.data_root
        source_dir = source_root / dataset
        selection_path = args.selection_root / f"{dataset}_downstream_predictions.jsonl"
        selections = list(iter_jsonl(selection_path))
        if len(selections) != 100:
            raise ValueError(f"{dataset}: expected 100 selection rows")
        qids = [str(row["qid"]) for row in selections]
        chunk_ids: set[str] = set()
        for row in selections:
            for method in METHODS:
                chunk_ids.update(str(value) for value in row["contexts"][method]["chunk_ids"])

        all_questions = keyed(source_dir / "questions.jsonl", "qid")
        all_corpus = keyed(source_dir / "corpus.jsonl", "id")
        missing_questions = set(qids).difference(all_questions)
        missing_chunks = chunk_ids.difference(all_corpus)
        if missing_questions or missing_chunks:
            raise ValueError(
                f"{dataset}: missing {len(missing_questions)} questions "
                f"and {len(missing_chunks)} chunks"
            )

        output_dir = args.output_dir / dataset
        image_dir = output_dir / "images"
        output_dir.mkdir(parents=True)
        output_corpus: list[dict[str, Any]] = []
        image_count = 0
        image_bytes = 0
        for chunk_id in sorted(chunk_ids):
            chunk = dict(all_corpus[chunk_id])
            if str(chunk.get("modality", "text")) == "image":
                source = locate_image(str(chunk["content"]), source_root)
                extension = source.suffix.lower() or ".img"
                filename = hashlib.sha256(chunk_id.encode("utf-8")).hexdigest()[:24] + extension
                target = image_dir / filename
                image_dir.mkdir(exist_ok=True)
                shutil.copy2(source, target)
                chunk["content"] = f"{dataset}/images/{filename}"
                image_count += 1
                image_bytes += target.stat().st_size
            output_corpus.append(chunk)

        write_jsonl(output_dir / "questions.jsonl", (all_questions[qid] for qid in qids))
        write_jsonl(output_dir / "corpus.jsonl", output_corpus)
        manifest["datasets"][dataset] = {
            "questions": len(qids),
            "chunks": len(output_corpus),
            "images": image_count,
            "image_bytes": image_bytes,
            "selection_sha256": sha256(selection_path),
        }

    manifest_path = args.output_dir / "BUNDLE_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
