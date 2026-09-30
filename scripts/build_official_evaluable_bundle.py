"""Materialize the complete locally evaluable official QA splits.

The output contains exactly the qids declared by
``official_evaluable_test_2026_09_30``.  It reuses already downloaded official
metadata and image archives, and writes the normalized ``questions.jsonl`` /
``corpus.jsonl`` schema consumed by the retrieval and downstream runners.
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq
from tqdm import tqdm

from scripts.prepare_official_conformal_bundle import (
    MMQA_IMAGE_ARCHIVE,
    MMQA_RAW_ROOT,
    WEBQA_IMAGE_ARCHIVE,
    download_url,
    stable_chunk_id,
    table_to_markdown,
    tatqa_answers,
)

DATASETS = ("hotpotqa", "mmqa", "tatqa", "webqa")


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    temporary.replace(path)
    return count


def planned_qids(split_root: Path, dataset: str) -> list[str]:
    payload = json.loads((split_root / dataset / "test_manifest.json").read_text())
    qids = [str(value) for value in payload["plan"]]
    if len(qids) != int(payload["count"]) or len(qids) != len(set(qids)):
        raise ValueError(f"{dataset}: invalid test manifest")
    return qids


def ensure_image_link(output: Path, shared: Path) -> None:
    shared.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_symlink():
        if output.resolve() != shared.resolve():
            raise RuntimeError(f"Image symlink points elsewhere: {output}")
        return
    if output.exists():
        raise RuntimeError(f"Refusing to replace existing image path: {output}")
    output.symlink_to(shared.resolve(), target_is_directory=True)


def build_hotpot(split_root: Path, output: Path, parquet_path: Path) -> dict[str, Any]:
    wanted = planned_qids(split_root, "hotpotqa")
    wanted_set = set(wanted)
    by_qid = {str(row["id"]): row for row in pq.read_table(parquet_path).to_pylist()}
    if missing := wanted_set - by_qid.keys():
        raise ValueError(f"HotpotQA source misses {len(missing)} planned qids")
    corpus: dict[str, dict[str, Any]] = {}
    questions = []
    for qid in wanted:
        row = by_qid[qid]
        support_titles = set(map(str, row["supporting_facts"]["title"]))
        candidates, support = [], []
        for title, sentences in zip(row["context"]["title"], row["context"]["sentences"]):
            content = f"{title}\n{' '.join(map(str, sentences))}".strip()
            chunk_id = stable_chunk_id("hotpot", str(title), content)
            corpus.setdefault(
                chunk_id,
                {
                    "id": chunk_id,
                    "source_doc_id": str(title),
                    "modality": "text",
                    "content": content,
                },
            )
            candidates.append(chunk_id)
            if str(title) in support_titles:
                support.append(chunk_id)
        questions.append(
            {
                "qid": qid,
                "question": str(row["question"]),
                "gold_answers": [str(row["answer"])],
                "candidate_ids": candidates,
                "support_ids": list(dict.fromkeys(support)),
                "metadata": {
                    "source_split": "distractor/validation",
                    "split_role": "test",
                    "level": row.get("level"),
                    "type": row.get("type"),
                },
            }
        )
    root = output / "hotpotqa"
    write_jsonl(root / "corpus.jsonl", (corpus[key] for key in sorted(corpus)))
    write_jsonl(root / "questions.jsonl", questions)
    return {"questions": len(questions), "corpus": len(corpus)}


def build_tatqa(split_root: Path, output: Path, source: Path) -> dict[str, Any]:
    wanted = planned_qids(split_root, "tatqa")
    wanted_set = set(wanted)
    documents = json.loads(source.read_text(encoding="utf-8"))
    corpus: dict[str, dict[str, Any]] = {}
    by_qid: dict[str, dict[str, Any]] = {}
    for document_index, document in enumerate(documents):
        report_id = str(document.get("table", {}).get("uid") or f"dev:{document_index}")
        table_id = stable_chunk_id("tatqa-table", report_id)
        corpus[table_id] = {
            "id": table_id,
            "source_doc_id": report_id,
            "modality": "table",
            "content": table_to_markdown(
                document.get("table", {}).get("table", document.get("table", {}))
            ),
        }
        paragraph_ids: dict[str, str] = {}
        for paragraph_index, paragraph in enumerate(document.get("paragraphs", [])):
            order = str(paragraph.get("order", paragraph_index + 1))
            content = str(paragraph.get("text", ""))
            chunk_id = stable_chunk_id("tatqa-text", report_id, order, content)
            paragraph_ids[order] = chunk_id
            corpus[chunk_id] = {
                "id": chunk_id,
                "source_doc_id": report_id,
                "modality": "text",
                "content": content,
            }
        candidates = [table_id, *paragraph_ids.values()]
        for question in document.get("questions", []):
            qid = str(question["uid"])
            if qid not in wanted_set:
                continue
            mapping = question.get("mapping") or {}
            related = {str(value) for value in question.get("rel_paragraphs", [])}
            if isinstance(mapping.get("paragraph"), dict):
                related.update(str(value) for value in mapping["paragraph"])
            answer_from = str(question.get("answer_from", "")).lower()
            support = [paragraph_ids[key] for key in related if key in paragraph_ids]
            if "table" in answer_from or mapping.get("table"):
                support.insert(0, table_id)
            by_qid[qid] = {
                "qid": qid,
                "question": str(question["question"]),
                "gold_answers": tatqa_answers(question),
                "candidate_ids": candidates,
                "support_ids": list(dict.fromkeys(support)),
                "metadata": {
                    "source_split": "dev",
                    "split_role": "test",
                    "answer_type": question.get("answer_type"),
                    "answer_from": answer_from,
                    "scale": question.get("scale"),
                },
            }
    if missing := wanted_set - by_qid.keys():
        raise ValueError(f"TAT-QA source misses {len(missing)} planned qids")
    used = {chunk_id for qid in wanted for chunk_id in by_qid[qid]["candidate_ids"]}
    root = output / "tatqa"
    write_jsonl(root / "corpus.jsonl", (corpus[key] for key in sorted(used)))
    write_jsonl(root / "questions.jsonl", (by_qid[qid] for qid in wanted))
    return {"questions": len(wanted), "corpus": len(used)}


def build_mmqa(
    split_root: Path, output: Path, source_dir: Path, shared_images: Path
) -> dict[str, Any]:
    wanted = planned_qids(split_root, "mmqa")
    wanted_set = set(wanted)
    downloads = output / ".downloads" / "mmqa"
    names = ("MMQA_texts.jsonl.gz", "MMQA_tables.jsonl.gz", "MMQA_images.jsonl.gz")
    paths = {
        name: download_url(f"{MMQA_RAW_ROOT}/{name}", downloads / name, f"MMQA {name}")
        for name in names
    }
    by_qid = {str(row["qid"]): row for row in iter_jsonl(source_dir / "MMQA_dev.jsonl.gz")}
    if missing := wanted_set - by_qid.keys():
        raise ValueError(f"MMQA source misses {len(missing)} planned qids")
    questions_source = [by_qid[qid] for qid in wanted]
    text_ids = {
        str(value) for row in questions_source for value in row["metadata"].get("text_doc_ids", [])
    }
    image_ids = {
        str(value) for row in questions_source for value in row["metadata"].get("image_doc_ids", [])
    }
    table_ids = {
        str(row["metadata"]["table_id"])
        for row in questions_source
        if row["metadata"].get("table_id") is not None
    }
    texts = {
        str(row["id"]): row
        for row in iter_jsonl(paths["MMQA_texts.jsonl.gz"])
        if str(row["id"]) in text_ids
    }
    tables = {
        str(row["id"]): row
        for row in iter_jsonl(paths["MMQA_tables.jsonl.gz"])
        if str(row["id"]) in table_ids
    }
    images = {
        str(row["id"]): row
        for row in iter_jsonl(paths["MMQA_images.jsonl.gz"])
        if str(row["id"]) in image_ids
    }
    missing_meta = (
        (text_ids - texts.keys()) | (table_ids - tables.keys()) | (image_ids - images.keys())
    )
    if missing_meta:
        raise ValueError(f"MMQA metadata misses {len(missing_meta)} candidate ids")
    needed = [row for row in images.values() if not (shared_images / str(row["path"])).is_file()]
    if needed:
        archive_path = download_url(
            MMQA_IMAGE_ARCHIVE, downloads / "final_dataset_images.zip", "MMQA image archive"
        )
        with zipfile.ZipFile(archive_path) as archive:
            members = set(archive.namelist())
            for row in tqdm(needed, desc="Extract MMQA evaluation images", unit="image"):
                relative = Path(str(row["path"]))
                member = f"final_dataset_images/{relative.as_posix()}"
                if member not in members:
                    raise FileNotFoundError(member)
                destination = shared_images / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source_handle, destination.open("wb") as target:
                    shutil.copyfileobj(source_handle, target)
    ensure_image_link(output / "mmqa" / "images", shared_images)
    corpus: dict[str, dict[str, Any]] = {}
    for doc_id, row in texts.items():
        corpus[doc_id] = {
            "id": doc_id,
            "source_doc_id": doc_id,
            "modality": "text",
            "content": f"{row.get('title', '')}\n{row.get('text', '')}".strip(),
        }
    for doc_id, row in tables.items():
        title = " — ".join(
            filter(
                None, [str(row.get("title", "")), str(row.get("table", {}).get("table_name", ""))]
            )
        )
        corpus[doc_id] = {
            "id": doc_id,
            "source_doc_id": doc_id,
            "modality": "table",
            "content": table_to_markdown(row, title=title),
        }
    for doc_id, row in images.items():
        corpus[doc_id] = {
            "id": doc_id,
            "source_doc_id": doc_id,
            "modality": "image",
            "content": f"mmqa/images/{Path(str(row['path'])).as_posix()}",
            "caption": str(row.get("title", "")),
        }
    questions = []
    for row in questions_source:
        metadata = row["metadata"]
        candidates = [str(value) for value in metadata.get("text_doc_ids", [])]
        if metadata.get("table_id") is not None:
            candidates.append(str(metadata["table_id"]))
        candidates.extend(str(value) for value in metadata.get("image_doc_ids", []))
        support = [
            str(item["doc_id"])
            for item in row.get("supporting_context", [])
            if item.get("doc_id") is not None
        ]
        questions.append(
            {
                "qid": str(row["qid"]),
                "question": str(row["question"]),
                "gold_answers": [
                    str(answer.get("answer", answer)) for answer in row.get("answers", [])
                ],
                "candidate_ids": list(dict.fromkeys(candidates)),
                "support_ids": list(dict.fromkeys(support)),
                "metadata": {
                    "source_split": "dev",
                    "split_role": "test",
                    "type": metadata.get("type"),
                    "modalities": metadata.get("modalities", []),
                },
            }
        )
    root = output / "mmqa"
    write_jsonl(root / "corpus.jsonl", (corpus[key] for key in sorted(corpus)))
    write_jsonl(root / "questions.jsonl", questions)
    return {
        "questions": len(questions),
        "corpus": len(corpus),
        "images": len(images),
        "images_extracted": len(needed),
    }


def webqa_existing_images(directory: Path, selected: set[str]) -> dict[str, Path]:
    found: dict[str, Path] = {}
    if directory.is_dir():
        for path in directory.iterdir():
            if path.is_file() and path.stem in selected and path.stat().st_size:
                found[path.stem] = path
    return found


def build_webqa(
    split_root: Path, output: Path, annotations: Path, archive: Path, shared_images: Path
) -> dict[str, Any]:
    wanted = planned_qids(split_root, "webqa")
    wanted_set = set(wanted)
    by_qid = {str(row["qid"]): row for row in iter_jsonl(annotations / "test.json")}
    if missing := wanted_set - by_qid.keys():
        raise ValueError(f"WebQA source misses {len(missing)} planned qids")
    questions_source = [by_qid[qid] for qid in wanted]
    text_ids = {
        str(value)
        for row in questions_source
        for field in ("txt_posFacts", "txt_negFacts")
        for value in row.get(field, [])
    }
    image_ids = {
        str(value)
        for row in questions_source
        for field in ("img_posFacts", "img_negFacts")
        for value in row.get(field, [])
    }
    docs = {
        str(row["snippet_id"]): row
        for row in iter_jsonl(annotations / "all_docs.json")
        if str(row["snippet_id"]) in text_ids
    }
    images = {
        str(row["image_id"]): row
        for row in iter_jsonl(annotations / "all_imgs.json")
        if str(row["image_id"]) in image_ids
    }
    if (text_ids - docs.keys()) or (image_ids - images.keys()):
        raise ValueError("WebQA candidate metadata is incomplete")
    found = webqa_existing_images(shared_images, image_ids)
    reused_images = len(found)
    remaining = image_ids - found.keys()
    if remaining:
        with archive.open("rb") as raw, tarfile.open(fileobj=raw, mode="r|gz") as tar:
            for member in tqdm(tar, desc="Scan WebQA image archive", unit="member"):
                if not member.isfile() or Path(member.name).stem not in remaining:
                    continue
                image_id = Path(member.name).stem
                suffix = Path(member.name).suffix.lower() or ".jpg"
                destination = shared_images / f"{image_id}{suffix}"
                source_handle = tar.extractfile(member)
                if source_handle is None:
                    continue
                with destination.open("wb") as target:
                    shutil.copyfileobj(source_handle, target)
                found[image_id] = destination
                remaining.remove(image_id)
                if not remaining:
                    break
    if remaining:
        raise FileNotFoundError(f"WebQA archive misses {len(remaining)} selected images")
    ensure_image_link(output / "webqa" / "images", shared_images)
    corpus: dict[str, dict[str, Any]] = {}
    for chunk_id, row in docs.items():
        corpus[chunk_id] = {
            "id": chunk_id,
            "source_doc_id": chunk_id,
            "modality": "text",
            "content": f"{row.get('title', '')}\n{row.get('fact', '')}".strip(),
        }
    for chunk_id, row in images.items():
        corpus[chunk_id] = {
            "id": chunk_id,
            "source_doc_id": chunk_id,
            "modality": "image",
            "content": f"webqa/images/{found[chunk_id].name}",
            "caption": " ".join(
                dict.fromkeys(
                    filter(None, [str(row.get("title", "")), str(row.get("caption", ""))])
                )
            ),
        }
    questions = []
    for row in questions_source:
        support = [
            str(value) for field in ("txt_posFacts", "img_posFacts") for value in row.get(field, [])
        ]
        candidates = support + [
            str(value) for field in ("txt_negFacts", "img_negFacts") for value in row.get(field, [])
        ]
        questions.append(
            {
                "qid": str(row["qid"]),
                "question": str(row["Q"]),
                "gold_answers": [str(value) for value in row.get("A", [])],
                "candidate_ids": list(dict.fromkeys(candidates)),
                "support_ids": list(dict.fromkeys(support)),
                "metadata": {
                    "source_split": "validation",
                    "split_role": "test",
                    "candidate_construction": "official WebQA positive+negative facts",
                },
            }
        )
    root = output / "webqa"
    write_jsonl(root / "corpus.jsonl", (corpus[key] for key in sorted(corpus)))
    write_jsonl(root / "questions.jsonl", questions)
    return {
        "questions": len(questions),
        "corpus": len(corpus),
        "images": len(images),
        "images_reused": reused_images,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-root",
        type=Path,
        default=Path("research/internal_state_rag/splits/official_evaluable_test_2026_09_30"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/official_evaluable_test_2026_09_30")
    )
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument(
        "--source-cache", type=Path, default=Path(".cache/official_evaluable_test_sources")
    )
    parser.add_argument(
        "--mmqa-shared-images",
        type=Path,
        default=Path("data/zip_calibration_split_20_09/mmqa/images"),
    )
    parser.add_argument(
        "--webqa-shared-images",
        type=Path,
        default=Path("/dev/shm/uncertainty_rag_webqa_stage_20260922/webqa/images"),
    )
    parser.add_argument(
        "--webqa-downloads",
        type=Path,
        default=Path("data/zip_calibration_split_20_09_webqa/.downloads"),
    )
    args = parser.parse_args()
    selected = [value.strip() for value in args.datasets.split(",") if value.strip()]
    if not selected or set(selected) - set(DATASETS):
        parser.error(f"--datasets must be a subset of {DATASETS}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    if "hotpotqa" in selected:
        parquet = next(
            (args.source_cache / "huggingface/datasets--hotpotqa--hotpot_qa/snapshots").rglob(
                "validation-*.parquet"
            )
        )
        results["hotpotqa"] = build_hotpot(args.split_root, args.output_dir, parquet)
    if "mmqa" in selected:
        results["mmqa"] = build_mmqa(
            args.split_root, args.output_dir, args.source_cache, args.mmqa_shared_images
        )
    if "tatqa" in selected:
        results["tatqa"] = build_tatqa(
            args.split_root, args.output_dir, args.source_cache / "tatqa_dataset_dev.json"
        )
    if "webqa" in selected:
        annotations = args.webqa_downloads / "Ancci--webqa_large"
        archive = args.webqa_downloads / "TreezzZ--WebQA" / WEBQA_IMAGE_ARCHIVE
        results["webqa"] = build_webqa(
            args.split_root, args.output_dir, annotations, archive, args.webqa_shared_images
        )
    manifest_path = args.output_dir / "manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    merged_results = dict(previous.get("datasets", {}))
    merged_results.update(results)
    manifest = {
        "schema_version": 1,
        "protocol": "official_evaluable_test_2026_09_30",
        "datasets": merged_results,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
