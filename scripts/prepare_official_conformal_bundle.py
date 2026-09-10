"""Build paper-scale retrieval inputs from official benchmark candidates.

The output uses a normalized schema:

* ``<dataset>/corpus.jsonl`` stores every unique official chunk once.
* ``<dataset>/questions.jsonl`` stores candidate IDs and gold support IDs.

No synthetic distractors are created.  Test splits without public support labels
are intentionally excluded.  WebQA annotations/images are pinned redistributions
of the official release because the authors' Google Drive ships images as 51
multipart archives; their original candidate IDs and labels are preserved.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import tarfile
import time
import zipfile
import concurrent.futures
from pathlib import Path
from typing import Any, Iterable, Iterator

import pyarrow.parquet as pq
import requests
from huggingface_hub import hf_hub_download
from remotezip import RemoteZip
from tqdm import tqdm

MMQA_GIT_REVISION = "4dd14328c6d02a4daa357cc6032915a0b14602e3"
MMQA_RAW_ROOT = (
    f"https://raw.githubusercontent.com/allenai/multimodalqa/{MMQA_GIT_REVISION}/dataset"
)
MMQA_IMAGE_ARCHIVE = (
    "https://multimodalqa-images.s3-us-west-2.amazonaws.com/"
    "final_dataset_images/final_dataset_images.zip"
)

WEBQA_ANNOTATION_REPO = "Ancci/webqa_large"
WEBQA_ANNOTATION_REVISION = "5d2222ebafe1443d2e31f396a5eafb8803394146"
WEBQA_IMAGE_REPO = "TreezzZ/WebQA"
WEBQA_IMAGE_REVISION = "58a6ed255b8ae02ddd048eb133f2bf7dfde340f5"
WEBQA_IMAGE_ARCHIVE = "images.tar.gz"
WEBQA_IMAGE_ARCHIVE_BYTES = 39_103_815_251
WEBQA_APPROX_EXTRACTED_BYTES = 55_100_000_000

HOTPOT_REPO = "hotpotqa/hotpot_qa"
HOTPOT_REVISION = "1908d6afbbead072334abe2965f91bd2709910ab"
HOTPOT_FILES = {
    "train": [
        "distractor/train-00000-of-00002.parquet",
        "distractor/train-00001-of-00002.parquet",
    ],
    "validation": ["distractor/validation-00000-of-00001.parquet"],
}

TATQA_GIT_REVISION = "870accc41953dcde885aabeb963d94aabdc0fbc3"
TATQA_RAW_ROOT = (
    f"https://raw.githubusercontent.com/NExTplusplus/TAT-QA/{TATQA_GIT_REVISION}/dataset_raw"
)
TATQA_FILES = {
    "train": "tatqa_dataset_train.json",
    "dev": "tatqa_dataset_dev.json",
}


def jsonl_rows(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]], *, total: int | None = None) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in tqdm(
            rows, total=total, desc=f"Write {path.parent.name}/{path.name}", unit="row"
        ):
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    temporary.replace(path)
    return count


def download_url(url: str, destination: Path, description: str) -> Path:
    if destination.is_file():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.part")
    with requests.get(
        url,
        stream=True,
        timeout=(30, 300),
        headers={"Accept-Encoding": "identity"},
    ) as response:
        response.raise_for_status()
        response.raw.decode_content = True
        total = int(response.headers.get("content-length", 0)) or None
        with (
            temporary.open("wb") as handle,
            tqdm.wrapattr(
                response.raw,
                "read",
                total=total,
                desc=description,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as source,
        ):
            shutil.copyfileobj(source, handle)
    temporary.replace(destination)
    return destination


def read_json_file(path: Path) -> Any:
    """Read JSON and tolerate a content-encoded file left by an older run."""

    with path.open("rb") as handle:
        is_gzip = handle.read(2) == b"\x1f\x8b"
    opener = gzip.open if is_gzip else Path.open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def hf_download(repo: str, revision: str, filename: str, download_dir: Path) -> Path:
    print(f"HF download: {repo}@{revision} / {filename}", flush=True)
    return Path(
        hf_hub_download(
            repo_id=repo,
            repo_type="dataset",
            revision=revision,
            filename=filename,
            local_dir=download_dir / repo.replace("/", "--"),
            token=os.environ.get("HF_TOKEN"),
        )
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_chunk_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()[:24]
    return f"{prefix}:{digest}"


def take_limit(rows: Iterable[dict[str, Any]], remaining: int | None) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        if remaining is not None and len(selected) >= remaining:
            break
        selected.append(row)
    return selected


def table_to_markdown(value: Any, *, title: str = "") -> str:
    if isinstance(value, dict):
        table = value.get("table", value)
        header = table.get("header", []) if isinstance(table, dict) else []
        rows = table.get("table_rows", []) if isinstance(table, dict) else []
        headers = [
            str(cell.get("column_name", cell)) if isinstance(cell, dict) else str(cell)
            for cell in header
        ]
        values = [
            [str(cell.get("text", cell)) if isinstance(cell, dict) else str(cell) for cell in row]
            for row in rows
        ]
    elif isinstance(value, list) and value:
        headers = [str(item) for item in value[0]]
        values = [[str(item) for item in row] for row in value[1:]]
    else:
        return f"{title}\n{value}".strip()
    width = max(len(headers), max((len(row) for row in values), default=0))
    headers = (headers + [""] * width)[:width]
    lines = [f"| {' | '.join(headers)} |", f"| {' | '.join(['---'] * width)} |"]
    lines.extend(f"| {' | '.join((row + [''] * width)[:width])} |" for row in values)
    return "\n".join(([title] if title else []) + lines)


def prepare_mmqa(root: Path, downloads: Path, max_questions: int) -> dict[str, Any]:
    started = time.perf_counter()
    names = [
        "MMQA_train.jsonl.gz",
        "MMQA_dev.jsonl.gz",
        "MMQA_texts.jsonl.gz",
        "MMQA_tables.jsonl.gz",
        "MMQA_images.jsonl.gz",
    ]
    paths = {
        name: download_url(f"{MMQA_RAW_ROOT}/{name}", downloads / "mmqa" / name, f"MMQA {name}")
        for name in names
    }
    questions: list[tuple[str, dict[str, Any]]] = []
    for split, filename in (("train", "MMQA_train.jsonl.gz"), ("dev", "MMQA_dev.jsonl.gz")):
        remaining = None if max_questions == 0 else max_questions - len(questions)
        if remaining is not None and remaining <= 0:
            break
        questions.extend((split, row) for row in take_limit(jsonl_rows(paths[filename]), remaining))

    text_ids = set()
    image_ids = set()
    table_ids = set()
    for _, row in questions:
        metadata = row.get("metadata", {})
        text_ids.update(str(value) for value in metadata.get("text_doc_ids", []))
        image_ids.update(str(value) for value in metadata.get("image_doc_ids", []))
        if metadata.get("table_id") is not None:
            table_ids.add(str(metadata["table_id"]))
        for item in row.get("supporting_context", []):
            doc_id = str(item.get("doc_id"))
            if not doc_id or doc_id == "None":
                continue
            doc_part = str(item.get("doc_part", ""))
            if doc_part == "text":
                text_ids.add(doc_id)
            elif doc_part == "image":
                image_ids.add(doc_id)
            elif doc_part == "table":
                table_ids.add(doc_id)
    texts = {
        str(row["id"]): row
        for row in jsonl_rows(paths["MMQA_texts.jsonl.gz"])
        if str(row["id"]) in text_ids
    }
    tables = {
        str(row["id"]): row
        for row in jsonl_rows(paths["MMQA_tables.jsonl.gz"])
        if str(row["id"]) in table_ids
    }
    image_meta = {
        str(row["id"]): row
        for row in jsonl_rows(paths["MMQA_images.jsonl.gz"])
        if str(row["id"]) in image_ids
    }

    missing_metadata = (
        (text_ids - texts.keys()) | (table_ids - tables.keys()) | (image_ids - image_meta.keys())
    )
    if missing_metadata:
        raise RuntimeError(
            f"MMQA official metadata is missing {len(missing_metadata)} referenced IDs"
        )

    image_dir = root / "mmqa" / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    needed = [
        row for key, row in image_meta.items() if not (image_dir / str(row["path"])).is_file()
    ]
    if needed:
        print(f"MMQA images: {len(needed)} official referenced assets (local parallel extraction)")
        local_zip_path = downloads / "mmqa" / "final_dataset_images.zip"
        download_url(MMQA_IMAGE_ARCHIVE, local_zip_path, "MMQA image archive")
        with zipfile.ZipFile(local_zip_path, "r") as archive:
            members = set(archive.namelist())
            
            def extract_image(row):
                relative = Path(str(row["path"]))
                member = f"final_dataset_images/{relative.as_posix()}"
                if member not in members:
                    raise FileNotFoundError(f"Missing MMQA official image member: {member}")
                destination = image_dir / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target)
                    
            with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
                list(tqdm(executor.map(extract_image, needed), total=len(needed), desc="Extract MMQA images", unit="image"))

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
    for doc_id, row in image_meta.items():
        corpus[doc_id] = {
            "id": doc_id,
            "source_doc_id": doc_id,
            "modality": "image",
            "content": f"mmqa/images/{Path(str(row['path'])).as_posix()}",
            "caption": str(row.get("title", "")),
        }

    query_rows = []
    for split, row in questions:
        metadata = row.get("metadata", {})
        candidates = [str(value) for value in metadata.get("text_doc_ids", [])]
        table_id = metadata.get("table_id")
        if table_id is not None:
            candidates.append(str(table_id))
        candidates.extend(str(value) for value in metadata.get("image_doc_ids", []))
        support = [
            str(item["doc_id"])
            for item in row.get("supporting_context", [])
            if item.get("doc_id") is not None
        ]
        candidates.extend(support)
        query_rows.append(
            {
                "qid": str(row["qid"]),
                "question": str(row["question"]),
                "gold_answers": [
                    str(answer.get("answer", answer)) for answer in row.get("answers", [])
                ],
                "candidate_ids": list(dict.fromkeys(candidates)),
                "support_ids": list(dict.fromkeys(support)),
                "metadata": {
                    "source_split": split,
                    "type": metadata.get("type"),
                    "modalities": metadata.get("modalities", []),
                },
            }
        )
    write_jsonl(
        root / "mmqa" / "corpus.jsonl", (corpus[key] for key in sorted(corpus)), total=len(corpus)
    )
    write_jsonl(root / "mmqa" / "questions.jsonl", query_rows, total=len(query_rows))
    return {
        "questions": len(query_rows),
        "corpus": len(corpus),
        "images": len(image_meta),
        "source": "official allenai/multimodalqa",
        "revision": MMQA_GIT_REVISION,
        "source_files": {name: sha256_file(path) for name, path in paths.items()},
        "seconds": round(time.perf_counter() - started, 3),
    }


class _ProgressReader:
    def __init__(self, handle, progress):
        self.handle = handle
        self.progress = progress

    def read(self, size=-1):
        data = self.handle.read(size)
        self.progress.update(len(data))
        return data

    def readable(self):
        return True


def _extract_webqa_images(
    archive: Path, selected_ids: set[str], destination: Path
) -> dict[str, str]:
    destination.mkdir(parents=True, exist_ok=True)
    found: dict[str, str] = {}
    for image_id in selected_ids:
        existing = list(destination.glob(f"{image_id}.*"))
        if existing:
            found[image_id] = f"webqa/images/{existing[0].name}"
    remaining = selected_ids - found.keys()
    if not remaining:
        return found
    print(
        f"Scanning the 39.1 GB official WebQA image archive for {len(remaining):,} assets. "
        "The byte progress bar below gives elapsed time and ETA.",
        flush=True,
    )
    with (
        archive.open("rb") as raw,
        tqdm(
            total=archive.stat().st_size,
            desc="Scan/extract WebQA images",
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as progress,
    ):
        reader = _ProgressReader(raw, progress)
        with tarfile.open(fileobj=reader, mode="r|gz") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                image_id = Path(member.name).stem
                if image_id not in remaining:
                    continue
                suffix = Path(member.name).suffix.lower() or ".jpg"
                output = destination / f"{image_id}{suffix}"
                source = tar.extractfile(member)
                if source is None:
                    continue
                with output.open("wb") as target:
                    shutil.copyfileobj(source, target)
                found[image_id] = f"webqa/images/{output.name}"
                remaining.remove(image_id)
                progress.set_postfix_str(f"found={len(found):,}/{len(selected_ids):,}")
                if not remaining:
                    break
    if remaining:
        raise FileNotFoundError(
            f"WebQA image archive lacks {len(remaining)} selected IDs; examples: "
            + ", ".join(sorted(remaining)[:5])
        )
    return found


def prepare_webqa(
    root: Path,
    downloads: Path,
    max_questions: int,
    *,
    keep_archive: bool,
    skip_disk_check: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    # The redistribution splits the official 36,766-example training file into
    # train(31,766)+dev(5,000), and names the official 4,966 validation examples
    # test.json. All three retain public positive/negative fact labels.
    redistributed_splits = {
        "train": "train",
        "train_holdout": "dev",
        "validation": "test",
    }
    split_files = {
        source_split: hf_download(
            WEBQA_ANNOTATION_REPO,
            WEBQA_ANNOTATION_REVISION,
            f"{filename_stem}.json",
            downloads,
        )
        for source_split, filename_stem in redistributed_splits.items()
    }
    questions: list[tuple[str, dict[str, Any]]] = []
    for split, path in split_files.items():
        remaining = None if max_questions == 0 else max_questions - len(questions)
        if remaining is not None and remaining <= 0:
            break
        questions.extend((split, row) for row in take_limit(jsonl_rows(path), remaining))

    selected_text_ids = {
        str(value)
        for _, row in questions
        for field in ("txt_posFacts", "txt_negFacts")
        for value in row.get(field, [])
    }
    selected_image_ids = {
        str(value)
        for _, row in questions
        for field in ("img_posFacts", "img_negFacts")
        for value in row.get(field, [])
    }
    docs_path = hf_download(
        WEBQA_ANNOTATION_REPO, WEBQA_ANNOTATION_REVISION, "all_docs.json", downloads
    )
    images_meta_path = hf_download(
        WEBQA_ANNOTATION_REPO, WEBQA_ANNOTATION_REVISION, "all_imgs.json", downloads
    )
    docs = {
        str(row["snippet_id"]): row
        for row in tqdm(jsonl_rows(docs_path), desc="Index selected WebQA text", unit="chunk")
        if str(row["snippet_id"]) in selected_text_ids
    }
    image_meta = {
        str(row["image_id"]): row
        for row in tqdm(
            jsonl_rows(images_meta_path), desc="Index selected WebQA image metadata", unit="image"
        )
        if str(row["image_id"]) in selected_image_ids
    }
    if selected_text_ids - docs.keys():
        raise RuntimeError(
            f"Missing {len(selected_text_ids - docs.keys())} official WebQA text candidates"
        )
    if selected_image_ids - image_meta.keys():
        raise RuntimeError(
            f"Missing {len(selected_image_ids - image_meta.keys())} official WebQA image candidates"
        )

    existing_count = sum(
        bool(list((root / "webqa" / "images").glob(f"{image_id}.*")))
        for image_id in selected_image_ids
    )
    if existing_count != len(selected_image_ids):
        estimated_extract = WEBQA_APPROX_EXTRACTED_BYTES * (
            len(selected_image_ids) / max(1, 400_000)
        )
        required = WEBQA_IMAGE_ARCHIVE_BYTES + estimated_extract + 5 * 2**30
        free = shutil.disk_usage(root).free
        print(
            f"WebQA disk preflight: free={free / 2**30:.1f} GiB, "
            f"estimated required={required / 2**30:.1f} GiB "
            f"({len(selected_image_ids):,} unique official candidate images)",
            flush=True,
        )
        if free < required and not skip_disk_check:
            raise RuntimeError(
                "Insufficient disk for official WebQA images. Use a larger disk, reduce "
                "--max-questions, or pass --skip-disk-check only if the estimate is known "
                "to be conservative."
            )
        archive = hf_download(
            WEBQA_IMAGE_REPO, WEBQA_IMAGE_REVISION, WEBQA_IMAGE_ARCHIVE, downloads
        )
        image_paths = _extract_webqa_images(archive, selected_image_ids, root / "webqa" / "images")
        if not keep_archive:
            archive.unlink(missing_ok=True)
    else:
        image_paths = {
            image_id: "webqa/images/"
            + list((root / "webqa" / "images").glob(f"{image_id}.*"))[0].name
            for image_id in selected_image_ids
        }

    corpus: dict[str, dict[str, Any]] = {}
    for chunk_id, row in docs.items():
        corpus[chunk_id] = {
            "id": chunk_id,
            "source_doc_id": chunk_id,
            "modality": "text",
            "content": f"{row.get('title', '')}\n{row.get('fact', '')}".strip(),
        }
    for chunk_id in image_meta:
        corpus[chunk_id] = {
            "id": chunk_id,
            "source_doc_id": chunk_id,
            "modality": "image",
            "content": image_paths[chunk_id],
            "caption": " ".join(
                dict.fromkeys(
                    filter(
                        None,
                        [
                            str(image_meta[chunk_id].get("title", "")),
                            str(image_meta[chunk_id].get("caption", "")),
                        ],
                    )
                )
            ),
        }

    query_rows = []
    for split, row in questions:
        support = [
            str(value) for field in ("txt_posFacts", "img_posFacts") for value in row.get(field, [])
        ]
        candidates = support + [
            str(value) for field in ("txt_negFacts", "img_negFacts") for value in row.get(field, [])
        ]
        query_rows.append(
            {
                "qid": str(row["qid"]),
                "question": str(row["Q"]),
                "gold_answers": [str(value) for value in row.get("A", [])],
                "candidate_ids": list(dict.fromkeys(candidates)),
                "support_ids": list(dict.fromkeys(support)),
                "metadata": {
                    "source_split": split,
                    "candidate_construction": "official WebQA positive+negative facts",
                },
            }
        )
    write_jsonl(
        root / "webqa" / "corpus.jsonl", (corpus[key] for key in sorted(corpus)), total=len(corpus)
    )
    write_jsonl(root / "webqa" / "questions.jsonl", query_rows, total=len(query_rows))
    return {
        "questions": len(query_rows),
        "corpus": len(corpus),
        "images": len(selected_image_ids),
        "source": "official WebQA fields via pinned redistributions; no synthetic candidates",
        "annotation_repo": WEBQA_ANNOTATION_REPO,
        "annotation_revision": WEBQA_ANNOTATION_REVISION,
        "image_repo": WEBQA_IMAGE_REPO,
        "image_revision": WEBQA_IMAGE_REVISION,
        "seconds": round(time.perf_counter() - started, 3),
    }


def prepare_hotpotqa(root: Path, downloads: Path, max_questions: int) -> dict[str, Any]:
    started = time.perf_counter()
    corpus: dict[str, dict[str, Any]] = {}
    query_rows: list[dict[str, Any]] = []
    sources: dict[str, str] = {}
    for split in ("train", "validation"):
        for filename in HOTPOT_FILES[split]:
            if max_questions and len(query_rows) >= max_questions:
                break
            path = hf_download(HOTPOT_REPO, HOTPOT_REVISION, filename, downloads)
            sources[filename] = sha256_file(path)
            parquet = pq.ParquetFile(path)
            progress = tqdm(
                total=parquet.metadata.num_rows, desc=f"Parse HotpotQA {split}", unit="question"
            )
            for batch in parquet.iter_batches(batch_size=1024):
                for row in batch.to_pylist():
                    if max_questions and len(query_rows) >= max_questions:
                        break
                    context = row.get("context") or {}
                    support_titles = {
                        str(value) for value in (row.get("supporting_facts") or {}).get("title", [])
                    }
                    candidates, support = [], []
                    for title, sentences in zip(
                        context.get("title", []), context.get("sentences", [])
                    ):
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
                    query_rows.append(
                        {
                            "qid": str(row["id"]),
                            "question": str(row["question"]),
                            "gold_answers": [str(row["answer"])],
                            "candidate_ids": candidates,
                            "support_ids": list(dict.fromkeys(support)),
                            "metadata": {
                                "source_split": f"distractor/{split}",
                                "level": row.get("level"),
                                "type": row.get("type"),
                            },
                        }
                    )
                progress.update(len(batch))
                if max_questions and len(query_rows) >= max_questions:
                    break
            progress.close()
    write_jsonl(
        root / "hotpotqa" / "corpus.jsonl",
        (corpus[key] for key in sorted(corpus)),
        total=len(corpus),
    )
    write_jsonl(root / "hotpotqa" / "questions.jsonl", query_rows, total=len(query_rows))
    return {
        "questions": len(query_rows),
        "corpus": len(corpus),
        "source": HOTPOT_REPO,
        "revision": HOTPOT_REVISION,
        "source_files": sources,
        "seconds": round(time.perf_counter() - started, 3),
    }


def tatqa_answers(question: dict[str, Any]) -> list[str]:
    answer = question.get("answer", "")
    values = [str(value) for value in answer] if isinstance(answer, list) else [str(answer)]
    scale = str(question.get("scale", "")).strip()
    if scale:
        values.extend(f"{value} {scale}" for value in list(values))
    return list(dict.fromkeys(value for value in values if value))


def prepare_tatqa(root: Path, downloads: Path, max_questions: int) -> dict[str, Any]:
    started = time.perf_counter()
    corpus: dict[str, dict[str, Any]] = {}
    query_rows: list[dict[str, Any]] = []
    sources: dict[str, str] = {}
    for split, filename in TATQA_FILES.items():
        if max_questions and len(query_rows) >= max_questions:
            break
        path = download_url(
            f"{TATQA_RAW_ROOT}/{filename}", downloads / "tatqa" / filename, f"TAT-QA {split}"
        )
        sources[filename] = sha256_file(path)
        documents = read_json_file(path)
        for document_index, document in enumerate(
            tqdm(documents, desc=f"Parse TAT-QA {split}", unit="report")
        ):
            report_id = str(document.get("table", {}).get("uid") or f"{split}:{document_index}")
            table_id = stable_chunk_id("tatqa-table", report_id)
            corpus.setdefault(
                table_id,
                {
                    "id": table_id,
                    "source_doc_id": report_id,
                    "modality": "table",
                    "content": table_to_markdown(
                        document.get("table", {}).get("table", document.get("table", {}))
                    ),
                },
            )
            paragraph_ids: dict[str, str] = {}
            for paragraph_index, paragraph in enumerate(document.get("paragraphs", [])):
                order = str(paragraph.get("order", paragraph_index + 1))
                content = str(paragraph.get("text", ""))
                chunk_id = stable_chunk_id("tatqa-text", report_id, order, content)
                paragraph_ids[order] = chunk_id
                corpus.setdefault(
                    chunk_id,
                    {
                        "id": chunk_id,
                        "source_doc_id": report_id,
                        "modality": "text",
                        "content": content,
                    },
                )
            candidates = [table_id, *paragraph_ids.values()]
            for question in document.get("questions", []):
                if max_questions and len(query_rows) >= max_questions:
                    break
                mapping = question.get("mapping") or {}
                related = {str(value) for value in question.get("rel_paragraphs", [])}
                if isinstance(mapping.get("paragraph"), dict):
                    related.update(str(value) for value in mapping["paragraph"])
                answer_from = str(question.get("answer_from", "")).lower()
                support = [paragraph_ids[key] for key in related if key in paragraph_ids]
                if "table" in answer_from or mapping.get("table"):
                    support.insert(0, table_id)
                query_rows.append(
                    {
                        "qid": str(question["uid"]),
                        "question": str(question["question"]),
                        "gold_answers": tatqa_answers(question),
                        "candidate_ids": candidates,
                        "support_ids": list(dict.fromkeys(support)),
                        "metadata": {
                            "source_split": split,
                            "answer_from": answer_from,
                            "answer_type": question.get("answer_type"),
                            "scale": question.get("scale"),
                        },
                    }
                )
            if max_questions and len(query_rows) >= max_questions:
                break
    write_jsonl(
        root / "tatqa" / "corpus.jsonl", (corpus[key] for key in sorted(corpus)), total=len(corpus)
    )
    write_jsonl(root / "tatqa" / "questions.jsonl", query_rows, total=len(query_rows))
    return {
        "questions": len(query_rows),
        "corpus": len(corpus),
        "source": "official NExTplusplus/TAT-QA",
        "revision": TATQA_GIT_REVISION,
        "source_files": sources,
        "seconds": round(time.perf_counter() - started, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", default="mmqa,webqa,hotpotqa,tatqa")
    parser.add_argument(
        "--max-questions",
        type=int,
        default=0,
        help="Per-dataset limit for a trial run; 0 uses every labelled train+dev question",
    )
    parser.add_argument("--keep-downloads", action="store_true")
    parser.add_argument("--skip-disk-check", action="store_true")
    args = parser.parse_args()
    if args.max_questions < 0:
        parser.error("--max-questions cannot be negative")
    selected = [value.strip() for value in args.datasets.split(",") if value.strip()]
    unknown = set(selected) - {"mmqa", "webqa", "hotpotqa", "tatqa"}
    if unknown:
        parser.error(f"Unknown datasets: {sorted(unknown)}")

    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    downloads = root / ".downloads"
    print(
        f"Official bundle output={root} datasets={selected} "
        f"max_questions={'ALL' if args.max_questions == 0 else args.max_questions}",
        flush=True,
    )
    builders = {
        "mmqa": lambda: prepare_mmqa(root, downloads, args.max_questions),
        "webqa": lambda: prepare_webqa(
            root,
            downloads,
            args.max_questions,
            keep_archive=args.keep_downloads,
            skip_disk_check=args.skip_disk_check,
        ),
        "hotpotqa": lambda: prepare_hotpotqa(root, downloads, args.max_questions),
        "tatqa": lambda: prepare_tatqa(root, downloads, args.max_questions),
    }
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "data_grade": "official",
        "candidate_scope": "dataset_provided_per_query",
        "max_questions_per_dataset": args.max_questions or None,
        "synthetic_candidates": False,
        "datasets": {},
    }
    total_started = time.perf_counter()
    for dataset in selected:
        print(f"\n===== PREPARE OFFICIAL {dataset.upper()} =====", flush=True)
        manifest["datasets"][dataset] = builders[dataset]()
    manifest["total_seconds"] = round(time.perf_counter() - total_started, 3)
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if downloads.exists() and not args.keep_downloads:
        resolved_downloads = downloads.resolve()
        if resolved_downloads.parent != root:
            raise RuntimeError(
                f"Refusing to remove unexpected download directory: {resolved_downloads}"
            )
        shutil.rmtree(resolved_downloads)
    print(f"Official bundle ready: {manifest_path}")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
