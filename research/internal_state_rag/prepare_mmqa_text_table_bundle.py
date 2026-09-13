"""Build an official MMQA text/table subset without downloading image assets.

The script takes disjoint queries from MMQA train and dev, retaining only examples
whose complete labelled supporting context is textual or tabular.  The corpus is
the union of official text/table candidates attached to the selected questions.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Iterable

from scripts.prepare_official_conformal_bundle import table_to_markdown


def iter_gzip_jsonl(path: Path) -> Iterable[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_text_table_complete(row: dict) -> bool:
    metadata = row.get("metadata", {})
    image_ids = {str(value) for value in metadata.get("image_doc_ids", [])}
    support = row.get("supporting_context", [])
    if not support:
        return False
    for item in support:
        doc_id = str(item.get("doc_id", ""))
        part = str(item.get("doc_part", "")).lower()
        if not doc_id or doc_id in image_ids or part == "image":
            return False
    return True


def select_queries(path: Path, count: int) -> list[dict]:
    selected = []
    for row in iter_gzip_jsonl(path):
        if is_text_table_complete(row):
            selected.append(row)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"Only found {len(selected)} eligible queries in {path}; need {count}")
    return selected


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--queries-per-role", type=int, default=1000)
    args = parser.parse_args()
    if args.queries_per_role < 1:
        parser.error("--queries-per-role must be positive")

    sources = {
        "train": args.raw_dir / "MMQA_train.jsonl.gz",
        "dev": args.raw_dir / "MMQA_dev.jsonl.gz",
        "texts": args.raw_dir / "MMQA_texts.jsonl.gz",
        "tables": args.raw_dir / "MMQA_tables.jsonl.gz",
    }
    for path in sources.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    selected = {
        split: select_queries(sources[split], args.queries_per_role)
        for split in ("train", "dev")
    }
    all_rows = [(split, row) for split in ("train", "dev") for row in selected[split]]
    qids = [str(row["qid"]) for _, row in all_rows]
    if len(qids) != len(set(qids)):
        raise ValueError("Duplicate qids across MMQA train/dev")

    text_ids: set[str] = set()
    table_ids: set[str] = set()
    for _, row in all_rows:
        metadata = row.get("metadata", {})
        text_ids.update(str(value) for value in metadata.get("text_doc_ids", []))
        if metadata.get("table_id") is not None:
            table_ids.add(str(metadata["table_id"]))

    texts = {
        str(row["id"]): row
        for row in iter_gzip_jsonl(sources["texts"])
        if str(row["id"]) in text_ids
    }
    tables = {
        str(row["id"]): row
        for row in iter_gzip_jsonl(sources["tables"])
        if str(row["id"]) in table_ids
    }
    missing = (text_ids - texts.keys()) | (table_ids - tables.keys())
    if missing:
        raise ValueError(f"Official metadata missing candidate document {sorted(missing)[0]}")

    corpus = []
    for doc_id in sorted(texts):
        row = texts[doc_id]
        corpus.append(
            {
                "id": doc_id,
                "source_doc_id": doc_id,
                "modality": "text",
                "content": f"{row.get('title', '')}\n{row.get('text', '')}".strip(),
            }
        )
    for doc_id in sorted(tables):
        row = tables[doc_id]
        corpus.append(
            {
                "id": doc_id,
                "source_doc_id": doc_id,
                "modality": "table",
                "content": table_to_markdown(row, title=str(row.get("title", ""))),
            }
        )

    questions = []
    candidate_universe = text_ids | table_ids
    for split, row in all_rows:
        metadata = row.get("metadata", {})
        candidates = [str(value) for value in metadata.get("text_doc_ids", [])]
        if metadata.get("table_id") is not None:
            candidates.append(str(metadata["table_id"]))
        support = [str(item["doc_id"]) for item in row["supporting_context"]]
        if not set(support) <= candidate_universe:
            raise ValueError(f"{row['qid']} has support outside the text/table universe")
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
                    "source_split": split,
                    "type": metadata.get("type"),
                    "modalities": metadata.get("modalities", []),
                    "subset": "complete_text_table_support",
                },
            }
        )

    dataset_dir = args.output_dir / "mmqa"
    write_jsonl(dataset_dir / "corpus.jsonl", corpus)
    write_jsonl(dataset_dir / "questions.jsonl", questions)
    manifest = {
        "schema_version": 2,
        "data_grade": "official_text_table_subset",
        "candidate_scope": "selected_query_official_candidate_union",
        "synthetic_candidates": False,
        "dataset": "mmqa",
        "queries_per_source_split": {
            split: len(rows) for split, rows in selected.items()
        },
        "questions": len(questions),
        "corpus": len(corpus),
        "corpus_modalities": {"text": len(texts), "table": len(tables)},
        "selection": "first eligible rows in official file order",
        "eligibility": "all labelled support documents are text or table",
        "qid_sha256": hashlib.sha256("\n".join(sorted(qids)).encode()).hexdigest(),
        "source_files": {name: sha256_file(path) for name, path in sources.items()},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
