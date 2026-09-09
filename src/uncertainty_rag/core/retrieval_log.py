"""Pure data helpers for reproducible conformal retrieval logs."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from uncertainty_rag.core.conformal_retrieval import ConformalDataError

QUERY_TYPE_RULE_IDS = {
    "pooled": "pooled-v1",
    "keyword_v1": "question-keyword-v1",
}


@dataclass(frozen=True)
class CorpusRecord:
    chunk_id: str
    modality: str
    content: str
    source_doc_id: str


@dataclass(frozen=True)
class QueryRecord:
    dataset: str
    qid: str
    question: str
    support_ids: frozenset[str]
    source_split: str


def stable_json_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_split_role(
    dataset: str,
    qid: str,
    *,
    seed: int,
    development_fraction: float,
    calibration_fraction: float,
) -> str:
    """Assign a whole query deterministically to one experimental role."""

    if development_fraction < 0 or calibration_fraction < 0:
        raise ConformalDataError("Split fractions cannot be negative")
    if development_fraction + calibration_fraction >= 1:
        raise ConformalDataError("Development + calibration fractions must be below 1")
    digest = hashlib.sha256(f"{seed}\0{dataset}\0{qid}".encode()).digest()
    unit_value = int.from_bytes(digest[:8], "big") / 2**64
    if unit_value < development_fraction:
        return "development"
    if unit_value < development_fraction + calibration_fraction:
        return "calibration"
    return "test"


def frozen_query_type(question: str, mode: str) -> str:
    """Return a deterministic question-only query group."""

    if mode == "pooled":
        return "pooled"
    if mode != "keyword_v1":
        raise ConformalDataError(f"Unsupported query-type mode: {mode}")

    normalized = f" {re.sub(r'[^a-z0-9%]+', ' ', question.lower())} "
    groups = {
        "visual_spatial": (
            " color ",
            " colour ",
            " image ",
            " picture ",
            " photo ",
            " shape ",
            " look like ",
            " shown ",
            " appear ",
            " located ",
            " left ",
            " right ",
        ),
        "number_table": (
            " how many ",
            " number ",
            " amount ",
            " percent ",
            " percentage ",
            " total ",
            " difference ",
            " ratio ",
            " average ",
            " value ",
        ),
        "time_sequence": (
            " when ",
            " year ",
            " date ",
            " before ",
            " after ",
            " first ",
            " last ",
            " during ",
        ),
    }
    matched = [name for name, terms in groups.items() if any(term in normalized for term in terms)]
    if len(matched) > 1:
        return "mixed"
    return matched[0] if matched else "fact_lookup"


def load_bundle_records(
    questions_path: Path,
    *,
    bundle_root: Path,
    dataset: str,
) -> tuple[list[CorpusRecord], list[QueryRecord]]:
    """Build a deduplicated corpus and query list from a benchmark bundle."""

    if not questions_path.is_file():
        raise FileNotFoundError(f"Missing questions file: {questions_path}")

    corpus_by_id: dict[str, CorpusRecord] = {}
    queries: list[QueryRecord] = []
    seen_qids: set[str] = set()
    with questions_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row.get("qid", "")).strip()
            question = str(row.get("question", "")).strip()
            if not qid or not question:
                raise ConformalDataError(
                    f"Missing qid/question at {questions_path}:{line_number}"
                )
            if qid in seen_qids:
                raise ConformalDataError(f"Duplicate qid in bundle: {dataset}/{qid}")
            seen_qids.add(qid)

            support_ids: set[str] = set()
            for chunk in row.get("chunks", []):
                chunk_id = str(chunk.get("id", "")).strip()
                modality = str(chunk.get("modality", "text")).strip().lower()
                content = str(chunk.get("content", ""))
                if not chunk_id or not content:
                    raise ConformalDataError(
                        f"Missing chunk id/content at {questions_path}:{line_number}"
                    )
                if modality == "image" and not content.startswith(("http://", "https://")):
                    content = str((bundle_root / content).resolve())
                source_doc_id = str(chunk.get("source_doc_id", chunk_id))
                record = CorpusRecord(chunk_id, modality, content, source_doc_id)
                previous = corpus_by_id.setdefault(chunk_id, record)
                if previous != record:
                    raise ConformalDataError(
                        f"Conflicting definitions for chunk_id={chunk_id} in {dataset}"
                    )
                if bool(chunk.get("is_support", False)):
                    support_ids.add(chunk_id)

            metadata = row.get("metadata") or {}
            queries.append(
                QueryRecord(
                    dataset=dataset,
                    qid=qid,
                    question=question,
                    support_ids=frozenset(support_ids),
                    source_split=str(metadata.get("source_split", "unknown")),
                )
            )

    corpus = [corpus_by_id[key] for key in sorted(corpus_by_id)]
    missing_images = [
        record.content
        for record in corpus
        if record.modality == "image" and not Path(record.content).is_file()
    ]
    if missing_images:
        preview = ", ".join(missing_images[:3])
        raise FileNotFoundError(f"Missing {len(missing_images)} image assets, including: {preview}")
    return corpus, queries


def corpus_revision(corpus: Iterable[CorpusRecord]) -> str:
    """Hash corpus identities and content bytes for an immutable revision."""

    digest = hashlib.sha256()
    for record in corpus:
        digest.update(
            f"{record.chunk_id}\0{record.source_doc_id}\0{record.modality}\0".encode("utf-8")
        )
        if record.modality == "image":
            digest.update(file_sha256(Path(record.content)).encode("ascii"))
        else:
            digest.update(hashlib.sha256(record.content.encode("utf-8")).hexdigest().encode())
        digest.update(b"\n")
    return f"sha256:{digest.hexdigest()}"


def retrieval_log_row(
    *,
    query: QueryRecord,
    chunk: CorpusRecord,
    split_role: str,
    query_type: str,
    rank: int,
    cosine_score: float,
    top_l: int,
    retriever_id: str,
    corpus_revision_id: str,
    preprocess_hash: str,
    query_type_rule_id: str,
    non_support_label: str,
) -> Mapping[str, Any]:
    if non_support_label not in {"false", "unknown"}:
        raise ConformalDataError("non_support_label must be 'false' or 'unknown'")
    support_label = "support" if chunk.chunk_id in query.support_ids else non_support_label
    return {
        "dataset": query.dataset,
        "qid": query.qid,
        "split_role": split_role,
        "query_text": query.question,
        "query_type": query_type,
        "chunk_id": chunk.chunk_id,
        "source_doc_id": chunk.source_doc_id,
        "modality": chunk.modality,
        "rank": rank,
        "cosine_score": float(max(-1.0, min(1.0, cosine_score))),
        "support_label": support_label,
        "top_l": top_l,
        "retriever_id": retriever_id,
        "corpus_revision": corpus_revision_id,
        "preprocess_hash": preprocess_hash,
        "query_type_rule_id": query_type_rule_id,
    }
