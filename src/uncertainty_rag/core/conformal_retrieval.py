"""Data contracts and reference banks for conformal backfill retrieval.

This module implements the data-preparation part of Section 2.3. It deliberately
does not run a retriever: callers must provide logs produced by the exact frozen
top-L retrieval pipeline that will be used at inference.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Any

VALID_SPLIT_ROLES = frozenset({"development", "calibration", "test"})
VALID_SUPPORT_LABELS = frozenset({"support", "false", "unknown"})
VALID_MODALITIES = frozenset({"text", "image", "table", "audio", "video"})
VALID_CONDITION_FIELDS = frozenset({"dataset", "query_type", "modality", "rank_bin"})
DEFAULT_CONDITION_FIELDS = ("dataset", "query_type", "modality", "rank_bin")
SCHEMA_VERSION = 1


class ConformalDataError(ValueError):
    """Raised when a retrieval log cannot support valid bank construction."""


@dataclass(frozen=True, order=True)
class RankBin:
    """Inclusive one-based rank interval."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 1 or self.end < self.start:
            raise ConformalDataError(f"Invalid rank bin {self.start}-{self.end}")

    @property
    def label(self) -> str:
        return f"{self.start}-{self.end}"

    def contains(self, rank: int) -> bool:
        return self.start <= rank <= self.end


@dataclass(frozen=True)
class RetrievalCandidate:
    """One candidate returned by a frozen top-L retrieval run."""

    dataset: str
    qid: str
    split_role: str
    query_text: str
    query_type: str
    chunk_id: str
    modality: str
    rank: int
    cosine_score: float
    support_label: str
    top_l: int
    retriever_id: str
    corpus_revision: str
    preprocess_hash: str
    query_type_rule_id: str
    source_doc_id: str | None = None
    retrieved_l: int | None = None

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> RetrievalCandidate:
        required = {
            "dataset",
            "qid",
            "split_role",
            "query_text",
            "query_type",
            "chunk_id",
            "modality",
            "rank",
            "cosine_score",
            "support_label",
            "top_l",
            "retriever_id",
            "corpus_revision",
            "preprocess_hash",
            "query_type_rule_id",
        }
        missing = sorted(name for name in required if name not in row)
        if missing:
            raise ConformalDataError(f"Missing required fields: {', '.join(missing)}")

        candidate = cls(
            dataset=str(row["dataset"]).strip(),
            qid=str(row["qid"]).strip(),
            split_role=str(row["split_role"]).strip().lower(),
            query_text=str(row["query_text"]),
            query_type=str(row["query_type"]).strip().lower(),
            chunk_id=str(row["chunk_id"]).strip(),
            source_doc_id=(
                str(row["source_doc_id"]).strip()
                if row.get("source_doc_id") is not None
                else None
            ),
            modality=str(row["modality"]).strip().lower(),
            rank=int(row["rank"]),
            cosine_score=float(row["cosine_score"]),
            support_label=str(row["support_label"]).strip().lower(),
            top_l=int(row["top_l"]),
            retriever_id=str(row["retriever_id"]).strip(),
            corpus_revision=str(row["corpus_revision"]).strip(),
            preprocess_hash=str(row["preprocess_hash"]).strip(),
            query_type_rule_id=str(row["query_type_rule_id"]).strip(),
            retrieved_l=(
                int(row["retrieved_l"])
                if row.get("retrieved_l") is not None
                else None
            ),
        )
        candidate.validate()
        return candidate

    def validate(self) -> None:
        text_fields = {
            "dataset": self.dataset,
            "qid": self.qid,
            "query_type": self.query_type,
            "chunk_id": self.chunk_id,
            "retriever_id": self.retriever_id,
            "corpus_revision": self.corpus_revision,
            "preprocess_hash": self.preprocess_hash,
            "query_type_rule_id": self.query_type_rule_id,
        }
        empty = sorted(name for name, value in text_fields.items() if not value)
        if empty:
            raise ConformalDataError(f"Empty required fields: {', '.join(empty)}")
        if self.split_role not in VALID_SPLIT_ROLES:
            raise ConformalDataError(f"Invalid split_role: {self.split_role}")
        if self.support_label not in VALID_SUPPORT_LABELS:
            raise ConformalDataError(f"Invalid support_label: {self.support_label}")
        if self.modality not in VALID_MODALITIES:
            raise ConformalDataError(f"Invalid modality: {self.modality}")
        if self.top_l < 1:
            raise ConformalDataError("top_l must be positive")
        if not 1 <= self.effective_retrieved_l <= self.top_l:
            raise ConformalDataError(
                f"retrieved_l={self.effective_retrieved_l} must be within 1..top_l={self.top_l}"
            )
        if not 1 <= self.rank <= self.effective_retrieved_l:
            raise ConformalDataError(
                f"rank {self.rank} is outside the retrieved_l={self.effective_retrieved_l}"
            )
        if not isfinite(self.cosine_score) or not -1.0 <= self.cosine_score <= 1.0:
            raise ConformalDataError(
                f"cosine_score must be in [-1, 1], got {self.cosine_score}"
            )

    @property
    def effective_retrieved_l(self) -> int:
        """Actual rows returned for this query; legacy fixed-L logs omit it."""

        return self.retrieved_l if self.retrieved_l is not None else self.top_l

    @property
    def pipeline_fingerprint(self) -> tuple[str, str, str, str, int]:
        return (
            self.retriever_id,
            self.corpus_revision,
            self.preprocess_hash,
            self.query_type_rule_id,
            self.top_l,
        )


def parse_rank_bins(specification: str) -> tuple[RankBin, ...]:
    """Parse a comma-separated rank-bin specification such as ``1-3,4-10``."""

    bins: list[RankBin] = []
    for part in specification.split(","):
        part = part.strip()
        if not part:
            continue
        pieces = part.split("-", maxsplit=1)
        if len(pieces) != 2:
            raise ConformalDataError(f"Rank bin must be START-END, got {part!r}")
        bins.append(RankBin(int(pieces[0]), int(pieces[1])))
    if not bins:
        raise ConformalDataError("At least one rank bin is required")
    ordered = sorted(bins)
    for previous, current in zip(ordered, ordered[1:]):
        if current.start <= previous.end:
            raise ConformalDataError(
                f"Overlapping rank bins: {previous.label} and {current.label}"
            )
    return tuple(ordered)


def parse_condition_fields(specification: str) -> tuple[str, ...]:
    """Parse and validate the fields used to condition a reference bank."""

    fields = tuple(part.strip() for part in specification.split(",") if part.strip())
    if not fields:
        raise ConformalDataError("At least one conditioning field is required")
    if len(set(fields)) != len(fields):
        raise ConformalDataError("Conditioning fields cannot be repeated")
    unsupported = sorted(set(fields) - VALID_CONDITION_FIELDS)
    if unsupported:
        raise ConformalDataError(
            f"Unsupported conditioning fields: {', '.join(unsupported)}"
        )
    if "dataset" not in fields:
        raise ConformalDataError("dataset must remain a conditioning field")
    return fields


def validate_rank_bin_coverage(rank_bins: Sequence[RankBin], top_l: int) -> None:
    """Require rank bins to cover every rank from 1 through top_l exactly once."""

    observed = [rank for rank_bin in rank_bins for rank in range(rank_bin.start, rank_bin.end + 1)]
    expected = list(range(1, top_l + 1))
    if observed != expected:
        raise ConformalDataError(
            f"Rank bins must cover 1..{top_l} without gaps; observed {observed}"
        )


def rank_bin_for(rank: int, rank_bins: Sequence[RankBin]) -> RankBin:
    for rank_bin in rank_bins:
        if rank_bin.contains(rank):
            return rank_bin
    raise ConformalDataError(f"Rank {rank} is not covered by the configured rank bins")


def conformal_p_value(cosine_score: float, sorted_false_scores: Sequence[float]) -> float:
    """Return the conservative upper-tail conformal p-value for a candidate."""

    if not sorted_false_scores:
        raise ConformalDataError("Cannot compute a p-value from an empty reference bank")
    if any(
        sorted_false_scores[index] > sorted_false_scores[index + 1]
        for index in range(len(sorted_false_scores) - 1)
    ):
        raise ConformalDataError("Reference-bank scores must be sorted ascending")
    first_greater_or_equal = bisect_left(sorted_false_scores, cosine_score)
    count_greater_or_equal = len(sorted_false_scores) - first_greater_or_equal
    return (1.0 + count_greater_or_equal) / (len(sorted_false_scores) + 1.0)


def build_reference_bank_artifact(
    rows: Iterable[Mapping[str, Any]],
    *,
    rank_bins: Sequence[RankBin],
    condition_fields: Sequence[str] = DEFAULT_CONDITION_FIELDS,
    min_bank_size: int = 1000,
    allow_small_banks: bool = False,
) -> dict[str, Any]:
    """Validate retrieval rows and build deterministic false-match banks.

    Banks contain only rows labelled ``false`` from the ``calibration`` split.
    Dataset is part of the bank condition to prevent accidental pooling across
    different corpora or benchmark distributions.
    """

    if min_bank_size < 1:
        raise ConformalDataError("min_bank_size must be positive")
    condition_fields = parse_condition_fields(",".join(condition_fields))

    candidates = [RetrievalCandidate.from_mapping(row) for row in rows]
    if not candidates:
        raise ConformalDataError("Retrieval log is empty")

    split_by_qid: dict[tuple[str, str], str] = {}
    seen_rank: set[tuple[str, str, int]] = set()
    seen_chunk: set[tuple[str, str, str]] = set()
    ranks_by_qid: dict[tuple[str, str], set[int]] = defaultdict(set)
    query_signatures: dict[tuple[str, str], set[tuple[str, str, int, int]]] = defaultdict(set)
    fingerprints: dict[str, set[tuple[str, str, str, str, int]]] = defaultdict(set)
    query_counts: dict[str, set[str]] = defaultdict(set)
    label_counts: dict[str, int] = defaultdict(int)

    for candidate in candidates:
        qid_key = (candidate.dataset, candidate.qid)
        previous_split = split_by_qid.setdefault(qid_key, candidate.split_role)
        if previous_split != candidate.split_role:
            raise ConformalDataError(
                f"Split leakage: {candidate.dataset}/{candidate.qid} appears in "
                f"both {previous_split} and {candidate.split_role}"
            )

        rank_key = (candidate.dataset, candidate.qid, candidate.rank)
        if rank_key in seen_rank:
            raise ConformalDataError(
                f"Duplicate retrieval rank: {candidate.dataset}/{candidate.qid} "
                f"rank={candidate.rank}"
            )
        seen_rank.add(rank_key)
        chunk_key = (candidate.dataset, candidate.qid, candidate.chunk_id)
        if chunk_key in seen_chunk:
            raise ConformalDataError(
                f"Duplicate candidate chunk: {candidate.dataset}/{candidate.qid} "
                f"chunk_id={candidate.chunk_id}"
            )
        seen_chunk.add(chunk_key)
        ranks_by_qid[qid_key].add(candidate.rank)
        query_signatures[qid_key].add(
            (
                candidate.query_text,
                candidate.query_type,
                candidate.top_l,
                candidate.effective_retrieved_l,
            )
        )
        fingerprints[candidate.dataset].add(candidate.pipeline_fingerprint)
        query_counts[candidate.split_role].add(f"{candidate.dataset}\0{candidate.qid}")
        label_counts[candidate.support_label] += 1

    drifted = {dataset: values for dataset, values in fingerprints.items() if len(values) != 1}
    if drifted:
        datasets = ", ".join(sorted(drifted))
        raise ConformalDataError(f"Pipeline drift within dataset(s): {datasets}")

    inconsistent_queries = [key for key, values in query_signatures.items() if len(values) != 1]
    if inconsistent_queries:
        dataset, qid = sorted(inconsistent_queries)[0]
        raise ConformalDataError(
            f"Inconsistent query metadata within top-L rows: {dataset}/{qid}"
        )

    for qid_key, ranks in sorted(ranks_by_qid.items()):
        query_retrieved_l = next(iter(query_signatures[qid_key]))[3]
        expected_ranks = set(range(1, query_retrieved_l + 1))
        if ranks != expected_ranks:
            dataset, qid = qid_key
            missing = sorted(expected_ranks - ranks)
            raise ConformalDataError(
                f"Incomplete top-L retrieval for {dataset}/{qid}; "
                f"retrieved_l={query_retrieved_l}, missing ranks={missing}"
            )

    top_l_values = {candidate.top_l for candidate in candidates}
    if len(top_l_values) != 1:
        raise ConformalDataError(f"All datasets must use the same top_l, got {top_l_values}")
    top_l = next(iter(top_l_values))
    if "rank_bin" in condition_fields:
        validate_rank_bin_coverage(rank_bins, top_l)

    grouped: dict[tuple[str, ...], list[RetrievalCandidate]] = defaultdict(list)
    for candidate in candidates:
        if candidate.split_role != "calibration" or candidate.support_label != "false":
            continue
        values = {
            "dataset": candidate.dataset,
            "query_type": candidate.query_type,
            "modality": candidate.modality,
        }
        if "rank_bin" in condition_fields:
            values["rank_bin"] = rank_bin_for(candidate.rank, rank_bins).label
        key = tuple(values[field] for field in condition_fields)
        grouped[key].append(candidate)

    if not grouped:
        raise ConformalDataError("No calibration rows labelled 'false' were found")

    banks: list[dict[str, Any]] = []
    underpowered: list[dict[str, Any]] = []
    for key in sorted(grouped):
        members = grouped[key]
        scores = sorted(candidate.cosine_score for candidate in members)
        condition = dict(zip(condition_fields, key))
        bank = {
            "condition": condition,
            "n_false_scores": len(scores),
            "n_calibration_queries": len({candidate.qid for candidate in members}),
            "scores": scores,
            "provenance": [
                {
                    "qid": candidate.qid,
                    "chunk_id": candidate.chunk_id,
                    "rank": candidate.rank,
                }
                for candidate in sorted(
                    members, key=lambda item: (item.qid, item.rank, item.chunk_id)
                )
            ],
        }
        banks.append(bank)
        if len(scores) < min_bank_size:
            underpowered.append({
                **condition,
                "n_false_scores": len(scores),
                "required": min_bank_size,
            })

    if underpowered and not allow_small_banks:
        description = "; ".join(
            "/".join(str(item[field]) for field in condition_fields)
            + f"={item['n_false_scores']}"
            for item in underpowered
        )
        raise ConformalDataError(
            f"Reference banks below min_bank_size={min_bank_size}: {description}"
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "top_l": top_l,
        "conditioning": list(condition_fields),
        "rank_bins": (
            [rank_bin.label for rank_bin in rank_bins]
            if "rank_bin" in condition_fields
            else []
        ),
        "min_bank_size": min_bank_size,
        "is_paper_ready": not underpowered,
        "summary": {
            "input_rows": len(candidates),
            "queries_by_split": {
                split: len(query_counts.get(split, set())) for split in sorted(VALID_SPLIT_ROLES)
            },
            "rows_by_support_label": {
                label: label_counts.get(label, 0) for label in sorted(VALID_SUPPORT_LABELS)
            },
            "reference_banks": len(banks),
            "underpowered_banks": len(underpowered),
        },
        "pipeline_fingerprints": {
            dataset: {
                "retriever_id": next(iter(values))[0],
                "corpus_revision": next(iter(values))[1],
                "preprocess_hash": next(iter(values))[2],
                "query_type_rule_id": next(iter(values))[3],
                "top_l": next(iter(values))[4],
            }
            for dataset, values in sorted(fingerprints.items())
        },
        "underpowered": underpowered,
        "banks": banks,
    }
