"""Conformal scoring and experimental context selection for retrieval logs.

The reference banks estimate the null distribution for the hypothesis that a
retrieved candidate is false evidence.  A small p-value therefore supports
rejecting that null and admitting the candidate to the context.

The Benjamini--Yekutieli (BY) step-up rule implemented here is valid under
arbitrary p-value dependence before a context-size cap is applied.  Choosing at
most K of the BY rejections is useful for the end-to-end experiment, but the
proposal still requires a proof for that capped procedure.  Outputs expose this
distinction instead of claiming a guarantee that has not been established.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping, Sequence

from uncertainty_rag.core.conformal_retrieval import (
    ConformalDataError,
    RetrievalCandidate,
    conformal_p_value,
    parse_condition_fields,
    parse_rank_bins,
    rank_bin_for,
)


@dataclass(frozen=True)
class BYResult:
    """Indices rejected by the Benjamini--Yekutieli step-up procedure."""

    rejected_indices: tuple[int, ...]
    cutoff_rank: int
    cutoff_p_value: float | None
    harmonic_number: float


def benjamini_yekutieli(p_values: Sequence[float], alpha: float) -> BYResult:
    """Apply BY to dependent p-values and return indices in input order."""

    if not 0 < alpha < 1:
        raise ConformalDataError("alpha must be strictly between 0 and 1")
    if any(not isfinite(value) or not 0 <= value <= 1 for value in p_values):
        raise ConformalDataError("p-values must be finite and in [0, 1]")
    if not p_values:
        return BYResult((), 0, None, 0.0)

    count = len(p_values)
    harmonic = sum(1.0 / index for index in range(1, count + 1))
    ordered = sorted(range(count), key=lambda index: (p_values[index], index))
    cutoff_rank = 0
    for rank, index in enumerate(ordered, start=1):
        threshold = rank * alpha / (count * harmonic)
        if p_values[index] <= threshold:
            cutoff_rank = rank

    rejected = frozenset(ordered[:cutoff_rank])
    return BYResult(
        rejected_indices=tuple(index for index in range(count) if index in rejected),
        cutoff_rank=cutoff_rank,
        cutoff_p_value=(p_values[ordered[cutoff_rank - 1]] if cutoff_rank else None),
        harmonic_number=harmonic,
    )


class ReferenceBankIndex:
    """Validated lookup from retrieval-row conditions to sorted false scores."""

    def __init__(self, artifact: Mapping[str, Any]) -> None:
        self.top_l = int(artifact["top_l"])
        self.condition_fields = parse_condition_fields(
            ",".join(str(value) for value in artifact["conditioning"])
        )
        rank_bin_labels = tuple(str(value) for value in artifact.get("rank_bins", ()))
        self.rank_bins = parse_rank_bins(",".join(rank_bin_labels)) if rank_bin_labels else ()
        if "rank_bin" in self.condition_fields and not self.rank_bins:
            raise ConformalDataError("rank-conditioned banks require rank_bins")

        self.min_bank_size = int(artifact.get("min_bank_size", 1))
        self.pipeline_fingerprints = artifact.get("pipeline_fingerprints", {})
        self._scores: dict[tuple[str, ...], tuple[float, ...]] = {}
        self._underpowered: set[tuple[str, ...]] = set()
        for bank in artifact.get("banks", ()):
            condition = bank["condition"]
            key = tuple(str(condition[field]) for field in self.condition_fields)
            if key in self._scores:
                raise ConformalDataError(f"Duplicate reference bank condition: {condition}")
            scores = tuple(float(value) for value in bank["scores"])
            if int(bank["n_false_scores"]) != len(scores):
                raise ConformalDataError(f"Bank size disagrees with scores: {condition}")
            if any(scores[index] > scores[index + 1] for index in range(len(scores) - 1)):
                raise ConformalDataError(f"Bank scores are not sorted: {condition}")
            self._scores[key] = scores
            if len(scores) < self.min_bank_size:
                self._underpowered.add(key)
        if not self._scores:
            raise ConformalDataError("Reference-bank artifact contains no banks")

    def _validate_fingerprint(self, candidate: RetrievalCandidate) -> None:
        expected = self.pipeline_fingerprints.get(candidate.dataset)
        if not expected:
            return
        observed = {
            "retriever_id": candidate.retriever_id,
            "corpus_revision": candidate.corpus_revision,
            "preprocess_hash": candidate.preprocess_hash,
            "query_type_rule_id": candidate.query_type_rule_id,
            "top_l": candidate.top_l,
        }
        mismatched = [field for field, value in observed.items() if expected.get(field) != value]
        if mismatched:
            raise ConformalDataError(
                f"Pipeline fingerprint mismatch for {candidate.dataset}: {', '.join(mismatched)}"
            )

    def condition_for(self, candidate: RetrievalCandidate) -> dict[str, str]:
        """Return the reference-bank condition for one candidate."""

        values = {
            "dataset": candidate.dataset,
            "query_type": candidate.query_type,
            "modality": candidate.modality,
        }
        if "rank_bin" in self.condition_fields:
            values["rank_bin"] = rank_bin_for(candidate.rank, self.rank_bins).label
        return {field: values[field] for field in self.condition_fields}

    def score(
        self,
        row: Mapping[str, Any],
        *,
        allow_underpowered: bool = False,
    ) -> dict[str, Any]:
        """Validate a retrieval row and assign its matched conformal p-value."""

        candidate = RetrievalCandidate.from_mapping(row)
        self._validate_fingerprint(candidate)
        condition = self.condition_for(candidate)
        key = tuple(condition[field] for field in self.condition_fields)
        scores = self._scores.get(key)
        if scores is None:
            return {
                "candidate": candidate,
                "condition": condition,
                "bank_size": 0,
                "bank_status": "missing",
                "p_value": 1.0,
            }
        underpowered = key in self._underpowered
        if underpowered and not allow_underpowered:
            return {
                "candidate": candidate,
                "condition": condition,
                "bank_size": len(scores),
                "bank_status": "underpowered_blocked",
                "p_value": 1.0,
            }
        return {
            "candidate": candidate,
            "condition": condition,
            "bank_size": len(scores),
            "bank_status": "underpowered_allowed" if underpowered else "ok",
            "p_value": conformal_p_value(candidate.cosine_score, scores),
        }


def select_query_context(
    rows: Sequence[Mapping[str, Any]],
    bank_index: ReferenceBankIndex,
    *,
    alpha: float,
    max_context: int,
    allow_underpowered: bool = False,
) -> dict[str, Any]:
    """Score one query, run BY, and form an experimental at-most-K context."""

    if max_context < 1:
        raise ConformalDataError("max_context must be positive")
    scored = [bank_index.score(row, allow_underpowered=allow_underpowered) for row in rows]
    if not scored:
        raise ConformalDataError("Cannot select a context from an empty query")

    candidates = [item["candidate"] for item in scored]
    query_keys = {(item.dataset, item.qid, item.split_role) for item in candidates}
    if len(query_keys) != 1:
        raise ConformalDataError("All rows passed to select_query_context must share one query")
    ranks = [item.rank for item in candidates]
    if len(set(ranks)) != len(ranks):
        raise ConformalDataError("Query contains duplicate retrieval ranks")

    by = benjamini_yekutieli([item["p_value"] for item in scored], alpha)
    by_rejected = set(by.rejected_indices)
    capped_order = sorted(
        by_rejected,
        key=lambda index: (
            scored[index]["p_value"],
            candidates[index].rank,
            candidates[index].chunk_id,
        ),
    )
    selected = set(capped_order[:max_context])
    selected_in_context_order = sorted(selected, key=lambda index: candidates[index].rank)

    decisions = []
    for index, (item, candidate) in enumerate(zip(scored, candidates)):
        if item["bank_status"] == "missing":
            reason = "missing_bank"
        elif item["bank_status"] == "underpowered_blocked":
            reason = "underpowered_bank"
        elif index not in by_rejected:
            reason = "by_not_rejected"
        elif index not in selected:
            reason = "context_cap"
        elif candidate.rank > max_context:
            reason = "backfill"
        else:
            reason = "accepted_top_k"
        decisions.append(
            {
                "chunk_id": candidate.chunk_id,
                "source_doc_id": candidate.source_doc_id,
                "modality": candidate.modality,
                "rank": candidate.rank,
                "cosine_score": candidate.cosine_score,
                "support_label": candidate.support_label,
                "condition": item["condition"],
                "bank_size": item["bank_size"],
                "bank_status": item["bank_status"],
                "p_value": item["p_value"],
                "by_rejected": index in by_rejected,
                "selected": index in selected,
                "decision_reason": reason,
            }
        )

    baseline_indices = sorted(range(len(candidates)), key=lambda index: candidates[index].rank)[
        :max_context
    ]
    reserve_supports = sum(item.support_label == "support" for item in candidates)
    selected_supports = sum(candidates[index].support_label == "support" for index in selected)
    selected_false = sum(candidates[index].support_label == "false" for index in selected)
    baseline_supports = sum(
        candidates[index].support_label == "support" for index in baseline_indices
    )
    baseline_false = sum(candidates[index].support_label == "false" for index in baseline_indices)
    dataset, qid, split_role = next(iter(query_keys))
    return {
        "dataset": dataset,
        "qid": qid,
        "split_role": split_role,
        "alpha": alpha,
        "max_context": max_context,
        "reserve_size": len(candidates),
        "reserve_supports": reserve_supports,
        "by_rejections": len(by_rejected),
        "by_cutoff_rank": by.cutoff_rank,
        "by_cutoff_p_value": by.cutoff_p_value,
        "by_harmonic_number": by.harmonic_number,
        "selected_count": len(selected),
        "selected_supports": selected_supports,
        "selected_false": selected_false,
        "selected_unknown": len(selected) - selected_supports - selected_false,
        "selected_ranks": [candidates[index].rank for index in selected_in_context_order],
        "selected_chunk_ids": [candidates[index].chunk_id for index in selected_in_context_order],
        "backfill_count": sum(candidates[index].rank > max_context for index in selected),
        "baseline_count": len(baseline_indices),
        "baseline_supports": baseline_supports,
        "baseline_false": baseline_false,
        "formal_capped_risk_guarantee": False,
        "procedure": "BY_then_pvalue_cap_experimental",
        "decisions": decisions,
    }
