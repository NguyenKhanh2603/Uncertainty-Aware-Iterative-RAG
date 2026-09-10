import pytest

from uncertainty_rag.core.conformal_retrieval import ConformalDataError
from uncertainty_rag.core.conformal_selection import (
    ReferenceBankIndex,
    benjamini_yekutieli,
    select_query_context,
)


def row(*, rank, score, modality="text", support_label="false"):
    return {
        "dataset": "mmqa",
        "qid": "q1",
        "split_role": "development",
        "query_text": "What is shown?",
        "query_type": "pooled",
        "chunk_id": f"chunk-{rank}",
        "source_doc_id": f"doc-{rank}",
        "modality": modality,
        "rank": rank,
        "cosine_score": score,
        "support_label": support_label,
        "top_l": 3,
        "retrieved_l": 3,
        "retriever_id": "retriever@rev1",
        "corpus_revision": "corpus@rev1",
        "preprocess_hash": "preprocess@rev1",
        "query_type_rule_id": "pooled-v1",
    }


def artifact(*, min_bank_size=1):
    return {
        "top_l": 3,
        "conditioning": ["dataset", "modality"],
        "rank_bins": [],
        "min_bank_size": min_bank_size,
        "pipeline_fingerprints": {
            "mmqa": {
                "retriever_id": "retriever@rev1",
                "corpus_revision": "corpus@rev1",
                "preprocess_hash": "preprocess@rev1",
                "query_type_rule_id": "pooled-v1",
                "top_l": 3,
            }
        },
        "banks": [
            {
                "condition": {"dataset": "mmqa", "modality": "text"},
                "n_false_scores": 100,
                "scores": [0.1] * 100,
            },
            {
                "condition": {"dataset": "mmqa", "modality": "image"},
                "n_false_scores": 3,
                "scores": [0.1, 0.2, 0.3],
            },
        ],
    }


def test_benjamini_yekutieli_step_up():
    result = benjamini_yekutieli([0.01, 0.03, 0.20], alpha=0.1)

    assert result.rejected_indices == (0, 1)
    assert result.cutoff_rank == 2
    assert result.harmonic_number == pytest.approx(11 / 6)


def test_reference_bank_matches_modality_and_computes_upper_tail_p_value():
    index = ReferenceBankIndex(artifact())

    result = index.score(row(rank=1, score=0.25, modality="image"))

    assert result["condition"] == {"dataset": "mmqa", "modality": "image"}
    assert result["p_value"] == pytest.approx(2 / 4)


def test_underpowered_bank_fails_closed_unless_explicitly_allowed():
    index = ReferenceBankIndex(artifact(min_bank_size=10))
    candidate = row(rank=1, score=0.35, modality="image")

    blocked = index.score(candidate)
    allowed = index.score(candidate, allow_underpowered=True)

    assert blocked["bank_status"] == "underpowered_blocked"
    assert blocked["p_value"] == 1.0
    assert allowed["bank_status"] == "underpowered_allowed"
    assert allowed["p_value"] == pytest.approx(1 / 4)


def test_pipeline_fingerprint_mismatch_is_rejected():
    index = ReferenceBankIndex(artifact())
    candidate = row(rank=1, score=0.5)
    candidate["query_type_rule_id"] = "question-keyword-v1"

    with pytest.raises(ConformalDataError, match="fingerprint mismatch"):
        index.score(candidate)


def test_selection_can_backfill_beyond_original_top_k():
    index = ReferenceBankIndex(artifact())
    rows = [
        row(rank=1, score=0.0),
        row(rank=2, score=1.0, support_label="support"),
        row(rank=3, score=1.0, support_label="support"),
    ]

    result = select_query_context(rows, index, alpha=0.2, max_context=2)

    assert result["selected_ranks"] == [2, 3]
    assert result["backfill_count"] == 1
    assert result["selected_supports"] == 2
    assert result["formal_capped_risk_guarantee"] is False


def test_backfill_preserves_accepted_top_k_before_using_reserve():
    index = ReferenceBankIndex(artifact())
    rows = [
        row(rank=1, score=1.0, support_label="support"),
        row(rank=2, score=0.0),
        row(rank=3, score=1.0, support_label="support"),
    ]

    result = select_query_context(rows, index, alpha=0.2, max_context=2)

    assert result["no_backfill_ranks"] == [1]
    assert result["selected_ranks"] == [1, 3]
    assert result["no_backfill_supports"] == 1
    assert result["selected_supports"] == 2
