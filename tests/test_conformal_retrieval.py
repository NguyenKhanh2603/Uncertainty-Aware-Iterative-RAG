import pytest

from uncertainty_rag.core.conformal_retrieval import (
    ConformalDataError,
    build_reference_bank_artifact,
    conformal_p_value,
    parse_rank_bins,
)


def candidate(
    *,
    qid="q1",
    split_role="calibration",
    rank=1,
    score=0.5,
    support_label="false",
    retriever_id="retriever@rev1",
    top_l=3,
):
    return {
        "dataset": "mmqa",
        "qid": qid,
        "split_role": split_role,
        "query_text": "What is shown?",
        "query_type": "visual",
        "chunk_id": f"{qid}-chunk-{rank}",
        "source_doc_id": f"doc-{rank}",
        "modality": "image",
        "rank": rank,
        "cosine_score": score,
        "support_label": support_label,
        "top_l": top_l,
        "retriever_id": retriever_id,
        "corpus_revision": "corpus@rev1",
        "preprocess_hash": "preprocess@rev1",
        "query_type_rule_id": "router@rev1",
    }


def test_conformal_p_value_counts_ties_conservatively():
    scores = [0.2, 0.5, 0.5, 0.9]

    assert conformal_p_value(0.5, scores) == pytest.approx(4 / 5)
    assert conformal_p_value(0.95, scores) == pytest.approx(1 / 5)


def test_reference_banks_keep_only_calibration_false_matches():
    rows = [
        candidate(qid="cal", rank=1, score=0.7),
        candidate(qid="cal", rank=2, score=0.8, support_label="support"),
        candidate(qid="cal", rank=3, score=0.6, support_label="unknown"),
        candidate(qid="test", split_role="test", rank=1, score=0.9),
        candidate(qid="test", split_role="test", rank=2, score=0.8),
        candidate(qid="test", split_role="test", rank=3, score=0.7),
    ]

    artifact = build_reference_bank_artifact(
        rows,
        rank_bins=parse_rank_bins("1-3"),
        min_bank_size=1,
    )

    assert artifact["is_paper_ready"] is True
    assert artifact["summary"]["queries_by_split"] == {
        "calibration": 1,
        "development": 0,
        "test": 1,
    }
    assert artifact["banks"][0]["scores"] == [0.7]
    assert artifact["banks"][0]["provenance"] == [
        {"qid": "cal", "chunk_id": "cal-chunk-1", "rank": 1}
    ]


def test_reference_bank_rejects_qid_split_leakage():
    rows = [
        candidate(qid="same", split_role="calibration", rank=1),
        candidate(qid="same", split_role="test", rank=2),
    ]

    with pytest.raises(ConformalDataError, match="Split leakage"):
        build_reference_bank_artifact(
            rows,
            rank_bins=parse_rank_bins("1-3"),
            min_bank_size=1,
        )


def test_reference_bank_rejects_pipeline_drift_within_dataset():
    rows = [
        candidate(qid="q1", rank=1, retriever_id="retriever@rev1", top_l=1),
        candidate(qid="q2", rank=1, retriever_id="retriever@rev2", top_l=1),
    ]

    with pytest.raises(ConformalDataError, match="Pipeline drift"):
        build_reference_bank_artifact(
            rows,
            rank_bins=parse_rank_bins("1-1"),
            min_bank_size=1,
        )


def test_small_bank_requires_explicit_smoke_override():
    rows = [candidate(rank=1, top_l=1)]

    with pytest.raises(ConformalDataError, match="below min_bank_size"):
        build_reference_bank_artifact(
            rows,
            rank_bins=parse_rank_bins("1-1"),
            min_bank_size=2,
        )

    artifact = build_reference_bank_artifact(
        rows,
        rank_bins=parse_rank_bins("1-1"),
        min_bank_size=2,
        allow_small_banks=True,
    )
    assert artifact["is_paper_ready"] is False
    assert artifact["summary"]["underpowered_banks"] == 1


def test_rank_bins_must_cover_declared_top_l():
    with pytest.raises(ConformalDataError, match="cover 1..3"):
        build_reference_bank_artifact(
            [candidate(rank=1), candidate(rank=2), candidate(rank=3)],
            rank_bins=parse_rank_bins("1-2"),
            min_bank_size=1,
        )


def test_reference_bank_rejects_incomplete_top_l_query():
    rows = [candidate(rank=1), candidate(rank=3)]

    with pytest.raises(ConformalDataError, match="Incomplete top-L"):
        build_reference_bank_artifact(
            rows,
            rank_bins=parse_rank_bins("1-3"),
            min_bank_size=1,
        )


def test_reference_bank_accepts_ragged_official_candidate_pool():
    rows = [candidate(rank=1), candidate(rank=2)]
    for row in rows:
        row["retrieved_l"] = 2

    artifact = build_reference_bank_artifact(
        rows,
        rank_bins=parse_rank_bins("1-3"),
        condition_fields=("dataset", "modality"),
        min_bank_size=1,
    )

    assert artifact["top_l"] == 3
    assert artifact["summary"]["input_rows"] == 2


def test_modality_conditioning_pools_ranks_without_a_rank_bin():
    rows = [
        candidate(qid="q1", rank=1, score=0.7),
        candidate(qid="q1", rank=2, score=0.6),
        candidate(qid="q1", rank=3, score=0.5),
    ]

    artifact = build_reference_bank_artifact(
        rows,
        rank_bins=parse_rank_bins("1-3"),
        condition_fields=("dataset", "modality"),
        min_bank_size=1,
    )

    assert artifact["conditioning"] == ["dataset", "modality"]
    assert artifact["rank_bins"] == []
    assert artifact["banks"][0]["condition"] == {
        "dataset": "mmqa",
        "modality": "image",
    }
    assert artifact["banks"][0]["n_false_scores"] == 3
