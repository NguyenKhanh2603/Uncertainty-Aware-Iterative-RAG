import pytest

from scripts.rescore_conformal_candidates import rerank_rows, score_metrics


def test_reranker_preserves_cosine_and_orders_by_selection_score():
    rows = [
        {
            "dataset": "tatqa",
            "qid": "q1",
            "chunk_id": "a",
            "modality": "text",
            "rank": 1,
            "cosine_score": 0.9,
        },
        {
            "dataset": "tatqa",
            "qid": "q1",
            "chunk_id": "b",
            "modality": "table",
            "rank": 2,
            "cosine_score": 0.8,
        },
    ]

    rescored = rerank_rows(rows, [0.1, 0.9], "probe@test")

    assert [row["chunk_id"] for row in rescored] == ["b", "a"]
    assert [row["rank"] for row in rescored] == [1, 2]
    assert [row["cosine_rank"] for row in rescored] == [2, 1]
    assert [row["cosine_score"] for row in rescored] == [0.8, 0.9]


def test_score_metrics_reports_perfect_separation():
    rows = [
        {"support_label": "false", "score": 0.1},
        {"support_label": "support", "score": 0.9},
        {"support_label": "unknown", "score": 1.0},
    ]

    metrics = score_metrics(rows, "score")

    assert metrics is not None
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["average_precision"] == pytest.approx(1.0)
