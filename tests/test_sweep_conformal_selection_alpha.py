import pytest

from scripts.sweep_conformal_selection_alpha import (
    SweepMetrics,
    parse_alpha_grid,
    selected_indices,
)
from uncertainty_rag.core.conformal_retrieval import ConformalDataError


def decisions():
    return [
        {"p_value": 0.001, "rank": 1, "chunk_id": "a", "support_label": "support"},
        {"p_value": 0.02, "rank": 3, "chunk_id": "b", "support_label": "false"},
        {"p_value": 0.5, "rank": 2, "chunk_id": "c", "support_label": "support"},
    ]


def test_parse_alpha_grid_sorts_and_deduplicates():
    assert parse_alpha_grid("0.1,0.05,0.1") == (0.05, 0.1)
    with pytest.raises(ConformalDataError):
        parse_alpha_grid("1.0")


def test_higher_alpha_can_select_more_saved_p_values():
    low, _ = selected_indices(decisions(), alpha=0.01, max_context=2)
    high, _ = selected_indices(decisions(), alpha=0.2, max_context=2)

    assert low == {0}
    assert high == {0, 1}


def test_sweep_metrics_reports_conditional_and_micro_false_share():
    metrics = SweepMetrics()
    base = {
        "reserve_supports": 2,
        "baseline_count": 2,
        "baseline_supports": 1,
        "baseline_false": 1,
    }
    metrics.update(base, decisions(), {0, 1}, by_rejections=2, max_context=2)
    metrics.update(base, decisions(), set(), by_rejections=0, max_context=2)

    result = metrics.finish()
    assert result["micro_evidence_precision"] == 0.5
    assert result["micro_false_share"] == 0.5
    assert result["mean_query_fdp"] == 0.25
    assert result["mean_query_fdp_given_nonempty"] == 0.5
