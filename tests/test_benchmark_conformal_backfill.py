from scripts.benchmark_conformal_backfill import BackfillDelta, policy_indices


def decisions():
    return [
        {"p_value": 0.001, "rank": 1, "chunk_id": "a", "support_label": "support"},
        {"p_value": 0.9, "rank": 2, "chunk_id": "b", "support_label": "false"},
        {"p_value": 0.001, "rank": 3, "chunk_id": "c", "support_label": "support"},
    ]


def test_policy_ablation_exposes_verified_backfill():
    policies = policy_indices(decisions(), alpha=0.2, max_context=2)

    assert policies["fixed_top_k"] == {0, 1}
    assert policies["conformal_no_backfill"] == {0}
    assert policies["conformal_backfill"] == {0, 2}


def test_backfill_delta_counts_support_rescue():
    delta = BackfillDelta()
    delta.update(decisions(), set(), {2})

    result = delta.finish()
    assert result["backfill_supports_added"] == 1
    assert result["backfill_false_added"] == 0
    assert result["support_rescue_queries"] == 1
