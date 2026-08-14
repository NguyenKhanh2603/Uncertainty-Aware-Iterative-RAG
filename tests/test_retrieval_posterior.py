import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "eval" / "UQ_Eval"))

from run_webq_retrieval_posterior import (  # noqa: E402
    calculate_rpp_scores,
    retrieval_logits,
    sample_context_indices,
)


def test_retrieval_logits_are_invariant_to_affine_score_scale():
    scores = [1.5, 1.4, 1.2, 0.9]

    original = retrieval_logits(scores, temperature=1.0)
    transformed = retrieval_logits([10 * value + 7 for value in scores], temperature=1.0)

    assert np.allclose(original, transformed)


def test_context_sampling_is_reproducible_and_without_replacement():
    scores = [1.6 - 0.05 * index for index in range(10)]

    first, probabilities = sample_context_indices(scores, 5, 5, 1.0, seed=10)
    second, _ = sample_context_indices(scores, 5, 5, 1.0, seed=10)

    assert first == second
    assert all(len(draw) == len(set(draw)) == 5 for draw in first)
    assert np.isclose(sum(probabilities), 1.0)


def test_rpp_anchor_risk_and_entropy_decomposition():
    # Anchor cluster is 0. Five contexts follow, each with two answer IDs.
    semantic_ids = [0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1]

    result = calculate_rpp_scores(semantic_ids, num_contexts=5, answers_per_context=2)

    assert result["rpp_anchor_mass"] == 0.5
    assert result["rpp_anchor_risk"] == 0.5
    assert result["rpp_retrieval_mi"] >= 0.0
    assert np.isclose(
        result["rpp_total_entropy"],
        result["rpp_generation_entropy"] + result["rpp_retrieval_mi"],
    )
