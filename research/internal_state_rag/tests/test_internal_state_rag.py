from collections import Counter
from math import isinf
from types import SimpleNamespace

import numpy as np
import torch

from research.internal_state_rag import (
    InterventionOutcome,
    RetrievalState,
    Trajectory,
    calibrate_stop_threshold,
    derive_retrieval_state,
)
from research.internal_state_rag.causal import mask_attention_heads, zero_head_slices
from research.internal_state_rag.run_causal_head_smoke import layer_matched_controls
from research.internal_state_rag.signals import (
    InternalTrace,
    contrast_trace,
    normalized_chunk_entropy,
)


def test_counterfactual_labels_distinguish_retrieval_from_reasoning_failure():
    retrieval = InterventionOutcome(False, False, False, True)
    reasoning = InterventionOutcome(False, False, False, False)

    assert derive_retrieval_state(retrieval) is RetrievalState.RETRIEVAL_INSUFFICIENT
    assert derive_retrieval_state(reasoning) is RetrievalState.REASONING_FAILURE


def test_context_that_breaks_known_answer_is_misleading():
    outcome = InterventionOutcome(False, True, True, True)
    assert derive_retrieval_state(outcome) is RetrievalState.CONTEXT_MISLED


def test_trajectory_calibration_controls_any_unsafe_round():
    rows = [
        Trajectory(scores=[0.10, 0.05], unsafe=[False, False]) for _ in range(18)
    ]
    rows += [
        Trajectory(scores=[0.90, 0.80], unsafe=[True, True]),
        Trajectory(scores=[0.95, 0.85], unsafe=[True, True]),
    ]

    result = calibrate_stop_threshold(rows, alpha=0.1)

    assert result.certified
    # The largest valid threshold admits one failed trajectory:
    # (1 empirical failure + 1 correction) / (20 + 1) <= 0.1.
    assert result.threshold == 0.80
    assert result.corrected_anytime_risk <= 0.1


def test_too_small_bank_cannot_certify_requested_alpha():
    result = calibrate_stop_threshold(
        [Trajectory(scores=[0.1], unsafe=[False])], alpha=0.1
    )
    assert not result.certified
    assert isinf(result.threshold) and result.threshold < 0


def _trace(logprob: float, residual: np.ndarray) -> InternalTrace:
    return InternalTrace(
        answer_token_ids=[7, 8],
        layer_ids=[2, 4],
        chunk_ids=["a", "b"],
        residual_mean=residual,
        target_logprob=np.full((2, 2), logprob, dtype=np.float32),
        target_margin=np.full((2, 2), logprob, dtype=np.float32),
        entropy=np.full((2, 2), 0.5, dtype=np.float32),
        top1_token_id=np.asarray([[7, 8], [7, 8]], dtype=np.int64),
        attention_mass=np.asarray([[[0.5, 0.5]], [[0.9, 0.1]]], dtype=np.float32),
    )


def test_trace_contrast_exposes_context_logprob_gain():
    with_context = _trace(-0.2, np.asarray([[1.0, 0.0], [0.0, 1.0]]))
    without_context = _trace(-1.2, np.asarray([[1.0, 0.0], [1.0, 0.0]]))

    contrast = contrast_trace(with_context, without_context)

    assert np.allclose(contrast["context_logprob_gain_by_layer"], [1.0, 1.0])
    assert np.allclose(contrast["residual_cosine_by_layer"], [1.0, 0.0])


def test_attention_entropy_preserves_layer_and_head_axes():
    mass = np.asarray([[[0.5, 0.5], [1.0, 0.0]]], dtype=np.float32)
    entropy = normalized_chunk_entropy(mass)
    assert entropy.shape == (1, 2)
    assert np.allclose(entropy, [[1.0, 0.0]], atol=1e-6)


def test_zero_head_slices_only_changes_selected_head():
    hidden = torch.arange(8, dtype=torch.float32).reshape(1, 1, 8)
    masked = zero_head_slices(hidden, [1, 3], num_heads=4)
    assert masked.tolist() == [[[0.0, 1.0, 0.0, 0.0, 4.0, 5.0, 0.0, 0.0]]]
    assert hidden.tolist() == [[[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]]]


def test_attention_head_mask_hook_is_removed_after_context():
    projection = torch.nn.Linear(8, 8, bias=False)
    projection.weight.data.copy_(torch.eye(8))
    layer = SimpleNamespace(self_attn=SimpleNamespace(o_proj=projection))
    model = SimpleNamespace(
        model=SimpleNamespace(layers=[layer]),
        config=SimpleNamespace(num_attention_heads=4),
    )
    hidden = torch.ones(1, 1, 8)

    with mask_attention_heads(model, [(0, 2)]):
        assert projection(hidden).tolist() == [[[1, 1, 1, 1, 0, 0, 1, 1]]]
    assert projection(hidden).tolist() == [[[1, 1, 1, 1, 1, 1, 1, 1]]]


def test_causal_controls_match_top_head_layer_counts():
    ranking = [
        {"layer": layer, "head": head, "support_fraction": 1 - head / 10}
        for layer in (2, 4)
        for head in range(5)
    ]
    top = [(2, 0), (2, 1), (4, 0)]
    random_heads, bottom_heads = layer_matched_controls(
        ranking, top, num_heads=5, seed=7
    )
    assert Counter(layer for layer, _ in random_heads) == Counter({2: 2, 4: 1})
    assert Counter(layer for layer, _ in bottom_heads) == Counter({2: 2, 4: 1})
    assert not set(random_heads) & set(top)
    assert bottom_heads == [(2, 4), (2, 3), (4, 4)]
