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
from research.internal_state_rag.directer_plausibility import distribution_plausibility
from research.internal_state_rag.contrastive_saliency import (
    jensen_shannon_from_logits,
    select_context_sensitive_tokens,
)
from research.internal_state_rag.edge_mask import zero_attention_edges
from research.internal_state_rag.run_causal_head_smoke import layer_matched_controls
from research.internal_state_rag.run_full_topl_validation import (
    eligible_qids,
    make_split_plan,
)
from research.internal_state_rag.run_full_topl_saliency import grouped_z
from research.internal_state_rag.analyze_hidden_chunk_probe import query_metrics
from research.internal_state_rag.analyze_causal_ranknet_probe import rank_loss
from research.internal_state_rag.analyze_hard_negative_calibration_curve import (
    hard_negative_mask,
)
from research.internal_state_rag.make_pairwise_feature_plan import make_plan
from research.internal_state_rag.analyze_pairwise_conformal_clean_split import (
    conformal_threshold,
    end_to_end_metrics,
    keep_mask,
)
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


def test_causal_rank_loss_prefers_the_teacher_order():
    target = torch.tensor([[0.5, 0.0, -0.5]])
    aligned = torch.tensor([[2.0, 0.0, -2.0]])
    reversed_scores = torch.tensor([[-2.0, 0.0, 2.0]])

    assert rank_loss(aligned, target) < rank_loss(reversed_scores, target)


def test_hard_negative_mask_keeps_support_and_highest_scoring_negatives():
    labels = np.asarray([[False, True, False, False]])
    scores = np.asarray([[0.8, 0.7, 0.9, 0.1]])

    mask = hard_negative_mask(labels, scores, hard_negatives=2)

    assert mask.tolist() == [[True, True, True, False]]


def test_pairwise_feature_plan_is_reproducible_and_excludes_prior_role():
    available = ["q3", "q1", "q2", "q4"]

    first = make_plan(available, n=2, seed=7, excluded={"q4"})
    second = make_plan(list(reversed(available)), n=2, seed=7, excluded={"q4"})

    assert first == second
    assert "q4" not in first


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


def test_chunk_edge_mask_only_changes_selected_head_last_query():
    weights = torch.full((1, 2, 2, 4), 0.25)
    masked = zero_attention_edges(weights, [1], start=1, end=3)

    assert torch.allclose(masked[:, 0], weights[:, 0])
    assert torch.allclose(masked[:, 1, 0], weights[:, 1, 0])
    assert torch.allclose(masked[:, 1, 1], torch.tensor([[0.5, 0.0, 0.0, 0.5]]))
    assert torch.allclose(masked.sum(dim=-1), torch.ones(1, 2, 2))


def test_directer_plausibility_scores_intervened_top_token_under_raw_model():
    raw = torch.tensor([[4.0, 3.0, 0.0], [3.0, 2.0, 0.0]])
    intervened = torch.tensor([[2.0, 5.0, 0.0], [4.0, 1.0, 0.0]])

    result = distribution_plausibility(raw, intervened, beta=0.5)

    assert result.probability_ratios[0] < 0.5
    assert result.probability_ratios[1] == 1.0
    assert result.rejection_rate == 0.5
    assert result.top1_change_rate == 0.5
    assert result.worst_rejection_score > 0


def test_context_sensitivity_is_zero_for_equal_distributions():
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    scores = jensen_shannon_from_logits(logits, logits.clone())
    assert torch.allclose(scores, torch.zeros(2), atol=1e-7)


def test_context_sensitive_selection_keeps_at_least_one_token():
    assert select_context_sensitive_tokens(torch.zeros(3)) == [0, 1, 2]
    assert select_context_sensitive_tokens(torch.tensor([0.0, 0.1, 0.9])) == [2]


def test_full_topl_validation_plan_keeps_roles_disjoint():
    plan = make_split_plan(
        [f"q{index}" for index in range(12)],
        ["q0", "q1"],
        {"q2", "q3"},
        n_calibration=3,
        n_evaluation=3,
        seed=4,
    )

    by_role = {
        role: {row["qid"] for row in plan if row["role"] == role}
        for role in ("scorer_train", "conformal_calibration", "evaluation")
    }
    assert by_role["scorer_train"] == {"q0", "q1"}
    assert len(by_role["conformal_calibration"]) == 3
    assert len(by_role["evaluation"]) == 3
    assert not by_role["conformal_calibration"] & by_role["evaluation"]
    assert not ({"q2", "q3"} & set().union(*by_role.values()))


def test_full_topl_eligibility_selects_requested_source_split():
    questions = {
        "train": {"metadata": {"source_split": "train"}},
        "dev": {"metadata": {"source_split": "dev"}},
    }
    retrieval = {
        "train": [{"support_label": "support"}],
        "dev": [{"support_label": "support"}],
    }

    assert eligible_qids(questions, retrieval, source_split="dev") == ["dev"]


def test_full_topl_saliency_fusion_uses_query_local_scores():
    values = grouped_z(np.asarray([1.0, 2.0, 3.0]))
    assert np.isclose(values.mean(), 0.0)
    assert np.isclose(values.std(), 1.0)


def test_hidden_probe_metrics_average_over_queries():
    labels = np.asarray([[True, False, False], [False, True, False]])
    scores = np.asarray([[0.9, 0.2, 0.1], [0.8, 0.7, 0.1]])
    metrics = query_metrics(labels, scores)
    assert metrics["mean_query_ap"] == 0.75
    assert metrics["mrr"] == 0.75
    assert metrics["top1_support_rate"] == 0.5


def test_query_conformal_any_support_threshold_uses_best_support():
    scores = np.asarray([[0.8, 0.7, 0.1], [0.6, 0.5, 0.2]])
    labels = np.asarray([[False, True, False], [True, True, False]])

    threshold, order = conformal_threshold(
        scores, labels, alpha=0.5, coverage_target="any_support"
    )

    assert order == 2
    assert threshold == 0.6


def test_end_to_end_coverage_counts_unretrievable_queries_as_failures():
    labels = np.asarray([[True, False], [False, False]])
    mask = keep_mask(np.asarray([[0.9, 0.1], [0.2, 0.1]]), threshold=0.5)

    metrics = end_to_end_metrics(labels, mask)

    assert metrics["retrieval_query_coverage_ceiling"] == 0.5
    assert metrics["query_any_support_coverage"] == 0.5
    assert metrics["query_all_support_coverage"] == 0.5
