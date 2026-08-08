"""Unit tests for core uncertainty computation."""

from __future__ import annotations

import pytest
from math import log2

from uncertainty_rag.core.uncertainty import UncertaintyEstimator, UncertaintyProfile
from uncertainty_rag.core.semantic_cluster import Concept
from uncertainty_rag.core.sampler import Sample
from uncertainty_rag.models.llm_client import TokenLogprob


class TestSETotal:
    """Test SE_total = H[C] = -Σ P(c) log₂ P(c)."""

    def test_single_concept(self):
        """All samples in one concept → SE_total = 0 (no uncertainty)."""
        concepts = [Concept(id=0, sample_indices=[0, 1, 2], probability=1.0)]
        se = UncertaintyEstimator.compute_se_total(concepts)
        assert se == pytest.approx(0.0, abs=1e-8)

    def test_two_equal_concepts(self):
        """Two equiprobable concepts → SE_total = log₂(2) = 1.0."""
        concepts = [
            Concept(id=0, sample_indices=[0, 1], probability=0.5),
            Concept(id=1, sample_indices=[2, 3], probability=0.5),
        ]
        se = UncertaintyEstimator.compute_se_total(concepts)
        assert se == pytest.approx(1.0, abs=1e-8)

    def test_three_equal_concepts(self):
        """Three equiprobable concepts → SE_total = log₂(3)."""
        concepts = [
            Concept(id=i, sample_indices=[i], probability=1 / 3) for i in range(3)
        ]
        se = UncertaintyEstimator.compute_se_total(concepts)
        assert se == pytest.approx(log2(3), abs=1e-6)

    def test_unequal_concepts(self):
        """Unequal probabilities → known entropy value."""
        concepts = [
            Concept(id=0, sample_indices=[0, 1, 2, 3, 4, 5, 6, 7], probability=0.8),
            Concept(id=1, sample_indices=[8, 9], probability=0.2),
        ]
        se = UncertaintyEstimator.compute_se_total(concepts)
        expected = -(0.8 * log2(0.8) + 0.2 * log2(0.2))
        assert se == pytest.approx(expected, abs=1e-6)

    def test_empty_concepts(self):
        """No concepts → SE_total = 0."""
        se = UncertaintyEstimator.compute_se_total([])
        assert se == 0.0


class TestSEAleatoric:
    """Test SE_aleatoric = (1/M) Σᵢ H(key_tokens_i)."""

    def test_low_entropy_tokens(self):
        """High-confidence tokens → low aleatoric uncertainty."""
        samples = [
            Sample(
                text="Paris",
                key_token_logprobs=[
                    TokenLogprob(token="Paris", logprob=-0.01),
                ],
            ),
            Sample(
                text="Paris",
                key_token_logprobs=[
                    TokenLogprob(token="Paris", logprob=-0.02),
                ],
            ),
        ]
        se_a = UncertaintyEstimator.compute_se_aleatoric_raw(samples)
        # Should be close to 0 (very confident)
        assert se_a < 0.1

    def test_high_entropy_tokens(self):
        """Low-confidence tokens → high aleatoric uncertainty."""
        samples = [
            Sample(
                text="Maybe Paris",
                key_token_logprobs=[
                    TokenLogprob(token="Maybe", logprob=-3.0),
                    TokenLogprob(token="Paris", logprob=-2.5),
                ],
            ),
        ]
        se_a = UncertaintyEstimator.compute_se_aleatoric_raw(samples)
        assert se_a > 1.0

    def test_no_key_tokens(self):
        """No key tokens → aleatoric = 0."""
        samples = [Sample(text="test", key_token_logprobs=[])]
        se_a = UncertaintyEstimator.compute_se_aleatoric_raw(samples)
        assert se_a == 0.0


class TestSEEpistemic:
    """Test SE_epistemic = SE_total - SE_aleatoric."""

    def test_epistemic_is_nonnegative(self):
        """Epistemic uncertainty must be ≥ 0 (mutual information)."""
        estimator = UncertaintyEstimator()
        concepts = [
            Concept(id=0, sample_indices=[0, 1], probability=0.5),
            Concept(id=1, sample_indices=[2, 3], probability=0.5),
        ]
        samples = [
            Sample(text=f"answer {i}", key_token_logprobs=[
                TokenLogprob(token=f"answer", logprob=-1.0),
            ])
            for i in range(4)
        ]
        profile = estimator.compute(samples, concepts)
        assert profile.se_epistemic >= 0


class TestNormalization:
    """Test SE_aleatoric normalization."""

    def test_calibrated_normalization(self):
        """With calibrated bounds, normalization maps to [0, SE_total]."""
        estimator = UncertaintyEstimator(norm_min=0.5, norm_max=5.0)
        normalized = estimator.normalize_aleatoric(raw_value=2.75, se_total=1.0)
        assert 0.0 <= normalized <= 1.0

    def test_clamped_normalization(self):
        """Without calibration, SE_aleatoric is clamped to [0, SE_total]."""
        estimator = UncertaintyEstimator()  # No calibration
        normalized = estimator.normalize_aleatoric(raw_value=100.0, se_total=1.5)
        assert normalized <= 1.5
        assert normalized >= 0.0


class TestAdaptiveThresholds:
    """U2: Test adaptive threshold computation."""

    def test_adaptive_thresholds_from_initial(self):
        from uncertainty_rag.config import ThresholdConfig

        config = ThresholdConfig(
            mode="adaptive", alpha=0.5, beta=0.5, adaptive_min_tau=0.05
        )
        tau_n, tau_m = config.compute_adaptive_thresholds(
            initial_se_aleatoric=1.0, initial_se_epistemic=0.8
        )
        assert tau_n == pytest.approx(0.5, abs=1e-8)
        assert tau_m == pytest.approx(0.4, abs=1e-8)

    def test_adaptive_floor(self):
        """Adaptive thresholds respect the minimum floor."""
        from uncertainty_rag.config import ThresholdConfig

        config = ThresholdConfig(
            mode="adaptive", alpha=0.5, beta=0.5, adaptive_min_tau=0.2
        )
        tau_n, tau_m = config.compute_adaptive_thresholds(
            initial_se_aleatoric=0.01, initial_se_epistemic=0.01
        )
        assert tau_n == 0.2  # Floored at adaptive_min_tau
        assert tau_m == 0.2
