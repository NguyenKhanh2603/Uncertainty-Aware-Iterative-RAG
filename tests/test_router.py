"""Unit tests for the Router with dual-condition stopping and U2 adaptive thresholds."""

from __future__ import annotations

import pytest

from uncertainty_rag.config import ThresholdConfig
from uncertainty_rag.core.router import Router, RoutingDecision
from uncertainty_rag.core.uncertainty import UncertaintyProfile


def _make_profile(se_total: float, se_aleatoric: float, se_epistemic: float) -> UncertaintyProfile:
    return UncertaintyProfile(
        se_total=se_total,
        se_aleatoric=se_aleatoric,
        se_epistemic=se_epistemic,
        num_concepts=2,
    )


class TestFixedThresholds:
    """Test router with fixed thresholds."""

    def setup_method(self):
        self.config = ThresholdConfig(mode="fixed", tau_noise=0.5, tau_missing=0.3)
        self.router = Router(self.config)

    def test_stop_both_low(self):
        """Both aleatoric and epistemic below thresholds → STOP."""
        profile = _make_profile(se_total=0.5, se_aleatoric=0.3, se_epistemic=0.2)
        assert self.router.decide(profile) == RoutingDecision.STOP

    def test_prune_high_aleatoric(self):
        """High aleatoric → PRUNE (priority over retrieve)."""
        profile = _make_profile(se_total=1.2, se_aleatoric=0.8, se_epistemic=0.4)
        assert self.router.decide(profile) == RoutingDecision.PRUNE

    def test_retrieve_high_epistemic(self):
        """Low aleatoric, high epistemic → RETRIEVE."""
        profile = _make_profile(se_total=0.8, se_aleatoric=0.2, se_epistemic=0.6)
        assert self.router.decide(profile) == RoutingDecision.RETRIEVE

    def test_prune_priority_over_retrieve(self):
        """Both high → PRUNE (noise before knowledge gap)."""
        profile = _make_profile(se_total=1.5, se_aleatoric=0.8, se_epistemic=0.7)
        assert self.router.decide(profile) == RoutingDecision.PRUNE

    def test_stop_at_zero(self):
        """Zero uncertainty → STOP."""
        profile = _make_profile(se_total=0.0, se_aleatoric=0.0, se_epistemic=0.0)
        assert self.router.decide(profile) == RoutingDecision.STOP


class TestAdaptiveThresholds:
    """Test router with U2 adaptive thresholds."""

    def test_calibration_from_initial_profile(self):
        """Adaptive thresholds are computed from the first iteration's profile."""
        config = ThresholdConfig(mode="adaptive", alpha=0.5, beta=0.5, adaptive_min_tau=0.05)
        router = Router(config)

        assert not router.is_calibrated

        initial_profile = _make_profile(se_total=2.0, se_aleatoric=1.0, se_epistemic=1.0)
        router.calibrate_adaptive(initial_profile)

        assert router.is_calibrated
        assert router.tau_noise == pytest.approx(0.5)  # 0.5 * 1.0
        assert router.tau_missing == pytest.approx(0.5)  # 0.5 * 1.0

    def test_adaptive_uses_calibrated_thresholds(self):
        """After calibration, routing uses the adaptive thresholds."""
        config = ThresholdConfig(mode="adaptive", alpha=0.3, beta=0.3, adaptive_min_tau=0.05)
        router = Router(config)

        initial = _make_profile(se_total=2.0, se_aleatoric=1.2, se_epistemic=0.8)
        router.calibrate_adaptive(initial)

        # tau_noise = 0.3 * 1.2 = 0.36, tau_missing = 0.3 * 0.8 = 0.24
        assert router.tau_noise == pytest.approx(0.36)
        assert router.tau_missing == pytest.approx(0.24)

        # Low uncertainty → STOP
        low = _make_profile(se_total=0.4, se_aleatoric=0.2, se_epistemic=0.2)
        assert router.decide(low) == RoutingDecision.STOP

        # High aleatoric → PRUNE
        noisy = _make_profile(se_total=1.0, se_aleatoric=0.5, se_epistemic=0.2)
        assert router.decide(noisy) == RoutingDecision.PRUNE


class TestDecisionRationale:
    """Test that decision rationale is informative."""

    def test_rationale_includes_thresholds(self):
        config = ThresholdConfig(mode="fixed", tau_noise=0.5, tau_missing=0.3)
        router = Router(config)
        profile = _make_profile(se_total=1.0, se_aleatoric=0.8, se_epistemic=0.2)
        rationale = router.get_decision_rationale(profile)

        assert rationale["decision"] == "PRUNE"
        assert rationale["tau_noise"] == 0.5
        assert "exceeds" in rationale["reason"]
