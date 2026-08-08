"""Uncertainty-based routing: dual-condition stopping with adaptive thresholds (U2).

Routing decisions:
  STOP     — both aleatoric AND epistemic below thresholds → confident answer
  PRUNE    — aleatoric is high → noise in context → remove conflicting chunks
  RETRIEVE — epistemic is high → knowledge gap → fetch new evidence

Priority: PRUNE before RETRIEVE (noise must be resolved first).

Supports:
  - Fixed thresholds: τ_noise and τ_missing are static hyperparameters.
  - Adaptive thresholds (U2): τ computed as fraction of initial uncertainty profile.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from uncertainty_rag.config import ThresholdConfig
from uncertainty_rag.core.uncertainty import UncertaintyProfile


class RoutingDecision(str, Enum):
    """Pipeline routing decision."""

    STOP = "STOP"
    PRUNE = "PRUNE"
    RETRIEVE = "RETRIEVE"


class Router:
    """Decide whether to stop, prune, or retrieve based on uncertainty profile.

    Implements dual-condition stopping (W3 fix):
    - STOP when both SE_aleatoric ≤ τ_noise AND SE_epistemic ≤ τ_missing
    - PRUNE first if noise is high (priority over retrieval)
    - RETRIEVE if knowledge gap is high (noise already handled)

    Supports adaptive thresholds (U2):
    - On the first iteration, record initial uncertainty as baseline
    - τ_noise = alpha * SE_aleatoric_initial
    - τ_missing = beta * SE_epistemic_initial
    """

    def __init__(self, config: ThresholdConfig) -> None:
        self.config = config
        # Effective thresholds (may be overridden by adaptive mode)
        self._tau_noise: float = config.tau_noise
        self._tau_missing: float = config.tau_missing
        self._is_calibrated: bool = (config.mode == "fixed")
        # Store initial profile for logging
        self._initial_profile: Optional[UncertaintyProfile] = None

    @property
    def tau_noise(self) -> float:
        return self._tau_noise

    @property
    def tau_missing(self) -> float:
        return self._tau_missing

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated

    def calibrate_adaptive(self, initial_profile: UncertaintyProfile) -> None:
        """U2: Calibrate adaptive thresholds from the first iteration's uncertainty profile.

        Called once at iteration 0. After calibration, thresholds are fixed for the
        remainder of the pipeline run.
        """
        if self.config.mode != "adaptive":
            return

        self._initial_profile = initial_profile
        self._tau_noise, self._tau_missing = self.config.compute_adaptive_thresholds(
            initial_se_aleatoric=initial_profile.se_aleatoric,
            initial_se_epistemic=initial_profile.se_epistemic,
        )
        self._is_calibrated = True

    def decide(self, profile: UncertaintyProfile) -> RoutingDecision:
        """Make a routing decision based on the current uncertainty profile.

        Args:
            profile: Current iteration's UncertaintyProfile.

        Returns:
            RoutingDecision: STOP, PRUNE, or RETRIEVE.
        """
        # STOP: both noise and knowledge gap are below thresholds
        if profile.se_aleatoric <= self._tau_noise and profile.se_epistemic <= self._tau_missing:
            return RoutingDecision.STOP

        # PRUNE first if noise is high (priority over retrieval)
        # Information Theory: noise must be resolved before we can meaningfully
        # evaluate whether knowledge is missing
        if profile.se_aleatoric > self._tau_noise:
            return RoutingDecision.PRUNE

        # RETRIEVE if knowledge gap is high (noise already handled)
        if profile.se_epistemic > self._tau_missing:
            return RoutingDecision.RETRIEVE

        # Fallback safety
        return RoutingDecision.STOP

    def get_decision_rationale(self, profile: UncertaintyProfile) -> dict:
        """Return a human-readable rationale for the routing decision."""
        decision = self.decide(profile)
        return {
            "decision": decision.value,
            "se_aleatoric": round(profile.se_aleatoric, 4),
            "se_epistemic": round(profile.se_epistemic, 4),
            "se_total": round(profile.se_total, 4),
            "tau_noise": round(self._tau_noise, 4),
            "tau_missing": round(self._tau_missing, 4),
            "threshold_mode": self.config.mode,
            "reason": self._explain(decision, profile),
        }

    def _explain(self, decision: RoutingDecision, profile: UncertaintyProfile) -> str:
        if decision == RoutingDecision.STOP:
            return (
                f"Both aleatoric ({profile.se_aleatoric:.4f} ≤ {self._tau_noise:.4f}) "
                f"and epistemic ({profile.se_epistemic:.4f} ≤ {self._tau_missing:.4f}) "
                f"are below thresholds → confident answer."
            )
        elif decision == RoutingDecision.PRUNE:
            return (
                f"Aleatoric uncertainty ({profile.se_aleatoric:.4f}) exceeds "
                f"τ_noise ({self._tau_noise:.4f}) → context has noise/conflicts, "
                f"pruning required before evaluating knowledge gaps."
            )
        else:
            return (
                f"Epistemic uncertainty ({profile.se_epistemic:.4f}) exceeds "
                f"τ_missing ({self._tau_missing:.4f}) → knowledge gap detected, "
                f"retrieving new evidence."
            )
