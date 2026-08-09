"""Two-Signal Independent Routing for Iterative RAG.

Routing decisions:
  RETRIEVE — semantic entropy is high → model is semantically confused, fetch new evidence
  PRUNE    — semantic entropy is low BUT token uncertainty is high → noise in context
  STOP     — both semantic entropy and token uncertainty below thresholds → confident answer

Supports:
  - Fixed thresholds: τ_token and τ_semantic are static hyperparameters.
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
    """Decide whether to stop, prune, or retrieve based on independent signals.

    Implements independent signal routing:
    - RETRIEVE if SE_semantic > τ_semantic
    - PRUNE if SE_semantic ≤ τ_semantic AND U_token > τ_token
    - STOP if both are below their respective thresholds

    Supports adaptive thresholds (U2):
    - On the first iteration, record initial uncertainty as baseline
    - τ_token = alpha * U_token_initial
    - τ_semantic = beta * SE_semantic_initial
    """

    def __init__(self, config: ThresholdConfig) -> None:
        self.config = config
        # Effective thresholds (may be overridden by adaptive mode)
        self._tau_token: float = config.tau_token
        self._tau_semantic: float = config.tau_semantic
        self._is_calibrated: bool = (config.mode == "fixed")
        # Store initial profile for logging
        self._initial_profile: Optional[UncertaintyProfile] = None

    @property
    def tau_token(self) -> float:
        return self._tau_token

    @property
    def tau_semantic(self) -> float:
        return self._tau_semantic

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
        self._tau_token, self._tau_semantic = self.config.compute_adaptive_thresholds(
            initial_u_token=initial_profile.u_token,
            initial_se_semantic=initial_profile.se_semantic,
        )
        self._is_calibrated = True

    def decide(self, profile: UncertaintyProfile) -> RoutingDecision:
        """Make a routing decision based on the current uncertainty profile.

        Args:
            profile: Current iteration's UncertaintyProfile.

        Returns:
            RoutingDecision: STOP, PRUNE, or RETRIEVE.
        """
        # RETRIEVE: Semantic confusion is high -> fetch external knowledge
        if profile.se_semantic > self._tau_semantic:
            return RoutingDecision.RETRIEVE

        # PRUNE: Semantic confusion is low (agrees on meaning) BUT token uncertainty is high (noisy context)
        if profile.u_token > self._tau_token:
            return RoutingDecision.PRUNE

        # STOP: Both are low
        return RoutingDecision.STOP

        # Fallback safety
        return RoutingDecision.STOP

    def get_decision_rationale(self, profile: UncertaintyProfile) -> dict:
        """Return a human-readable rationale for the routing decision."""
        decision = self.decide(profile)
        return {
            "decision": decision.value,
            "se_semantic": round(profile.se_semantic, 4),
            "u_token": round(profile.u_token, 4),
            "tau_token": round(self._tau_token, 4),
            "tau_semantic": round(self._tau_semantic, 4),
            "threshold_mode": self.config.mode,
            "reason": self._explain(decision, profile),
        }

    def _explain(self, decision: RoutingDecision, profile: UncertaintyProfile) -> str:
        if decision == RoutingDecision.STOP:
            return (
                f"Both Semantic Entropy ({profile.se_semantic:.4f} ≤ {self._tau_semantic:.4f}) "
                f"and Token Uncertainty ({profile.u_token:.4f} ≤ {self._tau_token:.4f}) "
                f"are below thresholds → confident answer."
            )
        elif decision == RoutingDecision.PRUNE:
            return (
                f"Semantic Entropy is low ({profile.se_semantic:.4f} ≤ {self._tau_semantic:.4f}) but "
                f"Token Uncertainty ({profile.u_token:.4f}) exceeds τ_token ({self._tau_token:.4f}) "
                f"→ context has noise/conflicts, pruning required."
            )
        else:
            return (
                f"Semantic Entropy ({profile.se_semantic:.4f}) exceeds "
                f"τ_semantic ({self._tau_semantic:.4f}) → model is semantically confused, "
                f"retrieving new evidence."
            )
