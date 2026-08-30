"""Three-Signal Independent Routing for Iterative RAG.

Routing logic:
  S = Semantic Entropy       (answer disagreement)
  U = Token Uncertainty      (token-level variability)
  E = Evidence Ratio         (claim-context support)

  STOP     ← S ≤ τ_S AND U ≤ τ_U AND E ≥ τ_E
  RETRIEVE ← S > τ_S OR E < τ_E
  PRUNE    ← S ≤ τ_S AND U > τ_U AND E ≥ τ_E

Key insight: SE=0 only means "all answers agree", NOT "answer is correct".
Evidence Ratio checks whether context ACTUALLY SUPPORTS the claims.
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
    """Three-signal routing: SE + U_token + Evidence."""

    def __init__(self, config: ThresholdConfig) -> None:
        self.config = config
        self._tau_token: float = config.tau_token
        self._tau_semantic: float = config.tau_semantic
        self._tau_evidence: float = getattr(config, 'tau_evidence', 0.7)
        self._is_calibrated: bool = (config.mode == "fixed")
        self._initial_profile: Optional[UncertaintyProfile] = None

    @property
    def tau_token(self) -> float:
        return self._tau_token

    @property
    def tau_semantic(self) -> float:
        return self._tau_semantic

    @property
    def tau_evidence(self) -> float:
        return self._tau_evidence

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated

    def calibrate_adaptive(self, initial_profile: UncertaintyProfile) -> None:
        """U2: Calibrate adaptive thresholds from the first iteration."""
        if self.config.mode != "adaptive":
            return
        self._initial_profile = initial_profile
        self._tau_token, self._tau_semantic = self.config.compute_adaptive_thresholds(
            initial_u_token=initial_profile.u_token,
            initial_se_semantic=initial_profile.se_semantic,
        )
        self._is_calibrated = True

    def decide(self, profile: UncertaintyProfile) -> RoutingDecision:
        """Make routing decision based on 3 independent signals."""
        S = profile.se_semantic
        U = profile.u_token
        E = profile.evidence_ratio

        # RETRIEVE: semantic confusion OR evidence insufficient
        if S > self._tau_semantic:
            return RoutingDecision.RETRIEVE
        if E < self._tau_evidence:
            return RoutingDecision.RETRIEVE

        # PRUNE: semantically agrees, evidence OK, but token noise high
        if U > self._tau_token:
            return RoutingDecision.PRUNE

        # STOP: all three signals satisfactory
        return RoutingDecision.STOP

    def get_decision_rationale(self, profile: UncertaintyProfile) -> dict:
        """Return human-readable rationale."""
        decision = self.decide(profile)
        return {
            "decision": decision.value,
            "se_semantic": round(profile.se_semantic, 4),
            "u_token": round(profile.u_token, 4),
            "evidence_ratio": round(profile.evidence_ratio, 4),
            "tau_token": round(self._tau_token, 4),
            "tau_semantic": round(self._tau_semantic, 4),
            "tau_evidence": round(self._tau_evidence, 4),
            "threshold_mode": self.config.mode,
            "reason": self._explain(decision, profile),
        }

    def _explain(self, decision: RoutingDecision, profile: UncertaintyProfile) -> str:
        S, U, E = profile.se_semantic, profile.u_token, profile.evidence_ratio
        if decision == RoutingDecision.STOP:
            return (
                f"All signals OK: SE({S:.4f}≤{self._tau_semantic:.4f}), "
                f"U_token({U:.4f}≤{self._tau_token:.4f}), "
                f"Evidence({E:.4f}≥{self._tau_evidence:.4f}) → STOP"
            )
        elif decision == RoutingDecision.PRUNE:
            return (
                f"SE low({S:.4f}), Evidence OK({E:.4f}), "
                f"but U_token({U:.4f})>{self._tau_token:.4f} → PRUNE"
            )
        else:
            reasons = []
            if S > self._tau_semantic:
                reasons.append(f"SE({S:.4f})>{self._tau_semantic:.4f}")
            if E < self._tau_evidence:
                reasons.append(f"Evidence({E:.4f})<{self._tau_evidence:.4f}")
            return " AND ".join(reasons) + " → RETRIEVE"
