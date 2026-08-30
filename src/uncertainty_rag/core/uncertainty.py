"""Semantic Uncertainty and Token Uncertainty Decomposition.

Mathematical framework:
  SE_semantic(Y|Q,C) = -Σ P(c) log₂ P(c)              [Predictive Entropy]
  U_token(Y|Q,C)     = E_θ[-Σ P(y_j|θ) log₂ P(y_j|θ)] [Token-level variability]

Third signal: evidence_ratio (computed by EvidenceChecker).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import log2
from statistics import mean
from typing import Optional

import numpy as np

from uncertainty_rag.core.sampler import Sample
from uncertainty_rag.core.semantic_cluster import Concept


@dataclass
class UncertaintyProfile:
    """Complete uncertainty profile for a set of samples."""

    se_semantic: float
    u_token: float
    num_concepts: int
    concept_distribution: list[tuple[int, float]] = field(default_factory=list)
    # Signal B: Evidence Sufficiency
    evidence_ratio: float = 1.0  # 0.0 = no evidence, 1.0 = fully supported
    unsupported_claims: list[str] = field(default_factory=list)


class UncertaintyEstimator:
    """Compute Semantic Entropy and Token Uncertainty."""

    def __init__(self) -> None:
        pass

    @staticmethod
    def compute_se_semantic(concepts: list[Concept]) -> float:
        """Compute Semantic Entropy over concept distribution.

        SE_semantic = H[C] = -Σ P(c) log₂ P(c)
        """
        if not concepts:
            return 0.0

        se = 0.0
        for c in concepts:
            if c.probability > 0:
                se -= c.probability * log2(c.probability)
        return se

    @staticmethod
    def compute_u_token(samples: list[Sample]) -> float:
        """Compute raw token uncertainty from key-token log-probabilities.

        U_token = (1/M) Σᵢ H(key_tokens_i)
        """
        per_sample_entropy = []

        for sample in samples:
            key_logprobs = [t.logprob for t in sample.key_token_logprobs]
            if key_logprobs:
                h_i = -sum(lp / log2(2.718281828) for lp in key_logprobs) / len(key_logprobs)
                per_sample_entropy.append(h_i)

        if not per_sample_entropy:
            return 0.0

        return mean(per_sample_entropy)

    def compute(
        self,
        samples: list[Sample],
        concepts: list[Concept],
        evidence_ratio: float = 1.0,
        unsupported_claims: Optional[list[str]] = None,
    ) -> UncertaintyProfile:
        """Compute all uncertainty signals."""
        se_semantic = self.compute_se_semantic(concepts)
        u_token = self.compute_u_token(samples)

        concept_dist = [(c.id, c.probability) for c in concepts]

        return UncertaintyProfile(
            se_semantic=se_semantic,
            u_token=u_token,
            num_concepts=len(concepts),
            concept_distribution=concept_dist,
            evidence_ratio=evidence_ratio,
            unsupported_claims=unsupported_claims or [],
        )
