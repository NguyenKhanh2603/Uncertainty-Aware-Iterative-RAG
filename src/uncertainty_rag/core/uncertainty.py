"""Semantic Uncertainty and Token Uncertainty Decomposition.

Mathematical framework (from Methodology_Workflow.md):
  SE_semantic(Y|Q,C) = -Σ P(c) log₂ P(c)              [Predictive Entropy]
  U_token(Y|Q,C)     = E_θ[-Σ P(y_j|θ) log₂ P(y_j|θ)] [Data Noise]

These are treated as independent signals for routing.
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


class UncertaintyEstimator:
    """Compute Semantic Entropy and Token Uncertainty.
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def compute_se_semantic(concepts: list[Concept]) -> float:
        """Compute Semantic Entropy over concept distribution.

        SE_semantic = H[C] = -Σ P(c) log₂ P(c)

        This measures the system's overall ambiguity about the answer.
        Range: [0, log₂(M)] where M is the number of samples.
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

        where H(key_tokens_i) = -(1/K) Σⱼ log₂ P(token_j)

        This approximates the per-sample predictive entropy using only
        content-bearing tokens (filtering stop words and structural tokens).
        """
        per_sample_entropy = []

        for sample in samples:
            key_logprobs = [t.logprob for t in sample.key_token_logprobs]
            if key_logprobs:
                # Convert from natural log (ln) to log₂ for consistency with SE_semantic
                # logprobs from API are ln, so we divide by ln(2) to get log₂
                h_i = -sum(lp / log2(2.718281828) for lp in key_logprobs) / len(key_logprobs)
                # Note: lp is negative (log prob), so -lp is positive → h_i is positive
                per_sample_entropy.append(h_i)

        if not per_sample_entropy:
            return 0.0

        return mean(per_sample_entropy)

    def compute(self, samples: list[Sample], concepts: list[Concept]) -> UncertaintyProfile:
        """Compute Semantic Entropy and Token Uncertainty independently."""
        se_semantic = self.compute_se_semantic(concepts)
        u_token = self.compute_u_token(samples)

        concept_dist = [(c.id, c.probability) for c in concepts]

        return UncertaintyProfile(
            se_semantic=se_semantic,
            u_token=u_token,
            num_concepts=len(concepts),
            concept_distribution=concept_dist,
        )
