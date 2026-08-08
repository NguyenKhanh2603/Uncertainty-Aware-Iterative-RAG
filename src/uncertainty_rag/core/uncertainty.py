"""Semantic Uncertainty Decomposition: Total, Aleatoric, and Epistemic.

Mathematical framework (from Methodology_Paper_Draft.md):
  SE_total(Y|Q,C)    = -Σ P(c) log₂ P(c)              [Predictive Entropy]
  SE_aleatoric(Y|Q,C) = E_θ[-Σ P(c|θ) log₂ P(c|θ)]    [Data Noise]
  SE_epistemic(Y|Q,C) = SE_total - SE_aleatoric         [Knowledge Gap / Mutual Information]

SE_aleatoric is approximated using per-sample key-token entropy (W1 fix),
normalized to be on the same scale as SE_total.
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

    se_total: float
    se_aleatoric: float
    se_epistemic: float
    num_concepts: int
    concept_distribution: list[tuple[int, float]] = field(default_factory=list)
    # Raw values before normalization (for analysis/logging)
    se_aleatoric_raw: float = 0.0
    # Normalization parameters used (for reproducibility)
    norm_min: Optional[float] = None
    norm_max: Optional[float] = None


class UncertaintyEstimator:
    """Compute the three-component semantic uncertainty decomposition.

    The estimator supports a calibrated normalization mode where min/max bounds
    for SE_aleatoric are pre-computed on a dev set, ensuring stable comparison
    with SE_total across different datasets and modalities.
    """

    def __init__(
        self,
        norm_min: Optional[float] = None,
        norm_max: Optional[float] = None,
    ) -> None:
        """Initialize with optional normalization bounds.

        Args:
            norm_min: Minimum observed SE_aleatoric_raw (from dev set calibration).
            norm_max: Maximum observed SE_aleatoric_raw (from dev set calibration).
                      If both are None, uses simple scaling: SE_aleatoric_raw / log₂(V)
                      where V is the avg vocabulary size from top logprobs.
        """
        self.norm_min = norm_min
        self.norm_max = norm_max

    @staticmethod
    def compute_se_total(concepts: list[Concept]) -> float:
        """Compute Semantic Entropy over concept distribution.

        SE_total = H[C] = -Σ P(c) log₂ P(c)

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
    def compute_se_aleatoric_raw(samples: list[Sample]) -> float:
        """Compute raw aleatoric uncertainty from key-token log-probabilities.

        SE_aleatoric_raw = (1/M) Σᵢ H(key_tokens_i)

        where H(key_tokens_i) = -(1/K) Σⱼ log₂ P(token_j)

        This approximates the per-sample predictive entropy using only
        content-bearing tokens (filtering stop words and structural tokens).
        """
        per_sample_entropy = []

        for sample in samples:
            key_logprobs = [t.logprob for t in sample.key_token_logprobs]
            if key_logprobs:
                # Convert from natural log (ln) to log₂ for consistency with SE_total
                # logprobs from API are ln, so we divide by ln(2) to get log₂
                h_i = -sum(lp / log2(2.718281828) for lp in key_logprobs) / len(key_logprobs)
                # Note: lp is negative (log prob), so -lp is positive → h_i is positive
                per_sample_entropy.append(h_i)

        if not per_sample_entropy:
            return 0.0

        return mean(per_sample_entropy)

    def normalize_aleatoric(self, raw_value: float, se_total: float) -> float:
        """Normalize SE_aleatoric_raw to the same scale as SE_total.

        Three modes:
        1. Calibrated: Use pre-computed norm_min/norm_max from dev set.
        2. Clamped: Clamp to [0, SE_total] — ensures SE_epistemic ≥ 0.
        3. Scaling: Divide by estimated max token entropy.
        """
        if self.norm_min is not None and self.norm_max is not None:
            # Calibrated normalization: map [norm_min, norm_max] → [0, SE_total_max]
            if self.norm_max - self.norm_min < 1e-8:
                normalized = 0.0
            else:
                normalized = (raw_value - self.norm_min) / (self.norm_max - self.norm_min)
                # Scale to SE_total range: max possible SE_total = log₂(M)
                normalized = normalized * se_total
        else:
            # Simple clamped scaling: ensure SE_aleatoric ∈ [0, SE_total]
            # Use ratio-based approach: token entropy is typically much larger
            # than semantic entropy, so we scale down proportionally
            if raw_value > 0 and se_total > 0:
                # Heuristic: cap at SE_total (since epistemic must be ≥ 0)
                normalized = min(raw_value, se_total)
            else:
                normalized = 0.0

        # Final safety clamp: SE_aleatoric ∈ [0, SE_total]
        return max(0.0, min(normalized, se_total))

    def compute(self, samples: list[Sample], concepts: list[Concept]) -> UncertaintyProfile:
        """Compute the full three-component uncertainty decomposition.

        Returns an UncertaintyProfile with SE_total, SE_aleatoric, and SE_epistemic.
        SE_epistemic = SE_total - SE_aleatoric (Mutual Information).
        """
        se_total = self.compute_se_total(concepts)
        se_aleatoric_raw = self.compute_se_aleatoric_raw(samples)
        se_aleatoric = self.normalize_aleatoric(se_aleatoric_raw, se_total)
        se_epistemic = max(0.0, se_total - se_aleatoric)

        concept_dist = [(c.id, c.probability) for c in concepts]

        return UncertaintyProfile(
            se_total=se_total,
            se_aleatoric=se_aleatoric,
            se_epistemic=se_epistemic,
            num_concepts=len(concepts),
            concept_distribution=concept_dist,
            se_aleatoric_raw=se_aleatoric_raw,
            norm_min=self.norm_min,
            norm_max=self.norm_max,
        )

    def calibrate_from_dev(self, profiles: list[UncertaintyProfile]) -> None:
        """Calibrate normalization bounds from dev set profiles.

        Call this after running the pipeline on a dev set to set norm_min/norm_max.
        """
        raw_values = [p.se_aleatoric_raw for p in profiles if p.se_aleatoric_raw > 0]
        if raw_values:
            self.norm_min = float(np.percentile(raw_values, 5))
            self.norm_max = float(np.percentile(raw_values, 95))
