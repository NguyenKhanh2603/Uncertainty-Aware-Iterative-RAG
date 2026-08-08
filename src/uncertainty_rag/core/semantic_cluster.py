"""Semantic clustering via NLI bidirectional entailment.

Groups M generated samples into Semantic Equivalence Classes (Concepts).
Two samples belong to the same concept if their claim sets are bidirectionally entailed.
This is the core innovation from Kuhn et al. (2023) adapted for claim-level clustering.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from uncertainty_rag.core.sampler import Sample
from uncertainty_rag.models.nli_model import NLIModel


@dataclass
class Concept:
    """A Semantic Equivalence Class — a cluster of semantically equivalent samples."""

    id: int
    sample_indices: list[int] = field(default_factory=list)
    representative_claims: list[str] = field(default_factory=list)
    probability: float = 0.0  # P(c) = |samples_in_c| / M

    @property
    def size(self) -> int:
        return len(self.sample_indices)


class SemanticClusterer:
    """Cluster samples into Semantic Equivalence Classes (Concepts) using NLI.

    Algorithm:
    1. For each pair of samples (i, j), concatenate their claims and check
       bidirectional entailment using NLI.
    2. Build an adjacency graph where edges connect equivalent samples.
    3. Find connected components → each component is a Concept.
    4. P(c) = |samples_in_c| / M.

    Optimization: Use embedding-based pre-filter to skip NLI on obviously
    dissimilar pairs (cosine similarity < threshold).
    """

    def __init__(
        self,
        nli_model: NLIModel,
        embedding_prefilter_threshold: float = 0.3,
    ) -> None:
        self.nli = nli_model
        self.embedding_prefilter_threshold = embedding_prefilter_threshold

    def _claims_to_text(self, claims: list[str]) -> str:
        """Concatenate claims into a single text for NLI comparison."""
        return " ".join(claims) if claims else ""

    def _find_connected_components(self, n: int, adjacency: list[tuple[int, int]]) -> list[set]:
        """Find connected components using Union-Find."""
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]  # Path compression
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for i, j in adjacency:
            union(i, j)

        components: dict[int, set] = {}
        for i in range(n):
            root = find(i)
            if root not in components:
                components[root] = set()
            components[root].add(i)

        return list(components.values())

    def cluster(self, samples: list[Sample]) -> list[Concept]:
        """Cluster samples into Concepts based on semantic equivalence.

        Args:
            samples: List of Sample objects (must have claims extracted).

        Returns:
            List of Concept objects with probabilities.
        """
        n = len(samples)
        if n == 0:
            return []

        if n == 1:
            return [
                Concept(
                    id=0,
                    sample_indices=[0],
                    representative_claims=samples[0].claims,
                    probability=1.0,
                )
            ]

        # Prepare NLI pairs — check all sample pairs
        pairs_to_check: list[tuple[int, int]] = []
        texts = [self._claims_to_text(s.claims) for s in samples]

        for i in range(n):
            for j in range(i + 1, n):
                if texts[i] and texts[j]:
                    pairs_to_check.append((i, j))

        # Batch NLI for efficiency: check bidirectional entailment
        equivalent_pairs: list[tuple[int, int]] = []

        if pairs_to_check:
            # Forward direction: (i → j)
            forward_pairs = [(texts[i], texts[j]) for i, j in pairs_to_check]
            forward_scores = self.nli.predict_batch(forward_pairs)

            # Backward direction: (j → i)
            backward_pairs = [(texts[j], texts[i]) for i, j in pairs_to_check]
            backward_scores = self.nli.predict_batch(backward_pairs)

            for k, (i, j) in enumerate(pairs_to_check):
                _, _, entail_fwd = forward_scores[k]
                _, _, entail_bwd = backward_scores[k]
                if (
                    entail_fwd >= self.nli.entailment_threshold
                    and entail_bwd >= self.nli.entailment_threshold
                ):
                    equivalent_pairs.append((i, j))

        # Find connected components
        components = self._find_connected_components(n, equivalent_pairs)

        # Build Concept objects
        concepts = []
        for concept_id, member_indices in enumerate(components):
            indices = sorted(member_indices)
            # Representative claims: from the first sample in the cluster
            rep_claims = samples[indices[0]].claims if indices else []
            concepts.append(
                Concept(
                    id=concept_id,
                    sample_indices=indices,
                    representative_claims=rep_claims,
                    probability=len(indices) / n,
                )
            )

        # Sort by probability (descending) for deterministic ordering
        concepts.sort(key=lambda c: c.probability, reverse=True)
        # Re-assign IDs after sorting
        for i, c in enumerate(concepts):
            c.id = i

        return concepts
