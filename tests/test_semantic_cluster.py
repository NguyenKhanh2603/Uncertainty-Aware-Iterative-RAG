"""Unit tests for SemanticClusterer — NLI-based concept clustering."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from uncertainty_rag.core.semantic_cluster import SemanticClusterer, Concept
from uncertainty_rag.core.sampler import Sample
from uncertainty_rag.models.nli_model import NLIModel


class FakeNLIModel:
    """Fake NLI model returning pre-configured scores."""

    def __init__(self, entailment_threshold: float = 0.5) -> None:
        self.entailment_threshold = entailment_threshold
        self._pair_scores: dict[tuple[str, str], tuple[float, float, float]] = {}

    def set_pair_score(self, a: str, b: str, scores: tuple[float, float, float]):
        """Set (contradiction, neutral, entailment) for a pair."""
        self._pair_scores[(a, b)] = scores

    def predict(self, premise: str, hypothesis: str) -> tuple[float, float, float]:
        key = (premise, hypothesis)
        if key in self._pair_scores:
            return self._pair_scores[key]
        return (0.1, 0.8, 0.1)  # Default: neutral

    def predict_batch(
        self, pairs: list[tuple[str, str]], batch_size: int = 32
    ) -> list[tuple[float, float, float]]:
        return [self.predict(p, h) for p, h in pairs]

    def bidirectional_entailment(self, a: str, b: str) -> bool:
        _, _, e_ab = self.predict(a, b)
        _, _, e_ba = self.predict(b, a)
        return e_ab >= self.entailment_threshold and e_ba >= self.entailment_threshold


class TestSingleSample:
    def test_single_sample_one_concept(self):
        """One sample → one concept with P(c) = 1.0."""
        nli = FakeNLIModel()
        clusterer = SemanticClusterer(nli_model=nli)
        samples = [Sample(text="Paris", claims=["Paris is the capital"])]
        concepts = clusterer.cluster(samples)
        assert len(concepts) == 1
        assert concepts[0].probability == 1.0
        assert concepts[0].sample_indices == [0]


class TestMultipleSamples:
    def test_all_equivalent(self):
        """All samples equivalent → one concept."""
        nli = FakeNLIModel()
        # All pairs entail each other
        nli._pair_scores = {}
        for a in ["claim A", "claim B", "claim C"]:
            for b in ["claim A", "claim B", "claim C"]:
                nli._pair_scores[(a, b)] = (0.0, 0.1, 0.9)

        clusterer = SemanticClusterer(nli_model=nli)
        samples = [
            Sample(text="A", claims=["claim A"]),
            Sample(text="B", claims=["claim B"]),
            Sample(text="C", claims=["claim C"]),
        ]
        concepts = clusterer.cluster(samples)
        assert len(concepts) == 1
        assert concepts[0].probability == pytest.approx(1.0)

    def test_all_different(self):
        """No pairs equivalent → M concepts (one per sample)."""
        nli = FakeNLIModel()
        # Default is neutral (no entailment)
        clusterer = SemanticClusterer(nli_model=nli)
        samples = [
            Sample(text="Paris", claims=["Paris is capital"]),
            Sample(text="London", claims=["London is capital"]),
            Sample(text="Berlin", claims=["Berlin is capital"]),
        ]
        concepts = clusterer.cluster(samples)
        assert len(concepts) == 3
        for c in concepts:
            assert c.probability == pytest.approx(1 / 3, abs=1e-6)

    def test_two_clusters(self):
        """Two groups of equivalent samples → two concepts."""
        nli = FakeNLIModel()
        # Samples 0,1 entail each other; sample 2 is different
        for a, b in [
            ("claim Paris", "claim Paris too"),
            ("claim Paris too", "claim Paris"),
        ]:
            nli._pair_scores[(a, b)] = (0.0, 0.1, 0.9)

        clusterer = SemanticClusterer(nli_model=nli)
        samples = [
            Sample(text="A", claims=["claim Paris"]),
            Sample(text="B", claims=["claim Paris too"]),
            Sample(text="C", claims=["claim London"]),
        ]
        concepts = clusterer.cluster(samples)
        assert len(concepts) == 2
        # Larger cluster should have P=2/3
        probs = sorted([c.probability for c in concepts], reverse=True)
        assert probs[0] == pytest.approx(2 / 3, abs=1e-6)
        assert probs[1] == pytest.approx(1 / 3, abs=1e-6)


class TestConnectedComponents:
    def test_transitive_clustering(self):
        """A≡B and B≡C → A, B, C in same concept (transitivity via Union-Find)."""
        nli = FakeNLIModel()
        # A↔B and B↔C (but not A↔C directly)
        for a, b in [("A", "B"), ("B", "A"), ("B", "C"), ("C", "B")]:
            nli._pair_scores[(a, b)] = (0.0, 0.1, 0.9)

        clusterer = SemanticClusterer(nli_model=nli)
        samples = [
            Sample(text="x", claims=["A"]),
            Sample(text="y", claims=["B"]),
            Sample(text="z", claims=["C"]),
        ]
        concepts = clusterer.cluster(samples)
        # All three should be in one concept due to transitivity
        assert len(concepts) == 1


class TestEmptyInput:
    def test_empty_samples(self):
        nli = FakeNLIModel()
        clusterer = SemanticClusterer(nli_model=nli)
        concepts = clusterer.cluster([])
        assert concepts == []

    def test_empty_claims(self):
        """Samples with no claims → separate concepts."""
        nli = FakeNLIModel()
        clusterer = SemanticClusterer(nli_model=nli)
        samples = [
            Sample(text="A", claims=[]),
            Sample(text="B", claims=[]),
        ]
        concepts = clusterer.cluster(samples)
        # Empty claims → no text for NLI → each sample is its own concept
        assert len(concepts) == 2
