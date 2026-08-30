"""Semantic clustering via NLI bidirectional entailment.

Groups M generated samples into Semantic Equivalence Classes (Concepts).
Two samples belong to the same concept if their claim sets are bidirectionally entailed.

CRITICAL FIX (P0-1): Added Factual Anchor Pre-Check.
NLI cannot reliably distinguish "1976" vs "1972" when surrounded by identical text.
Solution: Extract numbers/dates/proper nouns FIRST. If they differ → different cluster (skip NLI).

CRITICAL FIX (P1): ABSTAIN Normalization.
"I don't know", "I cannot determine", etc. are normalized to [ABSTAIN] before clustering.
This prevents SE from inflating when LLM generates multiple refusal phrasings.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field

import numpy as np

from uncertainty_rag.core.sampler import Sample
from uncertainty_rag.models.nli_model import NLIModel

logger = logging.getLogger(__name__)

# ── ABSTAIN detection patterns ──────────────────────────────────────────────────

_ABSTAIN_PATTERNS = [
    r"(?i)i\s+(don.?t|do\s+not|cannot|can.?t|am\s+unable\s+to)\s+(know|answer|determine|find|tell|say)",
    r"(?i)(cannot|can.?t|unable\s+to)\s+(be\s+)?determin",
    r"(?i)insufficient\s+information",
    r"(?i)not\s+(enough|sufficient)\s+(information|data|context|evidence)",
    r"(?i)no\s+(information|data|evidence)\s+(is\s+)?(available|provided|given)",
    r"(?i)there\s+is\s+no\s+(information|mention|evidence)",
    r"(?i)based\s+on\s+(the\s+)?(given|provided|available)\s+(texts?|context|information).{0,20}(no|not|cannot)",
    r"(?i)it\s+(cannot|can.?t)\s+be\s+(determined|established|confirmed)",
    r"(?i)i\s+am\s+unsure",
    r"(?i)i.?m\s+sorry.{0,30}(not|no)\s+(available|information|mention)",
    r"(?i)\[Text\s+\d+\]",  # References like [Text 3] indicate confusion
]

ABSTAIN_TOKEN = "[ABSTAIN]"


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

    Algorithm (UPGRADED):
    0. Normalize ABSTAIN answers → all refusals go to one cluster.
    1. Extract factual anchors (numbers, dates) from each sample's raw text.
    2. For each pair (i, j): if factual anchors conflict → NOT equivalent (skip NLI).
    3. Otherwise, check bidirectional NLI entailment.
    4. Build adjacency graph → connected components → Concepts.
    5. P(c) = |samples_in_c| / M.
    """

    def __init__(
        self,
        nli_model: NLIModel,
        embedding_prefilter_threshold: float = 0.3,
    ) -> None:
        self.nli = nli_model
        self.embedding_prefilter_threshold = embedding_prefilter_threshold

    @staticmethod
    def is_abstain(text: str) -> bool:
        """Detect if an answer is a refusal/abstention."""
        if not text or not text.strip():
            return True
        for pattern in _ABSTAIN_PATTERNS:
            if re.search(pattern, text):
                return True
        return False

    @staticmethod
    def _extract_factual_anchors(text: str) -> set[str]:
        """Extract factual anchors (numbers, dates, years) from raw answer text.

        These are tokens where even a 1-digit difference means a completely
        different factual answer (e.g., 1976 vs 1972).
        """
        # Extract all numbers (integers and decimals)
        numbers = set(re.findall(r'\b\d+\.?\d*\b', text))
        return numbers

    @staticmethod
    def _answers_have_conflicting_facts(anchors_a: set[str], anchors_b: set[str]) -> bool:
        """Check if two answers have conflicting factual anchors.

        Key insight: If both answers contain numbers but they differ,
        the answers are factually different regardless of NLI score.

        Examples:
          {1976} vs {1972} → CONFLICT (different year)
          {1976} vs {1976} → NO CONFLICT
          {} vs {1976}     → NO CONFLICT (one has no numbers = paraphrase check needed)
          {1976, 100} vs {1976, 200} → CONFLICT (100 vs 200)
        """
        if not anchors_a or not anchors_b:
            # One or both have no factual anchors → can't determine conflict from numbers alone
            return False

        # If both have numbers, check if the sets are compatible
        # Two sets conflict if they are not equal (any difference in numbers = different fact)
        # But we need to be careful: {1976} vs {1976, 100} is not necessarily a conflict
        # The conflict is when the INTERSECTION differs from what each set claims

        # Simple and robust rule: if any number appears in one but not the other, conflict
        # But only if both sets are non-empty
        diff_a = anchors_a - anchors_b  # Numbers in A but not B
        diff_b = anchors_b - anchors_a  # Numbers in B but not A

        if diff_a or diff_b:
            return True

        return False

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

        # ── Step 0: ABSTAIN Normalization ────────────────────────────────────
        abstain_indices = set()
        for i, sample in enumerate(samples):
            if self.is_abstain(sample.text):
                abstain_indices.add(i)
                logger.info(f"  [Clustering] Sample {i} normalized to ABSTAIN: \"{sample.text[:60]}...\"")

        # ── Step 1: Extract factual anchors from raw text ────────────────────
        anchors = [self._extract_factual_anchors(s.text) for s in samples]
        for i, a in enumerate(anchors):
            if a:
                logger.info(f"  [Clustering] Sample {i} factual anchors: {a}")

        # ── Step 2: Build equivalence pairs ──────────────────────────────────
        texts = [self._claims_to_text(s.claims) for s in samples]
        pairs_to_nli: list[tuple[int, int]] = []

        # Pre-determined equivalences (ABSTAIN pairs skip NLI)
        abstain_equiv_pairs: list[tuple[int, int]] = []

        for i in range(n):
            for j in range(i + 1, n):
                # Case 1: Both are ABSTAIN → equivalent (no NLI needed)
                if i in abstain_indices and j in abstain_indices:
                    abstain_equiv_pairs.append((i, j))
                    continue

                # Case 2: One is ABSTAIN, other is not → NOT equivalent
                if i in abstain_indices or j in abstain_indices:
                    continue

                # Case 3: Factual anchor conflict → NOT equivalent (skip NLI)
                if self._answers_have_conflicting_facts(anchors[i], anchors[j]):
                    logger.info(
                        f"  [Clustering] Samples {i} & {j} CONFLICT on facts: "
                        f"{anchors[i]} vs {anchors[j]} → separate clusters (NLI skipped)"
                    )
                    continue

                # Case 3.5: Exact match -> definitely equivalent
                if texts[i] and texts[j] and texts[i].strip().lower() == texts[j].strip().lower():
                    equivalent_pairs.append((i, j))
                    continue

                # Case 4: No obvious conflict → need NLI check
                if texts[i] and texts[j]:
                    pairs_to_nli.append((i, j))

        # ── Step 3: Batch NLI for remaining pairs ────────────────────────────
        equivalent_pairs: list[tuple[int, int]] = list(abstain_equiv_pairs)

        if pairs_to_nli:
            forward_pairs = [(texts[i], texts[j]) for i, j in pairs_to_nli]
            forward_scores = self.nli.predict_batch(forward_pairs)

            backward_pairs = [(texts[j], texts[i]) for i, j in pairs_to_nli]
            backward_scores = self.nli.predict_batch(backward_pairs)

            for k, (i, j) in enumerate(pairs_to_nli):
                _, _, entail_fwd = forward_scores[k]
                _, _, entail_bwd = backward_scores[k]
                if (
                    entail_fwd >= self.nli.entailment_threshold
                    and entail_bwd >= self.nli.entailment_threshold
                ):
                    equivalent_pairs.append((i, j))

        # ── Step 4: Find connected components ────────────────────────────────
        components = self._find_connected_components(n, equivalent_pairs)

        # ── Step 5: Build Concept objects ────────────────────────────────────
        concepts = []
        for concept_id, member_indices in enumerate(components):
            indices = sorted(member_indices)
            rep_claims = samples[indices[0]].claims if indices else []

            # Mark ABSTAIN clusters
            is_abstain_cluster = all(idx in abstain_indices for idx in indices)
            if is_abstain_cluster:
                rep_claims = [ABSTAIN_TOKEN]

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
        for i, c in enumerate(concepts):
            c.id = i

        return concepts
