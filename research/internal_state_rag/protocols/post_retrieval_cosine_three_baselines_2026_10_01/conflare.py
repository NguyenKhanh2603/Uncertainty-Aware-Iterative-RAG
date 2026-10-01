"""Post-retrieval CONFLARE cosine calibration and selection adapter."""
from __future__ import annotations
from typing import Any, Sequence
import numpy as np

def conflare_threshold(per_query: Sequence[np.ndarray], alpha: float) -> tuple[float, dict[str, Any]]:
    """Mirror CONFLARE's one-record-per-question distance calibration."""

    # The official evaluator walks a ranked corpus and stops at the first
    # relevant chunk.  With frozen labels, that is the support with maximum
    # similarity / minimum cosine distance for each retrievable query.
    first_relevant_similarity = np.asarray(
        [scores.max() for scores in per_query if len(scores)], dtype=float
    )
    if not len(first_relevant_similarity):
        raise ValueError("CONFLARE calibration has no retrievable questions")
    distances = 1.0 - first_relevant_similarity
    distance_threshold = float(np.percentile(distances, 100.0 * (1.0 - alpha)))
    return 1.0 - distance_threshold, {
        "calibration_records": int(len(first_relevant_similarity)),
        "distance_threshold": distance_threshold,
        "similarity_threshold": 1.0 - distance_threshold,
        "calibration_unit": "one_first_known_relevant_chunk_per_question",
        "question_source": "benchmark_human_question_with_labelled_support",
    }


def select(scores, similarity_threshold):
    """Keep candidates in their supplied order; no reranking or fallback."""
    return np.asarray(scores, dtype=float) > similarity_threshold
