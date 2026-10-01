"""Post-retrieval CCE cosine calibration and selection adapter."""
from __future__ import annotations
from typing import Any, Sequence
import numpy as np

def cce_embedding_threshold(per_query: Sequence[np.ndarray], alpha: float) -> tuple[float, dict[str, Any]]:
    """Apply CCE's positive-pair nonconformity quantile on raw Jina cosine."""

    positive = np.concatenate([scores for scores in per_query if len(scores)])
    if not len(positive):
        raise ValueError("CCE calibration has no retrieved positive pairs")
    nonconformity = 1.0 - positive
    # CCE retains A(q,c) <= tau. ``higher`` is the conservative empirical
    # quantile corresponding to the finite calibration bank.
    tau = float(np.quantile(nonconformity, 1.0 - alpha, method="higher"))
    return 1.0 - tau, {
        "positive_pairs": int(len(positive)),
        "nonconformity_threshold": tau,
        "similarity_threshold": 1.0 - tau,
        "calibration_unit": "labelled_relevant_question_chunk_pair",
    }


def select(scores, similarity_threshold):
    """Keep candidates in their supplied order; no reranking or fallback."""
    return np.asarray(scores, dtype=float) >= similarity_threshold
