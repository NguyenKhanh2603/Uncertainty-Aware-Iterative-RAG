"""Post-retrieval TRAQ cosine calibration and selection adapter."""
from __future__ import annotations
from typing import Any, Sequence
import numpy as np

def traq_retrieval_threshold(
    per_query: Sequence[np.ndarray], alpha: float
) -> tuple[float, dict[str, Any]]:
    """Mirror TRAQ's Bonferroni retrieval-score threshold.

    TRAQ's retrieval component calibrates a single best true-retrieval score
    per question.  Its Bonferroni baseline splits total alpha equally between
    retrieval and answer prediction sets, hence alpha_R = alpha / 2.
    """

    best_true = np.asarray([scores.max() for scores in per_query if len(scores)], dtype=float)
    if not len(best_true):
        raise ValueError("TRAQ calibration has no retrievable questions")
    alpha_retrieval = alpha / 2.0
    threshold = float(np.quantile(best_true, alpha_retrieval, method="lower"))
    return threshold, {
        "retrievable_calibration_queries": int(len(best_true)),
        "similarity_threshold": threshold,
        "total_alpha": alpha,
        "alpha_retrieval": alpha_retrieval,
        "alpha_answer": alpha_retrieval,
        "calibration_unit": "one_best_true_retrieval_score_per_question",
        "scope": "retrieval_component_only; no_answer_prediction_set",
    }


def select(scores, similarity_threshold):
    """Keep candidates in their supplied order; no reranking or fallback."""
    return np.asarray(scores, dtype=float) >= similarity_threshold
