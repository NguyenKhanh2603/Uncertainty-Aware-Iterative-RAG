"""Standalone RAGU semantic-entropy implementation.

Derived from related_repos/ragu/semantic_uncertainty/uncertainty/
uncertainty_measures/semantic_entropy.py (BSD 3-Clause).  It deliberately uses
RAGU's length-normalized sequence likelihood and probability-weighted semantic
clusters, rather than this project's frequency-only uncertainty score.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Protocol

import numpy as np


class EntailmentModel(Protocol):
    def check_implication(self, text1: str, text2: str) -> int:
        """Return 0=contradiction, 1=neutral, 2=entailment."""


class DebertaMNLI:
    """The same NLI model and class interpretation used by RAGU."""

    def __init__(self, model_name: str = "microsoft/deberta-v2-xlarge-mnli") -> None:
        import torch
        import torch.nn.functional as functional
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self._functional = functional
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def check_implication(self, text1: str, text2: str) -> int:
        inputs = self.tokenizer(text1, text2, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            logits = self.model(**inputs).logits
        return int(self._torch.argmax(self._functional.softmax(logits, dim=1)).cpu().item())


def get_semantic_ids(strings: list[str], model: EntailmentModel, strict_entailment: bool = True) -> list[int]:
    """Copy of RAGU's greedy bidirectional entailment clustering."""
    semantic_ids = [-1] * len(strings)
    next_id = 0
    for index, first in enumerate(strings):
        if semantic_ids[index] != -1:
            continue
        semantic_ids[index] = next_id
        for other_index in range(index + 1, len(strings)):
            forward = model.check_implication(first, strings[other_index])
            backward = model.check_implication(strings[other_index], first)
            if forward not in (0, 1, 2) or backward not in (0, 1, 2):
                raise ValueError("NLI model must return 0, 1, or 2")
            if strict_entailment:
                equivalent = forward == 2 and backward == 2
            else:
                equivalent = 0 not in (forward, backward) and (forward, backward) != (1, 1)
            if equivalent:
                semantic_ids[other_index] = next_id
        next_id += 1
    return semantic_ids


def predictive_entropy(log_probs: list[float]) -> float:
    """RAGU regular entropy: negative mean length-normalized log likelihood."""
    return float(-np.sum(log_probs) / len(log_probs))


def _logsumexp(values: list[float]) -> float:
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def logsumexp_by_id(semantic_ids: list[int], log_likelihoods: list[float]) -> list[float]:
    """RAGU's normalized probability mass per semantic cluster."""
    normalizer = _logsumexp(log_likelihoods)
    clusters: dict[int, list[float]] = defaultdict(list)
    for semantic_id, likelihood in zip(semantic_ids, log_likelihoods):
        clusters[semantic_id].append(likelihood - normalizer)
    return [_logsumexp(clusters[semantic_id]) for semantic_id in sorted(clusters)]


def predictive_entropy_rao(log_probs: list[float]) -> float:
    return float(-sum(math.exp(log_prob) * log_prob for log_prob in log_probs))


def cluster_assignment_entropy(semantic_ids: list[int]) -> float:
    counts = np.bincount(semantic_ids)
    probabilities = counts / len(semantic_ids)
    return float(-(probabilities * np.log(probabilities)).sum())


def compute_entropies(
    responses: list[str], token_logprobs: list[list[float]], model: EntailmentModel,
    question: str, strict_entailment: bool = True,
) -> dict[str, object]:
    """Compute the exact RAGU regular and semantic entropy formulas."""
    if len(responses) != len(token_logprobs) or len(responses) < 2:
        raise ValueError("At least two responses with matching token log probabilities are required")
    if any(not log_probs for log_probs in token_logprobs):
        raise ValueError("Every sampled response needs at least one output-token log probability")
    conditioned_responses = [f"{question} {response}" for response in responses]
    semantic_ids = get_semantic_ids(conditioned_responses, model, strict_entailment)
    mean_log_likelihoods = [float(np.mean(log_probs)) for log_probs in token_logprobs]
    semantic_log_likelihoods = logsumexp_by_id(semantic_ids, mean_log_likelihoods)
    return {
        "semantic_ids": semantic_ids,
        "mean_token_log_likelihoods": mean_log_likelihoods,
        "regular_entropy": predictive_entropy(mean_log_likelihoods),
        "semantic_entropy": predictive_entropy_rao(semantic_log_likelihoods),
        "cluster_assignment_entropy": cluster_assignment_entropy(semantic_ids),
    }
