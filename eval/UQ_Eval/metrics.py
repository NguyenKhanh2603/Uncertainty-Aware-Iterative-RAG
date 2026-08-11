"""Metric helpers copied locally from RAGU's retrieval_qa/metrics.py."""

from __future__ import annotations

import re
import string


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(char for char in text if char not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, ground_truths: list[str]) -> int:
    normalized_prediction = normalize_answer(prediction)
    return int(any(normalized_prediction == normalize_answer(answer) for answer in ground_truths))


def ragqa_match(prediction: str, ground_truths: list[str]) -> int:
    """RAGU ``Acc``: a normalized gold answer is contained in the prediction."""
    normalized_prediction = normalize_answer(prediction)
    return int(any(normalize_answer(answer) in normalized_prediction for answer in ground_truths))
