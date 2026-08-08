"""QA evaluation metrics: EM, Token F1, ROUGE-L, Numerical Accuracy."""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Union


def normalize_answer(answer: str) -> str:
    """Lower-case, strip articles, punctuation, and extra whitespace."""
    answer = answer.lower()
    # Remove articles
    answer = re.sub(r"\b(a|an|the)\b", " ", answer)
    # Remove punctuation
    answer = answer.translate(str.maketrans("", "", string.punctuation))
    # Collapse whitespace
    answer = " ".join(answer.split())
    return answer.strip()


def exact_match(prediction: str, gold_answers: list[str]) -> float:
    """Exact match: 1.0 if normalized prediction matches any gold answer."""
    pred_norm = normalize_answer(prediction)
    return float(any(normalize_answer(g) == pred_norm for g in gold_answers))


def token_f1(prediction: str, gold_answers: list[str]) -> float:
    """Token-level F1 score — max over all gold answers."""
    pred_tokens = normalize_answer(prediction).split()

    if not pred_tokens:
        return 0.0

    best_f1 = 0.0
    for gold in gold_answers:
        gold_tokens = normalize_answer(gold).split()
        if not gold_tokens:
            continue

        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_common = sum(common.values())

        if num_common == 0:
            continue

        precision = num_common / len(pred_tokens)
        recall = num_common / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        best_f1 = max(best_f1, f1)

    return best_f1


def numerical_accuracy(prediction: str, gold_answers: list[str], tolerance: float = 1e-6) -> float:
    """Exact numerical match for TAT-QA arithmetic answers."""
    pred_numbers = _extract_numbers(prediction)
    if not pred_numbers:
        return exact_match(prediction, gold_answers)

    for gold in gold_answers:
        gold_numbers = _extract_numbers(gold)
        if gold_numbers and pred_numbers:
            # Check if any predicted number matches any gold number
            for pn in pred_numbers:
                for gn in gold_numbers:
                    if abs(pn - gn) <= tolerance or (
                        gn != 0 and abs((pn - gn) / gn) <= tolerance
                    ):
                        return 1.0

    return 0.0


def _extract_numbers(text: str) -> list[float]:
    """Extract all numbers from text."""
    # Match integers, decimals, percentages, negatives
    pattern = r"-?\d+\.?\d*%?"
    matches = re.findall(pattern, text)
    numbers = []
    for m in matches:
        try:
            if m.endswith("%"):
                numbers.append(float(m[:-1]) / 100)
            else:
                numbers.append(float(m))
        except ValueError:
            continue
    return numbers


def rouge_l(prediction: str, gold_answers: list[str]) -> float:
    """ROUGE-L F1 score — max over all gold answers."""
    pred_tokens = normalize_answer(prediction).split()
    if not pred_tokens:
        return 0.0

    best_score = 0.0
    for gold in gold_answers:
        gold_tokens = normalize_answer(gold).split()
        if not gold_tokens:
            continue

        lcs_len = _lcs_length(pred_tokens, gold_tokens)
        if lcs_len == 0:
            continue

        precision = lcs_len / len(pred_tokens)
        recall = lcs_len / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        best_score = max(best_score, f1)

    return best_score


def _lcs_length(x: list[str], y: list[str]) -> int:
    """Longest Common Subsequence length."""
    m, n = len(x), len(y)
    # Space-optimized LCS
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


class MetricSuite:
    """Compute all metrics for a set of predictions."""

    def __init__(self, metrics: list[str] | None = None) -> None:
        self.metrics = metrics or ["em", "f1", "rouge_l"]

    def compute(
        self, prediction: str, gold_answers: list[str]
    ) -> dict[str, float]:
        """Compute all requested metrics for a single example."""
        results = {}
        for m in self.metrics:
            if m == "em":
                results["em"] = exact_match(prediction, gold_answers)
            elif m == "f1":
                results["f1"] = token_f1(prediction, gold_answers)
            elif m == "rouge_l":
                results["rouge_l"] = rouge_l(prediction, gold_answers)
            elif m == "numerical_accuracy":
                results["numerical_accuracy"] = numerical_accuracy(prediction, gold_answers)
        return results

    def compute_batch(
        self, predictions: list[str], gold_answers_list: list[list[str]]
    ) -> dict[str, float]:
        """Compute averaged metrics over a batch."""
        all_results: dict[str, list[float]] = {m: [] for m in self.metrics}

        for pred, golds in zip(predictions, gold_answers_list):
            single = self.compute(pred, golds)
            for k, v in single.items():
                all_results[k].append(v)

        return {
            k: sum(v) / len(v) if v else 0.0 for k, v in all_results.items()
        }
