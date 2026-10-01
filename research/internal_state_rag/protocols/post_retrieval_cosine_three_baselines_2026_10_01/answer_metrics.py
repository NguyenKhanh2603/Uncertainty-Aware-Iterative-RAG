"""Shared answer metrics copied from eval/metrics.py at fa6dd68."""
import re
import string
from collections import Counter

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
