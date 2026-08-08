"""Tests for evaluation metrics: EM, F1, ROUGE-L, Numerical Accuracy."""

from __future__ import annotations

import pytest

from eval.metrics import exact_match, token_f1, rouge_l, numerical_accuracy, MetricSuite


class TestExactMatch:
    def test_exact_match(self):
        assert exact_match("Paris", ["Paris"]) == 1.0

    def test_case_insensitive(self):
        assert exact_match("PARIS", ["paris"]) == 1.0

    def test_article_removal(self):
        assert exact_match("the Eiffel Tower", ["Eiffel Tower"]) == 1.0

    def test_multiple_gold(self):
        assert exact_match("NYC", ["New York City", "NYC", "New York"]) == 1.0

    def test_no_match(self):
        assert exact_match("London", ["Paris"]) == 0.0


class TestTokenF1:
    def test_perfect_match(self):
        assert token_f1("the capital of France is Paris", ["the capital of France is Paris"]) == 1.0

    def test_partial_overlap(self):
        f1 = token_f1("Paris is the capital", ["Paris is the capital of France"])
        assert 0.5 < f1 < 1.0

    def test_no_overlap(self):
        assert token_f1("London", ["Tokyo"]) == 0.0


class TestNumericalAccuracy:
    def test_exact_number(self):
        assert numerical_accuracy("42", ["42"]) == 1.0

    def test_decimal(self):
        assert numerical_accuracy("3.14", ["3.14"]) == 1.0

    def test_percentage(self):
        assert numerical_accuracy("50%", ["0.5"]) == 1.0

    def test_in_text(self):
        assert numerical_accuracy("The answer is 42 million", ["42"]) == 1.0


class TestROUGEL:
    def test_perfect(self):
        assert rouge_l("the quick brown fox", ["the quick brown fox"]) == 1.0

    def test_partial(self):
        r = rouge_l("the brown fox", ["the quick brown fox jumps"])
        assert 0.3 < r < 1.0

    def test_no_overlap(self):
        assert rouge_l("hello world", ["foo bar baz"]) == 0.0


class TestCalibration:
    """U3: Test ECE computation."""

    def test_perfect_calibration(self):
        from eval.analysis.calibration import UncertaintyCalibrator

        calibrator = UncertaintyCalibrator(num_bins=5)
        # Perfect calibration: 100% confidence → 100% accuracy, 0% → 0%
        confidences = [1.0] * 50 + [0.0] * 50
        accuracies = [1.0] * 50 + [0.0] * 50
        result = calibrator.compute_calibration(confidences, accuracies)
        assert result.ece < 0.1  # Should be near-zero

    def test_overconfident(self):
        from eval.analysis.calibration import UncertaintyCalibrator

        calibrator = UncertaintyCalibrator(num_bins=5)
        # Overconfident: 90% confidence but only 50% accuracy
        confidences = [0.9] * 100
        accuracies = [1.0] * 50 + [0.0] * 50
        result = calibrator.compute_calibration(confidences, accuracies)
        assert result.ece > 0.3  # High ECE = poorly calibrated
