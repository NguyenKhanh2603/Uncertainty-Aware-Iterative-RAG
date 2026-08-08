"""U3: Uncertainty Calibration Curve and Expected Calibration Error (ECE).

Measures how well-calibrated the uncertainty estimates are:
  - A well-calibrated system: when SE_total is low → answer is likely correct
  - ECE quantifies the gap between confidence and accuracy across bins

This is a key analysis for the paper — demonstrates the uncertainty signal
is meaningful and actionable, not just a heuristic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class CalibrationBin:
    """A single bin in the calibration analysis."""

    bin_lower: float
    bin_upper: float
    avg_confidence: float
    avg_accuracy: float
    count: int
    gap: float  # |accuracy - confidence|


@dataclass
class CalibrationResult:
    """Full calibration analysis results."""

    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    bins: list[CalibrationBin] = field(default_factory=list)
    # Per-example data for plotting
    confidences: list[float] = field(default_factory=list)
    accuracies: list[float] = field(default_factory=list)
    se_totals: list[float] = field(default_factory=list)
    # Comparison data (optional — for comparing methods)
    method_name: str = ""


class UncertaintyCalibrator:
    """Compute uncertainty calibration metrics and generate plots.

    Analysis:
    1. For each example, record (confidence=1-SE_total, accuracy=EM or F1)
    2. Bin examples by confidence level
    3. Compute ECE = weighted avg |accuracy_bin - confidence_bin|
    4. Generate reliability diagram (calibration plot)

    A perfectly calibrated system: 80% confidence → 80% accuracy.
    """

    def __init__(
        self,
        num_bins: int = 10,
        output_dir: str = "results/calibration",
    ) -> None:
        self.num_bins = num_bins
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def compute_calibration(
        self,
        confidences: list[float],
        accuracies: list[float],
        se_totals: Optional[list[float]] = None,
        method_name: str = "Ours",
    ) -> CalibrationResult:
        """Compute ECE and calibration bins.

        Args:
            confidences: List of confidence scores (1 - SE_total) per example.
            accuracies: List of accuracy scores (EM=0/1 or F1∈[0,1]) per example.
            se_totals: Raw SE_total values (for analysis).
            method_name: Name of the method (for comparison plots).

        Returns:
            CalibrationResult with ECE, MCE, and per-bin data.
        """
        assert len(confidences) == len(accuracies), "confidences and accuracies must match"
        n = len(confidences)

        if n == 0:
            return CalibrationResult(ece=0.0, mce=0.0, method_name=method_name)

        conf_array = np.array(confidences)
        acc_array = np.array(accuracies)

        # Clamp confidences to [0, 1]
        conf_array = np.clip(conf_array, 0.0, 1.0)

        # Bin edges
        bin_edges = np.linspace(0.0, 1.0, self.num_bins + 1)
        bins: list[CalibrationBin] = []
        total_ece = 0.0
        max_gap = 0.0

        for i in range(self.num_bins):
            lower = bin_edges[i]
            upper = bin_edges[i + 1]

            # Find examples in this bin
            if i < self.num_bins - 1:
                mask = (conf_array >= lower) & (conf_array < upper)
            else:
                # Include upper bound for last bin
                mask = (conf_array >= lower) & (conf_array <= upper)

            count = int(mask.sum())
            if count == 0:
                bins.append(CalibrationBin(
                    bin_lower=lower, bin_upper=upper,
                    avg_confidence=0.0, avg_accuracy=0.0,
                    count=0, gap=0.0,
                ))
                continue

            avg_conf = float(conf_array[mask].mean())
            avg_acc = float(acc_array[mask].mean())
            gap = abs(avg_acc - avg_conf)

            bins.append(CalibrationBin(
                bin_lower=lower, bin_upper=upper,
                avg_confidence=avg_conf, avg_accuracy=avg_acc,
                count=count, gap=gap,
            ))

            # Weighted contribution to ECE
            total_ece += (count / n) * gap
            max_gap = max(max_gap, gap)

        return CalibrationResult(
            ece=total_ece,
            mce=max_gap,
            bins=bins,
            confidences=confidences,
            accuracies=accuracies,
            se_totals=se_totals or [],
            method_name=method_name,
        )

    def plot_reliability_diagram(
        self,
        results: list[CalibrationResult],
        output_filename: str = "reliability_diagram.png",
    ) -> Path:
        """Generate a reliability diagram (calibration plot).

        Plots perfect calibration line vs. actual calibration for each method.

        Args:
            results: List of CalibrationResult objects to compare.
            output_filename: Output filename.

        Returns:
            Path to saved plot.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # ── Left: Reliability Diagram ──
        ax = axes[0]
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")

        colors = plt.cm.Set2(np.linspace(0, 1, len(results)))
        for result, color in zip(results, colors):
            bin_mids = [(b.bin_lower + b.bin_upper) / 2 for b in result.bins if b.count > 0]
            bin_accs = [b.avg_accuracy for b in result.bins if b.count > 0]
            label = f"{result.method_name} (ECE={result.ece:.3f})"
            ax.bar(
                bin_mids, bin_accs, width=1.0 / self.num_bins * 0.8,
                alpha=0.6, color=color, label=label, edgecolor="black", linewidth=0.5,
            )

        ax.set_xlabel("Confidence (1 - SE_total)", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title("Reliability Diagram", fontsize=14)
        ax.legend(fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

        # ── Right: SE_total vs Accuracy scatter ──
        ax2 = axes[1]
        for result, color in zip(results, colors):
            if result.se_totals:
                ax2.scatter(
                    result.se_totals, result.accuracies,
                    alpha=0.4, s=20, color=color, label=result.method_name,
                )

        ax2.set_xlabel("SE_total", fontsize=12)
        ax2.set_ylabel("Accuracy", fontsize=12)
        ax2.set_title("Uncertainty vs. Accuracy", fontsize=14)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / output_filename
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        return output_path

    def plot_ece_comparison(
        self,
        results: list[CalibrationResult],
        output_filename: str = "ece_comparison.png",
    ) -> Path:
        """Bar chart comparing ECE across methods."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = [r.method_name for r in results]
        eces = [r.ece for r in results]
        mces = [r.mce for r in results]

        x = np.arange(len(names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar(x - width / 2, eces, width, label="ECE", color="#4ECDC4", edgecolor="black")
        bars2 = ax.bar(x + width / 2, mces, width, label="MCE", color="#FF6B6B", edgecolor="black")

        ax.set_xlabel("Method", fontsize=12)
        ax.set_ylabel("Calibration Error", fontsize=12)
        ax.set_title("Calibration Error Comparison", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, ha="right")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)

        # Add value labels on bars
        for bar in bars1:
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9,
            )
        for bar in bars2:
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9,
            )

        plt.tight_layout()
        output_path = self.output_dir / output_filename
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        return output_path

    def save_results(self, result: CalibrationResult, filename: str = "calibration.json") -> Path:
        """Save calibration results as JSON."""
        output_path = self.output_dir / filename
        data = {
            "method_name": result.method_name,
            "ece": result.ece,
            "mce": result.mce,
            "num_examples": len(result.confidences),
            "bins": [
                {
                    "bin_lower": b.bin_lower,
                    "bin_upper": b.bin_upper,
                    "avg_confidence": round(b.avg_confidence, 4),
                    "avg_accuracy": round(b.avg_accuracy, 4),
                    "count": b.count,
                    "gap": round(b.gap, 4),
                }
                for b in result.bins
            ],
        }
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        return output_path
