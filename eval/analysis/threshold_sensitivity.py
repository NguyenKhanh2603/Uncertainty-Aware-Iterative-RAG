"""Threshold sensitivity analysis — grid search and visualization.

Analyzes how performance varies with τ_noise and τ_missing thresholds.
Reports for both fixed and adaptive modes to demonstrate U2's value.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class ThresholdPoint:
    """Result for a single threshold configuration."""

    tau_noise: float
    tau_missing: float
    mode: str  # "fixed" or "adaptive"
    em: float = 0.0
    f1: float = 0.0
    avg_iterations: float = 0.0
    avg_llm_calls: float = 0.0


@dataclass
class SensitivityResult:
    """Full sensitivity analysis result."""

    points: list[ThresholdPoint] = field(default_factory=list)
    best_point: Optional[ThresholdPoint] = None


class ThresholdSensitivityAnalyzer:
    """Analyze threshold sensitivity for the paper.

    Key analysis:
    1. Grid search over τ_noise × τ_missing
    2. Compare fixed vs. adaptive thresholds (U2)
    3. Show ±20% perturbation impact
    4. Plot heatmaps of performance vs. thresholds
    """

    def __init__(
        self,
        output_dir: str = "results/sensitivity",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def generate_threshold_grid(
        tau_noise_range: Optional[list[float]] = None,
        tau_missing_range: Optional[list[float]] = None,
    ) -> list[tuple[float, float]]:
        """Generate grid of (τ_noise, τ_missing) values to evaluate."""
        if tau_noise_range is None:
            tau_noise_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        if tau_missing_range is None:
            tau_missing_range = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]

        return list(product(tau_noise_range, tau_missing_range))

    @staticmethod
    def perturbation_analysis(
        base_tau_noise: float,
        base_tau_missing: float,
        perturbation_pct: float = 0.2,
        steps: int = 5,
    ) -> list[tuple[float, float]]:
        """Generate ±perturbation_pct variations around base thresholds."""
        points = []
        deltas = np.linspace(-perturbation_pct, perturbation_pct, steps)

        for dn in deltas:
            for dm in deltas:
                tn = base_tau_noise * (1 + dn)
                tm = base_tau_missing * (1 + dm)
                points.append((max(0.01, tn), max(0.01, tm)))

        return points

    def plot_sensitivity_heatmap(
        self,
        result: SensitivityResult,
        metric: str = "f1",
        output_filename: str = "threshold_heatmap.png",
    ) -> Path:
        """Plot a heatmap of performance vs. threshold values."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Extract unique threshold values
        tau_noises = sorted(set(p.tau_noise for p in result.points))
        tau_missings = sorted(set(p.tau_missing for p in result.points))

        # Build matrix
        matrix = np.full((len(tau_missings), len(tau_noises)), np.nan)
        for p in result.points:
            i = tau_missings.index(p.tau_missing)
            j = tau_noises.index(p.tau_noise)
            matrix[i, j] = getattr(p, metric, 0.0)

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", origin="lower")

        ax.set_xticks(range(len(tau_noises)))
        ax.set_xticklabels([f"{t:.2f}" for t in tau_noises])
        ax.set_yticks(range(len(tau_missings)))
        ax.set_yticklabels([f"{t:.2f}" for t in tau_missings])

        ax.set_xlabel("τ_noise (aleatoric threshold)", fontsize=12)
        ax.set_ylabel("τ_missing (epistemic threshold)", fontsize=12)
        ax.set_title(f"Threshold Sensitivity: {metric.upper()}", fontsize=14)

        # Add text annotations
        for i in range(len(tau_missings)):
            for j in range(len(tau_noises)):
                if not np.isnan(matrix[i, j]):
                    ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=8)

        plt.colorbar(im, ax=ax, label=metric.upper())
        plt.tight_layout()

        output_path = self.output_dir / output_filename
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        return output_path

    def plot_fixed_vs_adaptive(
        self,
        fixed_results: SensitivityResult,
        adaptive_results: SensitivityResult,
        output_filename: str = "fixed_vs_adaptive.png",
    ) -> Path:
        """Compare fixed vs. adaptive thresholds (U2 key figure).

        Shows that adaptive thresholds achieve stable performance across
        a wider range of α/β values compared to fixed thresholds.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Fixed thresholds: scatter F1 vs. avg_iterations
        ax = axes[0]
        f1s = [p.f1 for p in fixed_results.points]
        iters = [p.avg_iterations for p in fixed_results.points]
        ax.scatter(iters, f1s, alpha=0.6, s=40, c="#FF6B6B", label="Fixed")
        ax.set_xlabel("Avg. Iterations", fontsize=12)
        ax.set_ylabel("F1 Score", fontsize=12)
        ax.set_title("Fixed Thresholds", fontsize=14)
        ax.grid(True, alpha=0.3)

        # Adaptive thresholds
        ax2 = axes[1]
        f1s_a = [p.f1 for p in adaptive_results.points]
        iters_a = [p.avg_iterations for p in adaptive_results.points]
        ax2.scatter(iters_a, f1s_a, alpha=0.6, s=40, c="#4ECDC4", label="Adaptive")
        ax2.set_xlabel("Avg. Iterations", fontsize=12)
        ax2.set_ylabel("F1 Score", fontsize=12)
        ax2.set_title("Adaptive Thresholds (U2)", fontsize=14)
        ax2.grid(True, alpha=0.3)

        # Match y-axis scales
        ymin = min(min(f1s, default=0), min(f1s_a, default=0))
        ymax = max(max(f1s, default=1), max(f1s_a, default=1))
        for ax_item in axes:
            ax_item.set_ylim(ymin - 0.05, min(ymax + 0.05, 1.0))

        plt.tight_layout()
        output_path = self.output_dir / output_filename
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        return output_path

    def save_results(self, result: SensitivityResult, filename: str = "sensitivity.json") -> Path:
        """Save sensitivity results as JSON."""
        output_path = self.output_dir / filename
        data = {
            "points": [
                {
                    "tau_noise": p.tau_noise,
                    "tau_missing": p.tau_missing,
                    "mode": p.mode,
                    "em": p.em,
                    "f1": p.f1,
                    "avg_iterations": p.avg_iterations,
                    "avg_llm_calls": p.avg_llm_calls,
                }
                for p in result.points
            ],
            "best_point": {
                "tau_noise": result.best_point.tau_noise,
                "tau_missing": result.best_point.tau_missing,
                "f1": result.best_point.f1,
            } if result.best_point else None,
        }
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        return output_path
