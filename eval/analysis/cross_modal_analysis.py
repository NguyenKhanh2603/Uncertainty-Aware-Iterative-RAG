"""Cross-modal uncertainty analysis — per-modality breakdown.

Key figures for the paper:
1. SE_total, SE_aleatoric, SE_epistemic distributions for text vs. table vs. image
2. Routing decision frequency per modality
3. Pruning effectiveness per modality (which modality's chunks get pruned most?)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class ModalityStats:
    """Uncertainty statistics for a single modality."""

    modality: str
    se_totals: list[float] = field(default_factory=list)
    se_aleatorics: list[float] = field(default_factory=list)
    se_epistemics: list[float] = field(default_factory=list)
    prune_counts: list[int] = field(default_factory=list)
    retrieve_counts: list[int] = field(default_factory=list)
    stop_counts: list[int] = field(default_factory=list)
    avg_iterations: list[float] = field(default_factory=list)
    accuracies: list[float] = field(default_factory=list)

    @property
    def mean_se_total(self) -> float:
        return float(np.mean(self.se_totals)) if self.se_totals else 0.0

    @property
    def mean_se_aleatoric(self) -> float:
        return float(np.mean(self.se_aleatorics)) if self.se_aleatorics else 0.0

    @property
    def mean_se_epistemic(self) -> float:
        return float(np.mean(self.se_epistemics)) if self.se_epistemics else 0.0


class CrossModalAnalyzer:
    """Analyze uncertainty patterns across modalities.

    This analysis is unique to our framework and a key differentiator:
    - Shows that SE decomposition captures modality-specific noise patterns
    - Image distractors → high aleatoric (visual noise)
    - Table reasoning → high epistemic (numerical knowledge gaps)
    - Text QA → balanced distribution
    """

    def __init__(self, output_dir: str = "results/cross_modal") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_uncertainty_distributions(
        self,
        modality_stats: dict[str, ModalityStats],
        output_filename: str = "uncertainty_distributions.png",
    ) -> Path:
        """Box plots of SE components across modalities."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        modalities = list(modality_stats.keys())
        n_mod = len(modalities)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        metrics = [
            ("SE_total", "se_totals", "#FF6B6B"),
            ("SE_aleatoric", "se_aleatorics", "#4ECDC4"),
            ("SE_epistemic", "se_epistemics", "#45B7D1"),
        ]

        for ax, (label, attr, color) in zip(axes, metrics):
            data = []
            labels = []
            for mod in modalities:
                stats = modality_stats[mod]
                values = getattr(stats, attr)
                if values:
                    data.append(values)
                    labels.append(mod.capitalize())

            if data:
                bp = ax.boxplot(data, labels=labels, patch_artist=True)
                for patch in bp["boxes"]:
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)

            ax.set_title(label, fontsize=14, fontweight="bold")
            ax.set_ylabel("Value", fontsize=12)
            ax.grid(True, alpha=0.3, axis="y")

        plt.suptitle("Uncertainty Decomposition by Modality", fontsize=16, fontweight="bold")
        plt.tight_layout()

        output_path = self.output_dir / output_filename
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        return output_path

    def plot_routing_decisions(
        self,
        modality_stats: dict[str, ModalityStats],
        output_filename: str = "routing_decisions.png",
    ) -> Path:
        """Stacked bar chart of routing decisions per modality."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        modalities = list(modality_stats.keys())
        prune_totals = [sum(modality_stats[m].prune_counts) for m in modalities]
        retrieve_totals = [sum(modality_stats[m].retrieve_counts) for m in modalities]
        stop_totals = [sum(modality_stats[m].stop_counts) for m in modalities]

        x = np.arange(len(modalities))
        width = 0.6

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x, prune_totals, width, label="PRUNE", color="#FF6B6B")
        ax.bar(x, retrieve_totals, width, bottom=prune_totals, label="RETRIEVE", color="#4ECDC4")
        bottom_both = [p + r for p, r in zip(prune_totals, retrieve_totals)]
        ax.bar(x, stop_totals, width, bottom=bottom_both, label="STOP", color="#95E1D3")

        ax.set_xlabel("Modality", fontsize=12)
        ax.set_ylabel("Total Decisions", fontsize=12)
        ax.set_title("Routing Decision Distribution by Modality", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([m.capitalize() for m in modalities])
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        output_path = self.output_dir / output_filename
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        return output_path

    def save_results(
        self, modality_stats: dict[str, ModalityStats], filename: str = "cross_modal.json"
    ) -> Path:
        """Save cross-modal analysis as JSON."""
        output_path = self.output_dir / filename
        data = {}
        for mod, stats in modality_stats.items():
            data[mod] = {
                "mean_se_total": round(stats.mean_se_total, 4),
                "mean_se_aleatoric": round(stats.mean_se_aleatoric, 4),
                "mean_se_epistemic": round(stats.mean_se_epistemic, 4),
                "total_prune_decisions": sum(stats.prune_counts),
                "total_retrieve_decisions": sum(stats.retrieve_counts),
                "total_stop_decisions": sum(stats.stop_counts),
                "num_examples": len(stats.se_totals),
            }
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        return output_path
