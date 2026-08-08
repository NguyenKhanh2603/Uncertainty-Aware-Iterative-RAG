"""Computational cost analysis and Pareto frontier visualization.

Shows cost-performance tradeoff for our method vs. baselines.
Key figure: Pareto frontier of accuracy vs. LLM calls per query.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class CostPoint:
    """Single data point in cost-performance analysis."""

    method: str
    accuracy: float  # EM or F1
    avg_llm_calls: float
    avg_iterations: float
    avg_cost_usd: float
    avg_latency_s: float


class CostAnalyzer:
    """Analyze and visualize computational cost vs. performance tradeoff."""

    def __init__(self, output_dir: str = "results/cost") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_pareto_frontier(
        self,
        points: list[CostPoint],
        output_filename: str = "pareto_frontier.png",
    ) -> Path:
        """Plot accuracy vs. LLM calls with Pareto frontier."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 7))

        colors = {
            "Ours": "#FF6B6B",
            "Ours (Adaptive M)": "#FF4444",
            "Naive RAG": "#95E1D3",
            "FLARE": "#4ECDC4",
            "Self-RAG": "#45B7D1",
            "Iterative RAG": "#F38181",
        }

        for p in points:
            color = colors.get(p.method, "#888888")
            ax.scatter(
                p.avg_llm_calls, p.accuracy,
                s=150, c=color, edgecolors="black", linewidths=1.5,
                zorder=5, label=p.method,
            )
            ax.annotate(
                p.method, (p.avg_llm_calls, p.accuracy),
                textcoords="offset points", xytext=(10, 5), fontsize=9,
            )

        ax.set_xlabel("Avg. LLM Calls per Query", fontsize=12)
        ax.set_ylabel("Accuracy (F1)", fontsize=12)
        ax.set_title("Cost-Performance Tradeoff", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=10)

        plt.tight_layout()
        output_path = self.output_dir / output_filename
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        return output_path

    def plot_adaptive_m_comparison(
        self,
        fixed_m_points: list[CostPoint],
        adaptive_m_points: list[CostPoint],
        output_filename: str = "adaptive_m_comparison.png",
    ) -> Path:
        """U4 key figure: compare fixed M vs. adaptive M on cost and accuracy."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Cost comparison
        ax = axes[0]
        methods = [p.method for p in fixed_m_points]
        fixed_costs = [p.avg_llm_calls for p in fixed_m_points]
        adaptive_costs = [p.avg_llm_calls for p in adaptive_m_points]

        x = np.arange(len(methods))
        width = 0.35
        ax.bar(x - width / 2, fixed_costs, width, label="Fixed M", color="#FF6B6B")
        ax.bar(x + width / 2, adaptive_costs, width, label="Adaptive M (U4)", color="#4ECDC4")
        ax.set_ylabel("Avg. LLM Calls", fontsize=12)
        ax.set_title("Cost: Fixed vs. Adaptive M", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=15)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        # Accuracy comparison
        ax2 = axes[1]
        fixed_accs = [p.accuracy for p in fixed_m_points]
        adaptive_accs = [p.accuracy for p in adaptive_m_points]
        ax2.bar(x - width / 2, fixed_accs, width, label="Fixed M", color="#FF6B6B")
        ax2.bar(x + width / 2, adaptive_accs, width, label="Adaptive M (U4)", color="#4ECDC4")
        ax2.set_ylabel("Accuracy (F1)", fontsize=12)
        ax2.set_title("Accuracy: Fixed vs. Adaptive M", fontsize=14)
        ax2.set_xticks(x)
        ax2.set_xticklabels(methods, rotation=15)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        output_path = self.output_dir / output_filename
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        return output_path

    def save_results(self, points: list[CostPoint], filename: str = "cost_analysis.json") -> Path:
        """Save cost analysis results."""
        output_path = self.output_dir / filename
        data = [
            {
                "method": p.method,
                "accuracy": round(p.accuracy, 4),
                "avg_llm_calls": round(p.avg_llm_calls, 1),
                "avg_iterations": round(p.avg_iterations, 2),
                "avg_cost_usd": round(p.avg_cost_usd, 4),
                "avg_latency_s": round(p.avg_latency_s, 2),
            }
            for p in points
        ]
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        return output_path
