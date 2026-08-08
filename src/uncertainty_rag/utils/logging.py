"""Structured logging for per-iteration pipeline metrics."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class IterationLog:
    """Snapshot of a single pipeline iteration."""

    iteration: int
    se_total: float
    se_aleatoric: float
    se_epistemic: float
    num_concepts: int
    decision: str
    num_context_chunks: int
    llm_calls_this_iter: int
    samples_used: int  # M for this iteration (may vary with adaptive M)
    wall_time_s: float
    cost_so_far_usd: float
    # Adaptive threshold info (U2)
    effective_tau_noise: Optional[float] = None
    effective_tau_missing: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {k: round(v, 6) if isinstance(v, float) else v for k, v in asdict(self).items()}


class PipelineLogger:
    """JSON-lines logger for pipeline execution traces."""

    def __init__(self, output_dir: str = "results/logs", level: str = "INFO") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._logger = logging.getLogger("uncertainty_rag")
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
            self._logger.addHandler(handler)
        self._logger.setLevel(getattr(logging, level.upper()))
        self._logs: list[IterationLog] = []

    def log_iteration(self, entry: IterationLog) -> None:
        self._logs.append(entry)
        self._logger.info(
            f"Iter {entry.iteration}: SE_total={entry.se_total:.4f} "
            f"SE_a={entry.se_aleatoric:.4f} SE_e={entry.se_epistemic:.4f} "
            f"concepts={entry.num_concepts} → {entry.decision} "
            f"(M={entry.samples_used}, chunks={entry.num_context_chunks})"
        )

    def log_message(self, msg: str, level: str = "INFO") -> None:
        getattr(self._logger, level.lower())(msg)

    def save(self, query_id: str) -> Path:
        """Save iteration logs as JSON-lines file."""
        path = self.output_dir / f"{query_id}.jsonl"
        with open(path, "w") as f:
            for entry in self._logs:
                f.write(json.dumps(entry.to_dict()) + "\n")
        return path

    @property
    def history(self) -> list[IterationLog]:
        return list(self._logs)

    def reset(self) -> None:
        self._logs.clear()
