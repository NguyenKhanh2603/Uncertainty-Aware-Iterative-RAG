"""Cost tracking for LLM API calls — tracks calls, tokens, and estimated USD cost."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


# Approximate pricing per 1M tokens (as of mid-2026, update as needed)
_PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
}


@dataclass
class CallRecord:
    """Single LLM API call record."""

    model: str
    input_tokens: int
    output_tokens: int
    num_completions: int  # n parameter
    latency_s: float
    timestamp: float = field(default_factory=time.time)

    @property
    def estimated_cost_usd(self) -> float:
        pricing = _PRICING.get(self.model, {"input": 5.0, "output": 15.0})
        return (
            self.input_tokens * pricing["input"] / 1_000_000
            + self.output_tokens * pricing["output"] / 1_000_000
        )


class CostTracker:
    """Accumulates cost and usage metrics across a pipeline run."""

    def __init__(self) -> None:
        self._records: list[CallRecord] = []

    def record_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        num_completions: int = 1,
        latency_s: float = 0.0,
    ) -> CallRecord:
        rec = CallRecord(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            num_completions=num_completions,
            latency_s=latency_s,
        )
        self._records.append(rec)
        return rec

    @property
    def total_calls(self) -> int:
        return len(self._records)

    @property
    def total_input_tokens(self) -> int:
        return sum(r.input_tokens for r in self._records)

    @property
    def total_output_tokens(self) -> int:
        return sum(r.output_tokens for r in self._records)

    @property
    def total_completions(self) -> int:
        return sum(r.num_completions for r in self._records)

    @property
    def total_cost_usd(self) -> float:
        return sum(r.estimated_cost_usd for r in self._records)

    @property
    def total_latency_s(self) -> float:
        return sum(r.latency_s for r in self._records)

    def summary(self) -> dict:
        return {
            "total_calls": self.total_calls,
            "total_completions": self.total_completions,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "total_latency_s": round(self.total_latency_s, 2),
        }

    def reset(self) -> None:
        self._records.clear()
