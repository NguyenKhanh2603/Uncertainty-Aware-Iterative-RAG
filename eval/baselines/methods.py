"""Baseline implementations for comparison: Naive RAG, FLARE, Iterative RAG.

All baselines use the same ModalityHandler for fair comparison.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from uncertainty_rag.modality.base import ContextChunk, ModalityHandler
from uncertainty_rag.models.llm_client import BaseLLMClient
from uncertainty_rag.core.retriever import ActiveRetriever
from uncertainty_rag.utils.cost_tracker import CostTracker


@dataclass
class BaselineResult:
    """Result from a baseline method."""

    answer: str
    iterations: int
    llm_calls: int
    cost_usd: float


class BaselineMethod(ABC):
    """Abstract baseline interface."""

    @abstractmethod
    def run(
        self,
        query: str,
        context: list[ContextChunk],
    ) -> BaselineResult:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


class NaiveRAG(BaselineMethod):
    """Single retrieval → single generation. No iteration."""

    def __init__(
        self,
        llm: BaseLLMClient,
        handler: ModalityHandler,
        cost_tracker: Optional[CostTracker] = None,
    ) -> None:
        self.llm = llm
        self.handler = handler
        self.cost_tracker = cost_tracker or CostTracker()

    @property
    def name(self) -> str:
        return "Naive RAG"

    def run(self, query: str, context: list[ContextChunk]) -> BaselineResult:
        messages = self.handler.build_prompt_messages(
            query=query,
            chunks=context,
            system_prompt="Answer the question based on the context. Be concise and factual.",
        )
        results = self.llm.generate(messages=messages, n=1, temperature=0.0, logprobs=False)
        answer = results[0].text if results else ""

        return BaselineResult(
            answer=answer,
            iterations=1,
            llm_calls=1,
            cost_usd=self.cost_tracker.total_cost_usd,
        )


class IterativeRAG(BaselineMethod):
    """Fixed K iterations of retrieve → generate, no uncertainty routing."""

    def __init__(
        self,
        llm: BaseLLMClient,
        handler: ModalityHandler,
        retriever: ActiveRetriever,
        num_iterations: int = 3,
        cost_tracker: Optional[CostTracker] = None,
    ) -> None:
        self.llm = llm
        self.handler = handler
        self.retriever = retriever
        self.num_iterations = num_iterations
        self.cost_tracker = cost_tracker or CostTracker()

    @property
    def name(self) -> str:
        return "Iterative RAG"

    def run(self, query: str, context: list[ContextChunk]) -> BaselineResult:
        llm_calls = 0

        for _ in range(self.num_iterations):
            # Generate
            messages = self.handler.build_prompt_messages(
                query=query, chunks=context,
                system_prompt="Answer the question based on the context.",
            )
            results = self.llm.generate(messages=messages, n=1, temperature=0.0, logprobs=False)
            answer = results[0].text if results else ""
            llm_calls += 1

            # Retrieve using answer as query
            new_chunks = self.retriever.retrieve(
                hypothesis_claims=[answer],
                existing_context=context,
            )
            context.extend(new_chunks)

        # Final generation
        messages = self.handler.build_prompt_messages(
            query=query, chunks=context,
            system_prompt="Answer the question based on the context. Be concise.",
        )
        results = self.llm.generate(messages=messages, n=1, temperature=0.0, logprobs=False)
        llm_calls += 1

        return BaselineResult(
            answer=results[0].text if results else "",
            iterations=self.num_iterations,
            llm_calls=llm_calls,
            cost_usd=self.cost_tracker.total_cost_usd,
        )


class FLARE(BaselineMethod):
    """FLARE-style: token-confidence-based iterative retrieval.

    When low-confidence tokens are generated, re-retrieve and regenerate.
    Simplified implementation using logprobs.
    """

    def __init__(
        self,
        llm: BaseLLMClient,
        handler: ModalityHandler,
        retriever: ActiveRetriever,
        confidence_threshold: float = -1.0,
        max_iterations: int = 3,
        cost_tracker: Optional[CostTracker] = None,
    ) -> None:
        self.llm = llm
        self.handler = handler
        self.retriever = retriever
        self.confidence_threshold = confidence_threshold
        self.max_iterations = max_iterations
        self.cost_tracker = cost_tracker or CostTracker()

    @property
    def name(self) -> str:
        return "FLARE"

    def run(self, query: str, context: list[ContextChunk]) -> BaselineResult:
        llm_calls = 0

        for iteration in range(self.max_iterations):
            messages = self.handler.build_prompt_messages(
                query=query, chunks=context,
                system_prompt="Answer the question based on the context.",
            )
            results = self.llm.generate(
                messages=messages, n=1, temperature=0.0, logprobs=True
            )
            llm_calls += 1

            if not results:
                break

            answer = results[0].text

            # Check if any token has low confidence
            low_conf_tokens = [
                t for t in results[0].token_logprobs
                if t.logprob < self.confidence_threshold
            ]

            if not low_conf_tokens:
                # All tokens are confident → stop
                break

            # Use low-confidence span as retrieval query
            low_conf_text = " ".join(t.token for t in low_conf_tokens[:10])
            new_chunks = self.retriever.retrieve(
                hypothesis_claims=[low_conf_text],
                existing_context=context,
            )
            context.extend(new_chunks)

        return BaselineResult(
            answer=answer if results else "",
            iterations=iteration + 1,
            llm_calls=llm_calls,
            cost_usd=self.cost_tracker.total_cost_usd,
        )
