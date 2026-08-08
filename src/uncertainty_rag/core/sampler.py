"""M-sample generation with logprobs — supports adaptive M (U4)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from uncertainty_rag.config import SamplingConfig
from uncertainty_rag.modality.base import ContextChunk, ModalityHandler
from uncertainty_rag.models.llm_client import BaseLLMClient, SampleResult, TokenLogprob


SYSTEM_PROMPT = (
    "You are a precise and knowledgeable assistant. Answer the question based on the "
    "provided context. If the context is insufficient, provide your best answer based "
    "on your knowledge. Be concise and factual."
)


@dataclass
class Sample:
    """A single generated sample with its claims and token logprobs."""

    text: str
    token_logprobs: list[TokenLogprob] = field(default_factory=list)
    claims: list[str] = field(default_factory=list)
    key_token_logprobs: list[TokenLogprob] = field(default_factory=list)
    finish_reason: str = "stop"


class Sampler:
    """Generate M stochastic samples for uncertainty estimation.

    Supports:
    - Fixed M: Generate exactly M samples every iteration.
    - Adaptive M (U4): Start with M_initial, escalate to M_max if SE_total > threshold.
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        config: SamplingConfig,
    ) -> None:
        self.llm = llm_client
        self.config = config

    def generate_samples(
        self,
        query: str,
        context_chunks: list[ContextChunk],
        modality_handler: ModalityHandler,
        adaptive_phase: str = "initial",
    ) -> list[Sample]:
        """Generate M samples with logprobs.

        Args:
            query: The user query.
            context_chunks: Current context chunks.
            modality_handler: Modality-specific formatter.
            adaptive_phase: "initial" for M_initial, "full" for M_max (U4).
                           Ignored when adaptive_M is disabled.

        Returns:
            List of Sample objects with text and token logprobs.
        """
        # Determine M based on adaptive mode (U4)
        if self.config.adaptive_M_enabled:
            m = self.config.M_initial if adaptive_phase == "initial" else self.config.M_max
        else:
            m = self.config.M

        # Build prompt using modality handler
        messages = modality_handler.build_prompt_messages(
            query=query,
            chunks=context_chunks,
            system_prompt=SYSTEM_PROMPT,
        )

        # Generate M samples
        results: list[SampleResult] = self.llm.generate(
            messages=messages,
            n=m,
            temperature=self.config.temperature,
            logprobs=True,
            top_logprobs=5,
        )

        # Convert to Sample objects
        samples = []
        for result in results:
            samples.append(
                Sample(
                    text=result.text,
                    token_logprobs=result.token_logprobs,
                    finish_reason=result.finish_reason,
                )
            )

        return samples

    def should_escalate_m(self, se_total: float) -> bool:
        """U4: Check if we should escalate from M_initial to M_max.

        Returns True if SE_total from the initial quick probe exceeds the threshold.
        """
        if not self.config.adaptive_M_enabled:
            return False
        return se_total > self.config.adaptive_M_se_threshold
