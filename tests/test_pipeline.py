"""Integration tests for the full IterativeRAGPipeline.

Uses mock LLM/NLI to test the pipeline flow without API calls.
Tests: fixed vs. adaptive thresholds (U2), fixed vs. adaptive M (U4),
convergence, stopping conditions.
"""

from __future__ import annotations

import json
import pytest
from unittest.mock import MagicMock, patch

from uncertainty_rag.config import Config, SamplingConfig, ThresholdConfig, PipelineConfig
from uncertainty_rag.core.claim_extractor import ClaimExtractor
from uncertainty_rag.core.retriever import ActiveRetriever, DenseRetriever
from uncertainty_rag.core.router import Router, RoutingDecision
from uncertainty_rag.core.pruner import Pruner
from uncertainty_rag.core.sampler import Sample, Sampler
from uncertainty_rag.core.semantic_cluster import Concept, SemanticClusterer
from uncertainty_rag.core.uncertainty import UncertaintyEstimator, UncertaintyProfile
from uncertainty_rag.modality.base import ContextChunk
from uncertainty_rag.modality.text_handler import TextHandler
from uncertainty_rag.models.llm_client import BaseLLMClient, SampleResult, TokenLogprob
from uncertainty_rag.pipeline import IterativeRAGPipeline, PipelineResult
from uncertainty_rag.utils.cost_tracker import CostTracker


class MockLLM(BaseLLMClient):
    """Mock LLM for testing — returns deterministic responses."""

    def __init__(self, answers: list[str] | None = None):
        self._answers = answers or ["The answer is Paris."]
        self._idx = 0

    def generate(self, messages, n=1, temperature=0.7, max_tokens=1024,
                 logprobs=True, top_logprobs=5, json_mode=False):
        results = []
        for _ in range(n):
            text = self._answers[self._idx % len(self._answers)]
            self._idx += 1
            token_lps = [
                TokenLogprob(token="answer", logprob=-0.5),
                TokenLogprob(token=" Paris", logprob=-0.3),
            ] if logprobs else []
            results.append(SampleResult(text=text, token_logprobs=token_lps))
        return results


class MockClaimExtractor:
    """Mock claim extractor — returns text as a single claim."""

    def extract_all(self, samples):
        for s in samples:
            s.claims = [s.text]
            s.key_token_logprobs = s.token_logprobs
        return samples


class MockClusterer:
    """Mock clusterer — puts all samples in one concept."""

    def cluster(self, samples):
        if not samples:
            return []
        return [
            Concept(
                id=0,
                sample_indices=list(range(len(samples))),
                representative_claims=samples[0].claims,
                probability=1.0,
            )
        ]


class MockRetriever:
    """Mock retriever — returns nothing (no new docs)."""

    def retrieve(self, hypothesis_claims, existing_context, top_k=5):
        return []


class MockPruner:
    """Mock pruner — returns all chunks (no pruning)."""

    def prune(self, query, chunks, current_se_total=None):
        from uncertainty_rag.core.pruner import PruningReport
        return chunks, PruningReport(original_count=len(chunks), surviving_count=len(chunks))


class TestPipelineStopCondition:
    """Test that pipeline STOPS when both uncertainties are low."""

    def test_confident_stop_at_iteration_0(self):
        """All samples agree → SE_total ≈ 0 → STOP immediately."""
        config = Config()
        config.thresholds.mode = "fixed"
        config.thresholds.tau_noise = 0.5
        config.thresholds.tau_missing = 0.3
        config.pipeline.max_iterations = 5

        llm = MockLLM(["Paris"] * 10)
        handler = TextHandler()

        pipeline = IterativeRAGPipeline(
            config=config,
            sampler=Sampler(llm_client=llm, config=config.sampling),
            claim_extractor=MockClaimExtractor(),
            clusterer=MockClusterer(),
            uncertainty_estimator=UncertaintyEstimator(),
            router=Router(config.thresholds),
            pruner=MockPruner(),
            retriever=MockRetriever(),
            modality_handler=handler,
        )

        context = [ContextChunk(id="1", content="France's capital is Paris.", modality="text")]
        result = pipeline.run("What is the capital of France?", context)

        assert result.final_decision == "CONFIDENT_STOP"
        assert result.iterations == 1
        assert result.se_total == pytest.approx(0.0)


class TestPipelineMaxIterations:
    """Test that pipeline respects MAX_ITERATIONS."""

    def test_max_iterations_cap(self):
        """Pipeline should not exceed max_iterations."""
        config = Config()
        config.pipeline.max_iterations = 2
        config.thresholds.mode = "fixed"
        config.thresholds.tau_noise = 0.0  # Never prune
        config.thresholds.tau_missing = 0.0  # Never retrieve

        llm = MockLLM(["Paris"] * 20)
        handler = TextHandler()

        pipeline = IterativeRAGPipeline(
            config=config,
            sampler=Sampler(llm_client=llm, config=config.sampling),
            claim_extractor=MockClaimExtractor(),
            clusterer=MockClusterer(),
            uncertainty_estimator=UncertaintyEstimator(),
            router=Router(config.thresholds),
            pruner=MockPruner(),
            retriever=MockRetriever(),
            modality_handler=handler,
        )

        context = [ContextChunk(id="1", content="Context.", modality="text")]
        result = pipeline.run("Question?", context)

        assert result.iterations <= config.pipeline.max_iterations


class TestAdaptiveThresholdsPipeline:
    """U2: Test adaptive threshold calibration in the pipeline."""

    def test_adaptive_mode_calibrates(self):
        """In adaptive mode, thresholds should be calibrated from iteration 0."""
        config = Config()
        config.thresholds.mode = "adaptive"
        config.thresholds.alpha = 0.5
        config.thresholds.beta = 0.5
        config.pipeline.max_iterations = 1

        llm = MockLLM(["Paris"] * 10)
        handler = TextHandler()
        router = Router(config.thresholds)

        pipeline = IterativeRAGPipeline(
            config=config,
            sampler=Sampler(llm_client=llm, config=config.sampling),
            claim_extractor=MockClaimExtractor(),
            clusterer=MockClusterer(),
            uncertainty_estimator=UncertaintyEstimator(),
            router=router,
            pruner=MockPruner(),
            retriever=MockRetriever(),
            modality_handler=handler,
        )

        context = [ContextChunk(id="1", content="Context.", modality="text")]
        result = pipeline.run("Q?", context)

        # Router should have calibrated thresholds
        assert result.threshold_mode == "adaptive"


class TestAdaptiveMPipeline:
    """U4: Test adaptive M sampling behavior."""

    def test_adaptive_m_config(self):
        """Adaptive M should be configurable."""
        config = Config()
        config.sampling.adaptive_M_enabled = True
        config.sampling.M_initial = 3
        config.sampling.M_max = 10
        config.sampling.adaptive_M_se_threshold = 0.5

        assert config.sampling.adaptive_M_enabled is True
        assert config.sampling.M_initial == 3
        assert config.sampling.M_max == 10

    def test_fixed_m_config(self):
        """Fixed M disables adaptive behavior."""
        config = Config()
        config.sampling.adaptive_M_enabled = False
        config.sampling.M = 10

        sampler = Sampler(llm_client=MockLLM(), config=config.sampling)
        assert not sampler.should_escalate_m(1.0)  # Always False when disabled


class TestPipelineResult:
    """Test PipelineResult structure."""

    def test_result_has_all_fields(self):
        result = PipelineResult(
            answer="Paris",
            confidence=0.95,
            se_total=0.05,
            se_aleatoric=0.02,
            se_epistemic=0.03,
            iterations=2,
            final_decision="CONFIDENT_STOP",
        )
        assert result.answer == "Paris"
        assert result.confidence == 0.95
        assert result.iterations == 2
        assert result.final_decision == "CONFIDENT_STOP"

    def test_result_cost_summary(self):
        result = PipelineResult(
            answer="x", confidence=0.5, se_total=0.5,
            se_aleatoric=0.2, se_epistemic=0.3,
            iterations=1, final_decision="STOP",
            cost_summary={"total_calls": 10, "total_cost_usd": 0.001},
        )
        assert result.cost_summary["total_calls"] == 10
