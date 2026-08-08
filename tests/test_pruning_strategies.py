"""Unit tests for the 5 advanced pruning optimization strategies."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from uncertainty_rag.config import PruningConfig, PruningStrategy
from uncertainty_rag.core.pruner import (
    TwoPhasePruner,
    GrayZonePruner,
    PrefixCachingPruner,
    AttentionMaskingPruner,
    AttentionSaliencyPruner,
    PrunerFactory,
)
from uncertainty_rag.core.sampler import Sample
from uncertainty_rag.modality.base import ContextChunk


class MockNLIModel:
    def predict_batch(self, pairs):
        # Return entailment=1.0 for first chunk, contradiction (c=1.0) for the rest
        return [(0.0, 0.0, 1.0)] + [(1.0, 0.0, 0.0)] * (len(pairs) - 1)


class MockRerankerModel:
    def predict_batch(self, query, texts):
        # 1st chunk: Keep (0.9)
        # 2nd chunk: Trash (0.1)
        # 3rd chunk: Gray (0.5)
        return [0.9, 0.1, 0.5][:len(texts)]


def mock_eval_se(chunks: list[ContextChunk]) -> float:
    # Let's say keeping chunk "1" reduces SE to 0.5, removing it increases SE to 1.5
    if any(c.id == "1" for c in chunks):
        return 0.5
    return 1.5


@pytest.fixture
def sample_chunks():
    return [
        ContextChunk(id="1", content="Chunk 1", modality="text"),
        ContextChunk(id="2", content="Chunk 2", modality="text"),
        ContextChunk(id="3", content="Chunk 3", modality="text"),
    ]


@pytest.fixture
def current_samples():
    return [Sample(text="Answer", claims=["Claim 1"])]


class TestPrunerFactory:
    def test_factory_creation(self):
        config = PruningConfig(strategy=PruningStrategy.GRAY_ZONE)
        pruner = PrunerFactory.create(config, reranker_model=MockRerankerModel())
        assert isinstance(pruner, GrayZonePruner)

    def test_factory_requires_models(self):
        with pytest.raises(ValueError, match="requires an NLIModel"):
            PrunerFactory.create(PruningConfig(strategy=PruningStrategy.TWO_PHASE))
        
        with pytest.raises(ValueError, match="requires a RerankerModel"):
            PrunerFactory.create(PruningConfig(strategy=PruningStrategy.GRAY_ZONE))


class TestTwoPhasePruner:
    def test_two_phase_pruning(self, sample_chunks, current_samples):
        config = PruningConfig(pre_filter_enabled=True, contradiction_threshold=0.8)
        pruner = TwoPhasePruner(config, nli_model=MockNLIModel())
        
        final_chunks, report = pruner.prune(
            query="Q?",
            current_chunks=sample_chunks,
            current_se_total=1.0,
            eval_se_fn=mock_eval_se,
            current_samples=current_samples
        )
        
        # NLI filters out chunk 2 and 3 because mock returns contradiction for them
        assert report.pre_filtered_count == 2
        # LOO evaluates chunk 1. Removing chunk 1 gives SE=1.5 > 1.0 (useful chunk -> kept)
        assert len(final_chunks) == 1
        assert final_chunks[0].id == "1"


class TestGrayZonePruner:
    def test_gray_zone_pruning(self, sample_chunks, current_samples):
        config = PruningConfig(reranker_thresholds=(0.2, 0.8))
        pruner = GrayZonePruner(config, reranker=MockRerankerModel())
        
        final_chunks, report = pruner.prune(
            query="Q?",
            current_chunks=sample_chunks,
            current_se_total=1.0,
            eval_se_fn=mock_eval_se,
            current_samples=current_samples
        )
        
        # Chunk 1 (0.9) -> Kept automatically
        # Chunk 2 (0.1) -> Trashed
        # Chunk 3 (0.5) -> Gray zone -> Evaluated by LOO
        assert report.pre_filtered_count == 1  # Trash
        assert report.loo_evaluations == 1     # Gray zone evaluated
        
        # When evaluating chunk 3, chunk 1 is present in test_chunks. 
        # mock_eval_se returns 0.5 because chunk 1 is present. 
        # 0.5 <= 1.0, so chunk 3 removing it reduced SE -> it's noise -> pruned.
        assert len(final_chunks) == 1
        assert final_chunks[0].id == "1"


class TestPrefixCachingPruner:
    def test_prefix_caching_pruning(self, sample_chunks, current_samples):
        pruner = PrefixCachingPruner(PruningConfig())
        
        final_chunks, report = pruner.prune(
            query="Q?",
            current_chunks=sample_chunks,
            current_se_total=1.0,
            eval_se_fn=mock_eval_se,
            current_samples=current_samples
        )
        
        # APC evaluates all chunks via LOO
        assert report.loo_evaluations == 3
        assert report.strategy_used == "PREFIX_CACHING"


class TestMockPruners:
    def test_attention_masking_mock(self, sample_chunks, current_samples):
        pruner = AttentionMaskingPruner(PruningConfig())
        final_chunks, report = pruner.prune("Q", sample_chunks, 1.0, mock_eval_se, current_samples)
        assert report.strategy_used == "ATTENTION_MASKING"
        assert len(final_chunks) == len(sample_chunks)

    def test_attention_saliency_mock(self, sample_chunks, current_samples):
        pruner = AttentionSaliencyPruner(PruningConfig())
        final_chunks, report = pruner.prune("Q", sample_chunks, 1.0, mock_eval_se, current_samples)
        assert report.strategy_used == "ATTENTION_SALIENCY"
