"""Context pruning optimization strategies."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional

from uncertainty_rag.config import PruningConfig, PruningStrategy
from uncertainty_rag.core.sampler import Sample
from uncertainty_rag.modality.base import ContextChunk
from uncertainty_rag.models.nli_model import NLIModel
from uncertainty_rag.models.reranker import RerankerModel

logger = logging.getLogger(__name__)


@dataclass
class PruningReport:
    """Statistics about a pruning operation."""

    original_count: int
    surviving_count: int
    strategy_used: str
    pre_filtered_count: int = 0
    loo_evaluations: int = 0
    reranker_evaluations: int = 0


class BasePruner(ABC):
    """Abstract base class for context pruning strategies."""

    def __init__(self, config: PruningConfig):
        self.config = config

    @abstractmethod
    def prune(
        self,
        query: str,
        current_chunks: list[ContextChunk],
        current_se_semantic: float,
        eval_se_fn: Callable[[list[ContextChunk]], float],
        current_samples: list[Sample],
        eval_samples_fn: Optional[Callable[[list[Sample]], float]] = None,
    ) -> tuple[list[ContextChunk], PruningReport]:
        """Execute pruning and return surviving chunks and a report."""
        pass


class TwoPhasePruner(BasePruner):
    """Strategy 1: Two-phase Pruning (NLI Pre-filter + LOO)."""

    def __init__(self, config: PruningConfig, nli_model: NLIModel):
        super().__init__(config)
        self.nli_model = nli_model

    def prune(
        self,
        query: str,
        current_chunks: list[ContextChunk],
        current_se_semantic: float,
        eval_se_fn: Callable[[list[ContextChunk]], float],
        current_samples: list[Sample],
        eval_samples_fn: Optional[Callable[[list[Sample]], float]] = None,
    ) -> tuple[list[ContextChunk], PruningReport]:
        
        report = PruningReport(
            original_count=len(current_chunks),
            surviving_count=0,
            strategy_used="TWO_PHASE"
        )
        
        candidates = current_chunks.copy()

        # Phase 1: NLI Pre-filter
        if self.config.pre_filter_enabled and current_samples:
            # We use the most probable concept's claims as hypothesis
            # (In a real system, you'd aggregate claims from high-probability concepts)
            hypothesis = " ".join(current_samples[0].claims)
            
            pairs = [(c.content if isinstance(c.content, str) else str(c.content), hypothesis) for c in candidates]
            scores = self.nli_model.predict_batch(pairs)
            
            surviving_candidates = []
            for chunk, (c_score, n_score, e_score) in zip(candidates, scores):
                if c_score >= self.config.contradiction_threshold:
                    report.pre_filtered_count += 1
                    logger.info(f"  [Pre-filter] Pruned Chunk {chunk.id} (Contradiction: {c_score:.2f} >= {self.config.contradiction_threshold})")
                else:
                    surviving_candidates.append(chunk)
                    logger.info(f"  [Pre-filter] Kept Chunk {chunk.id} (Contradiction: {c_score:.2f} < {self.config.contradiction_threshold})")
            candidates = surviving_candidates

        # Phase 2: Leave-One-Out (LOO)
        logger.info(f"  [LOO Phase] Evaluating {len(candidates)} chunks...")
        final_chunks = []
        for i, chunk in enumerate(candidates):
            if report.loo_evaluations >= self.config.max_chunks_for_loo:
                final_chunks.append(chunk)
                continue

            test_chunks = candidates[:i] + candidates[i + 1:]
            new_se = eval_se_fn(test_chunks)
            report.loo_evaluations += 1

            if new_se <= current_se_semantic:
                # Removing it reduced or maintained uncertainty -> it's noise
                logger.info(f"  [LOO] Pruned Chunk {chunk.id}: Removing it decreased/maintained uncertainty (SE: {current_se_semantic:.4f} -> {new_se:.4f})")
            else:
                # Removing it increased uncertainty -> it's useful
                final_chunks.append(chunk)
                logger.info(f"  [LOO] Kept Chunk {chunk.id}: Removing it increased uncertainty (SE: {current_se_semantic:.4f} -> {new_se:.4f})")

        report.surviving_count = len(final_chunks)
        return final_chunks, report


class GrayZonePruner(BasePruner):
    """Strategy 2: Attention-Guided Gray-Zone Pruning using a Reranker."""

    def __init__(self, config: PruningConfig, reranker: RerankerModel):
        super().__init__(config)
        self.reranker = reranker

    def prune(
        self,
        query: str,
        current_chunks: list[ContextChunk],
        current_se_semantic: float,
        eval_se_fn: Callable[[list[ContextChunk]], float],
        current_samples: list[Sample],
        eval_samples_fn: Optional[Callable[[list[Sample]], float]] = None,
    ) -> tuple[list[ContextChunk], PruningReport]:
        
        report = PruningReport(
            original_count=len(current_chunks),
            surviving_count=0,
            strategy_used="GRAY_ZONE"
        )
        
        if not current_chunks:
            return [], report

        # 1. Reranker Scoring
        texts = [c.content if isinstance(c.content, str) else str(c.content) for c in current_chunks]
        scores = self.reranker.predict_batch(query, texts)
        report.reranker_evaluations = len(current_chunks)

        tau_trash, tau_keep = self.config.reranker_thresholds
        
        keep_chunks = []
        gray_zone_chunks = []

        for chunk, score in zip(current_chunks, scores):
            if score >= tau_keep:
                keep_chunks.append(chunk)
                logger.info(f"  [Reranker] Kept Chunk {chunk.id} (Score: {score:.4f} >= {tau_keep})")
            elif score <= tau_trash:
                report.pre_filtered_count += 1
                logger.info(f"  [Reranker] Pruned Chunk {chunk.id} (Score: {score:.4f} <= {tau_trash})")
            else:
                gray_zone_chunks.append(chunk)
                logger.info(f"  [Reranker] Gray-Zone Chunk {chunk.id} (Score: {score:.4f})")

        # 2. LOO on Gray-Zone only
        logger.info(f"  [LOO Phase] Evaluating {len(gray_zone_chunks)} Gray-Zone chunks...")
        final_gray = []
        for chunk in gray_zone_chunks:
            test_chunks = keep_chunks + [c for c in gray_zone_chunks if c != chunk]
            new_se = eval_se_fn(test_chunks)
            report.loo_evaluations += 1

            if new_se <= current_se_semantic:
                logger.info(f"  [LOO] Pruned Gray-Zone Chunk {chunk.id} (SE: {current_se_semantic:.4f} -> {new_se:.4f})")
            else:
                final_gray.append(chunk)
                logger.info(f"  [LOO] Kept Gray-Zone Chunk {chunk.id} (SE: {current_se_semantic:.4f} -> {new_se:.4f})")

        final_chunks = keep_chunks + final_gray
        report.surviving_count = len(final_chunks)
        return final_chunks, report


class PrefixCachingPruner(BasePruner):
    """Strategy 4: Automatic Prefix Caching (APC) Pruning.
    
    This is logically identical to LOO, but it signals to the pipeline/handler
    that the context should be formatted at the BOTTOM of the prompt to maximize
    cache hits in vLLM/API.
    """

    def prune(
        self,
        query: str,
        current_chunks: list[ContextChunk],
        current_se_semantic: float,
        eval_se_fn: Callable[[list[ContextChunk]], float],
        current_samples: list[Sample],
        eval_samples_fn: Optional[Callable[[list[Sample]], float]] = None,
    ) -> tuple[list[ContextChunk], PruningReport]:
        
        report = PruningReport(
            original_count=len(current_chunks),
            surviving_count=0,
            strategy_used="PREFIX_CACHING"
        )
        
        final_chunks = []
        for i, chunk in enumerate(current_chunks):
            if report.loo_evaluations >= self.config.max_chunks_for_loo:
                final_chunks.append(chunk)
                continue

            test_chunks = current_chunks[:i] + current_chunks[i + 1:]
            new_se = eval_se_fn(test_chunks)
            report.loo_evaluations += 1

            if new_se <= current_se_semantic:
                logger.info(f"  [APC LOO] Pruned Chunk {chunk.id} (SE: {current_se_semantic:.4f} -> {new_se:.4f})")
            else:
                final_chunks.append(chunk)
                logger.info(f"  [APC LOO] Kept Chunk {chunk.id} (SE: {current_se_semantic:.4f} -> {new_se:.4f})")

        report.surviving_count = len(final_chunks)
        return final_chunks, report


class AttentionMaskingPruner(BasePruner):
    """Strategy 3: Zero-Cost LOO via Attention Masking."""

    def __init__(self, config: PruningConfig, llm_client=None):
        super().__init__(config)
        self.llm_client = llm_client

    def prune(
        self,
        query: str,
        current_chunks: list[ContextChunk],
        current_se_semantic: float,
        eval_se_fn: Callable[[list[ContextChunk]], float],
        current_samples: list[Sample],
        eval_samples_fn: Optional[Callable[[list[Sample]], float]] = None,
    ) -> tuple[list[ContextChunk], PruningReport]:
        
        report = PruningReport(
            original_count=len(current_chunks),
            surviving_count=0,
            strategy_used="ATTENTION_MASKING"
        )

        if not self.llm_client or not hasattr(self.llm_client, "generate_with_custom_mask"):
            logger.warning("AttentionMaskingPruner requires HuggingFaceLocalClient. Falling back to keeping all chunks.")
            report.surviving_count = len(current_chunks)
            return current_chunks, report

        logger.info(f"  [Attention Masking Phase] Evaluating {len(current_chunks)} chunks simultaneously...")
        
        try:
            results_per_chunk = self.llm_client.generate_with_custom_mask(
                query=query,
                chunks=current_chunks,
                max_tokens=50,
                n=3 # Generate 3 samples per chunk for real SE clustering
            )
            
            final_chunks = []
            for chunk, samples in zip(current_chunks, results_per_chunk):
                logger.debug(f"  [Attention Masking] Output when Chunk {chunk.id} is masked (Sample 1): {samples[0].text[:50]}...")
                
                if eval_samples_fn:
                    # Convert SampleResult to pipeline's Sample format
                    mapped_samples = [
                        Sample(
                            id=f"mask_{chunk.id}_{i}",
                            text=res.text,
                            logprobs=0.0,
                            token_logprobs=[],
                            claims=[]
                        ) for i, res in enumerate(samples)
                    ]
                    
                    new_se = eval_samples_fn(mapped_samples)
                    report.loo_evaluations += 1
                    
                    if new_se <= current_se_semantic:
                        logger.info(f"  [Attention Masking] Pruned Chunk {chunk.id} (SE: {current_se_semantic:.4f} -> {new_se:.4f})")
                    else:
                        final_chunks.append(chunk)
                        logger.info(f"  [Attention Masking] Kept Chunk {chunk.id} (SE: {current_se_semantic:.4f} -> {new_se:.4f})")
                else:
                    # Fallback (mock) if no pipeline evaluator provided
                    final_chunks.append(chunk)
                    logger.info(f"  [Attention Masking] Kept Chunk {chunk.id} (eval_samples_fn missing)")
                    
            report.surviving_count = len(final_chunks)
            return final_chunks, report
            
        except NotImplementedError as e:
            logger.warning(f"AttentionMaskingPruner not fully implemented in client: {e}")
            report.surviving_count = len(current_chunks)
            return current_chunks, report


class AttentionSaliencyPruner(BasePruner):
    """Strategy 5: Attention-based Saliency Pruning."""

    def __init__(self, config: PruningConfig, llm_client=None):
        super().__init__(config)
        self.llm_client = llm_client

    def prune(
        self,
        query: str,
        current_chunks: list[ContextChunk],
        current_se_semantic: float,
        eval_se_fn: Callable[[list[ContextChunk]], float],
        current_samples: list[Sample],
        eval_samples_fn: Optional[Callable[[list[Sample]], float]] = None,
    ) -> tuple[list[ContextChunk], PruningReport]:
        
        report = PruningReport(
            original_count=len(current_chunks),
            surviving_count=0,
            strategy_used="ATTENTION_SALIENCY"
        )
        
        if not self.llm_client or not hasattr(self.llm_client, "extract_attention_saliency"):
            logger.warning("AttentionSaliencyPruner requires HuggingFaceLocalClient. Falling back to keeping all chunks.")
            report.surviving_count = len(current_chunks)
            return current_chunks, report
            
        logger.info(f"  [Attention Saliency Phase] Evaluating {len(current_chunks)} chunks in 1 forward pass...")
        
        try:
            scores = self.llm_client.extract_attention_saliency(
                query=query,
                chunks=current_chunks,
            )
            
            # Prune based on a threshold (e.g., 0.02)
            threshold = 0.02
            final_chunks = []
            for chunk, score in zip(current_chunks, scores):
                if score >= threshold:
                    final_chunks.append(chunk)
                    logger.info(f"  [Saliency] Kept Chunk {chunk.id} (Score: {score:.4f} >= {threshold})")
                else:
                    logger.info(f"  [Saliency] Pruned Chunk {chunk.id} (Score: {score:.4f} < {threshold})")
                    
            report.surviving_count = len(final_chunks)
            return final_chunks, report
            
        except (NotImplementedError, ValueError) as e:
            logger.warning(f"AttentionSaliencyPruner not fully implemented/supported in client: {e}")
            report.surviving_count = len(current_chunks)
            return current_chunks, report


class PrunerFactory:
    """Factory to instantiate the correct pruner based on config."""

    @staticmethod
    def create(
        config: PruningConfig,
        nli_model: Optional[NLIModel] = None,
        reranker_model: Optional[RerankerModel] = None,
        llm_client = None,
    ) -> BasePruner:
        if config.strategy == PruningStrategy.TWO_PHASE:
            if not nli_model:
                raise ValueError("TwoPhasePruner requires an NLIModel.")
            return TwoPhasePruner(config, nli_model)
            
        elif config.strategy == PruningStrategy.GRAY_ZONE:
            if not reranker_model:
                raise ValueError("GrayZonePruner requires a RerankerModel.")
            return GrayZonePruner(config, reranker_model)
            
        elif config.strategy == PruningStrategy.PREFIX_CACHING:
            return PrefixCachingPruner(config)
            
        elif config.strategy == PruningStrategy.ATTENTION_MASKING:
            return AttentionMaskingPruner(config, llm_client=llm_client)
            
        elif config.strategy == PruningStrategy.ATTENTION_SALIENCY:
            return AttentionSaliencyPruner(config, llm_client=llm_client)
            
        raise ValueError(f"Unknown pruning strategy: {config.strategy}")
