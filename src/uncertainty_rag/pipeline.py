"""Main Iterative RAG Pipeline with Uncertainty-Aware Routing.

Orchestrates the full loop:
  Sampling → Claim Extraction → Semantic Clustering → Uncertainty Decomposition
  → Routing (Stop/Prune/Retrieve) → Loop

Features:
  - Dual-condition stopping (W3): STOP when both aleatoric AND epistemic are low
  - Non-circular retrieval (W5): Implicit EIG via iterative loop evaluation
  - Adaptive thresholds (U2): τ calibrated from initial uncertainty profile
  - Adaptive M (U4): Two-phase sampling — quick probe then full sampling
  - Convergence check: abort if SE_total stagnates
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from uncertainty_rag.config import Config
from uncertainty_rag.core.claim_extractor import ClaimExtractor
from uncertainty_rag.core.pruner import BasePruner, PrunerFactory
from uncertainty_rag.core.retriever import ActiveRetriever, BaseRetriever, DenseRetriever
from uncertainty_rag.core.router import Router, RoutingDecision
from uncertainty_rag.core.sampler import Sample, Sampler
from uncertainty_rag.core.semantic_cluster import Concept, SemanticClusterer
from uncertainty_rag.core.uncertainty import UncertaintyEstimator, UncertaintyProfile
from uncertainty_rag.modality.base import ContextChunk, ModalityHandler
from uncertainty_rag.modality.text_handler import TextHandler
from uncertainty_rag.models.llm_client import BaseLLMClient, OpenAIClient
from uncertainty_rag.models.nli_model import NLIModel
from uncertainty_rag.models.reranker import RerankerModel
from uncertainty_rag.utils.cost_tracker import CostTracker
from uncertainty_rag.utils.logging import IterationLog, PipelineLogger


@dataclass
class PipelineResult:
    """Complete result of a pipeline run with full transparency."""

    answer: str
    confidence: float
    se_semantic: float
    u_token: float
    iterations: int
    final_decision: str  # "CONFIDENT_STOP" | "MAX_ITER_REACHED" | "CONVERGENCE_ABORT"
    history: list[IterationLog] = field(default_factory=list)
    cost_summary: dict = field(default_factory=dict)
    # Per-iteration concept/claim data for analysis
    final_concepts: list[Concept] = field(default_factory=list)
    final_context_count: int = 0
    # Adaptive info
    effective_tau_token: float = 0.0
    effective_tau_semantic: float = 0.0
    threshold_mode: str = "fixed"
    samples_used_per_iter: list[int] = field(default_factory=list)


class IterativeRAGPipeline:
    """Uncertainty-aware iterative RAG pipeline.

    The core insight: the pipeline operates identically regardless of input modality.
    Modality-specific logic is confined to:
      1. ModalityHandler: formats context for LLM/VLM prompts
      2. ClaimExtractor: uses modality-aware prompts

    Once claims are extracted, the uncertainty computation, routing, pruning,
    and retrieval logic are the same for text, table, and image inputs.
    """

    def __init__(
        self,
        config: Config,
        sampler: Sampler,
        claim_extractor: ClaimExtractor,
        clusterer: SemanticClusterer,
        uncertainty_estimator: UncertaintyEstimator,
        router: Router,
        pruner: Optional[BasePruner] = None,
        retriever: Optional[BaseRetriever] = None,
        modality_handler: Optional[ModalityHandler] = None,
        nli_model: Optional[NLIModel] = None,
        reranker_model: Optional[RerankerModel] = None,
        cost_tracker: Optional[CostTracker] = None,
        logger: Optional[PipelineLogger] = None,
    ) -> None:
        self.config = config
        self.sampler = sampler
        self.claim_extractor = claim_extractor
        self.clusterer = clusterer
        self.uncertainty = uncertainty_estimator
        self.router = router
        
        self.nli_model = nli_model or NLIModel(model_name=self.config.model.nli_name)
        self.reranker_model = reranker_model
        if self.config.pruning.strategy == "gray_zone" and not self.reranker_model:
            self.reranker_model = RerankerModel(model_name=self.config.model.reranker_name)
            
        self.pruner = pruner or PrunerFactory.create(
            config=self.config.pruning,
            nli_model=self.nli_model,
            reranker_model=self.reranker_model,
        )
        self.retriever = retriever or DenseRetriever(config=self.config.retrieval)
        self.handler = modality_handler or TextHandler()
        self.cost_tracker = cost_tracker or CostTracker()
        self.logger = logger or PipelineLogger(
            output_dir=config.logging.output_dir,
            level=config.logging.level,
        )

    def run(self, query: str, initial_context: list[ContextChunk]) -> PipelineResult:
        """Execute the iterative uncertainty-aware RAG pipeline.

        Args:
            query: The user's question.
            initial_context: Initial set of context chunks (from first retrieval).

        Returns:
            PipelineResult with answer, confidence, and full execution trace.
        """
        context = list(initial_context)
        history: list[IterationLog] = []
        samples_used_per_iter: list[int] = []
        profile: Optional[UncertaintyProfile] = None
        concepts: list[Concept] = []
        final_decision = "MAX_ITER_REACHED"

        self.logger.log_message(
            f"Starting pipeline: query='{query[:80]}...', "
            f"initial_chunks={len(context)}, max_iter={self.config.pipeline.max_iterations}"
        )

        for iteration in range(self.config.pipeline.max_iterations):
            iter_start = time.time()

            # ── Step 1: Semantic Uncertainty Profiling ────────────────────────
            self.logger.log_message(f"\n=======================================================")
            self.logger.log_message(f"--- [STEP 1: SAMPLING & CLAIM EXTRACTION] ---")
            
            # Print all current chunks in full
            self.logger.log_message(f"Current Context ({len(context)} chunks):")
            for i, c in enumerate(context):
                self.logger.log_message(f"  [Chunk {i+1} | ID: {c.id}]\n{c.content}\n")

            # U4: Adaptive M — two-phase sampling
            if self.sampler.config.adaptive_M_enabled and iteration == 0:
                self.logger.log_message(f"Adaptive M Enabled. Starting Phase 1 (Quick Probe) with M={self.sampler.config.M_initial}...")
                # Phase 1: Quick probe with M_initial
                samples = self.sampler.generate_samples(
                    query, context, self.handler, adaptive_phase="initial"
                )
                self.logger.log_message(f"Generated {len(samples)} initial samples. Extracting claims...")
                samples = self.claim_extractor.extract_all(samples)
                for i, s in enumerate(samples):
                    self.logger.log_message(f"  [Sample {i+1} Claims]: {s.claims}")
                    
                self.logger.log_message(f"\n--- [STEP 2: SEMANTIC UNCERTAINTY PROFILING] ---")
                probe_concepts = self.clusterer.cluster(samples)
                probe_se = self.uncertainty.compute_se_semantic(probe_concepts)
                self.logger.log_message(f"Probe Clustering: Found {len(probe_concepts)} concepts. Initial SE_semantic={probe_se:.4f}")

                if self.sampler.should_escalate_m(probe_se):
                    # Phase 2: Full sampling with M_max
                    self.logger.log_message(
                        f"Adaptive M: SE_semantic={probe_se:.4f} > threshold="
                        f"{self.sampler.config.adaptive_M_se_threshold:.4f} → "
                        f"escalating from M={self.sampler.config.M_initial} to "
                        f"M={self.sampler.config.M_max}"
                    )
                    self.logger.log_message(f"Generating full samples with M={self.sampler.config.M_max}...")
                    samples = self.sampler.generate_samples(
                        query, context, self.handler, adaptive_phase="full"
                    )
                    samples = self.claim_extractor.extract_all(samples)
                    for i, s in enumerate(samples):
                        self.logger.log_message(f"  [Sample {i+1} Claims]: {s.claims}")
                        
                    concepts = self.clusterer.cluster(samples)
                    samples_used_per_iter.append(self.sampler.config.M_max)
                else:
                    # Quick probe was sufficient
                    concepts = probe_concepts
                    samples_used_per_iter.append(self.sampler.config.M_initial)
                    self.logger.log_message(
                        f"Adaptive M: SE_semantic={probe_se:.4f} ≤ threshold → "
                        f"using M={self.sampler.config.M_initial} (no escalation)"
                    )
            else:
                # Fixed M or non-first iteration with adaptive M
                adaptive_phase = "full" if self.sampler.config.adaptive_M_enabled else "initial"
                m_used = (
                    self.sampler.config.M_max
                    if self.sampler.config.adaptive_M_enabled
                    else self.sampler.config.M
                )
                self.logger.log_message(f"Fixed/Full M Mode. Generating {m_used} samples...")
                samples = self.sampler.generate_samples(
                    query, context, self.handler, adaptive_phase=adaptive_phase
                )
                samples = self.claim_extractor.extract_all(samples)
                for i, s in enumerate(samples):
                    self.logger.log_message(f"  [Sample {i+1} Claims]: {s.claims}")
                    
                self.logger.log_message(f"\n--- [STEP 2: SEMANTIC UNCERTAINTY PROFILING] ---")
                concepts = self.clusterer.cluster(samples)
                samples_used_per_iter.append(m_used)

            # Log grouped concepts
            self.logger.log_message(f"Clustering complete. Grouped into {len(concepts)} concepts:")
            for i, c in enumerate(concepts):
                self.logger.log_message(f"  [Concept {i+1} | Prob: {c.probability:.4f}]: {c.representative_claims}")

            # Compute uncertainty profile
            profile = self.uncertainty.compute(samples, concepts)
            self.logger.log_message(
                f"Uncertainty Profile Calculated:\n"
                f"  SE_semantic:  {profile.se_semantic:.4f} (Ambiguity)\n"
                f"  U_token:      {profile.u_token:.4f} (Noise/Contradiction)"
            )

            # ── U2: Adaptive Threshold Calibration (iteration 0 only) ────────
            if iteration == 0 and self.config.thresholds.mode == "adaptive":
                self.router.calibrate_adaptive(profile)
                self.logger.log_message(
                    f"Adaptive thresholds calibrated: "
                    f"τ_token={self.router.tau_token:.4f}, "
                    f"τ_semantic={self.router.tau_semantic:.4f}"
                )

            # Log iteration
            iter_time = time.time() - iter_start
            log_entry = IterationLog(
                iteration=iteration,
                se_semantic=profile.se_semantic,
                u_token=profile.u_token,
                num_concepts=profile.num_concepts,
                decision="",  # Will be set below
                num_context_chunks=len(context),
                llm_calls_this_iter=samples_used_per_iter[-1],
                samples_used=samples_used_per_iter[-1],
                wall_time_s=iter_time,
                cost_so_far_usd=self.cost_tracker.total_cost_usd,
                effective_tau_token=self.router.tau_token,
                effective_tau_semantic=self.router.tau_semantic,
            )

            # ── Step 2: Routing (dual-condition stopping) ────────────────────
            decision = self.router.decide(profile)
            log_entry.decision = decision.value
            self.logger.log_iteration(log_entry)
            history.append(log_entry)

            if decision == RoutingDecision.STOP:
                final_decision = "CONFIDENT_STOP"
                break

            # ── Step 2a: Prune if noise is high ──────────────────────────────
            if decision == RoutingDecision.PRUNE:
                self.logger.log_message(f"--- [PRUNING PHASE START] ---")
                context, prune_report = self.pruner.prune(
                    query, context, current_se_semantic=profile.se_semantic,
                    eval_se_fn=lambda chunks: self.uncertainty.compute_se_semantic(
                        self.clusterer.cluster(
                            self.claim_extractor.extract_all(
                                self.sampler.generate_samples(query, chunks, self.handler, adaptive_phase="initial")
                            )
                        )
                    ),
                    current_samples=samples
                )
                self.logger.log_message(
                    f"--- [PRUNING PHASE END] ---\n"
                    f"Result: removed {prune_report.original_count - prune_report.surviving_count} "
                    f"chunks (Pre-filter: {prune_report.pre_filtered_count}, LOO Evals: {prune_report.loo_evaluations})"
                    f" → {prune_report.surviving_count} chunks remaining."
                )
                continue

            # ── Step 2b: Retrieve if knowledge gap is high ───────────────────
            if decision == RoutingDecision.RETRIEVE:
                self.logger.log_message(f"--- [RETRIEVAL PHASE START] ---")
                best_concept = max(concepts, key=lambda c: c.probability)
                hypothesis = " ".join(best_concept.representative_claims)
                self.logger.log_message(f"Selected Hypothesis for Retrieval: '{hypothesis}'")
                
                new_chunks = self.retriever.retrieve(
                    hypothesis_claims=best_concept.representative_claims,
                    existing_context=context,
                    top_k=self.config.retrieval.top_k,
                )
                
                if new_chunks:
                    for i, nc in enumerate(new_chunks):
                        self.logger.log_message(f"  + Retrieved Chunk {i+1}: ID={nc.id}\n{nc.content}\n")
                else:
                    self.logger.log_message("  + No new chunks found by retriever. Stopping early to prevent infinite loop.")
                    final_decision = "NO_NEW_KNOWLEDGE_STOP"
                    break
                
                context.extend(new_chunks)
                self.logger.log_message(
                    f"--- [RETRIEVAL PHASE END] ---\n"
                    f"Total Context Chunks now: {len(context)}"
                )
                continue

            # ── Convergence check ────────────────────────────────────────────
            if self._check_convergence(history):
                final_decision = "CONVERGENCE_ABORT"
                self.logger.log_message(
                    "Convergence detected: SE_semantic not improving → aborting"
                )
                break

        # ── Generate Final Answer ────────────────────────────────────────────

        if profile is None:
            # Edge case: max_iterations = 0
            return PipelineResult(
                answer="",
                confidence=0.0,
                se_semantic=0.0,
                u_token=0.0,
                iterations=0,
                final_decision="NO_ITERATIONS",
            )

        best_concept = max(concepts, key=lambda c: c.probability) if concepts else None
        answer = self._generate_final_answer(query, context, best_concept)

        return PipelineResult(
            answer=answer,
            confidence=max(0.0, 1.0 - profile.se_semantic),
            se_semantic=profile.se_semantic,
            u_token=profile.u_token,
            iterations=len(history),
            final_decision=final_decision,
            history=history,
            cost_summary=self.cost_tracker.summary(),
            final_concepts=concepts,
            final_context_count=len(context),
            effective_tau_token=self.router.tau_token,
            effective_tau_semantic=self.router.tau_semantic,
            threshold_mode=self.config.thresholds.mode,
            samples_used_per_iter=samples_used_per_iter,
        )

    def _generate_final_answer(
        self,
        query: str,
        context: list[ContextChunk],
        best_concept: Optional[Concept],
    ) -> str:
        """Generate the final answer using the best concept as guidance."""
        self.logger.log_message("\n--- [FINAL ANSWER GENERATION] ---")
        # Build final generation prompt with guidance from best concept
        guidance = ""
        if best_concept and best_concept.representative_claims:
            claims_text = "; ".join(best_concept.representative_claims[:5])
            guidance = f"\n\nBased on the available evidence, the key facts are: {claims_text}"
            self.logger.log_message(f"Guiding LLM with top claims: {claims_text}")
        else:
            self.logger.log_message("No highly probable claims found to guide LLM.")

        system_msg = (
            "You are an expert Question Answering system. Answer the user's question based STRICTLY "
            "on the provided context.\n\n"
            "CRITICAL RULES:\n"
            "1. Output ONLY the exact final answer string, number, or span.\n"
            "2. DO NOT include any conversational filler (e.g., 'The answer is...', 'Based on the table...').\n"
            "3. If the answer is a list of items, separate them with a comma.\n"
            "4. For numerical answers, output exactly the number and its scale (e.g., '15.2 million', '10%')."
        )
        
        messages = self.handler.build_prompt_messages(
            query=query + guidance,
            chunks=context,
            system_prompt=system_msg,
        )

        self.logger.log_message(f"Sending prompt to LLM (Context size: {len(context)} chunks)")
        
        results = self.sampler.llm.generate(
            messages=messages,
            n=1,
            temperature=0.0,  # Greedy for final answer
            logprobs=False,
        )

        final_ans = results[0].text if results else ""
        self.logger.log_message(f"Raw LLM Output (Final Answer): '{final_ans}'")
        self.logger.log_message("--- [END PIPELINE] ---\n")
        return final_ans

    def _check_convergence(self, history: list[IterationLog]) -> bool:
        """Check if SE_semantic hasn't improved in last `patience` iterations."""
        patience = self.config.pipeline.convergence_patience
        threshold = self.config.pipeline.convergence_threshold

        if len(history) < patience + 1:
            return False

        recent = [h.se_semantic for h in history[-patience:]]
        previous = history[-(patience + 1)].se_semantic

        if previous == 0:
            return True  # Already at zero, no improvement possible

        # Check if any of the last `patience` iterations improved by > threshold
        return all(se >= previous * (1.0 - threshold) for se in recent)
