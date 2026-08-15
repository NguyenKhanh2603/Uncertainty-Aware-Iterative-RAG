"""Main evaluation script — runs the pipeline on datasets and computes metrics.

Usage:
    # Quick dev run — text-only
    python eval/run_eval.py --dataset nq --split validation --max_examples 50

    # Quick dev run — multimodal
    python eval/run_eval.py --dataset tatqa --split validation --max_examples 50

    # Full evaluation
    python eval/run_eval.py --dataset hotpotqa --split test

    # Compare fixed vs. adaptive thresholds (U2)
    python eval/run_eval.py --dataset nq --threshold_mode fixed
    python eval/run_eval.py --dataset nq --threshold_mode adaptive

    # Compare fixed vs. adaptive M (U4)
    python eval/run_eval.py --dataset nq --adaptive_m
    python eval/run_eval.py --dataset nq --no-adaptive_m

    # Run calibration analysis (U3)
    python eval/run_eval.py --dataset nq --calibrate
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

# Remove the script's directory (eval) from sys.path to prevent eval/datasets
# from shadowing the HuggingFace `datasets` package!
script_dir = str(Path(__file__).parent.resolve())
sys.path = [p for p in sys.path if not (p and Path(p).resolve() == Path(script_dir))]

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm

from uncertainty_rag.config import Config
from uncertainty_rag.core.claim_extractor import ClaimExtractor
from uncertainty_rag.core.retriever import ActiveRetriever, DenseRetriever
from uncertainty_rag.core.router import Router
from uncertainty_rag.core.pruner import PrunerFactory
from uncertainty_rag.core.sampler import Sampler
from uncertainty_rag.core.semantic_cluster import SemanticClusterer
from uncertainty_rag.core.uncertainty import UncertaintyEstimator
from uncertainty_rag.models.llm_client import OpenAIClient
from uncertainty_rag.models.nli_model import NLIModel
from uncertainty_rag.models.reranker import RerankerModel
from uncertainty_rag.pipeline import IterativeRAGPipeline, PipelineResult
from uncertainty_rag.config import PruningStrategy
from uncertainty_rag.utils.cost_tracker import CostTracker
from uncertainty_rag.utils.logging import PipelineLogger

from eval.datasets.loader import DatasetLoader, EvalExample
from eval.datasets.text_datasets import TEXT_DATASETS
from eval.datasets.tatqa_loader import TATQALoader
from eval.datasets.webqa_loader import WebQALoader
from eval.datasets.multimodalqa_loader import MultiModalQALoader
from eval.metrics import MetricSuite
from eval.analysis.calibration import UncertaintyCalibrator, CalibrationResult


# ── Dataset Registry ────────────────────────────────────────────────────────────

DATASET_REGISTRY: dict[str, type[DatasetLoader]] = {
    **TEXT_DATASETS,
    "tatqa": TATQALoader,
    "webqa": WebQALoader,
    "multimodalqa": MultiModalQALoader,
}


def build_pipeline(config: Config) -> IterativeRAGPipeline:
    """Build the complete pipeline from config."""
    cost_tracker = CostTracker()

    # LLM client
    if config.pruning.strategy in [PruningStrategy.ATTENTION_MASKING, PruningStrategy.ATTENTION_SALIENCY]:
        from uncertainty_rag.models.llm_client import HuggingFaceLocalClient
        llm = HuggingFaceLocalClient(model_name=config.effective_llm_name)
        if config.effective_llm_name == config.model.claim_model:
            claim_llm = llm
        else:
            claim_llm = HuggingFaceLocalClient(model_name=config.model.claim_model)
    else:
        llm = OpenAIClient(model=config.effective_llm_name, cost_tracker=cost_tracker)
        claim_llm = OpenAIClient(model=config.model.claim_model, cost_tracker=cost_tracker)

    # NLI model
    nli = NLIModel(model_name=config.model.nli_name)

    # Modality handler
    dataset_name = ""  # Will be set per-dataset
    loader_cls = DATASET_REGISTRY.get(dataset_name)
    handler = None  # Will be set per-dataset

    # Core modules
    sampler = Sampler(llm_client=llm, config=config.sampling)
    claim_extractor = ClaimExtractor(llm_client=claim_llm, modality_type=config.modality.type)
    clusterer = SemanticClusterer(nli_model=nli)
    uncertainty = UncertaintyEstimator()
    router = Router(config=config.thresholds)

    # Retriever
    dense_retriever = DenseRetriever(
        embedding_model_name=config.model.embedding_name,
        config=config.retrieval,
    )

    # These will be set per-dataset in the eval loop
    return llm, claim_llm, nli, sampler, claim_extractor, clusterer, uncertainty, router, dense_retriever, cost_tracker


def run_evaluation(
    dataset_name: str,
    config: Config,
    split: str = "validation",
    max_examples: Optional[int] = None,
    start_index: int = 0,
    run_calibration: bool = False,
    output_dir: str = "results",
) -> dict:
    """Run full evaluation on a dataset."""
    # Load dataset
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_REGISTRY)}")

    loader = DATASET_REGISTRY[dataset_name]()
    examples = loader.load(split=split, max_examples=max_examples)
    if start_index > 0:
        examples = examples[start_index:]
    handler = loader.get_modality_handler()
    metrics_list = loader.get_metrics()
    metric_suite = MetricSuite(metrics=metrics_list)

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name} | Split: {split} | Examples: {len(examples)}")
    print(f"Modality: {config.modality.type} | Thresholds: {config.thresholds.mode}")
    print(f"Adaptive M: {config.sampling.adaptive_M_enabled} | M: {config.sampling.M}")
    print(f"{'='*60}\n")

    # Build pipeline components
    cost_tracker = CostTracker()
    if config.pruning.strategy in [PruningStrategy.ATTENTION_MASKING, PruningStrategy.ATTENTION_SALIENCY]:
        from uncertainty_rag.models.llm_client import HuggingFaceLocalClient
        print(f"Loading HuggingFaceLocalClient for strategy: {config.pruning.strategy.value}...")
        llm = HuggingFaceLocalClient(model_name=config.effective_llm_name)
        if config.effective_llm_name == config.model.claim_model:
            claim_llm = llm
        else:
            claim_llm = HuggingFaceLocalClient(model_name=config.model.claim_model)
    else:
        llm = OpenAIClient(model=config.effective_llm_name, cost_tracker=cost_tracker)
        claim_llm = OpenAIClient(model=config.model.claim_model, cost_tracker=cost_tracker)
    nli = NLIModel(model_name=config.model.nli_name)

    sampler = Sampler(llm_client=llm, config=config.sampling)
    claim_extractor = ClaimExtractor(llm_client=claim_llm, modality_type=config.modality.type)
    clusterer = SemanticClusterer(nli_model=nli)
    uncertainty = UncertaintyEstimator()
    router = Router(config=config.thresholds)

    dense_retriever = DenseRetriever(
        embedding_model_name=config.model.embedding_name,
        config=config.retrieval,
    )
    active_retriever = ActiveRetriever(
        retriever=dense_retriever,
        modality_handler=handler,
        config=config.retrieval,
    )

    reranker = None
    if config.pruning.strategy == PruningStrategy.GRAY_ZONE:
        reranker = RerankerModel(model_name=config.pruning.reranker_model)

    pruner = PrunerFactory.create(
        config=config.pruning,
        nli_model=nli,
        reranker_model=reranker,
    )

    logger = PipelineLogger(output_dir=f"{output_dir}/logs/{dataset_name}", level=config.logging.level)

    pipeline = IterativeRAGPipeline(
        config=config,
        sampler=sampler,
        claim_extractor=claim_extractor,
        clusterer=clusterer,
        uncertainty_estimator=uncertainty,
        router=router,
        pruner=pruner,
        retriever=active_retriever,
        modality_handler=handler,
        cost_tracker=cost_tracker,
        logger=logger,
    )

    # Run pipeline on all examples
    all_predictions = []
    all_gold_answers = []
    all_results: list[PipelineResult] = []

    for example in tqdm(examples, desc=f"Evaluating {dataset_name}"):
        # BƯỚC 1: Index toàn bộ kho dữ liệu
        if example.context_chunks:
            dense_retriever.index(
                example.context_chunks,
                text_fn=lambda c: handler.get_chunk_text_repr(c),
            )
            # BƯỚC 2: Chỉ Retrieve Top 10 đưa cho LLM ban đầu
            initial_chunks = dense_retriever.retrieve(
                query_text=example.query,
                existing_chunk_ids=set(),
                top_k=10
            )
            if not initial_chunks:
                initial_chunks = example.context_chunks[:10]
        else:
            initial_chunks = []

        # Reset per-query state
        router = Router(config=config.thresholds)
        pipeline.router = router
        cost_tracker.reset()
        logger.reset()

        # Run pipeline
        result = pipeline.run(query=example.query, initial_context=initial_chunks)

        # Save logs for this query
        if hasattr(example, 'query_id') and example.query_id:
            logger.save(query_id=str(example.query_id))
        else:
            logger.save(query_id=f"query_{len(all_results)}")

        all_predictions.append(result.answer)
        all_gold_answers.append(example.gold_answers)
        all_results.append(result)

    # Compute metrics
    batch_metrics = metric_suite.compute_batch(all_predictions, all_gold_answers)

    # Aggregate pipeline stats
    avg_iterations = sum(r.iterations for r in all_results) / len(all_results) if all_results else 0
    avg_se_total = sum(r.se_total for r in all_results) / len(all_results) if all_results else 0
    stop_decisions = sum(1 for r in all_results if r.final_decision == "CONFIDENT_STOP")

    results_summary = {
        "dataset": dataset_name,
        "split": split,
        "num_examples": len(examples),
        "config": {
            "threshold_mode": config.thresholds.mode,
            "adaptive_M": config.sampling.adaptive_M_enabled,
            "M": config.sampling.M,
            "M_initial": config.sampling.M_initial,
            "max_iterations": config.pipeline.max_iterations,
        },
        "metrics": {k: round(v, 4) for k, v in batch_metrics.items()},
        "pipeline_stats": {
            "avg_iterations": round(avg_iterations, 2),
            "avg_se_total": round(avg_se_total, 4),
            "confident_stops_pct": round(stop_decisions / len(all_results) * 100, 1) if all_results else 0,
        },
    }

    # U3: Calibration Analysis
    if run_calibration and all_results:
        calibrator = UncertaintyCalibrator(
            num_bins=config.calibration.num_bins,
            output_dir=f"{output_dir}/calibration/{dataset_name}",
        )
        # Compute per-example accuracy (EM)
        per_example_accuracy = [
            metric_suite.compute(pred, gold).get("em", 0.0)
            for pred, gold in zip(all_predictions, all_gold_answers)
        ]
        confidences = [max(0.0, 1.0 - r.se_total) for r in all_results]
        se_totals = [r.se_total for r in all_results]

        cal_result = calibrator.compute_calibration(
            confidences=confidences,
            accuracies=per_example_accuracy,
            se_totals=se_totals,
            method_name=f"Ours ({config.thresholds.mode})",
        )
        calibrator.plot_reliability_diagram([cal_result])
        calibrator.save_results(cal_result)

        results_summary["calibration"] = {
            "ece": round(cal_result.ece, 4),
            "mce": round(cal_result.mce, 4),
        }

    # Save results
    output_path = Path(output_dir) / f"{dataset_name}_{config.thresholds.mode}_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results_summary, f, indent=2)

    # Dump TAT-QA compatible predictions format if dataset is tatqa
    if dataset_name == "tatqa":
        tatqa_predictions = {}
        for example, result in zip(examples, all_results):
            # Format: { "uid": ["answer_string", "scale"] }
            # We assume scale is "" (baseline eval script handles parsing)
            tatqa_predictions[example.query_id] = [result.answer, ""]
        
        tatqa_pred_path = Path(output_dir) / f"{dataset_name}_{config.thresholds.mode}_predictions.json"
        with open(tatqa_pred_path, "w", encoding="utf-8") as f:
            json.dump(tatqa_predictions, f, indent=2, ensure_ascii=False)
        print(f"  TAT-QA Predictions saved to: {tatqa_pred_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"Results: {dataset_name}")
    for k, v in batch_metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"  Avg. iterations: {avg_iterations:.2f}")
    print(f"  Confident stops: {stop_decisions}/{len(all_results)}")
    if "calibration" in results_summary:
        print(f"  ECE: {results_summary['calibration']['ece']:.4f}")
    print(f"  Saved to: {output_path}")
    print(f"{'='*60}\n")

    return results_summary


def main():
    parser = argparse.ArgumentParser(description="Run Uncertainty-Aware RAG evaluation")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--start_index", type=int, default=0, help="Index to start evaluation from")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--config_override", type=str, default=None, help="Additional config YAML")
    parser.add_argument("--threshold_mode", type=str, choices=["fixed", "adaptive"], default=None)
    parser.add_argument("--pruning_strategy", type=str, choices=["two_phase", "gray_zone", "prefix_caching", "attention_masking", "attention_saliency"], default=None)
    parser.add_argument("--adaptive_m", action="store_true", default=None)
    parser.add_argument("--no-adaptive_m", dest="adaptive_m", action="store_false")
    parser.add_argument("--calibrate", action="store_true", help="Run U3 calibration analysis")
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    # Load config
    if args.config_override:
        config = Config.from_yamls(args.config, args.config_override)
    else:
        config = Config.from_yaml(args.config)

    # Apply CLI overrides
    if args.threshold_mode:
        config.thresholds.mode = args.threshold_mode
    if args.pruning_strategy:
        config.pruning.strategy = PruningStrategy(args.pruning_strategy)
    if args.adaptive_m is not None:
        config.sampling.adaptive_M_enabled = args.adaptive_m

    run_evaluation(
        dataset_name=args.dataset,
        config=config,
        split=args.split,
        max_examples=args.max_examples,
        start_index=args.start_index,
        run_calibration=args.calibrate,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
