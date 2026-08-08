# Uncertainty-Aware Iterative RAG via Semantic Information Gain

A modality-agnostic framework that decomposes semantic uncertainty into **aleatoric** (data noise) and **epistemic** (knowledge gap) components to drive differentiated corrective actions — **pruning** noisy context and **retrieving** missing evidence.

## Key Features

- **Semantic Uncertainty Decomposition**: SE_total, SE_aleatoric, SE_epistemic via NLI-based concept clustering
- **Dual-Condition Stopping**: Stops when *both* noise and knowledge gap are resolved
- **Adaptive Thresholds (U2)**: Self-calibrating thresholds from initial uncertainty profile
- **Adaptive Sampling (U4)**: Two-phase M sampling — quick probe then full sampling
- **Calibration Analysis (U3)**: ECE score and reliability diagrams
- **Modality-Agnostic**: Same pipeline for text, table, and image inputs

## Architecture

```
Query + Context → Sampling (M samples) → Claim Extraction → NLI Clustering
→ SE Decomposition → Routing (Stop/Prune/Retrieve) → Loop
```

## Context Pruning Strategies

The framework implements 5 distinct context pruning strategies (configured via `PrunerFactory`) to efficiently remove noisy information from the retrieved context:

1. **Two-Phase Pruning (Default)**: Reranker (coarse filtering) + NLI (fine-grained filtering) based on semantic conflict (not recommend)
2. **Gray-Zone Pruning**: Re-evaluates uncertain (gray-zone) chunks using a powerful reranker (e.g., `BAAI/bge-reranker-v2-m`). (not recommend)
3. **Prefix-Caching (LOO)**: Leave-One-Out uncertainty measurement leveraging KV-Cache for speed. (recommend)
4. **Attention Masking (LOO)**: Masks out chunk tokens at the tensor level to compute uncertainty impact without modifying the prompt (Requires Local VLM). (recommend)
5. **Attention Saliency**: Prunes chunks with the lowest attention weights directly from the attention matrix (Requires Local VLM). (suggestion of Mr.Hung)

## Running Multimodal Datasets (TAT-QA & MultimodalQA)

To run on multimodal datasets, our framework takes a modern **on-the-fly VLM approach**. We do not pre-process the entire dataset upfront.

- **Images (MultimodalQA / WebQA)**: We bypass legacy pre-extracted feature vectors (e.g., `img_features.tar.gz`). We load raw images directly from the dataset's candidate list for a specific question, converting them to Base64 for modern VLMs (like GPT-4o or LLaVA).
- **Tables (TAT-QA)**: Tables are converted to Markdown format and treated as structured text chunks. When the Pruner decides to remove a table, it removes the entire Markdown table block.

> [!IMPORTANT]
> Because the iterative pruning algorithm (LOO) calls the Vision-Language Model multiple times per question, evaluating on the entire 44GB image dataset is extremely costly. For ablation studies, it is highly recommended to run on a random sub-sample (e.g., `--max_examples 200`) first.

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Run unit tests
pytest tests/ -v

# Quick evaluation (50 examples)
python eval/run_eval.py --dataset nq --max_examples 50

# Compare fixed vs. adaptive thresholds (U2)
python eval/run_eval.py --dataset nq --threshold_mode fixed --max_examples 100
python eval/run_eval.py --dataset nq --threshold_mode adaptive --max_examples 100

# Compare fixed vs. adaptive M (U4)
python eval/run_eval.py --dataset nq --adaptive_m --max_examples 100
python eval/run_eval.py --dataset nq --no-adaptive_m --max_examples 100

# Run with calibration analysis (U3)
python eval/run_eval.py --dataset nq --calibrate --max_examples 100

# Multimodal evaluation (TAT-QA & WebQA/MMQA)
# We recommend using --max_examples 200 for initial testing due to VLM cost.
python eval/run_eval.py --dataset tatqa --config configs/multimodal.yaml --max_examples 200
python eval/run_eval.py --dataset webqa --config configs/multimodal.yaml --max_examples 200

# Run specific Pruning Strategies (Ablation Study)
# Options: two_phase (default), gray_zone, prefix_caching, attention_masking, attention_saliency
python eval/run_eval.py --dataset tatqa --pruning_strategy gray_zone
python eval/run_eval.py --dataset tatqa --pruning_strategy attention_masking
```

## Project Structure

```
paper/
├── configs/
│   ├── default.yaml          # Default hyperparameters
│   └── multimodal.yaml       # Multimodal overrides
├── src/uncertainty_rag/
│   ├── config.py             # Pydantic config system
│   ├── pipeline.py           # Main iterative pipeline
│   ├── models/
│   │   ├── llm_client.py     # OpenAI API wrapper
│   │   └── nli_model.py      # NLI cross-encoder
│   ├── modality/
│   │   ├── base.py           # Abstract ModalityHandler
│   │   ├── text_handler.py   # Text passages
│   │   ├── table_handler.py  # TAT-QA tables
│   │   └── image_handler.py  # WebQA images
│   ├── core/
│   │   ├── sampler.py        # M-sample generation (U4 adaptive M)
│   │   ├── claim_extractor.py
│   │   ├── semantic_cluster.py
│   │   ├── uncertainty.py    # SE decomposition
│   │   ├── router.py         # Dual-condition routing (U2 adaptive τ)
│   │   ├── pruner.py         # Two-phase pruning,....
│   │   └── retriever.py      # Hypothesis-driven retrieval
│   └── utils/
│       ├── cost_tracker.py
│       └── logging.py
├── eval/
│   ├── run_eval.py           # Main evaluation script
│   ├── metrics.py            # EM, F1, ROUGE-L, Numerical Accuracy
│   ├── datasets/
│   ├── baselines/
│   └── analysis/
│       ├── calibration.py    # U3: ECE + reliability diagrams
│       ├── ablation.py
│       ├── threshold_sensitivity.py
│       ├── cost_analysis.py
│       └── cross_modal_analysis.py
└── tests/
```

## Configuration

See `configs/default.yaml` for all hyperparameters. Key settings:

| Parameter | Description | Default |
|---|---|---|
| `thresholds.mode` | `"fixed"` or `"adaptive"` (U2) | `"fixed"` |
| `sampling.adaptive_M_enabled` | Enable two-phase sampling (U4) | `false` |
| `sampling.M` | Fixed sample count | `10` |
| `sampling.M_initial` | Initial samples for adaptive M | `3` |
| `calibration.enabled` | Enable U3 calibration analysis | `true` |
