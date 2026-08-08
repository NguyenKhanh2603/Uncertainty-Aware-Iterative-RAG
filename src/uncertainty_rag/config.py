"""Configuration dataclasses mirroring YAML structure with validation."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, model_validator


# ── Sub-configs ─────────────────────────────────────────────────────────────────


class PruningStrategy(str, Enum):
    """Available context pruning strategies."""
    TWO_PHASE = "two_phase"
    GRAY_ZONE = "gray_zone"
    ATTENTION_MASKING = "attention_masking"
    PREFIX_CACHING = "prefix_caching"
    ATTENTION_SALIENCY = "attention_saliency"


class SamplingConfig(BaseModel):
    """Controls stochastic sampling for uncertainty estimation."""

    M: int = Field(default=10, ge=2, description="Number of samples per iteration")
    temperature: float = Field(default=0.7, gt=0.0, le=2.0)
    # U4: Adaptive M
    M_initial: int = Field(default=3, ge=2, description="Initial M for adaptive sampling")
    M_max: int = Field(default=10, ge=2, description="Max M for adaptive sampling")
    adaptive_M_enabled: bool = Field(default=False, description="Enable two-phase adaptive M")
    adaptive_M_se_threshold: float = Field(
        default=0.5, ge=0.0, description="SE_total threshold to escalate M"
    )

    @model_validator(mode="after")
    def validate_adaptive_m(self) -> "SamplingConfig":
        if self.adaptive_M_enabled and self.M_initial >= self.M_max:
            raise ValueError(
                f"M_initial ({self.M_initial}) must be < M_max ({self.M_max}) "
                "when adaptive_M is enabled"
            )
        return self


class ThresholdConfig(BaseModel):
    """Routing thresholds — supports fixed and adaptive modes (U2)."""

    mode: Literal["fixed", "adaptive"] = "fixed"
    # Fixed thresholds
    tau_noise: float = Field(default=0.5, ge=0.0, description="Aleatoric threshold for pruning")
    tau_missing: float = Field(default=0.3, ge=0.0, description="Epistemic threshold for retrieval")
    # Adaptive parameters (U2)
    alpha: float = Field(default=0.5, gt=0.0, le=1.0, description="Fraction of initial aleatoric")
    beta: float = Field(default=0.5, gt=0.0, le=1.0, description="Fraction of initial epistemic")
    adaptive_min_tau: float = Field(default=0.05, ge=0.0, description="Minimum threshold floor")

    def compute_adaptive_thresholds(
        self, initial_se_aleatoric: float, initial_se_epistemic: float
    ) -> tuple[float, float]:
        """Compute adaptive thresholds from initial uncertainty profile.

        Returns:
            (tau_noise, tau_missing) calibrated from the first iteration.
        """
        tau_noise = max(self.alpha * initial_se_aleatoric, self.adaptive_min_tau)
        tau_missing = max(self.beta * initial_se_epistemic, self.adaptive_min_tau)
        return tau_noise, tau_missing


class PipelineConfig(BaseModel):
    """Iteration control and convergence."""

    max_iterations: int = Field(default=5, ge=1)
    convergence_patience: int = Field(default=2, ge=1)
    convergence_threshold: float = Field(default=0.05, ge=0.0, le=1.0)


class PruningConfig(BaseModel):
    """Context pruning parameters."""

    strategy: PruningStrategy = PruningStrategy.TWO_PHASE
    pre_filter_enabled: bool = True
    contradiction_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    max_chunks_for_loo: int = Field(default=10, ge=1)
    reranker_thresholds: tuple[float, float] = (0.2, 0.8)


class RetrievalConfig(BaseModel):
    """Active retrieval parameters."""

    top_k: int = Field(default=5, ge=1)
    dedup_cosine_threshold: float = Field(default=0.95, ge=0.0, le=1.0)


class ModelConfig(BaseModel):
    """Model identifiers."""

    llm_name: str = "gpt-4o-mini"
    vlm_name: str = "gpt-4o"
    nli_name: str = "cross-encoder/nli-deberta-v3-base"
    embedding_name: str = "all-MiniLM-L6-v2"
    reranker_name: str = "BAAI/bge-reranker-v2-m"
    claim_extractor_model: Optional[str] = None

    @property
    def claim_model(self) -> str:
        return self.claim_extractor_model or self.llm_name


class ModalityConfig(BaseModel):
    """Modality-specific settings."""

    type: Literal["text", "table", "multimodal"] = "text"
    image_detail: Literal["low", "high", "auto"] = "high"
    table_format: Literal["markdown", "html", "linearized"] = "markdown"


class CalibrationConfig(BaseModel):
    """U3: Uncertainty calibration settings."""

    enabled: bool = True
    num_bins: int = Field(default=10, ge=2)
    plot_output_dir: str = "results/calibration"


class LoggingConfig(BaseModel):
    """Logging settings."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    output_dir: str = "results/logs"
    log_per_iteration: bool = True


# ── Main Config ─────────────────────────────────────────────────────────────────


class Config(BaseModel):
    """Root configuration — fully validated."""

    sampling: SamplingConfig = SamplingConfig()
    thresholds: ThresholdConfig = ThresholdConfig()
    pipeline: PipelineConfig = PipelineConfig()
    pruning: PruningConfig = PruningConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    model: ModelConfig = ModelConfig()
    modality: ModalityConfig = ModalityConfig()
    calibration: CalibrationConfig = CalibrationConfig()
    logging: LoggingConfig = LoggingConfig()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load config from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    @classmethod
    def from_yamls(cls, *paths: str | Path) -> "Config":
        """Load and merge multiple YAML files (later files override earlier ones)."""
        merged: dict = {}
        for p in paths:
            with open(p) as f:
                data = yaml.safe_load(f) or {}
            for section, values in data.items():
                if isinstance(values, dict) and section in merged and isinstance(
                    merged[section], dict
                ):
                    merged[section].update(values)
                else:
                    merged[section] = values
        return cls(**merged)

    @property
    def effective_llm_name(self) -> str:
        """Return the LLM name appropriate for the current modality."""
        if self.modality.type == "multimodal":
            return self.model.vlm_name
        return self.model.llm_name
