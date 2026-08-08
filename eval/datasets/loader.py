"""Unified dataset loader interface with modality-aware formatting."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from uncertainty_rag.modality.base import ContextChunk, ModalityHandler


@dataclass
class EvalExample:
    """A single evaluation example (question + gold answers + context)."""

    query_id: str
    query: str
    gold_answers: list[str]
    context_chunks: list[ContextChunk] = field(default_factory=list)
    modality: str = "text"
    metadata: dict = field(default_factory=dict)


class DatasetLoader(ABC):
    """Abstract dataset loader interface."""

    @abstractmethod
    def load(self, split: str = "validation", max_examples: Optional[int] = None) -> list[EvalExample]:
        """Load evaluation examples from the dataset."""
        ...

    @abstractmethod
    def get_modality_handler(self) -> ModalityHandler:
        """Return the appropriate modality handler for this dataset."""
        ...

    @abstractmethod
    def get_metrics(self) -> list[str]:
        """Return the list of metric names appropriate for this dataset."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Dataset name."""
        ...
