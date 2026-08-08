"""Abstract ModalityHandler interface and shared data structures.

All modality-specific logic is encapsulated in handlers.
The core uncertainty pipeline only ever sees text claims.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ContextChunk:
    """A single unit of context — text passage, table, or image."""

    id: str
    content: Any  # str for text, dict for table, str (base64/path) for image
    modality: str  # "text" | "table" | "image"
    metadata: dict = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ContextChunk):
            return NotImplemented
        return self.id == other.id


class ModalityHandler(ABC):
    """Abstract interface for modality-specific operations.

    Concrete implementations handle the translation between raw modality-specific
    data and the LLM/VLM message format. Once claims are extracted from any modality,
    the downstream uncertainty pipeline is identical.
    """

    @abstractmethod
    def format_context_for_prompt(self, chunks: list[ContextChunk]) -> list[dict]:
        """Convert context chunks into LLM/VLM message content blocks.

        Returns:
            A list of content blocks suitable for the OpenAI messages API.
            For text-only, this is [{"type": "text", "text": "..."}].
            For multimodal, this may include {"type": "image_url", ...} blocks.
        """
        ...

    @abstractmethod
    def get_chunk_text_repr(self, chunk: ContextChunk) -> str:
        """Return a text representation of a chunk for NLI/embedding operations."""
        ...

    @abstractmethod
    def format_for_retrieval_query(self, claims: list[str]) -> str:
        """Format claims into a retrieval query appropriate for the modality."""
        ...

    def build_prompt_messages(
        self, query: str, chunks: list[ContextChunk], system_prompt: Optional[str] = None
    ) -> list[dict]:
        """Build complete message list for the LLM/VLM.

        Combines system prompt, context content blocks, and user query.
        """
        messages: list[dict] = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Build user message with context + query
        content_blocks = self.format_context_for_prompt(chunks)
        content_blocks.append({"type": "text", "text": f"\nQuestion: {query}\nAnswer:"})

        messages.append({"role": "user", "content": content_blocks})
        return messages
