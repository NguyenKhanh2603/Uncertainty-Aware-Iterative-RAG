"""Text-only modality handler for standard text passages."""

from __future__ import annotations

from uncertainty_rag.modality.base import ContextChunk, ModalityHandler


class TextHandler(ModalityHandler):
    """Handles pure text passages (NQ, TriviaQA, HotpotQA, PopQA, ASQA)."""

    def format_context_for_prompt(self, chunks: list[ContextChunk]) -> list[dict]:
        if not chunks:
            return [{"type": "text", "text": "[No context provided]"}]

        parts = ["Context:\n"]
        for i, chunk in enumerate(chunks):
            parts.append(f"[Doc {i + 1}] {chunk.content}\n")

        return [{"type": "text", "text": "\n".join(parts)}]

    def get_chunk_text_repr(self, chunk: ContextChunk) -> str:
        return str(chunk.content)

    def format_for_retrieval_query(self, claims: list[str]) -> str:
        return " ".join(claims)
