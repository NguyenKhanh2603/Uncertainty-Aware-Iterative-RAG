"""Active retriever — hypothesis-driven document retrieval using ChromaDB/FAISS.

Retrieval workflow (non-circular, addressing W5):
1. Take the highest-probability concept's claims from the uncertainty profiling step
2. Use claims as the search query — zero extra cost (reuses existing output)
3. Retrieve top-k documents from vector DB
4. Deduplicate against existing context
5. Return new chunks to add to context

The implicit EIG evaluation happens in the next iteration when SE_total
is recomputed — the iterative loop IS the information gain evaluator.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from uncertainty_rag.config import RetrievalConfig
from uncertainty_rag.modality.base import ContextChunk, ModalityHandler


class BaseRetriever(ABC):
    """Abstract retriever interface."""

    @abstractmethod
    def retrieve(
        self,
        query_text: str,
        existing_chunk_ids: set[str],
        top_k: int = 5,
    ) -> list[ContextChunk]:
        """Retrieve documents, excluding already-seen chunks."""
        ...


class DenseRetriever(BaseRetriever):
    """Dense retriever using sentence-transformers embeddings and ChromaDB.

    For text-only: searches Wikipedia/corpus
    For TAT-QA: searches within document tables/paragraphs
    For WebQA: searches within candidate source pool
    """

    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        config: Optional[RetrievalConfig] = None,
    ) -> None:
        self.config = config or RetrievalConfig()
        self.encoder = SentenceTransformer(embedding_model_name)
        # In-memory document store (for simplicity; swap with ChromaDB for scale)
        self._documents: list[ContextChunk] = []
        self._embeddings: Optional[np.ndarray] = None

    def index(self, documents: list[ContextChunk], text_fn=None) -> None:
        """Index a set of documents for retrieval.

        Args:
            documents: Documents to index.
            text_fn: Optional function to extract text from a chunk for embedding.
                     Defaults to str(chunk.content).
        """
        self._documents = documents
        texts = [text_fn(d) if text_fn else str(d.content) for d in documents]
        self._embeddings = self.encoder.encode(texts, normalize_embeddings=True)

    def retrieve(
        self,
        query_text: str,
        existing_chunk_ids: set[str],
        top_k: int = 5,
    ) -> list[ContextChunk]:
        """Retrieve top-k documents most relevant to the query.

        Args:
            query_text: Search query (typically concatenated hypothesis claims).
            existing_chunk_ids: IDs of chunks already in context (for dedup).
            top_k: Number of documents to return.

        Returns:
            List of new ContextChunk objects to add to context.
        """
        if self._embeddings is None or len(self._documents) == 0:
            return []

        # Encode query
        query_emb = self.encoder.encode([query_text], normalize_embeddings=True)

        # Compute cosine similarities
        similarities = (self._embeddings @ query_emb.T).squeeze()

        # Sort by similarity (descending)
        sorted_indices = np.argsort(-similarities)

        # Filter out existing chunks and deduplicate
        results: list[ContextChunk] = []
        for idx in sorted_indices:
            if len(results) >= top_k:
                break

            doc = self._documents[int(idx)]

            # Skip if already in context
            if doc.id in existing_chunk_ids:
                continue

            # Skip if too similar to an existing context chunk (cosine dedup)
            if self._is_duplicate(int(idx), existing_chunk_ids):
                continue

            results.append(doc)

        return results

    def _is_duplicate(self, candidate_idx: int, existing_ids: set[str]) -> bool:
        """Check if candidate is too similar to any existing chunk."""
        if self._embeddings is None:
            return False

        candidate_emb = self._embeddings[candidate_idx]
        for i, doc in enumerate(self._documents):
            if doc.id in existing_ids:
                existing_emb = self._embeddings[i]
                sim = float(np.dot(candidate_emb, existing_emb))
                if sim >= self.config.dedup_cosine_threshold:
                    return True
        return False


class ActiveRetriever:
    """Hypothesis-driven active retriever.

    Wraps a BaseRetriever and handles:
    1. Converting concept claims to search queries
    2. Deduplication against existing context
    3. Formatting results as ContextChunks
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        modality_handler: ModalityHandler,
        config: Optional[RetrievalConfig] = None,
    ) -> None:
        self.retriever = retriever
        self.handler = modality_handler
        self.config = config or RetrievalConfig()

    def retrieve(
        self,
        hypothesis_claims: list[str],
        existing_context: list[ContextChunk],
        top_k: Optional[int] = None,
    ) -> list[ContextChunk]:
        """Retrieve new evidence based on hypothesis claims.

        Args:
            hypothesis_claims: Claims from the highest-probability concept.
            existing_context: Current context chunks (for dedup).
            top_k: Override top_k from config.

        Returns:
            New ContextChunks to add to context.
        """
        if not hypothesis_claims:
            return []

        # Build query from claims
        query_text = self.handler.format_for_retrieval_query(hypothesis_claims)

        # Get existing IDs for dedup
        existing_ids = {chunk.id for chunk in existing_context}

        # Retrieve
        k = top_k or self.config.top_k
        return self.retriever.retrieve(query_text, existing_ids, k)
