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

import re
import numpy as np
from rank_bm25 import BM25Okapi
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
    """Hybrid retriever using Dense (Sentence-Transformers) + Sparse (BM25) via RRF."""

    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2", config=None, alpha: float = 0.5) -> None:
        self.config = config
        self.encoder = SentenceTransformer(embedding_model_name)
        self.alpha = alpha
        self._documents = []
        self._embeddings = None
        self.bm25 = None

    def index(self, documents: list[ContextChunk], text_fn=None) -> None:
        self._documents = documents
        texts = [text_fn(d) if text_fn else str(d.content) for d in documents]
        # Vector Index
        self._embeddings = self.encoder.encode(texts, normalize_embeddings=True)
        # BM25 Index
        tokenized_corpus = [re.findall(r"\w+", doc.lower()) for doc in texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query_text: str, existing_chunk_ids: set[str], top_k: int = 5) -> list[ContextChunk]:
        if not self._documents: return []

        # 1. Vector Search
        query_emb = self.encoder.encode([query_text], normalize_embeddings=True)
        similarities = (self._embeddings @ query_emb.T).squeeze()
        # Handle single document case
        if similarities.ndim == 0:
            similarities = np.array([similarities])
        vector_ranks = np.argsort(-similarities)

        # 2. BM25 Search
        tokenized_query = re.findall(r"\w+", query_text.lower())
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_ranks = np.argsort(-bm25_scores)

        # 3. RRF (Reciprocal Rank Fusion)
        k_rrf = 60
        hybrid_scores = np.zeros(len(self._documents))

        for rank, doc_idx in enumerate(vector_ranks):
            hybrid_scores[doc_idx] += self.alpha * (1.0 / (k_rrf + rank + 1))
        for rank, doc_idx in enumerate(bm25_ranks):
            hybrid_scores[doc_idx] += (1.0 - self.alpha) * (1.0 / (k_rrf + rank + 1))

        sorted_indices = np.argsort(-hybrid_scores)

        # 4. Tránh Vòng lặp Vô hạn (Memory-Aware Retrieval)
        results = []
        for idx in sorted_indices:
            if len(results) >= top_k: break
            doc = self._documents[int(idx)]
            if doc.id in existing_chunk_ids:
                continue  # Bỏ qua các chunk đã thấy
            results.append(doc)
        return results


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
