"""Reranker model wrapper for scoring context chunk relevance and contradiction."""

from __future__ import annotations

from typing import Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class RerankerModel:
    """Cross-encoder Reranker model for Gray-Zone pruning.
    
    Default: BAAI/bge-reranker-v2-m (Multilingual, SOTA).
    Scores the relevance/contradiction of a chunk given the query and current claims.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m",
        device: Optional[str] = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_batch(
        self, query: str, texts: list[str], batch_size: int = 16
    ) -> list[float]:
        """Compute relevance scores for a query and a list of texts.
        
        For BGE Reranker, output is a single logit. We apply sigmoid to map to [0, 1].
        """
        if not texts:
            return []

        all_scores = []
        pairs = [[query, text] for text in texts]

        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)

            logits = self.model(**inputs, return_dict=True).logits.view(-1).float()
            
            # Apply sigmoid to normalize scores to [0, 1] range for thresholding
            scores = torch.sigmoid(logits).cpu().tolist()
            if isinstance(scores, float):
                scores = [scores]
                
            all_scores.extend(scores)

        return all_scores
