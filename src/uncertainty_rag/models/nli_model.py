"""NLI model wrapper for bidirectional entailment checking and contradiction detection."""

from __future__ import annotations

from typing import Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class NLIModel:
    """Cross-encoder NLI model for semantic equivalence and contradiction detection.

    Uses DeBERTa-v3-base fine-tuned on NLI. Labels: 0=contradiction, 1=neutral, 2=entailment.
    NLI operates on extracted *text claims* regardless of source modality — this is
    the key design choice that makes the framework modality-agnostic.
    """

    # Label indices for cross-encoder/nli-deberta-v3-base
    CONTRADICTION = 0
    ENTAILMENT = 1
    NEUTRAL = 2

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-base",
        device: Optional[str] = None,
        entailment_threshold: float = 0.5,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.entailment_threshold = entailment_threshold

    @torch.no_grad()
    def predict(self, premise: str, hypothesis: str) -> tuple[float, float, float]:
        """Predict NLI scores for a single (premise, hypothesis) pair.

        Returns:
            (contradiction, neutral, entailment) probabilities.
        """
        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)

        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze().cpu().tolist()

        if isinstance(probs, float):
            # Edge case: single-class output
            return (0.0, 0.0, probs)

        return (probs[self.CONTRADICTION], probs[self.NEUTRAL], probs[self.ENTAILMENT])

    @torch.no_grad()
    def predict_batch(
        self, pairs: list[tuple[str, str]], batch_size: int = 32
    ) -> list[tuple[float, float, float]]:
        """Batch prediction for efficiency.

        Args:
            pairs: List of (premise, hypothesis) tuples.
            batch_size: Inference batch size.

        Returns:
            List of (contradiction, neutral, entailment) probability tuples.
        """
        all_results = []

        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            premises = [p for p, _ in batch]
            hypotheses = [h for _, h in batch]

            inputs = self.tokenizer(
                premises,
                hypotheses,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            ).to(self.device)

            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().tolist()

            for prob_row in probs:
                all_results.append(
                    (prob_row[self.CONTRADICTION], prob_row[self.NEUTRAL], prob_row[self.ENTAILMENT])
                )

        return all_results

    def bidirectional_entailment(self, text_a: str, text_b: str) -> bool:
        """Check if text_a and text_b are semantically equivalent via bidirectional entailment.

        Two texts are equivalent if A entails B AND B entails A.
        """
        _, _, entail_ab = self.predict(text_a, text_b)
        _, _, entail_ba = self.predict(text_b, text_a)
        return (
            entail_ab >= self.entailment_threshold and entail_ba >= self.entailment_threshold
        )

    def is_contradiction(self, text_a: str, text_b: str) -> bool:
        """Check if two texts contradict each other (either direction)."""
        contra_ab, _, _ = self.predict(text_a, text_b)
        return contra_ab >= self.entailment_threshold

    def contradiction_score(self, text_a: str, text_b: str) -> float:
        """Return the contradiction probability between two texts."""
        contra_ab, _, _ = self.predict(text_a, text_b)
        return contra_ab
