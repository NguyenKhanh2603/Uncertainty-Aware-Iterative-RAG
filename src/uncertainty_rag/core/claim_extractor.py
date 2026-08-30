"""Atomic claim extraction from LLM-generated text using structured JSON output.

Claim extraction bridges the modality gap: VLMs produce text claims from images,
table-aware prompts produce numerical claims from tables. Once extracted, all claims
are plain text — enabling modality-agnostic downstream processing.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from uncertainty_rag.core.sampler import Sample
from uncertainty_rag.models.llm_client import BaseLLMClient, TokenLogprob


# ── Extraction Prompts (modality-aware) ─────────────────────────────────────────

TEXT_CLAIM_PROMPT = """\
Extract all distinct, atomic factual claims explicitly stated in the following Answer.

CRITICAL RULES:
1. Each claim MUST be a complete, self-contained factual sentence with SUBJECT + RELATIONSHIP + OBJECT/ATTRIBUTE.
2. You MUST use the context from the Question to understand what the Answer refers to, especially if the Answer is short.
3. DO NOT introduce external knowledge or unstated assumptions. Extract ONLY facts derived from combining the Question and the Answer.
4. DO NOT use pronouns such as "he", "she", "it", or "they". Use explicit entity names.
5. If a statement contains multiple independent facts, decompose it into separate atomic claims.
6. Return a JSON object with key "claims" containing an array of strings.

Question: {query}
Answer: {text}
"""

TABLE_CLAIM_PROMPT = """\
Extract all distinct, atomic factual claims explicitly represented in the following Answer about tabular/numerical data.

CRITICAL RULES:
1. Each claim MUST be a complete, self-contained factual sentence with SUBJECT + RELATIONSHIP + VALUE/OBJECT.
2. You MUST use the context from the Question to understand what the numerical/tabular Answer refers to.
3. DO NOT introduce external knowledge. Extract ONLY information explicitly represented in the Question and Answer.
4. Preserve numerical information exactly, including values, units, and dates.
5. Return a JSON object with key "claims" containing an array of strings.

Question: {query}
Answer: {text}
"""

IMAGE_CLAIM_PROMPT = """\
Extract all distinct, atomic factual claims explicitly stated in the following Answer about visual content.

CRITICAL RULES:
1. Each claim MUST be a complete, self-contained factual sentence with SUBJECT + ATTRIBUTE/RELATIONSHIP + DESCRIPTION.
2. You MUST use the context from the Question to understand what the Answer refers to.
3. DO NOT infer, speculate, or introduce visual information that is not explicitly stated in the Answer or Question.
4. DO NOT use pronouns. Use explicit object names.
5. Return a JSON object with key "claims" containing an array of strings.

Question: {query}
Answer: {text}
"""


# ── Stop words for key-token filtering ──────────────────────────────────────────

_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such", "no",
    "not", "only", "own", "same", "so", "than", "too", "very", "just",
    "because", "but", "and", "or", "if", "while", "although", "this",
    "that", "these", "those", "it", "its", "i", "me", "my", "we", "our",
    "you", "your", "he", "him", "his", "she", "her", "they", "them",
    "their", "what", "which", "who", "whom", "whose",
})


class ClaimExtractor:
    """Extract atomic factual claims from generated text.

    Uses structured JSON-mode extraction to get reliable claim lists.
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        modality_type: str = "text",
    ) -> None:
        self.llm = llm_client
        self.modality_type = modality_type

    def _get_prompt_template(self) -> str:
        if self.modality_type == "table":
            return TABLE_CLAIM_PROMPT
        elif self.modality_type in ("image", "multimodal"):
            return IMAGE_CLAIM_PROMPT
        return TEXT_CLAIM_PROMPT

    def extract_claims(self, query: str, text: str) -> list[str]:
        """Extract atomic claims from a text using LLM with JSON mode."""
        if not text.strip():
            return []

        prompt = self._get_prompt_template().format(query=query, text=text)
        results = self.llm.generate(
            messages=[{"role": "user", "content": prompt}],
            n=1,
            temperature=0.0,
            max_tokens=800,
            logprobs=False,
            json_mode=True,
        )

        if not results:
            return [text]  # Fallback: treat entire text as one claim

        try:
            parsed = json.loads(results[0].text)
            if isinstance(parsed, list):
                claims = parsed
            else:
                claims = parsed.get("claims", [])
                
            if isinstance(claims, list) and all(isinstance(c, str) for c in claims):
                return [c.strip() for c in claims if c.strip()]
        except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
            pass

        # Fallback: try to extract from text directly
        return self._fallback_extract(results[0].text)

    def _fallback_extract(self, text: str) -> list[str]:
        """Fallback extraction when JSON parsing fails."""
        # Try to find JSON array in the text
        match = re.search(r'\[([^\]]+)\]', text, re.DOTALL)
        if match:
            try:
                claims = json.loads(f"[{match.group(1)}]")
                return [str(c).strip() for c in claims if str(c).strip()]
            except json.JSONDecodeError:
                pass

        # Last resort: split by sentence
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    def extract_all(self, query: str, samples: list[Sample]) -> list[Sample]:
        """Extract claims for all samples and identify key tokens."""
        for sample in samples:
            sample.claims = self.extract_claims(query, sample.text)
            sample.key_token_logprobs = self.identify_key_tokens(
                sample.claims, sample.token_logprobs
            )
        return samples

    @staticmethod
    def identify_key_tokens(
        claims: list[str], token_logprobs: list[TokenLogprob]
    ) -> list[TokenLogprob]:
        """Identify content-bearing tokens (nouns, verbs, numbers) for aleatoric estimation.

        Filters out stop words, punctuation, and structural tokens to focus on
        tokens that carry semantic meaning. Uses simple heuristics instead of
        a full POS tagger for speed.
        """
        if not token_logprobs:
            return []

        key_tokens = []
        for tlp in token_logprobs:
            token_lower = tlp.token.strip().lower()
            # Skip empty, whitespace-only, punctuation, and stop words
            if not token_lower:
                continue
            if all(c in ".,;:!?()-[]{}\"'`\n\t /\\|" for c in token_lower):
                continue
            if token_lower in _STOP_WORDS:
                continue
            if len(token_lower) <= 1:
                continue
            key_tokens.append(tlp)

        return key_tokens
