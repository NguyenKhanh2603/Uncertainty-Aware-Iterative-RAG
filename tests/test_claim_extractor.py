"""Unit tests for ClaimExtractor — JSON extraction, key token identification."""

from __future__ import annotations

import json
import pytest
from unittest.mock import MagicMock

from uncertainty_rag.core.claim_extractor import ClaimExtractor, _STOP_WORDS
from uncertainty_rag.core.sampler import Sample
from uncertainty_rag.models.llm_client import BaseLLMClient, SampleResult, TokenLogprob


class FakeLLMClient(BaseLLMClient):
    """Fake LLM that returns pre-configured responses."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self._call_idx = 0

    def generate(self, messages, n=1, temperature=0.7, max_tokens=1024,
                 logprobs=True, top_logprobs=5, json_mode=False) -> list[SampleResult]:
        if self._call_idx < len(self._responses):
            text = self._responses[self._call_idx]
            self._call_idx += 1
        else:
            text = self._responses[-1]
        return [SampleResult(text=text, token_logprobs=[], finish_reason="stop")]


class TestClaimExtraction:
    """Test claim extraction from LLM responses."""

    def test_valid_json_claims(self):
        """Well-formed JSON response → proper claim list."""
        response = json.dumps({"claims": [
            "Paris is the capital of France",
            "France is in Europe",
        ]})
        extractor = ClaimExtractor(llm_client=FakeLLMClient([response]), modality_type="text")
        claims = extractor.extract_claims("Paris is the capital of France. France is in Europe.")
        assert len(claims) == 2
        assert "Paris is the capital of France" in claims

    def test_malformed_json_fallback(self):
        """Malformed JSON → fallback to sentence splitting."""
        extractor = ClaimExtractor(
            llm_client=FakeLLMClient(["This is not valid JSON at all!"]),
            modality_type="text",
        )
        claims = extractor.extract_claims("Some text about something important.")
        # Should still return something via fallback
        assert isinstance(claims, list)

    def test_empty_text(self):
        """Empty text → empty claims."""
        extractor = ClaimExtractor(llm_client=FakeLLMClient(["[]"]), modality_type="text")
        claims = extractor.extract_claims("")
        assert claims == []

    def test_modality_prompt_selection(self):
        """Different modality types use different prompts."""
        text_ext = ClaimExtractor(llm_client=FakeLLMClient(["{}"]), modality_type="text")
        table_ext = ClaimExtractor(llm_client=FakeLLMClient(["{}"]), modality_type="table")
        image_ext = ClaimExtractor(llm_client=FakeLLMClient(["{}"]), modality_type="multimodal")

        # Each should select a different prompt template
        assert "factual claims" in text_ext._get_prompt_template().lower()
        assert "numerical" in table_ext._get_prompt_template().lower()
        assert "visual" in image_ext._get_prompt_template().lower()


class TestKeyTokenIdentification:
    """Test key token filtering for aleatoric estimation."""

    def test_stop_words_filtered(self):
        """Stop words and punctuation should be filtered out."""
        logprobs = [
            TokenLogprob(token="The", logprob=-0.1),
            TokenLogprob(token=" capital", logprob=-0.5),
            TokenLogprob(token=" is", logprob=-0.01),
            TokenLogprob(token=" Paris", logprob=-0.3),
            TokenLogprob(token=".", logprob=-0.001),
        ]
        key_tokens = ClaimExtractor.identify_key_tokens(["The capital is Paris"], logprobs)
        token_texts = [t.token.strip().lower() for t in key_tokens]
        assert "the" not in token_texts
        assert "is" not in token_texts
        assert "." not in token_texts
        assert "capital" in token_texts
        assert "paris" in token_texts

    def test_empty_logprobs(self):
        """Empty logprobs → empty key tokens."""
        key_tokens = ClaimExtractor.identify_key_tokens(["test"], [])
        assert key_tokens == []

    def test_numbers_preserved(self):
        """Numbers should be kept as key tokens."""
        logprobs = [
            TokenLogprob(token="42", logprob=-0.5),
            TokenLogprob(token=" million", logprob=-0.3),
        ]
        key_tokens = ClaimExtractor.identify_key_tokens(["42 million"], logprobs)
        token_texts = [t.token.strip() for t in key_tokens]
        assert "42" in token_texts
        assert "million" in token_texts


class TestExtractAll:
    """Test batch extraction for all samples."""

    def test_extract_all_populates_claims(self):
        """extract_all should populate claims for all samples."""
        responses = [
            json.dumps({"claims": ["claim A"]}),
            json.dumps({"claims": ["claim B"]}),
        ]
        extractor = ClaimExtractor(
            llm_client=FakeLLMClient(responses), modality_type="text"
        )
        samples = [
            Sample(text="Answer A", token_logprobs=[]),
            Sample(text="Answer B", token_logprobs=[]),
        ]
        result = extractor.extract_all(samples)
        assert len(result) == 2
        assert result[0].claims == ["claim A"]
        assert result[1].claims == ["claim B"]
