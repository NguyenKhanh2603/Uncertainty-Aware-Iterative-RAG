"""Standalone copy of this project's claim-level UQ scoring logic.

This module purposely has no import from ``src/uncertainty_rag``.  It scores
the sampled outputs already produced by ``run_webq_paper_baselines.py``.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from statistics import mean
from typing import Any

import torch
from openai import OpenAI
from transformers import AutoModelForSequenceClassification, AutoTokenizer


TEXT_CLAIM_PROMPT = """\
Extract all distinct, atomic factual claims from the following text.
Each claim should be a single, self-contained statement that can be independently verified.
Return a JSON object with key "claims" containing an array of strings.

Text: {text}"""

STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might", "shall", "can", "need",
    "dare", "ought", "to", "of", "in", "for", "on", "with", "at", "by", "from", "as", "into", "through",
    "during", "before", "after", "above", "below", "between", "out", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "each", "every",
    "both", "few", "more", "most", "other", "some", "such", "no", "not", "only", "own", "same", "so",
    "than", "too", "very", "just", "because", "but", "and", "or", "if", "while", "although", "this",
    "that", "these", "those", "it", "its", "i", "me", "my", "we", "our", "you", "your", "he", "him",
    "his", "she", "her", "they", "them", "their", "what", "which", "who", "whom", "whose",
})


@dataclass
class Sample:
    text: str
    token_logprobs: list[dict[str, Any]]
    claims: list[str]


class ClaimExtractor:
    """Standalone equivalent of the project's text ClaimExtractor."""

    def __init__(self, client: OpenAI, model: str, max_tokens: int = 256) -> None:
        self.client = client
        self.model = model
        self.max_tokens = max_tokens

    def extract(self, text: str) -> list[str]:
        if not text.strip():
            return []
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": TEXT_CLAIM_PROMPT.format(text=text)}],
            temperature=0.0,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or ""
        try:
            parsed = json.loads(content)
            claims = parsed.get("claims", [])
            if isinstance(claims, list) and all(isinstance(claim, str) for claim in claims):
                return [claim.strip() for claim in claims if claim.strip()]
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass
        match = re.search(r"\[([^\]]+)\]", content, re.DOTALL)
        if match:
            try:
                return [str(claim).strip() for claim in json.loads("[" + match.group(1) + "]") if str(claim).strip()]
            except json.JSONDecodeError:
                pass
        return [part.strip() for part in re.split(r"[.!?]+", content) if len(part.strip()) > 10] or [text]


class ClaimNLI:
    """Copy of the project's NLI model wrapper and label mapping."""

    def __init__(self, model_name: str, threshold: float, batch_size: int) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.threshold = threshold
        self.batch_size = batch_size

    @torch.no_grad()
    def equivalent_pairs(self, pairs: list[tuple[str, str]]) -> list[bool]:
        if not pairs:
            return []
        forward: list[float] = []
        backward: list[float] = []
        for firsts, seconds, target in (([a for a, _ in pairs], [b for _, b in pairs], forward), ([b for _, b in pairs], [a for a, _ in pairs], backward)):
            for index in range(0, len(firsts), self.batch_size):
                inputs = self.tokenizer(firsts[index:index + self.batch_size], seconds[index:index + self.batch_size], return_tensors="pt", truncation=True, max_length=512, padding=True).to(self.device)
                probabilities = torch.softmax(self.model(**inputs).logits, dim=-1).cpu().tolist()
                # Project NLIModel uses label 1 as entailment for this exact cross-encoder checkpoint.
                target.extend(float(row[1]) for row in probabilities)
        return [left >= self.threshold and right >= self.threshold for left, right in zip(forward, backward)]


def cluster_samples(samples: list[Sample], nli: ClaimNLI) -> list[list[int]]:
    texts = [" ".join(sample.claims) for sample in samples]
    pairs = [(texts[i], texts[j]) for i in range(len(texts)) for j in range(i + 1, len(texts)) if texts[i] and texts[j]]
    indices = [(i, j) for i in range(len(texts)) for j in range(i + 1, len(texts)) if texts[i] and texts[j]]
    parent = list(range(len(samples)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for (left, right), equivalent in zip(indices, nli.equivalent_pairs(pairs)):
        if equivalent:
            parent[find(left)] = find(right)
    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(samples)):
        groups[find(index)].append(index)
    return sorted(groups.values(), key=len, reverse=True)


def semantic_entropy(clusters: list[list[int]], total: int) -> float:
    return -sum((len(cluster) / total) * math.log2(len(cluster) / total) for cluster in clusters if cluster)


def token_uncertainty(samples: list[Sample]) -> float:
    per_sample: list[float] = []
    for sample in samples:
        logprobs = []
        for token_info in sample.token_logprobs:
            token = str(token_info.get("token", "")).strip().lower()
            logprob = float(token_info.get("logprob", -math.inf))
            if not token or token in STOP_WORDS or len(token) <= 1 or logprob <= -100:
                continue
            if all(character in ".,;:!?()-[]{}\"'`\n\t /\\|" for character in token):
                continue
            logprobs.append(logprob)
        if logprobs:
            per_sample.append(-mean(logprobs) / math.log(2))
    return mean(per_sample) if per_sample else 0.0
