"""Shared data, retrieval, and answer utilities for the FLARE benchmark."""

from __future__ import annotations

import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
from rank_bm25 import BM25Okapi


TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*")


def read_json(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return value


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    return rows


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


class BM25Corpus:
    """In-memory BM25 index over deterministic corpus chunks."""

    def __init__(self, chunks: list[dict]) -> None:
        if not chunks:
            raise ValueError("Cannot build a BM25 index over an empty corpus")
        self.chunks = chunks
        tokenized = [tokenize(f"{chunk['title']} {chunk['text']}") for chunk in chunks]
        self.index = BM25Okapi(tokenized)
        self.document_urls = []
        starts = []
        previous_url = None
        closed_urls: set[str] = set()
        for index, chunk in enumerate(chunks):
            url = chunk["url"]
            if url != previous_url:
                if url in closed_urls:
                    raise ValueError("Corpus chunks for each document must be contiguous")
                if previous_url is not None:
                    closed_urls.add(previous_url)
                self.document_urls.append(url)
                starts.append(index)
                previous_url = url
        self.document_starts = np.asarray(starts, dtype=np.int64)
        self.document_ends = np.asarray(starts[1:] + [len(chunks)], dtype=np.int64)

    @classmethod
    def from_jsonl(cls, path: Path) -> "BM25Corpus":
        return cls(read_jsonl(path))

    def search(self, query: str, top_k: int) -> list[dict]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        scores = self.index.get_scores(query_tokens)
        document_scores = np.maximum.reduceat(scores, self.document_starts)
        ranked_documents = np.argsort(-document_scores)[:top_k]
        results = []
        for document_index in ranked_documents:
            document_index = int(document_index)
            start = int(self.document_starts[document_index])
            end = int(self.document_ends[document_index])
            index = start + int(np.argmax(scores[start:end]))
            chunk = self.chunks[index]
            result = dict(chunk)
            result["score"] = float(scores[index])
            results.append(result)
        return results


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def token_f1(prediction: str, gold: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not prediction_tokens or not gold_tokens:
        return float(prediction_tokens == gold_tokens)
    common = Counter(prediction_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0
