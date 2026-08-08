"""Text-only dataset loaders: NQ, TriviaQA, HotpotQA, PopQA, ASQA."""

from __future__ import annotations

from typing import Optional

from datasets import load_dataset

from eval.datasets.loader import DatasetLoader, EvalExample
from uncertainty_rag.modality.base import ContextChunk, ModalityHandler
from uncertainty_rag.modality.text_handler import TextHandler


class NaturalQuestionsLoader(DatasetLoader):
    """NaturalQuestions Open — standard single-hop QA."""

    @property
    def name(self) -> str:
        return "natural_questions"

    def load(self, split: str = "validation", max_examples: Optional[int] = None) -> list[EvalExample]:
        ds = load_dataset("nq_open", split=split)
        examples = []
        for i, row in enumerate(ds):
            if max_examples and i >= max_examples:
                break
            answers = row["answer"] if isinstance(row["answer"], list) else [row["answer"]]
            examples.append(
                EvalExample(
                    query_id=f"nq_{i}",
                    query=row["question"],
                    gold_answers=answers,
                    modality="text",
                )
            )
        return examples

    def get_modality_handler(self) -> ModalityHandler:
        return TextHandler()

    def get_metrics(self) -> list[str]:
        return ["em", "f1"]


class TriviaQALoader(DatasetLoader):
    """TriviaQA — diverse trivia questions."""

    @property
    def name(self) -> str:
        return "triviaqa"

    def load(self, split: str = "validation", max_examples: Optional[int] = None) -> list[EvalExample]:
        ds = load_dataset("trivia_qa", "rc.nocontext", split=split)
        examples = []
        for i, row in enumerate(ds):
            if max_examples and i >= max_examples:
                break
            answers = row["answer"]["aliases"] if row.get("answer") else [row.get("answer", {}).get("value", "")]
            examples.append(
                EvalExample(
                    query_id=f"tqa_{i}",
                    query=row["question"],
                    gold_answers=answers,
                    modality="text",
                )
            )
        return examples

    def get_modality_handler(self) -> ModalityHandler:
        return TextHandler()

    def get_metrics(self) -> list[str]:
        return ["em", "f1"]


class HotpotQALoader(DatasetLoader):
    """HotpotQA — multi-hop reasoning (most relevant for iterative retrieval)."""

    @property
    def name(self) -> str:
        return "hotpotqa"

    def load(self, split: str = "validation", max_examples: Optional[int] = None) -> list[EvalExample]:
        ds = load_dataset("hotpot_qa", "distractor", split=split)
        examples = []
        for i, row in enumerate(ds):
            if max_examples and i >= max_examples:
                break
            # Build context from provided documents
            context_chunks = []
            if row.get("context"):
                titles = row["context"].get("title", [])
                sentences = row["context"].get("sentences", [])
                for j, (title, sents) in enumerate(zip(titles, sentences)):
                    text = f"{title}: {' '.join(sents)}"
                    context_chunks.append(
                        ContextChunk(id=f"hpqa_{i}_doc{j}", content=text, modality="text")
                    )

            examples.append(
                EvalExample(
                    query_id=f"hpqa_{i}",
                    query=row["question"],
                    gold_answers=[row["answer"]],
                    context_chunks=context_chunks,
                    modality="text",
                    metadata={"level": row.get("level", ""), "type": row.get("type", "")},
                )
            )
        return examples

    def get_modality_handler(self) -> ModalityHandler:
        return TextHandler()

    def get_metrics(self) -> list[str]:
        return ["em", "f1"]


class PopQALoader(DatasetLoader):
    """PopQA — long-tail entity knowledge."""

    @property
    def name(self) -> str:
        return "popqa"

    def load(self, split: str = "test", max_examples: Optional[int] = None) -> list[EvalExample]:
        ds = load_dataset("akariasai/PopQA", split=split)
        examples = []
        for i, row in enumerate(ds):
            if max_examples and i >= max_examples:
                break
            answers = row.get("possible_answers", [row.get("obj", "")])
            if isinstance(answers, str):
                answers = [answers]
            examples.append(
                EvalExample(
                    query_id=f"pqa_{i}",
                    query=row["question"],
                    gold_answers=answers,
                    modality="text",
                    metadata={"popularity": row.get("s_pop", 0)},
                )
            )
        return examples

    def get_modality_handler(self) -> ModalityHandler:
        return TextHandler()

    def get_metrics(self) -> list[str]:
        return ["em", "f1"]


# ── Registry ────────────────────────────────────────────────────────────────────

TEXT_DATASETS = {
    "nq": NaturalQuestionsLoader,
    "triviaqa": TriviaQALoader,
    "hotpotqa": HotpotQALoader,
    "popqa": PopQALoader,
}
