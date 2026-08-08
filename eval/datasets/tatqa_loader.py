"""TAT-QA dataset loader — hybrid table+text financial QA."""

from __future__ import annotations

from typing import Optional

from eval.datasets.loader import DatasetLoader, EvalExample
from uncertainty_rag.modality.base import ContextChunk, ModalityHandler
from uncertainty_rag.modality.table_handler import TableHandler


class TATQALoader(DatasetLoader):
    """TAT-QA: 16,552 questions over hybrid table+text financial documents.

    Answer types: span, multi-span, count, arithmetic.
    Tests numerical reasoning with noisy/contradictory context.
    """

    @property
    def name(self) -> str:
        return "tatqa"

    def load(
        self, split: str = "validation", max_examples: Optional[int] = None
    ) -> list[EvalExample]:
        try:
            from datasets import load_dataset
            ds = load_dataset("nextmove/tat-qa", split=split)
        except Exception:
            # Fallback: load from local file
            import json
            from pathlib import Path

            data_path = Path(f"data/tatqa/{split}.json")
            if not data_path.exists():
                raise FileNotFoundError(
                    f"TAT-QA data not found at {data_path}. "
                    "Download from https://github.com/NExTplusplus/TAT-QA"
                )
            with open(data_path) as f:
                ds = json.load(f)

        examples = []
        idx = 0

        for entry in ds:
            if max_examples and idx >= max_examples:
                break

            # Parse table
            table = entry.get("table", {})
            table_chunk = ContextChunk(
                id=f"tatqa_{idx}_table",
                content=table,
                modality="table",
                metadata={"source": "table"},
            )

            # Parse text paragraphs
            paragraphs = entry.get("paragraphs", [])
            text_chunks = []
            for j, para in enumerate(paragraphs):
                text = para.get("text", para) if isinstance(para, dict) else str(para)
                text_chunks.append(
                    ContextChunk(
                        id=f"tatqa_{idx}_text{j}",
                        content=text,
                        modality="text",
                        metadata={"source": "paragraph"},
                    )
                )

            # Parse questions for this context
            questions = entry.get("questions", [])
            for q in questions:
                if max_examples and idx >= max_examples:
                    break

                answer = q.get("answer", "")
                if isinstance(answer, list):
                    gold_answers = [str(a) for a in answer]
                else:
                    gold_answers = [str(answer)]

                examples.append(
                    EvalExample(
                        query_id=q.get("uid", f"tatqa_{idx}"),
                        query=q.get("question", ""),
                        gold_answers=gold_answers,
                        context_chunks=[table_chunk] + text_chunks,
                        modality="table",
                        metadata={
                            "answer_type": q.get("answer_type", ""),
                            "scale": q.get("scale", ""),
                            "derivation": q.get("derivation", ""),
                        },
                    )
                )
                idx += 1

        return examples

    def get_modality_handler(self) -> ModalityHandler:
        return TableHandler(table_format="markdown")

    def get_metrics(self) -> list[str]:
        return ["em", "f1", "numerical_accuracy"]
