"""WebQA dataset loader — multimodal multihop QA with images + text + distractors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from eval.datasets.loader import DatasetLoader, EvalExample
from uncertainty_rag.modality.base import ContextChunk, ModalityHandler
from uncertainty_rag.modality.image_handler import ImageHandler


class WebQALoader(DatasetLoader):
    """WebQA: Multimodal multihop QA with images + text snippets + distractors.

    Contains explicit distractor sources — directly tests pruning capability.
    Sources can be images (with captions) or text snippets.
    """

    def __init__(self, data_dir: str = "data/webqa", image_dir: Optional[str] = None) -> None:
        self.data_dir = Path(data_dir)
        self.image_dir = Path(image_dir) if image_dir else self.data_dir / "images"

    @property
    def name(self) -> str:
        return "webqa"

    def load(
        self, split: str = "validation", max_examples: Optional[int] = None
    ) -> list[EvalExample]:
        data_path = self.data_dir / f"{split}.json"
        if not data_path.exists():
            raise FileNotFoundError(
                f"WebQA data not found at {data_path}. "
                "Download from https://github.com/WebQnA/WebQA"
            )

        with open(data_path) as f:
            data = json.load(f)

        examples = []
        for idx, (qid, entry) in enumerate(data.items()):
            if max_examples and idx >= max_examples:
                break

            query = entry.get("Q", "")
            answer = entry.get("A", [""])
            if isinstance(answer, str):
                gold_answers = [answer]
            elif isinstance(answer, list):
                gold_answers = [str(a) for a in answer]
            else:
                gold_answers = [str(answer)]

            # Build context chunks from sources
            context_chunks = []

            # Positive sources (ground truth relevant)
            for i, src in enumerate(entry.get("img_posFacts", [])):
                context_chunks.append(
                    ContextChunk(
                        id=f"webqa_{qid}_img_pos_{i}",
                        content=src.get("image_url", src.get("image_id", "")),
                        modality="image",
                        metadata={
                            "caption": src.get("caption", ""),
                            "is_positive": True,
                            "source_type": "image",
                        },
                    )
                )

            for i, src in enumerate(entry.get("txt_posFacts", [])):
                context_chunks.append(
                    ContextChunk(
                        id=f"webqa_{qid}_txt_pos_{i}",
                        content=src.get("fact", src.get("snippet", "")),
                        modality="text",
                        metadata={"is_positive": True, "source_type": "text"},
                    )
                )

            # Negative sources (distractors — should be pruned)
            for i, src in enumerate(entry.get("img_negFacts", [])):
                context_chunks.append(
                    ContextChunk(
                        id=f"webqa_{qid}_img_neg_{i}",
                        content=src.get("image_url", src.get("image_id", "")),
                        modality="image",
                        metadata={
                            "caption": src.get("caption", ""),
                            "is_positive": False,
                            "source_type": "image",
                        },
                    )
                )

            for i, src in enumerate(entry.get("txt_negFacts", [])):
                context_chunks.append(
                    ContextChunk(
                        id=f"webqa_{qid}_txt_neg_{i}",
                        content=src.get("fact", src.get("snippet", "")),
                        modality="text",
                        metadata={"is_positive": False, "source_type": "text"},
                    )
                )

            examples.append(
                EvalExample(
                    query_id=f"webqa_{qid}",
                    query=query,
                    gold_answers=gold_answers,
                    context_chunks=context_chunks,
                    modality="multimodal",
                    metadata={
                        "qcate": entry.get("Qcate", ""),
                        "num_pos": len(entry.get("img_posFacts", []))
                        + len(entry.get("txt_posFacts", [])),
                        "num_neg": len(entry.get("img_negFacts", []))
                        + len(entry.get("txt_negFacts", [])),
                    },
                )
            )

        return examples

    def get_modality_handler(self) -> ModalityHandler:
        return ImageHandler(image_detail="high")

    def get_metrics(self) -> list[str]:
        return ["em", "f1", "rouge_l"]
