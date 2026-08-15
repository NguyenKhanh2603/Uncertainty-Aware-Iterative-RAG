from __future__ import annotations
from pathlib import Path

from uncertainty_rag.modality.base import ContextChunk, ModalityHandler


class MultimodalHandler(ModalityHandler):
    def __init__(self, image_detail: str = "high"):
        self.image_detail = image_detail

    def format_context_for_prompt(self, chunks: list[ContextChunk]) -> list[dict]:
        if not chunks:
            return [{"type": "text", "text": "[No context provided]"}]
            
        blocks = [{"type": "text", "text": "Context:\n"}]
        for i, chunk in enumerate(chunks):
            if chunk.modality == "image":
                blocks.append({"type": "text", "text": f"[Image {i + 1}]"})
                # Sử dụng file:// path thay vì base64 encode để tiết kiệm bộ nhớ
                image_data = chunk.content
                if Path(image_data).exists():
                    blocks.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"file://{image_data}",
                            "detail": self.image_detail,
                        },
                    })
            elif chunk.modality == "table":
                blocks.append({"type": "text", "text": f"[Table {i + 1}]\n{chunk.content}\n"})
            else:
                blocks.append({"type": "text", "text": f"[Text {i + 1}]\n{chunk.content}\n"})
        return blocks

    def get_chunk_text_repr(self, chunk: ContextChunk) -> str:
        if chunk.modality == "image":
            return "[Image]"
        return str(chunk.content)

    def format_for_retrieval_query(self, claims: list[str]) -> str:
        return " ".join(claims)
