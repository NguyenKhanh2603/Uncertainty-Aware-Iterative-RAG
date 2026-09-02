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
                caption = chunk.metadata.get("caption") or chunk.metadata.get("title")
                label = f"[Image {i + 1}]"
                if caption:
                    label += f" {caption}"
                blocks.append({"type": "text", "text": label})

                image_data = str(chunk.content)
                if image_data.startswith(("http://", "https://", "data:")):
                    image_url = image_data
                else:
                    local_path = image_data[7:] if image_data.startswith("file://") else image_data
                    image_url = (
                        f"file://{Path(local_path).resolve()}"
                        if Path(local_path).exists()
                        else None
                    )

                if image_url:
                    blocks.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url,
                                "detail": self.image_detail,
                            },
                        }
                    )
            elif chunk.modality == "table":
                blocks.append({"type": "text", "text": f"[Table {i + 1}]\n{chunk.content}\n"})
            else:
                blocks.append({"type": "text", "text": f"[Text {i + 1}]\n{chunk.content}\n"})
        return blocks

    def get_chunk_text_repr(self, chunk: ContextChunk) -> str:
        if chunk.modality == "image":
            caption = chunk.metadata.get("caption") or chunk.metadata.get("title")
            return f"[Image: {caption}]" if caption else "[Image]"
        return str(chunk.content)

    def format_for_retrieval_query(self, claims: list[str]) -> str:
        return " ".join(claims)
