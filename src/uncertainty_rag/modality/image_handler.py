"""Image+Text modality handler for WebQA style multimodal contexts."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Optional

from uncertainty_rag.modality.base import ContextChunk, ModalityHandler


def _encode_image_base64(image_path: str) -> str:
    """Read and base64-encode an image file."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _get_image_media_type(path: str) -> str:
    """Infer MIME type from file extension."""
    ext = Path(path).suffix.lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }.get(ext, "image/jpeg")


class ImageHandler(ModalityHandler):
    """Handles WebQA style image+text contexts.

    Images are base64-encoded and included as image_url content blocks in
    the OpenAI message format. Once the VLM generates claims about visual
    content, the downstream uncertainty pipeline is identical to text.
    """

    def __init__(self, image_detail: str = "high") -> None:
        self.image_detail = image_detail

    def format_context_for_prompt(self, chunks: list[ContextChunk]) -> list[dict]:
        if not chunks:
            return [{"type": "text", "text": "[No context provided]"}]

        content_blocks: list[dict] = []
        content_blocks.append({"type": "text", "text": "Context:\n"})

        for i, chunk in enumerate(chunks):
            if chunk.modality == "image":
                # Add image label
                caption = chunk.metadata.get("caption", "")
                label = f"[Image {i + 1}]"
                if caption:
                    label += f" Caption: {caption}"
                content_blocks.append({"type": "text", "text": label})

                # Add image content
                image_data = chunk.content
                if isinstance(image_data, str) and (
                    image_data.startswith("http://") or image_data.startswith("https://")
                ):
                    # URL-based image
                    content_blocks.append({
                        "type": "image_url",
                        "image_url": {"url": image_data, "detail": self.image_detail},
                    })
                elif isinstance(image_data, str) and Path(image_data).exists():
                    # File-based image → encode to base64
                    b64 = _encode_image_base64(image_data)
                    media_type = _get_image_media_type(image_data)
                    content_blocks.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{b64}",
                            "detail": self.image_detail,
                        },
                    })
                elif isinstance(image_data, str):
                    # Assume it's already base64-encoded
                    content_blocks.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}",
                            "detail": self.image_detail,
                        },
                    })
            else:
                # Text chunk
                content_blocks.append({
                    "type": "text",
                    "text": f"[Text {i + 1}] {chunk.content}\n",
                })

        return content_blocks

    def get_chunk_text_repr(self, chunk: ContextChunk) -> str:
        if chunk.modality == "image":
            caption = chunk.metadata.get("caption", "")
            return f"[Image: {caption}]" if caption else "[Image]"
        return str(chunk.content)

    def format_for_retrieval_query(self, claims: list[str]) -> str:
        return " ".join(claims)
