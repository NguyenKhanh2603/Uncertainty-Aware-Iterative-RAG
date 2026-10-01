"""Common modality-aware generator from the completed fa6dd68 experiment."""
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

@dataclass
class ContextChunk:
    id: str
    content: str
    modality: str

class QwenDirectAnswerGenerator:
    """Qwen's documented chat-template/generate/decode path.

    This deliberately does not use the attention-alignment helper: it inserts
    temporary markers intended for attribution, which produced malformed text
    during ordinary answer generation with the current transformers release.
    """

    def __init__(
        self,
        model_path: Path,
        *,
        min_pixels: int,
        max_pixels: int,
        revision: str | None = None,
    ):
        revision_kwargs = {"revision": revision} if revision is not None else {}
        self.processor = AutoProcessor.from_pretrained(
            str(model_path), min_pixels=min_pixels, max_pixels=max_pixels, **revision_kwargs
        )
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            str(model_path),
            # Match the feature-extraction path that completed on this A100;
            # it avoids a transient full-precision CPU copy while loading.
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
            **revision_kwargs,
        ).eval().to("cuda")

    @torch.inference_mode()
    def generate(self, prompt: str, context: list[ContextChunk], max_new_tokens: int) -> str:
        content: list[dict[str, str]] = [{"type": "text", "text": prompt + "\n\nContext:\n"}]
        for chunk in context:
            if chunk.modality == "image":
                content.append({"type": "image", "image": str(chunk.content)})
            else:
                content.append({"type": "text", "text": str(chunk.content) + "\n"})
        messages = [{"role": "user", "content": content}]
        rendered = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[rendered], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        ).to("cuda")
        generated = self.model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
        trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated)]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
