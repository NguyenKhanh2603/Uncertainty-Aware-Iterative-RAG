"""Extract residual, logit-lens, and per-head evidence-use signals.

The extractor performs a cached, teacher-forced answer pass.  The long prompt
is encoded once without retaining its attention matrices; answer positions are
then decoded one at a time, so retained attention is O(answer_length * context)
instead of O(context**2).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


@dataclass
class InternalTrace:
    """Compact signals for one question/context/answer triple."""

    answer_token_ids: list[int]
    layer_ids: list[int]
    chunk_ids: list[str]
    residual_mean: np.ndarray  # [layers, hidden]
    target_logprob: np.ndarray  # [layers, answer_tokens]
    target_margin: np.ndarray  # [layers, answer_tokens]
    entropy: np.ndarray  # [layers, answer_tokens]
    top1_token_id: np.ndarray  # [layers, answer_tokens]
    attention_mass: np.ndarray  # [layers, heads, chunks]

    def save(self, path: str | Path, *, metadata: dict[str, Any] | None = None) -> None:
        """Write numeric arrays plus JSON metadata to a compressed NPZ file."""

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "answer_token_ids": np.asarray(self.answer_token_ids, dtype=np.int64),
            "layer_ids": np.asarray(self.layer_ids, dtype=np.int64),
            "chunk_ids": np.asarray(self.chunk_ids, dtype=str),
            "residual_mean": self.residual_mean,
            "target_logprob": self.target_logprob,
            "target_margin": self.target_margin,
            "entropy": self.entropy,
            "top1_token_id": self.top1_token_id,
            "attention_mass": self.attention_mass,
            "metadata_json": np.asarray(json.dumps(metadata or {}, ensure_ascii=False)),
        }
        np.savez_compressed(output, **payload)

    def summary(self) -> dict[str, Any]:
        """Return scalar features suitable for a first lightweight probe."""

        layer_agreement = np.mean(
            self.top1_token_id == self.top1_token_id[-1:, :], axis=1, dtype=np.float64
        )
        attention_entropy = normalized_chunk_entropy(self.attention_mass)
        return {
            "n_answer_tokens": len(self.answer_token_ids),
            "layer_ids": self.layer_ids,
            "mean_target_logprob_by_layer": self.target_logprob.mean(axis=1).tolist(),
            "mean_target_margin_by_layer": self.target_margin.mean(axis=1).tolist(),
            "mean_entropy_by_layer": self.entropy.mean(axis=1).tolist(),
            "final_prediction_agreement_by_layer": layer_agreement.tolist(),
            "attention_entropy_by_layer_head": attention_entropy.tolist(),
        }


def normalized_chunk_entropy(attention_mass: np.ndarray) -> np.ndarray:
    """Entropy across chunks for each layer/head, normalized to [0, 1]."""

    values = np.asarray(attention_mass, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("attention_mass must have shape [layers, heads, chunks].")
    n_chunks = values.shape[-1]
    if n_chunks <= 1:
        return np.zeros(values.shape[:-1], dtype=np.float64)
    totals = values.sum(axis=-1, keepdims=True)
    probabilities = np.divide(values, totals, out=np.zeros_like(values), where=totals > 0)
    return -(probabilities * np.log(probabilities + 1e-12)).sum(axis=-1) / np.log(n_chunks)


def contrast_trace(with_context: InternalTrace, without_context: InternalTrace) -> dict[str, Any]:
    """Measure how evidence changes the model's latent answer trajectory.

    This contrast helps separate parametric confidence from evidence-induced
    confidence.  Both traces must use the same answer tokens and layer IDs.
    """

    if with_context.answer_token_ids != without_context.answer_token_ids:
        raise ValueError("Contrasted traces must teacher-force the same answer tokens.")
    if with_context.layer_ids != without_context.layer_ids:
        raise ValueError("Contrasted traces must use the same layers.")

    residual_cosine = _row_cosine(with_context.residual_mean, without_context.residual_mean)
    return {
        "context_logprob_gain_by_layer": (
            with_context.target_logprob.mean(axis=1)
            - without_context.target_logprob.mean(axis=1)
        ).tolist(),
        "context_margin_gain_by_layer": (
            with_context.target_margin.mean(axis=1)
            - without_context.target_margin.mean(axis=1)
        ).tolist(),
        "residual_cosine_by_layer": residual_cosine.tolist(),
    }


def _row_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.sum(left * right, axis=1)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator > 0,
    )


class QwenInternalStateExtractor:
    """Cached answer tracer for ``HuggingFaceLocalClient`` and Qwen2-VL.

    The existing client supplies exact chunk-token spans, including visual
    spans.  This class keeps the research code separate from the production
    client and can therefore be removed with one directory rollback.
    """

    def __init__(self, client: Any, *, layers: Sequence[int] | None = None) -> None:
        self.client = client
        self.model = client.model
        self.core = self.model.model
        self.lm_head = self.model.lm_head
        n_layers = int(self.model.config.num_hidden_layers)
        self.n_layers = n_layers
        if layers is None:
            start = max(0, n_layers // 3)
            layers = tuple(range(start, n_layers, 2))
            if n_layers - 1 not in layers:
                layers = (*layers, n_layers - 1)
        self.layers = sorted(set(int(layer) for layer in layers))
        if not self.layers or self.layers[0] < 0 or self.layers[-1] >= n_layers:
            raise ValueError(f"layers must be within [0, {n_layers - 1}].")

    @torch.inference_mode()
    def extract(
        self,
        query: str,
        chunks: list[Any],
        answer: str,
        *,
        max_answer_tokens: int = 64,
    ) -> InternalTrace:
        alignment = self.client.align_and_prepare_inputs(query, chunks)
        inputs = alignment.inputs
        spans = alignment.batch_chunk_spans[0]
        answer_ids = self.client.tokenizer.encode(answer, add_special_tokens=False)
        answer_ids = answer_ids[:max_answer_tokens]
        if not answer_ids:
            raise ValueError("answer must contain at least one model token.")

        input_ids = inputs["input_ids"]
        if input_ids.shape[0] != 1 or input_ids.shape[1] < 2:
            raise ValueError("The extractor currently expects one non-empty prompt.")
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
        prompt_embeddings = self._prompt_embeddings(inputs)
        prompt_positions = self._prompt_position_ids(inputs)

        prefix = self.core(
            input_ids=None,
            inputs_embeds=prompt_embeddings[:, :-1, :],
            attention_mask=attention_mask[:, :-1],
            position_ids=self._slice_positions(prompt_positions, 0, -1),
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )

        n_selected = len(self.layers)
        n_tokens = len(answer_ids)
        n_heads = int(self.model.config.num_attention_heads)
        n_chunks = len(spans)
        hidden_size = int(self.model.config.hidden_size)
        residual_sum = np.zeros((n_selected, hidden_size), dtype=np.float64)
        target_logprob = np.empty((n_selected, n_tokens), dtype=np.float32)
        target_margin = np.empty((n_selected, n_tokens), dtype=np.float32)
        entropy = np.empty((n_selected, n_tokens), dtype=np.float32)
        top1_token_id = np.empty((n_selected, n_tokens), dtype=np.int64)
        attention_mass = np.zeros((n_selected, n_heads, n_chunks), dtype=np.float64)

        cache = prefix.past_key_values
        current_embedding = prompt_embeddings[:, -1:, :]
        current_position = self._slice_positions(prompt_positions, -1, None)
        running_mask = attention_mask

        for token_index, target_id in enumerate(answer_ids):
            output = self.core(
                input_ids=None,
                inputs_embeds=current_embedding,
                attention_mask=running_mask,
                position_ids=current_position,
                past_key_values=cache,
                use_cache=True,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True,
            )
            cache = output.past_key_values

            for selected_index, layer in enumerate(self.layers):
                hidden = output.hidden_states[layer + 1][:, -1, :]
                residual_sum[selected_index] += hidden[0].float().cpu().numpy()
                # Hugging Face stores the already-normalized final decoder state
                # at hidden_states[-1]; intermediate layer outputs still need
                # the model's final norm before applying the unembedding.
                projected = (
                    hidden if layer == self.n_layers - 1 else self._final_norm(hidden)
                )
                logits = self.lm_head(projected).float()[0]
                log_probs = torch.log_softmax(logits, dim=-1)
                probs = log_probs.exp()
                target_logprob[selected_index, token_index] = log_probs[target_id].item()
                entropy[selected_index, token_index] = (
                    -(probs * log_probs).sum().item() / np.log(logits.numel())
                )
                top_values, top_indices = torch.topk(logits, k=2)
                top_id = int(top_indices[0].item())
                top1_token_id[selected_index, token_index] = top_id
                best_other = top_values[1] if top_id == target_id else top_values[0]
                target_margin[selected_index, token_index] = (
                    logits[target_id] - best_other
                ).item()

                layer_attention = output.attentions[layer][0, :, -1, :]
                for chunk_index, span in enumerate(spans):
                    if span.start is None or span.end is None:
                        continue
                    start = max(0, int(span.start))
                    end = min(int(span.end), layer_attention.shape[-1])
                    if end > start:
                        attention_mass[selected_index, :, chunk_index] += (
                            layer_attention[:, start:end].sum(dim=-1).float().cpu().numpy()
                        )

            next_id = torch.tensor([[target_id]], device=input_ids.device, dtype=input_ids.dtype)
            current_embedding = self.model.get_input_embeddings()(next_id)
            current_position = current_position + 1
            running_mask = torch.cat(
                [
                    running_mask,
                    torch.ones(
                        (running_mask.shape[0], 1),
                        device=running_mask.device,
                        dtype=running_mask.dtype,
                    ),
                ],
                dim=-1,
            )

        return InternalTrace(
            answer_token_ids=[int(token_id) for token_id in answer_ids],
            layer_ids=self.layers,
            chunk_ids=[str(span.chunk_id) for span in spans],
            residual_mean=(residual_sum / n_tokens).astype(np.float32),
            target_logprob=target_logprob,
            target_margin=target_margin,
            entropy=entropy,
            top1_token_id=top1_token_id,
            attention_mass=(attention_mass / n_tokens).astype(np.float32),
        )

    def _prompt_embeddings(self, inputs: Any) -> torch.Tensor:
        input_ids = inputs["input_ids"]
        embeddings = self.model.get_input_embeddings()(input_ids)
        if not getattr(self.client, "is_qwen_vl", False):
            return embeddings
        pixel_values = inputs.get("pixel_values")
        if pixel_values is None:
            return embeddings
        pixels = pixel_values.type(self.model.visual.get_dtype())
        image_embeddings = self.model.visual(pixels, grid_thw=inputs["image_grid_thw"])
        image_mask = (input_ids == self.model.config.image_token_id).unsqueeze(-1)
        image_mask = image_mask.expand_as(embeddings)
        return embeddings.masked_scatter(
            image_mask.to(embeddings.device),
            image_embeddings.to(embeddings.device, embeddings.dtype),
        )

    def _final_norm(self, hidden: torch.Tensor) -> torch.Tensor:
        norm = getattr(self.core, "norm", None)
        if norm is None:
            language_model = getattr(self.core, "language_model", None)
            norm = getattr(language_model, "norm", None)
        if norm is None:
            raise AttributeError("Could not locate the decoder final norm")
        return norm(hidden)

    def _prompt_position_ids(self, inputs: Any) -> torch.Tensor:
        input_ids = inputs["input_ids"]
        if not getattr(self.client, "is_qwen_vl", False):
            return torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)

        kwargs = {
            "input_ids": input_ids,
            "image_grid_thw": inputs.get("image_grid_thw"),
            "video_grid_thw": inputs.get("video_grid_thw"),
            "attention_mask": inputs.get("attention_mask"),
        }
        if "mm_token_type_ids" in inputs:
            kwargs["mm_token_type_ids"] = inputs["mm_token_type_ids"]
        rope_owner = (
            self.model
            if hasattr(self.model, "get_rope_index")
            else self.core
        )
        try:
            position_ids, _ = rope_owner.get_rope_index(**kwargs)
        except TypeError:
            kwargs.pop("mm_token_type_ids", None)
            position_ids, _ = rope_owner.get_rope_index(**kwargs)
        return position_ids

    @staticmethod
    def _slice_positions(
        position_ids: torch.Tensor, start: int, end: int | None
    ) -> torch.Tensor:
        if position_ids.ndim == 3:
            return position_ids[:, :, start:end]
        return position_ids[:, start:end]
