"""Chunk-specific answer-to-context attention interventions for Qwen2 models."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Sequence

import torch

from .causal import Head


def zero_attention_edges(
    weights: torch.Tensor,
    head_indices: Sequence[int],
    *,
    start: int,
    end: int,
) -> torch.Tensor:
    """Zero selected heads' last-query edges to ``[start, end)`` and renormalize."""

    if weights.ndim != 4:
        raise ValueError("weights must have shape [batch, heads, queries, keys].")
    if start < 0 or end <= start or end > weights.shape[-1]:
        raise ValueError("Invalid key span for attention-edge masking.")
    masked = weights.clone()
    original = weights[:, head_indices, -1:, :]
    changed = original.clone()
    changed[..., start:end] = 0
    normalizer = changed.sum(dim=-1, keepdim=True)
    changed = torch.where(normalizer > 0, changed / normalizer.clamp_min(1e-12), original)
    masked[:, head_indices, -1:, :] = changed
    return masked


@contextmanager
def mask_qwen2_answer_chunk_edges(
    model: Any,
    heads: Sequence[Head],
    *,
    start: int,
    end: int,
    prompt_length: int,
) -> Iterator[None]:
    """Temporarily mask answer-to-chunk edges at selected Qwen2 attention heads.

    Prefix calls shorter than ``prompt_length`` remain untouched. On a full
    generation prefill only the last prompt query is changed; on cached decode
    calls the single current answer query is changed. The patch is local to this
    context and restores Transformers' original eager attention function.
    """

    if getattr(model.config, "_attn_implementation", None) != "eager":
        raise ValueError("Chunk edge masking requires eager attention.")
    by_layer: dict[int, list[int]] = {}
    for layer, head in heads:
        by_layer.setdefault(int(layer), []).append(int(head))

    from transformers.models.qwen2 import modeling_qwen2

    original_attention = modeling_qwen2.eager_attention_forward

    def masked_attention(
        module: Any,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        scaling: float,
        dropout: float = 0.0,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output, weights = original_attention(
            module,
            query,
            key,
            value,
            attention_mask,
            scaling,
            dropout=dropout,
            **kwargs,
        )
        selected = by_layer.get(int(module.layer_idx))
        key_length = key.shape[-2]
        if not selected or key_length < prompt_length or end > key_length:
            return output, weights

        masked_weights = zero_attention_edges(weights, selected, start=start, end=end)
        value_states = modeling_qwen2.repeat_kv(value, module.num_key_value_groups)
        masked_output = torch.matmul(masked_weights, value_states)
        masked_output = masked_output.transpose(1, 2).contiguous()
        return masked_output, masked_weights

    modeling_qwen2.eager_attention_forward = masked_attention
    try:
        yield
    finally:
        modeling_qwen2.eager_attention_forward = original_attention
