"""Causal interventions for testing candidate evidence-use attention heads."""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Iterator, Sequence

import torch

Head = tuple[int, int]


def zero_head_slices(
    hidden: torch.Tensor,
    head_indices: Sequence[int],
    *,
    num_heads: int,
) -> torch.Tensor:
    """Zero selected concatenated attention-head outputs before output projection."""

    if hidden.shape[-1] % num_heads:
        raise ValueError("Attention projection width must be divisible by num_heads.")
    head_dim = hidden.shape[-1] // num_heads
    masked = hidden.clone()
    for head in head_indices:
        if head < 0 or head >= num_heads:
            raise ValueError(f"head index {head} is outside [0, {num_heads - 1}].")
        masked[..., head * head_dim : (head + 1) * head_dim] = 0
    return masked


@contextmanager
def mask_attention_heads(model: Any, heads: Sequence[Head]) -> Iterator[None]:
    """Temporarily ablate Qwen/Llama attention heads before each layer's o_proj.

    The hook is removed even when generation fails.  Masking happens during both
    prompt encoding and answer decoding, so the measured delta is a causal
    intervention on the complete context-integration computation.
    """

    by_layer: dict[int, list[int]] = defaultdict(list)
    for layer, head in heads:
        by_layer[int(layer)].append(int(head))

    decoder = model.model
    layers = decoder.layers
    num_heads = int(model.config.num_attention_heads)
    handles = []
    try:
        for layer_index, head_indices in by_layer.items():
            if layer_index < 0 or layer_index >= len(layers):
                raise ValueError(
                    f"layer index {layer_index} is outside [0, {len(layers) - 1}]."
                )
            projection = layers[layer_index].self_attn.o_proj

            def pre_hook(
                _module: Any,
                args: tuple[torch.Tensor, ...],
                selected: tuple[int, ...] = tuple(head_indices),
            ) -> tuple[torch.Tensor, ...]:
                return (
                    zero_head_slices(args[0], selected, num_heads=num_heads),
                    *args[1:],
                )

            handles.append(projection.register_forward_pre_hook(pre_hook))
        yield
    finally:
        for handle in handles:
            handle.remove()
