"""Layerwise contrastive saliency for generator-side chunk attribution.

This module follows MIRAGE's central idea: identify answer tokens whose
predictive distribution changes when context is supplied, then backpropagate a
contrastive answer objective to the supplied context.  In addition to input
saliency, it records first-order influence through residual states, attention
updates, and MLP updates at selected layers.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

import numpy as np
import torch


@dataclass
class ContrastiveSaliencyTrace:
    """Chunk-level saliency maps produced from one teacher-forced answer."""

    answer_token_ids: list[int]
    no_context_token_ids: list[int]
    cti_js_divergence: list[float]
    selected_answer_indices: list[int]
    layer_ids: list[int]
    chunk_ids: list[str]
    features: dict[str, np.ndarray]  # every value is [layers, chunks]


def jensen_shannon_from_logits(
    contextual_logits: torch.Tensor, no_context_logits: torch.Tensor
) -> torch.Tensor:
    """Return tokenwise Jensen-Shannon divergence for two logit matrices."""

    if contextual_logits.shape != no_context_logits.shape or contextual_logits.ndim != 2:
        raise ValueError("logits must have equal [answer_tokens, vocabulary] shapes")
    left = torch.log_softmax(contextual_logits.float(), dim=-1)
    right = torch.log_softmax(no_context_logits.float(), dim=-1)
    log_mix = torch.logaddexp(left, right) - np.log(2.0)
    return 0.5 * (
        (left.exp() * (left - log_mix)).sum(dim=-1)
        + (right.exp() * (right - log_mix)).sum(dim=-1)
    )


def select_context_sensitive_tokens(
    scores: torch.Tensor, *, minimum: int = 1
) -> list[int]:
    """Apply MIRAGE's example-local mean-plus-standard-deviation threshold."""

    values = scores.detach().float().cpu()
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("scores must be a non-empty vector")
    threshold = values.mean() + values.std(unbiased=False)
    selected = torch.nonzero(values >= threshold, as_tuple=False).flatten().tolist()
    if len(selected) < minimum:
        selected = torch.topk(values, k=min(minimum, values.numel())).indices.tolist()
    return sorted(int(index) for index in selected)


def _top_fraction_mean(values: torch.Tensor, fraction: float = 0.05) -> float:
    if values.numel() == 0:
        return 0.0
    count = max(1, int(np.ceil(values.numel() * fraction)))
    return float(torch.topk(values.float(), count).values.mean().item())


def _chunk_pool(
    activation: torch.Tensor,
    gradient: torch.Tensor,
    spans: Sequence[Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pool gradient and gradient-times-activation signals over chunk spans."""

    grad_norm = torch.linalg.vector_norm(gradient[0].float(), dim=-1)
    grad_x = (gradient[0].float() * activation[0].float()).sum(dim=-1).abs()
    norm_mean, norm_top, grad_x_mean, grad_x_top = [], [], [], []
    for span in spans:
        if span.start is None or span.end is None or int(span.end) <= int(span.start):
            norm_mean.append(0.0)
            norm_top.append(0.0)
            grad_x_mean.append(0.0)
            grad_x_top.append(0.0)
            continue
        start, end = int(span.start), int(span.end)
        norm_values = grad_norm[start:end]
        grad_x_values = grad_x[start:end]
        norm_mean.append(float(norm_values.mean().item()))
        norm_top.append(_top_fraction_mean(norm_values))
        grad_x_mean.append(float(grad_x_values.mean().item()))
        grad_x_top.append(_top_fraction_mean(grad_x_values))
    return tuple(
        np.asarray(values, dtype=np.float32)
        for values in (norm_mean, norm_top, grad_x_mean, grad_x_top)
    )


@contextmanager
def _capture_component_updates(
    core: Any, layer_ids: Sequence[int]
) -> Iterator[tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]]:
    attention: dict[int, torch.Tensor] = {}
    mlp: dict[int, torch.Tensor] = {}
    handles = []

    def save(target: dict[int, torch.Tensor], layer: int):
        def hook(_module: Any, _inputs: Any, output: torch.Tensor) -> None:
            target[layer] = output

        return hook

    try:
        for layer in layer_ids:
            block = core.layers[layer]
            handles.append(block.self_attn.o_proj.register_forward_hook(save(attention, layer)))
            handles.append(block.mlp.register_forward_hook(save(mlp, layer)))
        yield attention, mlp
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def _memory_efficient_attention(model: Any) -> Iterator[None]:
    """Use SDPA because this attribution path never requests attention maps."""

    original = getattr(model.config, "_attn_implementation", None)
    model.config._attn_implementation = "sdpa"
    try:
        yield
    finally:
        model.config._attn_implementation = original


def _answer_inputs(
    client: Any,
    query: str,
    chunks: list[Any],
    answer_ids: Sequence[int],
) -> tuple[Any, torch.Tensor, torch.Tensor]:
    alignment = client.align_and_prepare_inputs(query, chunks)
    prompt_ids = alignment.inputs["input_ids"]
    answer_prefix = torch.tensor(
        [list(answer_ids[:-1])], device=prompt_ids.device, dtype=prompt_ids.dtype
    )
    full_ids = torch.cat([prompt_ids, answer_prefix], dim=1)
    answer_positions = torch.arange(
        prompt_ids.shape[1] - 1,
        prompt_ids.shape[1] - 1 + len(answer_ids),
        device=prompt_ids.device,
    )
    return alignment, full_ids, answer_positions


def _teacher_forced_logits(
    client: Any,
    query: str,
    chunks: list[Any],
    answer_ids: Sequence[int],
) -> torch.Tensor:
    _, full_ids, answer_positions = _answer_inputs(client, query, chunks, answer_ids)
    output = client.model.model(
        input_ids=full_ids,
        attention_mask=torch.ones_like(full_ids),
        use_cache=False,
        output_hidden_states=False,
        return_dict=True,
    )
    return client.model.lm_head(output.last_hidden_state[:, answer_positions, :])[0]


def extract_contrastive_saliency(
    client: Any,
    query: str,
    chunks: list[Any],
    answer: str,
    *,
    max_answer_tokens: int = 12,
    layers: Sequence[int] | None = None,
) -> ContrastiveSaliencyTrace:
    """Attribute a generated answer to chunks through layerwise gradients.

    The language model is frozen.  No probe or LLM fine-tuning occurs here.
    Gradients are taken only with respect to the current example's activations.
    """

    answer_ids = client.tokenizer.encode(answer, add_special_tokens=False)[:max_answer_tokens]
    if not answer_ids:
        raise ValueError("answer must contain at least one model token")
    core = client.model.model
    n_layers = int(client.model.config.num_hidden_layers)
    if layers is None:
        layers = tuple(range(0, n_layers, 3))
    layer_ids = sorted(set(int(layer) for layer in layers))
    if not layer_ids or layer_ids[0] < 0 or layer_ids[-1] >= n_layers:
        raise ValueError(f"layers must be within [0, {n_layers - 1}]")

    client.model.requires_grad_(False)
    client.model.eval()
    with _memory_efficient_attention(client.model):
        with torch.inference_mode():
            no_context_logits = _teacher_forced_logits(client, query, [], answer_ids).float()

        alignment, full_ids, answer_positions = _answer_inputs(client, query, chunks, answer_ids)
        spans = alignment.batch_chunk_spans[0]
        embeddings = core.embed_tokens(full_ids).detach().requires_grad_(True)
        with torch.enable_grad(), _capture_component_updates(
            core, layer_ids
        ) as (attention_updates, mlp_updates):
            output = core(
                input_ids=None,
                inputs_embeds=embeddings,
                attention_mask=torch.ones_like(full_ids),
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            contextual_logits = client.model.lm_head(
                output.last_hidden_state[:, answer_positions, :]
            )[0].float()
            cti = jensen_shannon_from_logits(contextual_logits, no_context_logits).detach()
            selected = select_context_sensitive_tokens(cti)

            no_context_top2 = no_context_logits.topk(k=2, dim=-1).indices
            alternatives = no_context_top2[:, 0].clone()
            target_ids = torch.tensor(answer_ids, device=alternatives.device)
            same = alternatives == target_ids
            alternatives[same] = no_context_top2[same, 1]
            log_probs = torch.log_softmax(contextual_logits, dim=-1)
            target_logp = log_probs.gather(-1, target_ids[:, None]).squeeze(-1)
            alternative_logp = log_probs.gather(-1, alternatives[:, None]).squeeze(-1)
            weights = cti / cti[selected].mean().clamp_min(1e-8)
            objective = (weights[selected] * (target_logp - alternative_logp)[selected]).mean()

            residual_targets = [output.hidden_states[layer] for layer in layer_ids]
            attention_targets = [attention_updates[layer] for layer in layer_ids]
            mlp_targets = [mlp_updates[layer] for layer in layer_ids]
            targets = [*residual_targets, *attention_targets, *mlp_targets]
            gradients = torch.autograd.grad(objective, targets, allow_unused=False)

    features: dict[str, list[np.ndarray]] = {
        f"{component}_{metric}": []
        for component in ("residual", "attention", "mlp")
        for metric in ("grad_norm_mean", "grad_norm_top5", "grad_x_mean", "grad_x_top5")
    }
    width = len(layer_ids)
    component_data = (
        ("residual", residual_targets, gradients[:width]),
        ("attention", attention_targets, gradients[width : 2 * width]),
        ("mlp", mlp_targets, gradients[2 * width :]),
    )
    for component, activations, component_gradients in component_data:
        for activation, gradient in zip(activations, component_gradients, strict=True):
            pooled = _chunk_pool(activation, gradient, spans)
            for metric, values in zip(
                ("grad_norm_mean", "grad_norm_top5", "grad_x_mean", "grad_x_top5"),
                pooled,
                strict=True,
            ):
                features[f"{component}_{metric}"].append(values)

    return ContrastiveSaliencyTrace(
        answer_token_ids=[int(token_id) for token_id in answer_ids],
        no_context_token_ids=[int(value) for value in alternatives.detach().cpu().tolist()],
        cti_js_divergence=[float(value) for value in cti.cpu().tolist()],
        selected_answer_indices=selected,
        layer_ids=layer_ids,
        chunk_ids=[str(span.chunk_id) for span in spans],
        features={
            name: np.stack(values, axis=0).astype(np.float32)
            for name, values in features.items()
        },
    )
