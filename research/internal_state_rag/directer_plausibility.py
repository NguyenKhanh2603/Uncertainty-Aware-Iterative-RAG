"""DIRECTER-style distribution plausibility for chunk interventions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class PlausibilityStatistics:
    """Token-level and aggregate effects of an intervened distribution."""

    probability_ratios: list[float]
    min_probability_ratio: float
    geometric_mean_probability_ratio: float
    rejection_rate: float
    top1_change_rate: float
    mean_js_divergence: float

    @property
    def worst_rejection_score(self) -> float:
        """A larger score means the intervention proposed a less plausible token."""

        return float(-np.log(max(self.min_probability_ratio, 1e-30)))


def distribution_plausibility(
    raw_logits: torch.Tensor,
    intervened_logits: torch.Tensor,
    *,
    beta: float = 0.5,
) -> PlausibilityStatistics:
    """Apply DIRECTER's probability-ratio gate across teacher-forced positions.

    At each position, the candidate is the intervened distribution's top token.
    Its probability under the raw distribution is divided by the probability of
    the raw distribution's own top token. DIRECTER accepts when this ratio is at
    least ``beta``.
    """

    if raw_logits.shape != intervened_logits.shape or raw_logits.ndim != 2:
        raise ValueError("logits must have equal [tokens, vocabulary] shapes")
    if not 0 <= beta <= 1:
        raise ValueError("beta must lie in [0, 1]")

    raw_log_probs = torch.log_softmax(raw_logits.float(), dim=-1)
    intervened_log_probs = torch.log_softmax(intervened_logits.float(), dim=-1)
    raw_top = raw_log_probs.argmax(dim=-1)
    intervened_top = intervened_log_probs.argmax(dim=-1)
    raw_best = raw_log_probs.gather(-1, raw_top[:, None]).squeeze(-1)
    candidate_raw = raw_log_probs.gather(-1, intervened_top[:, None]).squeeze(-1)
    log_ratios = candidate_raw - raw_best
    ratios = log_ratios.exp()

    log_mixture = torch.logaddexp(raw_log_probs, intervened_log_probs) - np.log(2.0)
    raw_probs = raw_log_probs.exp()
    intervened_probs = intervened_log_probs.exp()
    js = 0.5 * (
        (raw_probs * (raw_log_probs - log_mixture)).sum(dim=-1)
        + (intervened_probs * (intervened_log_probs - log_mixture)).sum(dim=-1)
    )

    ratio_values = ratios.cpu().numpy().astype(np.float64)
    return PlausibilityStatistics(
        probability_ratios=ratio_values.tolist(),
        min_probability_ratio=float(ratio_values.min()),
        geometric_mean_probability_ratio=float(np.exp(log_ratios.mean().item())),
        rejection_rate=float(np.mean(ratio_values < beta)),
        top1_change_rate=float((raw_top != intervened_top).float().mean().item()),
        mean_js_divergence=float(js.mean().item()),
    )


@torch.inference_mode()
def teacher_forced_final_logits(
    extractor: Any,
    query: str,
    chunks: list[Any],
    answer: str,
    *,
    max_answer_tokens: int = 16,
) -> torch.Tensor:
    """Return final-layer logits at answer positions using the extractor cache path."""

    client = extractor.client
    alignment = client.align_and_prepare_inputs(query, chunks)
    inputs = alignment.inputs
    answer_ids = client.tokenizer.encode(answer, add_special_tokens=False)
    answer_ids = answer_ids[:max_answer_tokens]
    if not answer_ids:
        raise ValueError("answer must contain at least one model token")

    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
    prompt_embeddings = extractor._prompt_embeddings(inputs)
    prompt_positions = extractor._prompt_position_ids(inputs)
    prefix = extractor.core(
        input_ids=None,
        inputs_embeds=prompt_embeddings[:, :-1, :],
        attention_mask=attention_mask[:, :-1],
        position_ids=extractor._slice_positions(prompt_positions, 0, -1),
        use_cache=True,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )

    cache = prefix.past_key_values
    current_embedding = prompt_embeddings[:, -1:, :]
    current_position = extractor._slice_positions(prompt_positions, -1, None)
    running_mask = attention_mask
    logits = []
    for target_id in answer_ids:
        output = extractor.core(
            input_ids=None,
            inputs_embeds=current_embedding,
            attention_mask=running_mask,
            position_ids=current_position,
            past_key_values=cache,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        cache = output.past_key_values
        logits.append(extractor.lm_head(output.last_hidden_state[:, -1, :]).float()[0].cpu())
        next_id = torch.tensor(
            [[target_id]], device=input_ids.device, dtype=input_ids.dtype
        )
        current_embedding = extractor.core.embed_tokens(next_id)
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
    return torch.stack(logits)
