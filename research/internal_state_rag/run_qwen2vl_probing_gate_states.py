"""Extract Probing-RAG-style query states from a full retrieved context.

Probing-RAG uses intermediate hidden trajectories to decide whether another
retrieval round is needed.  This runner keeps that query-level decision separate
from candidate scoring: Qwen2-VL first produces a deterministic draft from the
full Top-L context, then its answer-token hidden trajectory is pooled into a
small state vector.  A downstream script fits only a lightweight gate on these
vectors and calibrates the pruning policy on a disjoint split.

The runner never uses the gold answer to form the state.  Gold support labels
are stored only as audit targets (for example, ``no_support_top3``) and are not
passed to the model.
"""

from __future__ import annotations

import argparse
import gzip
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from research.internal_state_rag.contrastive_saliency import _memory_efficient_attention
from research.internal_state_rag.run_tatqa_smoke import (
    ResearchTextClient,
    load_retrieval,
    load_rows,
    to_chunk,
)
from research.internal_state_rag.signals import QwenInternalStateExtractor, normalized_chunk_entropy


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_feature_projection(path: Path | None) -> dict[tuple[str, str], float]:
    """Load external scores only for audit; the gate itself is model-internal."""

    if path is None:
        return {}
    values = np.load(path, allow_pickle=False)
    qids = values["qids"].astype(str)
    chunk_ids = values["chunk_ids"].astype(str)
    scores = values["reranker_scores"].astype(np.float64)
    return {
        (str(qids[q_index]), str(chunk_ids[q_index, c_index])): float(
            scores[q_index, c_index]
        )
        for q_index in range(len(qids))
        for c_index in range(chunk_ids.shape[1])
        if np.isfinite(scores[q_index, c_index])
    }


def query_state_features(trace: Any, projection: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Pool intermediate hidden and answer-trajectory signals into one vector."""

    residual = np.asarray(trace.residual_mean, dtype=np.float32)
    normalized = residual / np.maximum(np.linalg.norm(residual, axis=1, keepdims=True), 1e-6)
    projected = normalized.astype(np.float32) @ projection.astype(np.float32)
    final_top = trace.top1_token_id[-1:, :]
    agreement = np.mean(trace.top1_token_id == final_top, axis=1, dtype=np.float64)
    attention_entropy = normalized_chunk_entropy(trace.attention_mass).mean(axis=1)

    blocks = []
    names = []
    for layer_index, layer in enumerate(trace.layer_ids):
        values = np.asarray(
            [
                np.mean(trace.target_logprob[layer_index]),
                np.min(trace.target_logprob[layer_index]),
                np.mean(trace.target_margin[layer_index]),
                np.min(trace.target_margin[layer_index]),
                np.mean(trace.entropy[layer_index]),
                float(agreement[layer_index]),
                float(np.linalg.norm(residual[layer_index])),
                float(np.mean(np.abs(residual[layer_index]))),
                float(attention_entropy[layer_index]),
            ],
            dtype=np.float32,
        )
        blocks.append(values)
        names.extend(
            [
                f"layer{layer}_target_logprob_mean",
                f"layer{layer}_target_logprob_min",
                f"layer{layer}_target_margin_mean",
                f"layer{layer}_target_margin_min",
                f"layer{layer}_entropy_mean",
                f"layer{layer}_final_token_agreement",
                f"layer{layer}_residual_norm",
                f"layer{layer}_residual_abs_mean",
                f"layer{layer}_attention_entropy",
            ]
        )
    blocks.append(np.asarray([len(trace.answer_token_ids)], dtype=np.float32))
    names.append("draft_answer_tokens")
    blocks.append(projected.reshape(-1).astype(np.float32))
    names.extend(
        f"layer{layer}_hidden_projection_{dimension}"
        for layer in trace.layer_ids
        for dimension in range(projection.shape[1])
    )
    vector = np.concatenate(blocks).astype(np.float32)
    if not np.isfinite(vector).all():
        raise FloatingPointError("Non-finite Probing-RAG state feature")
    return vector, {
        "layer_ids": [int(value) for value in trace.layer_ids],
        "feature_names": names,
        "mean_target_logprob_by_layer": trace.target_logprob.mean(axis=1).astype(float).tolist(),
        "mean_target_margin_by_layer": trace.target_margin.mean(axis=1).astype(float).tolist(),
        "mean_entropy_by_layer": trace.entropy.mean(axis=1).astype(float).tolist(),
        "residual_norm_by_layer": np.linalg.norm(residual, axis=1).astype(float).tolist(),
        "attention_entropy_by_layer": attention_entropy.astype(float).tolist(),
    }


@torch.inference_mode()
def generate_answer_last_token(
    client: ResearchTextClient,
    extractor: QwenInternalStateExtractor,
    query: str,
    chunks: list[Any],
    *,
    max_new_tokens: int,
) -> tuple[str, int]:
    """Greedy decode without materializing prompt-length vocabulary logits.

    ``model.generate`` calls the outer conditional-generation model on the
    complete multimodal prompt. Qwen2-VL 7B then applies its large LM head to
    every prompt position, which can fail for Top-30 contexts containing many
    images. This implementation builds the same KV cache through the decoder
    core and applies the LM head only to the final position at each step.
    """

    alignment = client.align_and_prepare_inputs(query, chunks)
    inputs = alignment.inputs
    input_ids = inputs["input_ids"]
    if input_ids.shape[0] != 1 or input_ids.shape[1] < 2:
        raise ValueError("Greedy decode expects one non-empty prompt")
    max_positions = int(client.model.config.max_position_embeddings)
    if input_ids.shape[1] > max_positions:
        raise ValueError(
            f"Prompt has {input_ids.shape[1]} tokens, exceeding model limit {max_positions}"
        )
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
    generated: list[int] = []
    eos = client.model.generation_config.eos_token_id
    eos_ids = {int(value) for value in eos} if isinstance(eos, (list, tuple)) else {int(eos)}

    for _ in range(max_new_tokens):
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
        final_hidden = output.last_hidden_state[:, -1:, :]
        next_id = int(torch.argmax(extractor.lm_head(final_hidden).float()[0, -1]).item())
        if next_id in eos_ids:
            break
        generated.append(next_id)
        token = torch.tensor([[next_id]], dtype=input_ids.dtype, device=input_ids.device)
        current_embedding = client.model.get_input_embeddings()(token)
        current_position = current_position + 1
        running_mask = torch.cat(
            [
                running_mask,
                torch.ones(
                    (running_mask.shape[0], 1),
                    dtype=running_mask.dtype,
                    device=running_mask.device,
                ),
            ],
            dim=-1,
        )
    return (
        client.tokenizer.decode(generated, skip_special_tokens=True).strip(),
        int(input_ids.shape[1]),
    )


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_existing(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        return list(json.loads(path.read_text(encoding="utf-8")).get("queries", []))
    except (OSError, json.JSONDecodeError):
        return []


def parse_args() -> argparse.Namespace:
    root = Path("data/conformal_global_run")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--feature-npz", type=Path, default=None)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--layers", default="3,7,11,15,19,23,27")
    parser.add_argument("--projection-dim", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--max-answer-tokens", type=int, default=12)
    parser.add_argument(
        "--max-image-pixels",
        type=int,
        default=None,
        help="Optional per-image pixel cap applied by the Qwen-VL processor.",
    )
    parser.add_argument(
        "--compute-dtype",
        choices=("bfloat16", "float16"),
        default="bfloat16",
    )
    parser.add_argument("--seed", type=int, default=20260914)
    return parser.parse_args()


def main(
    args: argparse.Namespace | None = None,
    *,
    client: ResearchTextClient | None = None,
) -> dict[str, Any]:
    """Extract one split, optionally reusing a model client across suite jobs."""

    if args is None:
        args = parse_args()
    manifest = json.loads(args.feature_manifest.read_text(encoding="utf-8"))
    full_plan = [str(value) for value in manifest["plan"]]
    if args.start < 0 or args.n < 1 or args.start + args.n > len(full_plan):
        raise ValueError("Requested slice lies outside the feature manifest plan")
    plan = full_plan[args.start : args.start + args.n]
    input_role = str(manifest["input_role"])
    top_l = int(manifest["top_l"])
    layers = [int(value) for value in args.layers.split(",") if value.strip()]
    if args.projection_dim < 1:
        raise ValueError("projection dimension must be positive")

    questions = load_rows(args.questions, "qid")
    corpus = load_rows(args.corpus, "id")
    retrieval = load_retrieval(args.retrieval, split_role=input_role)
    reranker_scores = load_feature_projection(args.feature_npz)
    completed = {str(row["qid"]): row for row in load_existing(args.output)}
    unexpected = set(completed).difference(plan)
    if unexpected:
        raise ValueError(f"Output contains qids outside requested plan: {sorted(unexpected)}")

    if client is None:
        client = ResearchTextClient(args.model, device="cuda", load_in_4bit=False)
    elif str(client.model_name) != str(args.model):
        raise ValueError(
            f"Reused client model {client.model_name!r} does not match {args.model!r}"
        )
    if args.max_image_pixels is not None:
        if args.max_image_pixels < 28 * 28:
            raise ValueError("max-image-pixels must be at least one 28x28 patch")
        if not client.is_qwen_vl:
            raise ValueError("max-image-pixels requires a Qwen-VL model")
        client.processor.image_processor.max_pixels = int(args.max_image_pixels)
    compute_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.compute_dtype]
    if next(client.model.parameters()).dtype != compute_dtype:
        client.model.to(dtype=compute_dtype)
    extractor = QwenInternalStateExtractor(client, layers=layers)
    generator = np.random.default_rng(args.seed)
    hidden_size = int(client.model.config.hidden_size)
    projection = generator.standard_normal((hidden_size, args.projection_dim)).astype(np.float32)
    projection /= np.sqrt(float(args.projection_dim))
    started = time.monotonic()

    def payload(status: str) -> dict[str, Any]:
        rows = [completed[qid] for qid in plan if qid in completed]
        return {
            "status": status,
            "method": "qwen2vl_probing_rag_query_hidden_state_gate",
            "model": args.model,
            "feature_manifest": str(args.feature_manifest),
            "input_role": input_role,
            "top_l": top_l,
            "start": args.start,
            "requested_queries": len(plan),
            "completed_queries": len(rows),
            "layers": extractor.layers,
            "projection_dim": args.projection_dim,
            "projection_seed": args.seed,
            "compute_dtype": args.compute_dtype,
            "max_new_tokens": args.max_new_tokens,
            "max_answer_tokens": args.max_answer_tokens,
            "max_image_pixels": args.max_image_pixels,
            "risk_definition": {
                "no_support_top1": "no labelled support among rank 1 candidate",
                "no_support_top3": "no labelled support among rank 1-3 candidates",
                "no_support_top5": "no labelled support among rank 1-5 candidates",
                "no_support_top10": "no labelled support among rank 1-10 candidates",
                "no_support_top30": "no labelled support in the retrieved pool",
            },
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "queries": rows,
        }

    with _memory_efficient_attention(client.model):
        for index, qid in enumerate(plan, start=1):
            if qid in completed:
                print(f"[{index}/{len(plan)}] resume-skip {qid}", flush=True)
                continue
            question = questions.get(qid)
            candidates = retrieval.get(qid, [])[:top_l]
            if question is None or not question.get("gold_answers"):
                print(f"[{index}/{len(plan)}] skip {qid}: missing question/gold metadata", flush=True)
                continue
            if len(candidates) != top_l:
                raise ValueError(f"{qid} has {len(candidates)} candidates; expected {top_l}")
            chunks = [to_chunk(row, corpus) for row in candidates]
            prompt = (
                "Answer using only the supplied context. Return only the short answer.\n"
                f"Question: {question['question']}"
            )
            started_query = time.monotonic()
            draft, prompt_tokens = generate_answer_last_token(
                client,
                extractor,
                prompt,
                chunks,
                max_new_tokens=args.max_new_tokens,
            )
            trace_answer = draft or "unknown"
            trace = extractor.extract(
                prompt,
                chunks,
                trace_answer,
                max_answer_tokens=args.max_answer_tokens,
            )
            vector, diagnostics = query_state_features(trace, projection)
            support = np.asarray(
                [row.get("support_label") == "support" for row in candidates], dtype=bool
            )
            row = {
                "qid": qid,
                "question": str(question["question"]),
                "draft_answer": draft,
                "trace_answer": trace_answer,
                "state_features": vector.astype(float).tolist(),
                "state_diagnostics": diagnostics,
                "draft_answer_token_count": len(trace.answer_token_ids),
                "prompt_tokens": prompt_tokens,
                "support_count_top30": int(support.sum()),
                "risk_labels": {
                    f"no_support_top{k}": bool(not support[:k].any())
                    for k in (1, 3, 5, 10, 30)
                },
                "top1_reranker_score": float(
                    reranker_scores.get((qid, str(candidates[0]["chunk_id"])), np.nan)
                ),
                "elapsed_seconds": round(time.monotonic() - started_query, 3),
            }
            completed[qid] = row
            atomic_write(args.output, payload("running"))
            print(
                f"[{index}/{len(plan)}] {qid} done {row['elapsed_seconds']:.1f}s "
                f"draft_tokens={row['draft_answer_token_count']}",
                flush=True,
            )

    output = payload("complete")
    output["elapsed_seconds_this_invocation"] = round(time.monotonic() - started, 3)
    atomic_write(args.output, output)
    print(json.dumps({key: value for key, value in output.items() if key != "queries"}, indent=2))
    return output


if __name__ == "__main__":
    main()
