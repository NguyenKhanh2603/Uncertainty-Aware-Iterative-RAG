"""Extract Qwen2-VL LM-head and hidden-state relevance features for fixed candidates."""

from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


def iter_jsonl(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_keyed(path: Path, key: str) -> dict[str, dict]:
    return {str(row[key]): row for row in iter_jsonl(path)}


def load_retrieval(path: Path, role: str) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {}
    for row in iter_jsonl(path):
        if str(row.get("split_role")) != role:
            continue
        result.setdefault(str(row["qid"]), []).append(row)
    for rows in result.values():
        rows.sort(key=lambda row: int(row["rank"]))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument(
        "--model-revision", default="895c3a49bc3fa70a340399125c650a463535e71c"
    )
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", default="3,7,11,15,19,23,27")
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--text-batch-size", type=int, default=24)
    parser.add_argument("--image-batch-size", type=int, default=6)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--min-pixels", type=int, default=3136)
    parser.add_argument("--max-pixels", type=int, default=200704)
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Optionally evaluate only the first N queries in the frozen plan.",
    )
    return parser.parse_args()


def verdict_instruction(question: str, kind: str, content: str = "") -> str:
    evidence = f"\nPassage:\n{content}" if content else ""
    return (
        f"Determine whether the {kind} contains information needed to answer the question. "
        "Reply with exactly Yes or No.\n"
        f"Question: {question}{evidence}\n"
        f"Does this {kind} contain useful evidence?"
    )


def text_prompt(processor, question: str, content: str) -> str:
    messages = [{"role": "user", "content": verdict_instruction(question, "passage", content)}]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def image_prompt(processor, question: str, path: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": path},
                {"type": "text", "text": verdict_instruction(question, "image")},
            ],
        }
    ]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def single_token_ids(tokenizer, strings: tuple[str, ...]) -> list[int]:
    result = []
    for value in strings:
        encoded = tokenizer.encode(value, add_special_tokens=False)
        if len(encoded) == 1:
            result.append(int(encoded[0]))
    if not result:
        raise ValueError(f"None of {strings!r} map to one token")
    return sorted(set(result))


def resolve_image(row: dict, bundle_root: Path) -> str:
    path = str(row["content"])
    if path.startswith(("http://", "https://")):
        return path
    resolved = (bundle_root / path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return str(resolved)


def main() -> None:
    args = parse_args()
    plan_payload = json.loads(args.feature_manifest.read_text(encoding="utf-8"))
    plan = [str(value) for value in plan_payload["plan"]]
    if args.max_queries is not None:
        if args.max_queries <= 0:
            raise ValueError("--max-queries must be positive")
        plan = plan[: args.max_queries]
    top_l = int(plan_payload["top_l"])
    layers = [int(value) for value in args.layers.split(",") if value.strip()]
    questions = load_keyed(args.questions, "qid")
    corpus = load_keyed(args.corpus, "id")
    retrieval = load_retrieval(args.retrieval, str(plan_payload["input_role"]))

    processor = AutoProcessor.from_pretrained(
        args.model,
        revision=args.model_revision,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model,
        revision=args.model_revision,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).eval().to("cuda")
    if max(layers) >= model.config.text_config.num_hidden_layers:
        raise ValueError(f"Layer outside 0..{model.config.text_config.num_hidden_layers - 1}")
    hidden_size = int(model.config.text_config.hidden_size)
    yes_ids = single_token_ids(processor.tokenizer, ("Yes", " yes", "YES"))
    no_ids = single_token_ids(processor.tokenizer, ("No", " no", "NO"))
    generator = torch.Generator(device="cpu").manual_seed(1949)
    projection = torch.randn(
        hidden_size, args.projection_dim, generator=generator, dtype=torch.float32
    ) / np.sqrt(args.projection_dim)
    projection = projection.to("cuda", dtype=torch.bfloat16)

    n_queries = len(plan)
    features = np.empty(
        (n_queries, top_l, len(layers), args.projection_dim), dtype=np.float16
    )
    lm_scores = np.empty((n_queries, top_l), dtype=np.float32)
    labels = np.empty((n_queries, top_l), dtype=np.bool_)
    cosine_scores = np.empty((n_queries, top_l), dtype=np.float32)
    reranker_scores = np.empty((n_queries, top_l), dtype=np.float32)
    bge_reranker_scores = np.full((n_queries, top_l), np.nan, dtype=np.float32)
    modalities = np.empty((n_queries, top_l), dtype="U8")
    chunk_ids = np.empty((n_queries, top_l), dtype="U128")
    prompt_lengths = np.empty((n_queries, top_l), dtype=np.int32)
    started = time.perf_counter()

    def run_batch(indices: list[int], prompts: list[str], images: list[Image.Image] | None):
        kwargs = dict(
            text=prompts,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        if images is not None:
            kwargs["images"] = images
        inputs = processor(**kwargs).to("cuda")
        with torch.inference_mode():
            outputs = model.model(
                **inputs,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            final = outputs.last_hidden_state[:, -1, :]
            logits = model.lm_head(final).float()
            yes_score = torch.logsumexp(logits[:, yes_ids], dim=-1)
            no_score = torch.logsumexp(logits[:, no_ids], dim=-1)
            for layer_offset, layer in enumerate(layers):
                state = torch.nn.functional.normalize(
                    outputs.hidden_states[layer + 1][:, -1, :].float(), p=2, dim=-1
                )
                compressed = state.to(projection.dtype) @ projection
                features[query_index, indices, layer_offset] = (
                    compressed.float().cpu().numpy().astype(np.float16)
                )
            lm_scores[query_index, indices] = (yes_score - no_score).cpu().numpy()
            prompt_lengths[query_index, indices] = (
                inputs["attention_mask"].sum(dim=-1).cpu().numpy()
            )
        del inputs, outputs, final, logits

    for query_index, qid in enumerate(plan):
        candidates = retrieval.get(qid, [])[:top_l]
        if len(candidates) != top_l:
            raise ValueError(f"{qid} has {len(candidates)} candidates; expected {top_l}")
        question = str(questions[qid]["question"])
        text_indices, text_prompts = [], []
        image_indices, image_prompts, image_inputs = [], [], []
        for candidate_index, candidate in enumerate(candidates):
            row = corpus[str(candidate["chunk_id"])]
            modality = str(row.get("modality", "text"))
            labels[query_index, candidate_index] = candidate["support_label"] == "support"
            cosine_scores[query_index, candidate_index] = float(candidate["cosine_score"])
            reranker_scores[query_index, candidate_index] = float(
                candidate.get("jina_reranker_score", candidate.get("selection_score", np.nan))
            )
            bge_reranker_scores[query_index, candidate_index] = float(
                candidate.get("bge_reranker_score", np.nan)
            )
            modalities[query_index, candidate_index] = modality
            chunk_ids[query_index, candidate_index] = str(candidate["chunk_id"])
            if modality == "image":
                path = resolve_image(row, args.bundle_root)
                image_indices.append(candidate_index)
                image_prompts.append(image_prompt(processor, question, path))
                image_inputs.append(Image.open(path).convert("RGB"))
            else:
                text_indices.append(candidate_index)
                text_prompts.append(text_prompt(processor, question, str(row["content"])))
        for start in range(0, len(text_indices), args.text_batch_size):
            stop = start + args.text_batch_size
            run_batch(text_indices[start:stop], text_prompts[start:stop], None)
        for start in range(0, len(image_indices), args.image_batch_size):
            stop = start + args.image_batch_size
            run_batch(
                image_indices[start:stop], image_prompts[start:stop], image_inputs[start:stop]
            )
        for image in image_inputs:
            image.close()
        elapsed = time.perf_counter() - started
        print(
            f"[{query_index + 1}/{n_queries}] {qid} "
            f"images={len(image_indices)} q/s={(query_index + 1) / elapsed:.3f}",
            flush=True,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "features.npz",
        features=features,
        lm_relevance_scores=lm_scores,
        labels=labels,
        cosine_scores=cosine_scores,
        reranker_scores=reranker_scores,
        bge_reranker_scores=bge_reranker_scores,
        modalities=modalities,
        chunk_ids=chunk_ids,
        qids=np.asarray(plan),
        prompt_lengths=prompt_lengths,
        layer_ids=np.asarray(layers, dtype=np.int16),
    )
    manifest = {
        "status": "complete",
        "method": "qwen2vl_pairwise_relevance_hidden_and_lm_head",
        "model": args.model,
        "model_revision": args.model_revision,
        "input_role": plan_payload["input_role"],
        "source_split": plan_payload["source_split"],
        "n_queries": n_queries,
        "top_l": top_l,
        "layers": layers,
        "projection_dim": args.projection_dim,
        "projection_seed": 1949,
        "text_batch_size": args.text_batch_size,
        "image_batch_size": args.image_batch_size,
        "max_length": args.max_length,
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "plan": plan,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in manifest.items() if k != "plan"}, indent=2))


if __name__ == "__main__":
    main()
