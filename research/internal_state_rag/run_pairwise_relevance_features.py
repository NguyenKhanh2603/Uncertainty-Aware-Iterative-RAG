"""Extract answer-position states for explicit question--chunk relevance prompts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from research.internal_state_rag.contrastive_saliency import _memory_efficient_attention
from research.internal_state_rag.run_tatqa_smoke import (
    ResearchTextClient,
    load_retrieval,
    load_rows,
)


def parse_args() -> argparse.Namespace:
    root = Path("data/conformal_global_run")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--questions",
        type=Path,
        default=root / "official_bundle_role_split_tatqa/tatqa/questions.jsonl",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=root / "official_bundle_role_split_tatqa/tatqa/corpus.jsonl",
    )
    parser.add_argument(
        "--retrieval",
        type=Path,
        default=root / "retrieval/tatqa_top30_bge_reranked.jsonl.gz",
    )
    parser.add_argument("--layers", default="6,12,18,24,30,35")
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-length", type=int, default=2048)
    return parser.parse_args()


def relevance_prompt(tokenizer: object, question: str, passage: str) -> str:
    instruction = (
        "Determine whether the passage contains information needed to answer the "
        "question. Reply with exactly Yes or No.\n"
        f"Question: {question}\n"
        f"Passage: {passage}\n"
        "Does this passage contain useful evidence?"
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}],
        tokenize=False,
        add_generation_prompt=True,
    )


def single_token_ids(tokenizer: object, strings: tuple[str, ...]) -> list[int]:
    ids = []
    for value in strings:
        encoded = tokenizer.encode(value, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"Expected one token for {value!r}, got {encoded}")
        ids.append(int(encoded[0]))
    return sorted(set(ids))


def main() -> None:
    args = parse_args()
    source_manifest = json.loads(args.feature_manifest.read_text(encoding="utf-8"))
    plan = [str(value) for value in source_manifest["plan"]]
    top_l = int(source_manifest["top_l"])
    layers = [int(value) for value in args.layers.split(",") if value.strip()]
    questions = load_rows(args.questions, "qid")
    corpus = load_rows(args.corpus, "id")
    retrieval = load_retrieval(
        args.retrieval, split_role=str(source_manifest["input_role"])
    )

    client = ResearchTextClient(args.model, device="cuda", load_in_4bit=False)
    tokenizer = client.tokenizer
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    yes_ids = single_token_ids(tokenizer, ("Yes", " yes", "YES"))
    no_ids = single_token_ids(tokenizer, ("No", " no", "NO"))
    hidden_size = int(client.model.config.hidden_size)
    generator = torch.Generator(device="cpu").manual_seed(1949)
    projection = torch.randn(
        hidden_size, args.projection_dim, generator=generator, dtype=torch.float32
    ) / np.sqrt(args.projection_dim)
    projection = projection.to(client.model.device, dtype=client.model.dtype)

    feature_rows, lm_score_rows, prompt_length_rows = [], [], []
    label_rows, bge_rows, rank_rows = [], [], []
    with torch.inference_mode(), _memory_efficient_attention(client.model):
        for query_index, qid in enumerate(plan, start=1):
            question = questions[qid]
            candidates = retrieval[qid][:top_l]
            if len(candidates) != top_l:
                raise ValueError(f"{qid} has {len(candidates)} candidates, expected {top_l}")
            prompts = [
                relevance_prompt(
                    tokenizer,
                    str(question["question"]),
                    str(corpus[str(candidate["chunk_id"])]["content"]),
                )
                for candidate in candidates
            ]
            query_features, query_scores, query_lengths = [], [], []
            for start in range(0, top_l, args.batch_size):
                batch_prompts = prompts[start : start + args.batch_size]
                inputs = tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=args.max_length,
                ).to(client.model.device)
                outputs = client.model.model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
                layer_features = []
                for layer in layers:
                    state = outputs.hidden_states[layer + 1][:, -1, :]
                    state = torch.nn.functional.normalize(state.float(), p=2, dim=-1)
                    compressed = state.to(projection.dtype) @ projection
                    layer_features.append(compressed.float().cpu().to(torch.float16).numpy())
                query_features.append(np.stack(layer_features, axis=1))
                logits = client.model.lm_head(outputs.last_hidden_state[:, -1, :]).float()
                yes_score = torch.logsumexp(logits[:, yes_ids], dim=-1)
                no_score = torch.logsumexp(logits[:, no_ids], dim=-1)
                query_scores.extend((yes_score - no_score).cpu().tolist())
                query_lengths.extend(inputs["attention_mask"].sum(dim=-1).cpu().tolist())
                del outputs, inputs, logits

            feature_rows.append(np.concatenate(query_features, axis=0))
            lm_score_rows.append(query_scores)
            prompt_length_rows.append(query_lengths)
            label_rows.append(
                [candidate["support_label"] == "support" for candidate in candidates]
            )
            bge_rows.append([float(candidate["selection_score"]) for candidate in candidates])
            rank_rows.append([int(candidate["rank"]) for candidate in candidates])
            print(
                f"[{query_index}/{len(plan)}] {qid} "
                f"tokens={min(query_lengths)}-{max(query_lengths)}",
                flush=True,
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "features.npz",
        features=np.stack(feature_rows),
        lm_relevance_scores=np.asarray(lm_score_rows, dtype=np.float32),
        labels=np.asarray(label_rows, dtype=np.bool_),
        bge_scores=np.asarray(bge_rows, dtype=np.float32),
        ranks=np.asarray(rank_rows, dtype=np.int16),
        qids=np.asarray(plan),
        prompt_lengths=np.asarray(prompt_length_rows, dtype=np.int32),
        layer_ids=np.asarray(layers, dtype=np.int16),
    )
    manifest = {
        "status": "complete",
        "method": "pairwise_question_chunk_relevance_prompt",
        "model": args.model,
        "source_feature_manifest": str(args.feature_manifest),
        "input_role": source_manifest["input_role"],
        "source_split": source_manifest["source_split"],
        "n_queries": len(plan),
        "top_l": top_l,
        "layers": layers,
        "hidden_size": hidden_size,
        "projection_dim": args.projection_dim,
        "projection_seed": 1949,
        "yes_token_ids": yes_ids,
        "no_token_ids": no_ids,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "plan": plan,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in manifest.items() if key != "plan"}, indent=2))


if __name__ == "__main__":
    main()
