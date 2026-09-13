"""Extract compressed layerwise chunk states for a frozen relevance probe."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from research.internal_state_rag.contrastive_saliency import _memory_efficient_attention
from research.internal_state_rag.run_full_topl_saliency import excluded_qids
from research.internal_state_rag.run_full_topl_validation import eligible_qids
from research.internal_state_rag.run_tatqa_smoke import (
    ResearchTextClient,
    load_retrieval,
    load_rows,
    to_chunk,
)


def parse_args() -> argparse.Namespace:
    root = Path("data/conformal_global_run")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
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
    parser.add_argument("--input-role", choices=("calibration", "test"), required=True)
    parser.add_argument("--source-split", required=True)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--top-l", type=int, default=30)
    parser.add_argument("--seed", type=int, default=401)
    parser.add_argument("--layers", default="6,12,18,24,30,35")
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--exclude", type=Path, action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    layers = [int(value) for value in args.layers.split(",") if value.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    questions = load_rows(args.questions, "qid")
    corpus = load_rows(args.corpus, "id")
    retrieval = load_retrieval(args.retrieval, split_role=args.input_role)
    eligible = eligible_qids(questions, retrieval, source_split=args.source_split)
    excluded = excluded_qids(args.exclude)
    available = [qid for qid in eligible if qid not in excluded]
    if len(available) < args.n:
        raise ValueError(f"Only {len(available)} eligible unexcluded queries for n={args.n}")
    plan = random.Random(args.seed).sample(available, args.n)

    client = ResearchTextClient(args.model, device="cuda", load_in_4bit=False)
    hidden_size = int(client.model.config.hidden_size)
    generator = torch.Generator(device="cpu").manual_seed(1729)
    projection = torch.randn(
        hidden_size, args.projection_dim, generator=generator, dtype=torch.float32
    ) / np.sqrt(args.projection_dim)
    projection = projection.to(client.model.device, dtype=torch.float32)

    feature_rows = []
    label_rows = []
    bge_rows = []
    rank_rows = []
    prompt_lengths = []
    strata = []
    with torch.inference_mode(), _memory_efficient_attention(client.model):
        for index, qid in enumerate(plan, start=1):
            question = questions[qid]
            retrieved = retrieval[qid][: args.top_l]
            chunks = [to_chunk(row, corpus) for row in retrieved]
            prompt = (
                "Answer using only the supplied context. Return only the short answer.\n"
                f"Question: {question['question']}"
            )
            alignment = client.align_and_prepare_inputs(prompt, chunks)
            inputs = alignment.inputs
            output = client.model.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            spans = alignment.batch_chunk_spans[0]
            per_layer = []
            for layer in layers:
                hidden = output.hidden_states[layer + 1][0].float()
                pooled = []
                for span in spans:
                    if span.start is None or span.end is None:
                        pooled.append(torch.zeros(hidden_size, device=hidden.device))
                    else:
                        pooled.append(hidden[int(span.start) : int(span.end)].mean(dim=0))
                states = torch.stack(pooled)
                states = torch.nn.functional.normalize(states, p=2, dim=-1)
                per_layer.append((states @ projection).cpu().to(torch.float16).numpy())
            feature_rows.append(np.stack(per_layer, axis=1))  # [chunks, layers, projection]
            label_rows.append([row["support_label"] == "support" for row in retrieved])
            bge_rows.append([float(row["selection_score"]) for row in retrieved])
            rank_rows.append([int(row["rank"]) for row in retrieved])
            prompt_lengths.append(int(inputs["input_ids"].shape[1]))
            support_ranks = [int(row["rank"]) for row in retrieved if row["support_label"] == "support"]
            strata.append("hard" if min(support_ranks) > 2 else "easy")
            print(
                f"[{index}/{len(plan)}] {qid} tokens={prompt_lengths[-1]} support={support_ranks}",
                flush=True,
            )

    np.savez_compressed(
        args.output_dir / "features.npz",
        features=np.stack(feature_rows),
        labels=np.asarray(label_rows, dtype=np.bool_),
        bge_scores=np.asarray(bge_rows, dtype=np.float32),
        ranks=np.asarray(rank_rows, dtype=np.int16),
        qids=np.asarray(plan),
        strata=np.asarray(strata),
        prompt_lengths=np.asarray(prompt_lengths, dtype=np.int32),
        layer_ids=np.asarray(layers, dtype=np.int16),
    )
    manifest = {
        "status": "complete",
        "model": args.model,
        "input_role": args.input_role,
        "source_split": args.source_split,
        "n_queries": len(plan),
        "top_l": args.top_l,
        "seed": args.seed,
        "layers": layers,
        "hidden_size": hidden_size,
        "projection_dim": args.projection_dim,
        "projection_seed": 1729,
        "excluded_queries": len(excluded),
        "plan": plan,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
