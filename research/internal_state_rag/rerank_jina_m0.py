"""Rerank a fixed retrieval reserve with Jina's multimodal reranker-m0."""

from __future__ import annotations

import argparse
import gzip
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor


def iter_jsonl(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_corpus(path: Path, bundle_root: Path) -> dict[str, dict]:
    rows = {}
    for row in iter_jsonl(path):
        row = dict(row)
        if row.get("modality") == "image":
            content = str(row["content"])
            if not content.startswith(("http://", "https://")):
                row["content"] = str((bundle_root / content).resolve())
        rows[str(row["id"])] = row
    return rows


def output_context(path: Path):
    temporary = path.with_name(path.name + ".tmp")
    opener = gzip.open if path.suffix == ".gz" else open
    return temporary, opener(temporary, "wt", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="jinaai/jina-reranker-m0")
    parser.add_argument(
        "--model-revision", default="94bfe0aeb2d4dd7978362699cddd5893d4e0adc8"
    )
    parser.add_argument("--text-batch-size", type=int, default=16)
    parser.add_argument("--image-batch-size", type=int, default=8)
    parser.add_argument("--text-max-length", type=int, default=1024)
    parser.add_argument("--image-max-length", type=int, default=2048)
    parser.add_argument("--max-image-pixels", type=int, default=200704)
    parser.add_argument("--checkpoint-every", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = list(iter_jsonl(args.retrieval))
    corpus = load_corpus(args.corpus, args.bundle_root)
    score_path = args.output.with_suffix(args.output.suffix + ".scores.npy")
    if score_path.is_file():
        scores = np.load(score_path, mmap_mode="r+")
        if scores.shape != (len(rows),):
            raise ValueError(f"Checkpoint shape {scores.shape} != {(len(rows),)}")
    else:
        score_path.parent.mkdir(parents=True, exist_ok=True)
        scores = np.lib.format.open_memmap(
            score_path, mode="w+", dtype=np.float32, shape=(len(rows),)
        )
        scores[:] = np.nan
        scores.flush()

    missing = np.flatnonzero(~np.isfinite(scores))
    started = time.perf_counter()
    if len(missing):
        model = AutoModel.from_pretrained(
            args.model,
            revision=args.model_revision,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).eval().to("cuda")
        model._processor = AutoProcessor.from_pretrained(
            args.model,
            revision=args.model_revision,
            max_pixels=args.max_image_pixels,
            min_pixels=3136,
            trust_remote_code=True,
        )
        for modality, batch_size, max_length in (
            ("text", args.text_batch_size, args.text_max_length),
            ("image", args.image_batch_size, args.image_max_length),
        ):
            indices = [
                int(index)
                for index in missing
                if ("image" if rows[int(index)]["modality"] == "image" else "text")
                == modality
            ]
            for offset in range(0, len(indices), args.checkpoint_every):
                block = indices[offset : offset + args.checkpoint_every]
                pairs = [
                    [
                        str(rows[index]["query_text"]),
                        str(corpus[str(rows[index]["chunk_id"])]["content"]),
                    ]
                    for index in block
                ]
                values = model.compute_score(
                    pairs,
                    batch_size=batch_size,
                    max_length=max_length,
                    doc_type=modality,
                    show_progress=True,
                )
                scores[block] = np.asarray(values, dtype=np.float32)
                scores.flush()
                print(
                    f"{modality}: {min(offset + len(block), len(indices))}/{len(indices)}",
                    flush=True,
                )

    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[str(row["qid"])].append(index)
    output_rows = []
    for indices in grouped.values():
        ordered = sorted(
            indices,
            key=lambda index: (-float(scores[index]), str(rows[index]["chunk_id"])),
        )
        for rank, index in enumerate(ordered, start=1):
            row = dict(rows[index])
            row["cosine_rank"] = int(row["rank"])
            row["rank"] = rank
            if "bge" in str(row.get("selection_score_id", "")).lower():
                row["bge_reranker_score"] = float(row["selection_score"])
            row["jina_reranker_score"] = float(scores[index])
            row["selection_score"] = float(scores[index])
            row["selection_score_id"] = (
                f"multimodal_cross_encoder:{args.model}"
                f"@{args.model_revision}"
                f"#text_max={args.text_max_length}#image_max={args.image_max_length}"
            )
            output_rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary, handle = output_context(args.output)
    with handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(args.output)
    manifest = {
        "status": "complete",
        "model": args.model,
        "model_revision": args.model_revision,
        "input": str(args.retrieval),
        "output": str(args.output),
        "rows": len(rows),
        "queries": len(grouped),
        "text_pairs": sum(row["modality"] != "image" for row in rows),
        "image_pairs": sum(row["modality"] == "image" for row in rows),
        "text_batch_size": args.text_batch_size,
        "image_batch_size": args.image_batch_size,
        "text_max_length": args.text_max_length,
        "image_max_length": args.image_max_length,
        "max_image_pixels": args.max_image_pixels,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    Path(str(args.output) + ".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
