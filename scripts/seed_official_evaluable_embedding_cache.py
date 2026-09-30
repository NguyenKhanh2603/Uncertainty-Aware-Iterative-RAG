"""Build the exact Jina-v4 corpus cache for an official evaluation bundle.

Vectors for chunk IDs already encoded by the ZIP-20-09 run are copied from
that frozen cache.  Only previously unseen chunks are sent through Jina-v4.
The resulting cache key is identical to ``generate_conformal_retrieval_log``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.generate_conformal_retrieval_log import (
    PREPROCESS_VERSION,
    encode_corpus,
    load_model,
    save_embedding_cache,
)
from uncertainty_rag.core.retrieval_log import (
    corpus_revision,
    load_bundle_records,
    stable_json_hash,
)

MODEL = "jinaai/jina-embeddings-v4"
REVISION = "853c867b65b749f3c3c72a06868140d842e04f06"


def old_vectors(cache_dirs: list[Path], target_ids: set[str]) -> dict[str, np.ndarray]:
    """Merge reusable vectors from the best matching cache in each directory."""

    candidates = []
    for cache_dir in cache_dirs:
        per_directory = []
        for metadata_path in cache_dir.glob("corpus-*.json"):
            vector_path = metadata_path.with_suffix(".npy")
            if not vector_path.is_file():
                continue
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            ids = payload.get("ids")
            if isinstance(ids, list):
                overlap = sum(str(chunk_id) in target_ids for chunk_id in ids)
                per_directory.append((overlap, len(ids), metadata_path, vector_path, ids))
        if per_directory:
            candidates.append(max(per_directory, key=lambda item: (item[0], item[1])))
    if not candidates:
        raise FileNotFoundError(f"No completed corpus cache in: {cache_dirs}")

    reusable: dict[str, np.ndarray] = {}
    for overlap, _, metadata_path, vector_path, ids in candidates:
        vectors = np.load(vector_path, mmap_mode="r")
        if len(vectors) != len(ids):
            raise ValueError(f"Cache metadata/array mismatch: {metadata_path}")
        print(
            f"Reuse source: {vector_path} ({len(ids):,} chunks; overlap={overlap:,})",
            flush=True,
        )
        for index, chunk_id_value in enumerate(ids):
            chunk_id = str(chunk_id_value)
            if chunk_id not in target_ids or chunk_id in reusable:
                continue
            reusable[chunk_id] = np.asarray(vectors[index], dtype=np.float32)
    return reusable


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("hotpotqa", "mmqa", "tatqa", "webqa"))
    parser.add_argument(
        "--bundle-dir", type=Path, default=Path("data/official_evaluable_test_2026_09_30")
    )
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument(
        "--reuse-cache-dir",
        type=Path,
        action="append",
        required=True,
        help="Completed compatible cache directory; repeat to merge disjoint corpus subsets.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--text-batch-size", type=int, default=8)
    parser.add_argument("--image-batch-size", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=128)
    parser.add_argument("--truncate-dim", type=int, default=512)
    parser.add_argument("--max-image-pixels", type=int, default=200704)
    parser.add_argument("--max-text-length", type=int, default=1024)
    args = parser.parse_args()

    corpus, _queries = load_bundle_records(
        args.bundle_dir / args.dataset / "questions.jsonl",
        bundle_root=args.bundle_dir,
        dataset=args.dataset,
    )
    policy = {
        "mode": "global",
        "top_l": 30,
        "candidate_scope": "dataset_provided_per_query",
        "image_caption_fusion": True,
        "max_image_pixels": args.max_image_pixels,
        "max_text_length": args.max_text_length,
        "embedding_checkpoint_every": args.checkpoint_every,
        "min_per_modality": None,
        "available_modalities": sorted({chunk.modality for chunk in corpus}),
    }
    preprocess_hash = (
        f"sha256:{stable_json_hash({'version': PREPROCESS_VERSION, 'retrieval_policy': policy})}"
    )
    retriever_id = (
        f"{MODEL}@{REVISION}#dim={args.truncate_dim}#dtype=bfloat16"
        f"#max_text_length={args.max_text_length}"
    )
    revision = corpus_revision(corpus)
    identity = stable_json_hash(
        {
            "dataset": args.dataset,
            "corpus_revision": revision,
            "retriever_id": retriever_id,
            "preprocess_hash": preprocess_hash,
        }
    )[:24]
    cache_key = f"corpus-{identity}"
    vectors_path = args.cache_dir / f"{cache_key}.npy"
    metadata_path = args.cache_dir / f"{cache_key}.json"
    ids = [chunk.chunk_id for chunk in corpus]
    if vectors_path.is_file() and metadata_path.is_file():
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        if payload.get("ids") == ids and len(np.load(vectors_path, mmap_mode="r")) == len(ids):
            print(f"Completed target cache already exists: {vectors_path}", flush=True)
            return

    reusable = old_vectors(args.reuse_cache_dir, set(ids))
    missing_indices = [
        index for index, chunk in enumerate(corpus) if chunk.chunk_id not in reusable
    ]
    vectors = np.empty((len(corpus), args.truncate_dim), dtype=np.float32)
    for index, chunk in enumerate(corpus):
        if chunk.chunk_id in reusable:
            vectors[index] = reusable[chunk.chunk_id]
    print(
        f"{args.dataset}: reuse={len(corpus) - len(missing_indices):,}, "
        f"encode={len(missing_indices):,}, total={len(corpus):,}",
        flush=True,
    )
    if missing_indices:
        model, _dtype = load_model(
            MODEL,
            REVISION,
            args.device,
            "bfloat16",
            args.max_image_pixels,
            args.max_text_length,
        )
        missing = [corpus[index] for index in missing_indices]
        encoded = encode_corpus(
            model,
            missing,
            truncate_dim=args.truncate_dim,
            text_batch_size=args.text_batch_size,
            image_batch_size=args.image_batch_size,
            image_caption_fusion=True,
            checkpoint_dir=args.cache_dir,
            checkpoint_prefix=f"{cache_key}.missing",
            checkpoint_every=args.checkpoint_every,
        )
        vectors[np.asarray(missing_indices)] = encoded
    save_embedding_cache(args.cache_dir, cache_key, ids, vectors)
    audit = {
        "dataset": args.dataset,
        "cache_key": cache_key,
        "corpus_revision": revision,
        "retriever_id": retriever_id,
        "preprocess_hash": preprocess_hash,
        "total_chunks": len(corpus),
        "reused_chunks": len(corpus) - len(missing_indices),
        "encoded_chunks": len(missing_indices),
    }
    (args.cache_dir / f"{cache_key}.seed_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
