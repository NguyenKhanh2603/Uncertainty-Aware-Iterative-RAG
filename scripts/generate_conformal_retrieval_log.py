"""Generate a scored top-L retrieval log for conformal calibration.

The default input is the public 800Q Hugging Face bundle and is explicitly marked
as smoke-grade. The same runner can consume a larger bundle with the same schema.
It uses one frozen Jina CLIP v2 embedding space for text, serialized tables, images,
and text queries. It does not load Qwen and does not use vLLM.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from tqdm import tqdm

from uncertainty_rag.core.retrieval_log import (
    QUERY_TYPE_RULE_IDS,
    CorpusRecord,
    corpus_revision,
    frozen_query_type,
    load_bundle_records,
    retrieval_log_row,
    stable_json_hash,
    stable_split_role,
)

DEFAULT_MODEL = "jinaai/jina-clip-v2"
DEFAULT_MODEL_REVISION = "e10d47f5691d0454a0fb5d13f46f2199b74cb436"
DEFAULT_HF_REPO = "danny2507/attention-uq-800q-colab"
DEFAULT_HF_REVISION = "26c3b8269d6ec5f17463f714bbc400658dd01313"
PREPROCESS_VERSION = "conformal-retrieval-log-v1"


def is_cuda_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def to_normalized_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        array = value.detach().float().cpu().numpy()
    else:
        array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise RuntimeError("Embedding model returned a zero vector")
    return (array / norms).astype(np.float32, copy=False)


def encode_with_backoff(
    items: Sequence[str],
    *,
    encode_batch: Callable[[Sequence[str]], Any],
    initial_batch_size: int,
    description: str,
) -> np.ndarray:
    """Encode all inputs and halve the batch automatically after CUDA OOM."""

    if not items:
        return np.empty((0, 0), dtype=np.float32)
    batch_size = max(1, initial_batch_size)
    vectors: list[np.ndarray] = []
    index = 0
    progress = tqdm(total=len(items), desc=description)
    while index < len(items):
        current_size = min(batch_size, len(items) - index)
        batch = items[index : index + current_size]
        try:
            with torch.inference_mode():
                encoded = to_normalized_numpy(encode_batch(batch))
        except BaseException as exc:
            if not is_cuda_oom(exc) or current_size == 1:
                progress.close()
                raise
            batch_size = max(1, current_size // 2)
            torch.cuda.empty_cache()
            print(f"[OOM RETRY] {description}: batch {current_size} -> {batch_size}")
            continue
        if len(encoded) != current_size:
            progress.close()
            raise RuntimeError(
                f"Encoder returned {len(encoded)} vectors for a batch of {current_size}"
            )
        vectors.append(encoded)
        index += current_size
        progress.update(current_size)
    progress.close()
    return np.concatenate(vectors, axis=0)


def resolve_dtype(device: str, dtype_name: str) -> torch.dtype:
    if dtype_name == "auto":
        return (
            torch.bfloat16
            if device.startswith("cuda") and torch.cuda.is_bf16_supported()
            else torch.float32
        )
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype_name]


def load_model(model_name: str, revision: str, device: str, dtype_name: str):
    from transformers import AutoModel

    dtype = resolve_dtype(device, dtype_name)
    model = AutoModel.from_pretrained(
        model_name,
        revision=revision,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    model = model.eval().to(device)
    return model, dtype


def cache_paths(cache_dir: Path, cache_key: str) -> tuple[Path, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{cache_key}.npy", cache_dir / f"{cache_key}.json"


def load_cached_embeddings(
    cache_dir: Path,
    cache_key: str,
    expected_ids: Sequence[str],
) -> np.ndarray | None:
    vectors_path, metadata_path = cache_paths(cache_dir, cache_key)
    if not vectors_path.is_file() or not metadata_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("ids") != list(expected_ids):
        return None
    vectors = np.load(vectors_path, mmap_mode="r")
    if len(vectors) != len(expected_ids):
        return None
    print(f"Loaded embedding cache: {vectors_path}")
    return np.asarray(vectors)


def save_embedding_cache(
    cache_dir: Path,
    cache_key: str,
    ids: Sequence[str],
    vectors: np.ndarray,
) -> None:
    vectors_path, metadata_path = cache_paths(cache_dir, cache_key)
    temporary_vectors = vectors_path.with_suffix(".npy.tmp")
    with temporary_vectors.open("wb") as handle:
        np.save(handle, vectors.astype(np.float32, copy=False))
    temporary_vectors.replace(vectors_path)
    temporary_metadata = metadata_path.with_suffix(".json.tmp")
    temporary_metadata.write_text(
        json.dumps({"ids": list(ids)}, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    temporary_metadata.replace(metadata_path)


def encode_corpus(
    model,
    corpus: Sequence[CorpusRecord],
    *,
    truncate_dim: int,
    text_batch_size: int,
    image_batch_size: int,
) -> np.ndarray:
    vectors: np.ndarray | None = None
    text_indices = [index for index, chunk in enumerate(corpus) if chunk.modality != "image"]
    image_indices = [index for index, chunk in enumerate(corpus) if chunk.modality == "image"]

    text_vectors = encode_with_backoff(
        [corpus[index].content for index in text_indices],
        encode_batch=lambda batch: model.encode_text(list(batch), truncate_dim=truncate_dim),
        initial_batch_size=text_batch_size,
        description="Encode text/table corpus",
    )
    if len(text_vectors):
        vectors = np.empty((len(corpus), text_vectors.shape[1]), dtype=np.float32)
        vectors[text_indices] = text_vectors

    image_vectors = encode_with_backoff(
        [corpus[index].content for index in image_indices],
        encode_batch=lambda batch: model.encode_image(list(batch), truncate_dim=truncate_dim),
        initial_batch_size=image_batch_size,
        description="Encode image corpus",
    )
    if len(image_vectors):
        if vectors is None:
            vectors = np.empty((len(corpus), image_vectors.shape[1]), dtype=np.float32)
        if image_vectors.shape[1] != vectors.shape[1]:
            raise RuntimeError("Text and image encoders returned different dimensions")
        vectors[image_indices] = image_vectors

    if vectors is None:
        raise RuntimeError("Corpus contains no encodable chunks")
    return vectors


def exact_top_l(
    query_vectors: np.ndarray,
    corpus_vectors: np.ndarray,
    *,
    top_l: int,
    device: str,
    query_batch_size: int,
    corpus_block_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact blockwise cosine top-L using normalized float32 vectors."""

    if top_l > len(corpus_vectors):
        raise ValueError(f"top_l={top_l} exceeds corpus size={len(corpus_vectors)}")
    all_scores: list[np.ndarray] = []
    all_indices: list[np.ndarray] = []
    for start in tqdm(range(0, len(query_vectors), query_batch_size), desc="Exact top-L"):
        query_tensor = torch.from_numpy(
            np.asarray(query_vectors[start : start + query_batch_size], dtype=np.float32)
        ).to(device)
        best_scores = torch.full(
            (len(query_tensor), top_l), -torch.inf, device=device, dtype=torch.float32
        )
        best_indices = torch.full(
            (len(query_tensor), top_l), -1, device=device, dtype=torch.long
        )
        for block_start in range(0, len(corpus_vectors), corpus_block_size):
            block = torch.from_numpy(
                np.asarray(
                    corpus_vectors[block_start : block_start + corpus_block_size],
                    dtype=np.float32,
                )
            ).to(device)
            scores = query_tensor @ block.T
            block_indices = torch.arange(
                block_start,
                block_start + len(block),
                device=device,
                dtype=torch.long,
            ).expand(len(query_tensor), -1)
            merged_scores = torch.cat((best_scores, scores), dim=1)
            merged_indices = torch.cat((best_indices, block_indices), dim=1)
            best_scores, positions = torch.topk(merged_scores, k=top_l, dim=1)
            best_indices = torch.gather(merged_indices, 1, positions)
            del block, scores, block_indices, merged_scores, merged_indices, positions
        all_scores.append(best_scores.cpu().numpy())
        all_indices.append(best_indices.cpu().numpy())
        del query_tensor, best_scores, best_indices
    return np.concatenate(all_scores), np.concatenate(all_indices)


def resolve_bundle(args: argparse.Namespace) -> Path:
    if args.bundle_dir is not None:
        return args.bundle_dir.resolve()
    from huggingface_hub import snapshot_download

    print(f"Downloading dataset bundle {args.hf_repo}@{args.hf_revision}")
    return Path(
        snapshot_download(
            repo_id=args.hf_repo,
            repo_type="dataset",
            revision=args.hf_revision,
            allow_patterns=["manifest.json", f"{args.dataset}/**"],
            token=os.environ.get("HF_TOKEN"),
        )
    )


def output_context(path: Path):
    temporary = path.with_name(f"{path.name}.tmp")
    if path.suffix == ".gz":
        return temporary, gzip.open(temporary, "wt", encoding="utf-8")
    return temporary, temporary.open("w", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--bundle-dir", type=Path)
    source.add_argument("--hf-repo", default=DEFAULT_HF_REPO)
    parser.add_argument("--hf-revision", default=DEFAULT_HF_REVISION)
    parser.add_argument("--dataset", default="mmqa")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/conformal_retrieval"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument("--truncate-dim", type=int, default=512)
    parser.add_argument("--top-l", type=int, default=20)
    parser.add_argument("--text-batch-size", type=int, default=32)
    parser.add_argument("--image-batch-size", type=int, default=8)
    parser.add_argument("--query-batch-size", type=int, default=64)
    parser.add_argument("--corpus-block-size", type=int, default=16384)
    parser.add_argument("--split-seed", type=int, default=8092026)
    parser.add_argument("--development-fraction", type=float, default=0.2)
    parser.add_argument("--calibration-fraction", type=float, default=0.6)
    parser.add_argument("--query-type-mode", choices=tuple(QUERY_TYPE_RULE_IDS), default="pooled")
    parser.add_argument("--non-support-label", choices=("false", "unknown"), default="false")
    parser.add_argument("--data-grade", choices=("smoke", "paper"), default="smoke")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if args.top_l < 1 or args.truncate_dim < 1:
        raise ValueError("top_l and truncate_dim must be positive")

    bundle_root = resolve_bundle(args)
    questions_path = bundle_root / args.dataset / "questions.jsonl"
    corpus, queries = load_bundle_records(
        questions_path,
        bundle_root=bundle_root,
        dataset=args.dataset,
    )
    corpus_revision_id = corpus_revision(corpus)
    resolved_dtype = resolve_dtype(args.device, args.dtype)
    dtype_id = str(resolved_dtype).removeprefix("torch.")
    retriever_id = (
        f"{args.model}@{args.model_revision}#dim={args.truncate_dim}#dtype={dtype_id}"
    )
    preprocess_hash = f"sha256:{stable_json_hash({'version': PREPROCESS_VERSION})}"
    query_type_rule_id = QUERY_TYPE_RULE_IDS[args.query_type_mode]
    print(
        f"dataset={args.dataset} queries={len(queries)} corpus={len(corpus)} "
        f"device={args.device} model={retriever_id}"
    )

    cache_identity = stable_json_hash(
        {
            "dataset": args.dataset,
            "corpus_revision": corpus_revision_id,
            "retriever_id": retriever_id,
            "preprocess_hash": preprocess_hash,
        }
    )[:24]
    corpus_ids = [chunk.chunk_id for chunk in corpus]
    corpus_vectors = load_cached_embeddings(args.cache_dir, f"corpus-{cache_identity}", corpus_ids)

    model = None
    if corpus_vectors is None:
        model, _ = load_model(args.model, args.model_revision, args.device, args.dtype)
        corpus_vectors = encode_corpus(
            model,
            corpus,
            truncate_dim=args.truncate_dim,
            text_batch_size=args.text_batch_size,
            image_batch_size=args.image_batch_size,
        )
        save_embedding_cache(args.cache_dir, f"corpus-{cache_identity}", corpus_ids, corpus_vectors)

    query_ids = [query.qid for query in queries]
    query_cache_identity = stable_json_hash(
        {
            "corpus_cache": cache_identity,
            "query_ids": query_ids,
            "query_text_hash": stable_json_hash([query.question for query in queries]),
            "task": "retrieval.query",
        }
    )[:24]
    query_vectors = load_cached_embeddings(
        args.cache_dir, f"queries-{query_cache_identity}", query_ids
    )
    if query_vectors is None:
        if model is None:
            model, _ = load_model(
                args.model, args.model_revision, args.device, args.dtype
            )
        query_vectors = encode_with_backoff(
            [query.question for query in queries],
            encode_batch=lambda batch: model.encode_text(
                list(batch), task="retrieval.query", truncate_dim=args.truncate_dim
            ),
            initial_batch_size=args.text_batch_size,
            description="Encode queries",
        )
        save_embedding_cache(
            args.cache_dir,
            f"queries-{query_cache_identity}",
            query_ids,
            query_vectors,
        )

    scores, indices = exact_top_l(
        np.asarray(query_vectors),
        np.asarray(corpus_vectors),
        top_l=args.top_l,
        device=args.device,
        query_batch_size=args.query_batch_size,
        corpus_block_size=args.corpus_block_size,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary, handle_context = output_context(args.output)
    role_counts = {"development": 0, "calibration": 0, "test": 0}
    support_hits = 0
    with handle_context as handle:
        for query_index, query in enumerate(queries):
            split_role = stable_split_role(
                query.dataset,
                query.qid,
                seed=args.split_seed,
                development_fraction=args.development_fraction,
                calibration_fraction=args.calibration_fraction,
            )
            role_counts[split_role] += 1
            query_type = frozen_query_type(query.question, args.query_type_mode)
            ranked = sorted(
                zip(scores[query_index], indices[query_index]),
                key=lambda pair: (-float(pair[0]), corpus[int(pair[1])].chunk_id),
            )
            for rank, (score, corpus_index) in enumerate(ranked, start=1):
                chunk = corpus[int(corpus_index)]
                row = retrieval_log_row(
                    query=query,
                    chunk=chunk,
                    split_role=split_role,
                    query_type=query_type,
                    rank=rank,
                    cosine_score=float(score),
                    top_l=args.top_l,
                    retriever_id=retriever_id,
                    corpus_revision_id=corpus_revision_id,
                    preprocess_hash=preprocess_hash,
                    query_type_rule_id=query_type_rule_id,
                    non_support_label=args.non_support_label,
                )
                support_hits += int(row["support_label"] == "support")
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(args.output)

    manifest = {
        "schema_version": 1,
        "data_grade": args.data_grade,
        "warning": (
            "The public 800Q bundle is smoke/development data, not paper-grade calibration."
            if args.data_grade == "smoke"
            else None
        ),
        "dataset": args.dataset,
        "bundle": {
            "local_path": str(bundle_root),
            "hf_repo": args.hf_repo if args.bundle_dir is None else None,
            "hf_revision": args.hf_revision if args.bundle_dir is None else None,
        },
        "queries": len(queries),
        "corpus_chunks": len(corpus),
        "rows": len(queries) * args.top_l,
        "top_l": args.top_l,
        "role_counts": role_counts,
        "retrieved_support_rows": support_hits,
        "retriever_id": retriever_id,
        "corpus_revision": corpus_revision_id,
        "preprocess_hash": preprocess_hash,
        "query_type_rule_id": query_type_rule_id,
        "non_support_label": args.non_support_label,
        "device": args.device,
        "model_dtype": str(resolved_dtype),
        "output": args.output.name,
    }
    manifest_path = Path(f"{args.output}.manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"Wrote {manifest['rows']} rows to {args.output}; "
        f"support hits={support_hits}; splits={role_counts}"
    )
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
