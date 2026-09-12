"""Rescore a frozen Top-L reserve with a candidate-level cross-encoder.

The dense retriever still defines the candidate set. This script adds a separate
``selection_score`` for conformal calibration and reranks only within that frozen
reserve. It is intended for targeted score-separation experiments, not corpus
retrieval.
"""

from __future__ import annotations

import argparse
import gzip
import itertools
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from uncertainty_rag.core.retrieval_log import load_bundle_records, stable_json_hash

DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"


def open_text(path: Path, mode: str):
    return gzip.open(path, mode, encoding="utf-8") if path.suffix == ".gz" else path.open(
        mode, encoding="utf-8"
    )


def read_rows(path: Path) -> list[dict[str, Any]]:
    with open_text(path, "rt") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def score_metrics(rows: Sequence[dict[str, Any]], score_field: str) -> dict[str, float] | None:
    labelled = [row for row in rows if row["support_label"] in {"support", "false"}]
    labels = np.asarray([row["support_label"] == "support" for row in labelled], dtype=np.int8)
    if not len(labels) or len(np.unique(labels)) != 2:
        return None
    scores = np.asarray([float(row[score_field]) for row in labelled], dtype=np.float64)
    return {
        "rows": int(len(labels)),
        "supports": int(labels.sum()),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
    }


def diagnostics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split_role in sorted({str(row["split_role"]) for row in rows}):
        split_rows = [row for row in rows if row["split_role"] == split_role]
        result[split_role] = {
            "cosine": score_metrics(split_rows, "cosine_score"),
            "selection": score_metrics(split_rows, "selection_score"),
        }
    return result


def rerank_rows(
    rows: Sequence[dict[str, Any]], scores: Sequence[float], score_id: str
) -> list[dict[str, Any]]:
    if len(rows) != len(scores):
        raise ValueError("rows and scores must align")
    indexed = [(index, dict(row)) for index, row in enumerate(rows)]
    output: list[dict[str, Any]] = []
    for _, group_iterator in itertools.groupby(
        indexed, key=lambda item: (item[1]["dataset"], item[1]["qid"])
    ):
        group = list(group_iterator)
        ordered = sorted(
            group,
            key=lambda item: (-float(scores[item[0]]), str(item[1]["chunk_id"])),
        )
        modality_ranks: dict[str, int] = {}
        for rank, (source_index, row) in enumerate(ordered, start=1):
            modality = str(row["modality"])
            modality_ranks[modality] = modality_ranks.get(modality, 0) + 1
            row["cosine_rank"] = row["rank"]
            row["rank"] = rank
            row["modality_rank"] = modality_ranks[modality]
            row["selection_score"] = float(scores[source_index])
            row["selection_score_id"] = score_id
            output.append(row)
    return output


def encode_pairs_with_backoff(
    model,
    tokenizer,
    queries: Sequence[str],
    documents: Sequence[str],
    *,
    batch_size: int,
    max_length: int,
    device: str,
) -> list[float]:
    scores: list[float] = []
    index = 0
    progress = tqdm(total=len(queries), desc="Cross-encoder rescore", unit="pair")
    while index < len(queries):
        current_size = min(batch_size, len(queries) - index)
        try:
            inputs = tokenizer(
                list(queries[index : index + current_size]),
                list(documents[index : index + current_size]),
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            with torch.inference_mode():
                logits = model(**inputs, return_dict=True).logits.view(-1).float()
                probabilities = torch.sigmoid(logits).cpu().tolist()
        except torch.cuda.OutOfMemoryError:
            if current_size == 1:
                raise
            batch_size = max(1, current_size // 2)
            torch.cuda.empty_cache()
            print(f"[OOM RETRY] cross-encoder batch {current_size} -> {batch_size}")
            continue
        scores.extend(float(value) for value in probabilities)
        index += current_size
        progress.update(current_size)
    progress.close()
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=DEFAULT_REVISION)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    if args.output.resolve() == args.input.resolve():
        parser.error("--output must differ from --input")
    if args.batch_size < 1 or args.max_length < 2:
        parser.error("--batch-size and --max-length must be positive")

    started = time.perf_counter()
    rows = read_rows(args.input)
    corpus, _ = load_bundle_records(
        args.bundle_dir / args.dataset / "questions.jsonl",
        bundle_root=args.bundle_dir,
        dataset=args.dataset,
    )
    content_by_id = {chunk.chunk_id: chunk.content for chunk in corpus}
    unsupported = sorted({row["modality"] for row in rows if row["modality"] == "image"})
    if unsupported:
        raise ValueError(
            "Image candidates require a multimodal reranker; caption-only is not implicit"
        )
    missing = [row["chunk_id"] for row in rows if row["chunk_id"] not in content_by_id]
    if missing:
        raise ValueError(f"Missing corpus content for chunk_id={missing[0]}")

    score_id = (
        f"cross_encoder_sigmoid:{args.model}@{args.model_revision}"
        f"#max_length={args.max_length}"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.model_revision)
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        revision=args.model_revision,
        torch_dtype=dtype,
    ).eval().to(args.device)
    scores = encode_pairs_with_backoff(
        model,
        tokenizer,
        [str(row["query_text"]) for row in rows],
        [content_by_id[str(row["chunk_id"])] for row in rows],
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
    )
    rescored = rerank_rows(rows, scores, score_id)
    old_preprocess_hashes = sorted({str(row["preprocess_hash"]) for row in rows})
    if len(old_preprocess_hashes) != 1:
        raise ValueError("Input contains multiple preprocessing fingerprints")
    preprocess_hash = "sha256:" + stable_json_hash(
        {"base_preprocess_hash": old_preprocess_hashes[0], "selection_score_id": score_id}
    )
    for row in rescored:
        row["preprocess_hash"] = preprocess_hash

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    output_context = (
        gzip.open(temporary, "wt", encoding="utf-8")
        if args.output.suffix == ".gz"
        else temporary.open("w", encoding="utf-8")
    )
    with output_context as handle:
        for row in rescored:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(args.output)

    base_manifest_path = Path(f"{args.input}.manifest.json")
    manifest = (
        json.loads(base_manifest_path.read_text(encoding="utf-8"))
        if base_manifest_path.is_file()
        else {}
    )
    manifest.update(
        {
            "output": args.output.name,
            "preprocess_hash": preprocess_hash,
            "selection_score_id": score_id,
            "selection_score_policy": "rerank frozen dense Top-L; higher is more relevant",
            "score_diagnostics": diagnostics(rescored),
            "rescore_seconds": round(time.perf_counter() - started, 3),
        }
    )
    Path(f"{args.output}.manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest["score_diagnostics"], indent=2))
    print(f"Wrote {len(rescored)} rescored rows to {args.output}")


if __name__ == "__main__":
    main()
