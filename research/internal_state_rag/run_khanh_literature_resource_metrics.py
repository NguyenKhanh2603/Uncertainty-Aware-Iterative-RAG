"""Measure latency and the requested FLOPs proxy for three literature adapters.

This runner reuses the exact selected chunk IDs stored by the completed
``splits_khanh_27_09`` run.  It follows the supplied evaluation loop:

* Qwen2-VL-7B-Instruct in bfloat16 with automatic device placement;
* text/table chunks inserted as ``Context: {content}`` blocks;
* image chunks inserted with min_pixels=3136 and max_pixels=200704;
* the short-answer prompt appended after all context blocks;
* greedy generation with max_new_tokens=24;
* append-only per-query progress for resumption.

Latency covers ``model.generate`` only.  CUDA is synchronized immediately
before and after generation.  ``approx_flops`` deliberately preserves the
requested proxy ``2 * parameter_count * generated_ids.shape[1]``; it is not a
hardware-counter measurement of actual transformer FLOPs.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from eval.metrics import exact_match, token_f1

DATASETS = ("hotpotqa", "mmqa", "tatqa", "webqa")
METHODS = (
    "cce_conformal_embedding_jina_alpha_0.10",
    "conflare_source_question_jina_alpha_0.10",
    "traq_retrieval_bonferroni_jina_alpha_0.10",
)
RESULT_ROOT = Path(
    "research/internal_state_rag/results/"
    "full_context_selection_splits_khanh_27_09_2026_09_27"
)
DATA_ROOT = Path("data/zip_calibration_split_20_09")
WEBQA_ROOT = Path("/dev/shm/uncertainty_rag_webqa_stage_20260922")
MODEL = Path(
    "/workspace/hf_cache/hub/models--Qwen--Qwen2-VL-7B-Instruct/"
    "snapshots/eed13092ef92e448dd6875b2a00151bd3f7db0ac"
)
PROMPT = "Answer using only the supplied context. Return only the short answer.\nQuestion: "


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--webqa-root", type=Path, default=WEBQA_ROOT)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def keyed(path: Path, key: str) -> dict[str, dict[str, Any]]:
    return {str(row[key]): row for row in iter_jsonl(path)}


def source_paths(
    dataset: str, data_root: Path, webqa_root: Path
) -> tuple[Path, Path, Path]:
    root = webqa_root if dataset == "webqa" else data_root
    return root / dataset / "questions.jsonl", root / dataset / "corpus.jsonl", root


def image_path(value: str, bundle_root: Path) -> Path:
    raw = Path(value)
    candidates = (raw, bundle_root / raw, bundle_root.parent / "official_bundle_1000_more" / raw)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Missing image context: {value}")


def validate_image(path: Path) -> None:
    with Image.open(path) as image:
        image.verify()


def completed_records(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    completed: dict[tuple[str, str, str], dict[str, Any]] = {}
    if not path.exists():
        return completed
    for row in iter_jsonl(path):
        key = (str(row["method"]), str(row["dataset"]), str(row["qid"]))
        if key in completed:
            raise ValueError(f"Duplicate progress key: {key}")
        completed[key] = row
    return completed


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def summary_rows(
    records: list[dict[str, Any]], methods: list[str], datasets: list[str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in methods:
        method_records = [row for row in records if row["method"] == method]
        for dataset in datasets + ["OVERALL"]:
            selected = (
                method_records
                if dataset == "OVERALL"
                else [row for row in method_records if row["dataset"] == dataset]
            )
            if not selected:
                continue
            rows.append(
                {
                    "method": method,
                    "dataset": dataset,
                    "queries": len(selected),
                    "em": fmean(float(row["em"]) for row in selected),
                    "f1": fmean(float(row["f1"]) for row in selected),
                    "latency_seconds": fmean(float(row["latency_seconds"]) for row in selected),
                    "approx_tflops": fmean(float(row["approx_flops"]) for row in selected)
                    / 1e12,
                    "mean_input_tokens": fmean(float(row["input_tokens"]) for row in selected),
                    "mean_output_tokens": fmean(float(row["output_tokens"]) for row in selected),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    datasets = [value.strip() for value in args.datasets.split(",") if value.strip()]
    methods = [value.strip() for value in args.methods.split(",") if value.strip()]
    if any(value not in DATASETS for value in datasets):
        raise ValueError(f"Unknown dataset in {datasets}")
    if any(value not in METHODS for value in methods):
        raise ValueError(f"Unknown method in {methods}")
    if args.limit < 0:
        raise ValueError("--limit must be nonnegative")
    if not args.model.is_dir():
        raise FileNotFoundError(args.model)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_dir / "progress_log.jsonl"
    completed = completed_records(progress_path)

    prepared: dict[str, dict[str, Any]] = {}
    for dataset in datasets:
        prediction_path = args.result_root / f"{dataset}_downstream_predictions.jsonl"
        stored = list(iter_jsonl(prediction_path))
        if args.limit:
            stored = stored[: args.limit]
        if len(stored) != (args.limit or 100):
            raise ValueError(f"{dataset}: expected {args.limit or 100} stored test queries")
        questions_path, corpus_path, bundle_root = source_paths(
            dataset, args.data_root, args.webqa_root
        )
        questions, corpus = keyed(questions_path, "qid"), keyed(corpus_path, "id")
        prepared[dataset] = {
            "stored": stored,
            "questions": questions,
            "corpus": corpus,
            "bundle_root": bundle_root,
        }
        for row in stored:
            qid = str(row["qid"])
            if qid not in questions:
                raise ValueError(f"{dataset}: missing question {qid}")
            for method in methods:
                if method not in row["contexts"]:
                    raise ValueError(f"{dataset}/{qid}: missing context for {method}")
                unknown = set(row["contexts"][method]["chunk_ids"]).difference(corpus)
                if unknown:
                    raise ValueError(f"{dataset}/{qid}: missing corpus chunk {sorted(unknown)[0]}")

    processor = AutoProcessor.from_pretrained(str(args.model))
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        str(args.model),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    input_device = next(model.parameters()).device
    run_config = {
        "model": str(args.model),
        "model_revision": args.model.name,
        "datasets": datasets,
        "methods": methods,
        "queries_per_dataset": args.limit or 100,
        "prompt": PROMPT + "{query}",
        "context_format": "text/table => Context: {content}; images => Qwen image block",
        "context_precedes_prompt": True,
        "decoding": "greedy",
        "max_new_tokens": 24,
        "min_pixels": 3136,
        "max_pixels": 200704,
        "latency_scope": "model.generate only, CUDA synchronized",
        "flops_proxy": "2 * parameter_count * generated_ids.shape[1]",
        "parameter_count": parameter_count,
        "gpu": torch.cuda.get_device_name(0),
    }
    config_path = args.output_dir / "RUN_CONFIG.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != run_config:
        raise ValueError(f"Existing config differs: {config_path}")
    if not config_path.exists():
        atomic_json(config_path, run_config)

    total = len(methods) * sum(len(prepared[dataset]["stored"]) for dataset in datasets)
    cursor = 0
    for method in methods:
        for dataset in datasets:
            bundle = prepared[dataset]
            for row in bundle["stored"]:
                cursor += 1
                qid = str(row["qid"])
                key = (method, dataset, qid)
                if key in completed:
                    print(f"[{cursor}/{total}] resume {method} {dataset} {qid}", flush=True)
                    continue
                content: list[dict[str, Any]] = []
                selected_ids = [str(value) for value in row["contexts"][method]["chunk_ids"]]
                for chunk_id in selected_ids:
                    chunk = bundle["corpus"][chunk_id]
                    modality = str(chunk.get("modality", "text"))
                    if modality in {"text", "table"}:
                        content.append(
                            {"type": "text", "text": f"Context: {chunk['content']}"}
                        )
                    elif modality == "image":
                        path = image_path(str(chunk["content"]), bundle["bundle_root"])
                        validate_image(path)
                        content.append(
                            {
                                "type": "image",
                                "image": f"file://{path}",
                                "min_pixels": 3136,
                                "max_pixels": 200704,
                            }
                        )
                    else:
                        raise ValueError(f"{dataset}/{qid}: unsupported modality {modality}")
                question = bundle["questions"][qid]
                content.append(
                    {"type": "text", "text": PROMPT + str(question["question"])}
                )
                messages = [{"role": "user", "content": content}]
                rendered = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[rendered],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(input_device)
                torch.cuda.synchronize()
                started = time.perf_counter()
                with torch.inference_mode():
                    generated = model.generate(
                        **inputs, max_new_tokens=24, do_sample=False
                    )
                torch.cuda.synchronize()
                latency = time.perf_counter() - started
                sequence_tokens = int(generated.shape[1])
                input_tokens = int(inputs.input_ids.shape[1])
                output_tokens = sequence_tokens - input_tokens
                approximate_flops = 2 * parameter_count * sequence_tokens
                trimmed = [
                    output[len(source) :]
                    for source, output in zip(inputs.input_ids, generated, strict=True)
                ]
                prediction = processor.batch_decode(
                    trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]
                gold = [str(answer) for answer in question["gold_answers"]]
                record = {
                    "method": method,
                    "dataset": dataset,
                    "qid": qid,
                    "selected_chunks": len(selected_ids),
                    "prediction": prediction,
                    "em": exact_match(prediction, gold),
                    "f1": token_f1(prediction, gold),
                    "latency_seconds": latency,
                    "approx_flops": approximate_flops,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                }
                append_jsonl(progress_path, record)
                completed[key] = record
                print(
                    f"[{cursor}/{total}] {method} {dataset} {qid} "
                    f"lat={latency:.3f}s approx={approximate_flops / 1e12:.3f} TFLOPs",
                    flush=True,
                )
                del inputs, generated, trimmed
                torch.cuda.empty_cache()

            records = list(completed.values())
            rows = summary_rows(records, methods, datasets)
            atomic_json(args.output_dir / "summary.json", {"status": "running", "rows": rows})
            write_csv(args.output_dir / "summary.csv", rows)

    rows = summary_rows(list(completed.values()), methods, datasets)
    atomic_json(args.output_dir / "summary.json", {"status": "complete", "rows": rows})
    write_csv(args.output_dir / "summary.csv", rows)


if __name__ == "__main__":
    main()
