#!/usr/bin/env python3
"""Run the three frozen Khanh-27-09 baseline contexts on Qwen2-VL-7B.

This file is intentionally standalone: it imports no code from the parent
repository.  It measures generation-only latency with CUDA synchronization and
preserves the requested FLOPs proxy:

    2 * model_parameter_count * total_returned_sequence_tokens

That quantity is an approximation for reporting consistency, not profiler
measured hardware FLOPs.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import string
import time
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

HERE = Path(__file__).resolve().parent
DATASETS = ("hotpotqa", "mmqa", "tatqa", "webqa")
METHODS = (
    "cce_conformal_embedding_jina_alpha_0.10",
    "conflare_source_question_jina_alpha_0.10",
    "traq_retrieval_bonferroni_jina_alpha_0.10",
)
METHOD_LABELS = {
    METHODS[0]: "CCE Conformal-Embedding, Jina adaptation (alpha=.10)",
    METHODS[1]: "CONFLARE source-question, Jina adaptation (alpha=.10)",
    METHODS[2]: "TRAQ retrieval Bonferroni, Jina adaptation (alpha=.10; alpha_R=.05)",
}
MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
MODEL_REVISION = "eed13092ef92e448dd6875b2a00151bd3f7db0ac"
PROMPT = "Answer using only the supplied context. Return only the short answer.\nQuestion: "


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=HERE / "data_bundle")
    parser.add_argument("--output-dir", type=Path, default=HERE / "colab_results")
    parser.add_argument("--selection-root", type=Path, default=HERE / "selections")
    parser.add_argument("--split-root", type=Path, default=HERE / "splits")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--limit", type=int, default=0, help="Smoke-test prefix; 0 means all 100.")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield value


def keyed(path: Path, key: str) -> dict[str, dict[str, Any]]:
    return {str(row[key]): row for row in iter_jsonl(path)}


def read_plan(path: Path) -> list[str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value.get("plan"), list):
        raise ValueError(f"{path}: expected list under plan")
    return [str(item) for item in value["plan"]]


def normalize_answer(value: str) -> str:
    value = value.lower()
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    value = value.translate(str.maketrans("", "", string.punctuation))
    return " ".join(value.split())


def exact_match(prediction: str, gold: list[str]) -> float:
    normalized = normalize_answer(prediction)
    return float(any(normalize_answer(answer) == normalized for answer in gold))


def token_f1(prediction: str, gold: list[str]) -> float:
    predicted = normalize_answer(prediction).split()
    if not predicted:
        return 0.0
    best = 0.0
    for answer in gold:
        expected = normalize_answer(answer).split()
        common = sum((Counter(predicted) & Counter(expected)).values())
        if common:
            precision, recall = common / len(predicted), common / len(expected)
            best = max(best, 2 * precision * recall / (precision + recall))
    return best


def completed_records(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    records: dict[tuple[str, str, str], dict[str, Any]] = {}
    if not path.exists():
        return records
    for row in iter_jsonl(path):
        key = (str(row["method"]), str(row["dataset"]), str(row["qid"]))
        if key in records:
            raise ValueError(f"Duplicate progress record: {key}")
        records[key] = row
    return records


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def summarize(
    records: Iterable[dict[str, Any]], methods: list[str], datasets: list[str]
) -> list[dict[str, Any]]:
    values = list(records)
    output: list[dict[str, Any]] = []
    for method in methods:
        method_rows = [row for row in values if row["method"] == method]
        for dataset in datasets + ["OVERALL"]:
            rows = method_rows if dataset == "OVERALL" else [
                row for row in method_rows if row["dataset"] == dataset
            ]
            if not rows:
                continue
            output.append(
                {
                    "Method": method,
                    "Method label": METHOD_LABELS[method],
                    "Dataset": dataset,
                    "Queries": len(rows),
                    "EM (%)": 100 * fmean(float(row["em"]) for row in rows),
                    "F1 (%)": 100 * fmean(float(row["f1"]) for row in rows),
                    "Latency (s)": fmean(float(row["latency_seconds"]) for row in rows),
                    "Approx. TFLOPs": fmean(float(row["approx_flops"]) for row in rows) / 1e12,
                    "Input tokens": fmean(float(row["input_tokens"]) for row in rows),
                    "Output tokens": fmean(float(row["output_tokens"]) for row in rows),
                }
            )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Qwen2-VL-7B resource metrics: Khanh 27/09 split",
        "",
        "Latency is CUDA-synchronized `model.generate` wall time per query. Approx. TFLOPs "
        "is `2 * parameter_count * returned_sequence_length`; it is not profiler FLOPs.",
        "",
        "| Method | Dataset | N | EM | F1 | Latency (s) | Approx. TFLOPs |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['Method label']} | {row['Dataset']} | {row['Queries']} | "
            f"{row['EM (%)']:.2f}% | {row['F1 (%)']:.2f}% | "
            f"{row['Latency (s)']:.3f} | {row['Approx. TFLOPs']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def resolve_image(data_root: Path, dataset: str, content: str) -> Path:
    raw = Path(content)
    candidates = (raw, data_root / raw, data_root / dataset / raw)
    for candidate in candidates:
        if candidate.is_file():
            with Image.open(candidate) as image:
                image.verify()
            return candidate.resolve()
    raise FileNotFoundError(f"Missing image for {dataset}: {content}")


def main() -> None:
    args = parse_args()
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    if not datasets or any(item not in DATASETS for item in datasets):
        raise ValueError(f"Unsupported datasets: {datasets}")
    if not methods or any(item not in METHODS for item in methods):
        raise ValueError(f"Unsupported methods: {methods}")
    if args.limit < 0 or args.limit > 100:
        raise ValueError("--limit must be from 0 through 100")
    prepared: dict[str, dict[str, Any]] = {}
    validation: dict[str, Any] = {"datasets": {}, "methods": methods}
    for dataset in datasets:
        calibration = read_plan(args.split_root / dataset / "calibration_manifest.json")
        test = read_plan(args.split_root / dataset / "test_manifest.json")
        if len(calibration) != 1000 or len(test) != 100 or set(calibration) & set(test):
            raise ValueError(f"{dataset}: invalid 1000-calibration / 100-test split")
        if args.limit:
            test = test[: args.limit]
        selections = {
            str(row["qid"]): row
            for row in iter_jsonl(args.selection_root / f"{dataset}_downstream_predictions.jsonl")
        }
        questions = keyed(args.data_root / dataset / "questions.jsonl", "qid")
        corpus = keyed(args.data_root / dataset / "corpus.jsonl", "id")
        for qid in test:
            if qid not in selections or qid not in questions:
                raise ValueError(f"{dataset}: missing frozen test qid {qid}")
            if str(questions[qid]["qid"]) != qid:
                raise ValueError(f"{dataset}: question/qid mismatch for {qid}")
            for method in methods:
                if method not in selections[qid]["contexts"]:
                    raise ValueError(f"{dataset}/{qid}: missing method {method}")
                ids = [str(value) for value in selections[qid]["contexts"][method]["chunk_ids"]]
                if len(ids) != len(set(ids)) or any(chunk_id not in corpus for chunk_id in ids):
                    raise ValueError(f"{dataset}/{qid}/{method}: invalid frozen chunk IDs")
        selected_ids = {
            str(chunk_id)
            for qid in test
            for method in methods
            for chunk_id in selections[qid]["contexts"][method]["chunk_ids"]
        }
        images = 0
        for chunk_id in selected_ids:
            chunk = corpus[chunk_id]
            if str(chunk.get("modality", "text")) == "image":
                resolve_image(args.data_root, dataset, str(chunk["content"]))
                images += 1
        prepared[dataset] = {
            "qids": test,
            "selections": selections,
            "questions": questions,
            "corpus": corpus,
        }
        validation["datasets"][dataset] = {
            "queries": len(test),
            "unique_selected_chunks": len(selected_ids),
            "unique_selected_images": images,
        }

    if args.validate_only:
        print(json.dumps({"status": "validated", **validation}, ensure_ascii=False, indent=2))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This fixed protocol requires a bfloat16-capable GPU (A100 recommended)")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_dir / "progress_log.jsonl"
    completed = completed_records(progress_path)

    processor = AutoProcessor.from_pretrained(args.model, revision=args.model_revision)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model,
        revision=args.model_revision,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    input_device = next(model.parameters()).device
    config = {
        "protocol": "khanh_27_09_colab_three_baselines_resource_v1",
        "model": args.model,
        "model_revision": args.model_revision,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "datasets": datasets,
        "methods": methods,
        "queries_per_dataset": args.limit or 100,
        "prompt": PROMPT + "{question}",
        "context_precedes_prompt": True,
        "decoding": "greedy",
        "max_new_tokens": 24,
        "min_pixels": 3136,
        "max_pixels": 200704,
        "latency_scope": "model.generate only, CUDA synchronized",
        "flops_proxy": "2 * parameter_count * generated_ids.shape[1]",
        "parameter_count": parameter_count,
    }
    config_path = args.output_dir / "RUN_CONFIG.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("Existing RUN_CONFIG.json differs; use a new output directory")
    atomic_json(config_path, config)

    total = len(methods) * sum(len(prepared[dataset]["qids"]) for dataset in datasets)
    cursor = 0
    for method in methods:
        for dataset in datasets:
            bundle = prepared[dataset]
            for qid in bundle["qids"]:
                cursor += 1
                key = (method, dataset, qid)
                if key in completed:
                    print(f"[{cursor}/{total}] resume {method} {dataset} {qid}", flush=True)
                    continue

                selected = bundle["selections"][qid]["contexts"][method]["chunk_ids"]
                content: list[dict[str, Any]] = []
                for chunk_id in selected:
                    chunk = bundle["corpus"][str(chunk_id)]
                    modality = str(chunk.get("modality", "text"))
                    if modality in {"text", "table"}:
                        content.append({"type": "text", "text": f"Context: {chunk['content']}"})
                    elif modality == "image":
                        image = resolve_image(args.data_root, dataset, str(chunk["content"]))
                        content.append(
                            {
                                "type": "image",
                                "image": f"file://{image}",
                                "min_pixels": 3136,
                                "max_pixels": 200704,
                            }
                        )
                    else:
                        raise ValueError(f"{dataset}/{qid}: unsupported modality {modality}")

                question = bundle["questions"][qid]
                content.append({"type": "text", "text": PROMPT + str(question["question"])})
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
                    generated = model.generate(**inputs, max_new_tokens=24, do_sample=False)
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
                gold = [str(value) for value in question["gold_answers"]]
                record = {
                    "method": method,
                    "dataset": dataset,
                    "qid": qid,
                    "selected_chunks": len(selected),
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

            rows = summarize(completed.values(), methods, datasets)
            atomic_json(args.output_dir / "summary.json", {"status": "running", "rows": rows})
            write_csv(args.output_dir / "final_downstream_results.csv", rows)
            write_markdown(args.output_dir / "RESOURCE_RESULTS.md", rows)

    rows = summarize(completed.values(), methods, datasets)
    atomic_json(args.output_dir / "summary.json", {"status": "complete", "rows": rows})
    write_csv(args.output_dir / "final_downstream_results.csv", rows)
    write_markdown(args.output_dir / "RESOURCE_RESULTS.md", rows)
    print(f"Complete: {len(completed)}/{total} rows; outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
