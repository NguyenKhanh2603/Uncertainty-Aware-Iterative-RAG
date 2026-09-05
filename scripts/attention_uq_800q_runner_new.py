"""Fast A100 800-question attention-pruning benchmark runner for Colab.

This file is copied into the Hugging Face data bundle by
prepare_attention_uq_800q_hf.py.  It expects PROJECT_DIR and BUNDLE_DIR to be
defined by the notebook bootstrap cell.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import random
import re
import string
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from IPython.display import display
from tqdm.auto import tqdm

from uncertainty_rag.modality.base import ContextChunk
from uncertainty_rag.models.llm_client import HuggingFaceLocalClient
from sentence_transformers import SentenceTransformer
from PIL import Image

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET_NAMES = ["mmqa", "webqa", "hotpotqa", "tatqa"]
N_PER_DATASET = int(os.environ.get("ATTN_UQ_N_PER_DATASET", "200"))
WINDOW_SIZE = int(os.environ.get("ATTN_UQ_WINDOW_SIZE", "10"))

SIM_THRESHOLD_TEXT = float(os.environ.get("ATTN_UQ_SIM_THRESHOLD_TEXT", "0.34"))
SIM_THRESHOLD_IMAGE = float(os.environ.get("ATTN_UQ_SIM_THRESHOLD_IMAGE", "0.23"))
SIM_THRESHOLD_TABLE = float(os.environ.get("ATTN_UQ_SIM_THRESHOLD_TABLE", "0.15"))

KEEP_THRESHOLD_TEXT = float(os.environ.get("ATTN_UQ_KEEP_THRESHOLD_TEXT", "0.12"))
KEEP_THRESHOLD_IMAGE = float(os.environ.get("ATTN_UQ_KEEP_THRESHOLD_IMAGE", "0.10"))
KEEP_THRESHOLD_TABLE = float(os.environ.get("ATTN_UQ_KEEP_THRESHOLD_TABLE", "0.10"))
ATTENTION_SCORE_MODE = "cascade_filtering"

MAX_NEW_TOKENS = int(os.environ.get("ATTN_UQ_MAX_NEW_TOKENS", "32"))
RANDOM_SEED = int(os.environ.get("ATTN_UQ_RANDOM_SEED", "42"))
CHECKPOINT_EVERY = int(os.environ.get("ATTN_UQ_CHECKPOINT_EVERY", "10"))
MODEL_NAME = os.environ.get("ATTN_UQ_MODEL", "Qwen/Qwen2-VL-7B-Instruct")

PROJECT_DIR = Path(globals().get("PROJECT_DIR", "/content/paper"))
BUNDLE_DIR = Path(globals().get("BUNDLE_DIR", "/content/attention_uq_800q"))
RESULT_DIR = Path(
    os.environ.get("ATTN_UQ_RESULT_DIR", "/content/attention_uq_800q_fast_a100_results_v5")
)
RESULT_DIR.mkdir(parents=True, exist_ok=True)
OOM_LOG = RESULT_DIR / "oom_log.jsonl"

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


@dataclass
class BenchmarkExample:
    dataset: str
    index: int
    qid: str
    question: str
    gold_answers: list[str]
    context_chunks: list[ContextChunk]
    support_ids: set[str]
    metadata: dict[str, Any] = field(default_factory=dict)


def load_examples(dataset: str) -> list[BenchmarkExample]:
    path = BUNDLE_DIR / dataset / "questions.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing bundled dataset: {path}")
    examples = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            chunks = []
            support_ids = set()
            for chunk in row.get("chunks", []):
                content = str(chunk.get("content", ""))
                if chunk.get("modality") == "image" and not content.startswith(
                    ("http://", "https://", "file://")
                ):
                    content = str(BUNDLE_DIR / content)
                chunk_id = str(chunk.get("id", f"{dataset}_{len(chunks)}"))
                if chunk.get("is_support"):
                    support_ids.add(chunk_id)
                chunks.append(
                    ContextChunk(
                        id=chunk_id,
                        content=content,
                        modality=str(chunk.get("modality", "text")),
                        metadata={"is_support": bool(chunk.get("is_support", False))},
                    )
                )
            examples.append(
                BenchmarkExample(
                    dataset=dataset,
                    index=int(row.get("index", len(examples))),
                    qid=str(row.get("qid", f"{dataset}_{len(examples)}")),
                    question=str(row.get("question", "")),
                    gold_answers=[str(x) for x in row.get("gold_answers", [])],
                    context_chunks=chunks,
                    support_ids=support_ids,
                    metadata=row.get("metadata", {}),
                )
            )
            if len(examples) >= N_PER_DATASET:
                break
    if len(examples) != N_PER_DATASET:
        raise RuntimeError(f"Expected {N_PER_DATASET} {dataset} questions, got {len(examples)}")
    return examples


all_examples = {name: load_examples(name) for name in DATASET_NAMES}
print("Loaded:", {name: len(rows) for name, rows in all_examples.items()})


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
llm = HuggingFaceLocalClient(model_name=MODEL_NAME, load_in_4bit=False)
device = llm.model.device
model_dtype = next(llm.model.parameters()).dtype
if model_dtype != torch.float16:
    raise RuntimeError(f"Expected unquantized FP16 model weights, got {model_dtype}")
device_map = getattr(llm.model, "hf_device_map", {})
offloaded = {name: target for name, target in device_map.items() if str(target) in {"cpu", "disk"}}
if offloaded:
    raise RuntimeError(f"Fast A100 runner refuses CPU/disk offload: {offloaded}")
print(f"Model={MODEL_NAME} dtype={model_dtype} quantized=False device={device}")
print(
    "Attention mode=single full eager forward | "
    f"score={ATTENTION_SCORE_MODE} | OOM policy=log and skip"
)



print("Loading Embedding Models for Similarity...")
text_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
clip_model = SentenceTransformer('clip-ViT-B-32', device='cuda')

def filter_chunks_by_similarity(question: str, chunks: list) -> list:
    if not chunks: return []
    q_emb_text = text_model.encode([question], normalize_embeddings=True)[0]
    q_emb_clip = clip_model.encode([question], normalize_embeddings=True)[0]
    filtered = []
    for chunk in chunks:
        mod = getattr(chunk, 'modality', chunk.metadata.get('modality', 'text')) if chunk else 'text'
        content = chunk.content
        sim = 0.0
        if mod == "image":
            img_path = content
            img = Image.open(img_path).convert("RGB")
            c_emb = clip_model.encode([img], normalize_embeddings=True)[0]
            sim = float(np.dot(q_emb_clip, c_emb))
            if sim >= SIM_THRESHOLD_IMAGE:
                filtered.append(chunk)
        elif mod == "table":
            c_emb = text_model.encode([str(content)], normalize_embeddings=True)[0]
            sim = float(np.dot(q_emb_text, c_emb))
            if sim >= SIM_THRESHOLD_TABLE:
                filtered.append(chunk)
        else:
            c_emb = text_model.encode([str(content)], normalize_embeddings=True)[0]
            sim = float(np.dot(q_emb_text, c_emb))
            if sim >= SIM_THRESHOLD_TEXT:
                filtered.append(chunk)
    return filtered

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def normalize_answer(value: Any) -> str:
    text = str(value).lower()
    text = "".join(char for char in text if char not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def token_f1(prediction: str, gold: str) -> float:
    predicted = normalize_answer(prediction).split()
    expected = normalize_answer(gold).split()
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def score_prediction(prediction: str, answers: list[str]) -> tuple[float, float]:
    if not answers:
        return np.nan, np.nan
    return (
        float(max(exact_match(prediction, answer) for answer in answers)),
        float(max(token_f1(prediction, answer) for answer in answers)),
    )


def support_metrics(chunks: list[ContextChunk], support_ids: set[str]) -> tuple[float, float]:
    if not support_ids:
        return np.nan, np.nan
    selected = {str(chunk.id) for chunk in chunks}
    overlap = len(selected & support_ids)
    return overlap / len(support_ids), overlap / len(selected) if selected else 0.0


def make_query(question: str) -> str:
    return str(question).strip() + "\nAnswer with only the final answer, as a concise phrase."


def cuda_cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def is_cuda_oom(exc: BaseException) -> bool:
    message = str(exc).lower()
    return isinstance(exc, torch.cuda.OutOfMemoryError) or any(
        marker in message
        for marker in (
            "cuda out of memory",
            "cuda error: out of memory",
            "cublas_status_alloc_failed",
            "outofmemoryerror",
        )
    )


def deterministic_rng(qid: str) -> random.Random:
    digest = hashlib.sha256(f"{RANDOM_SEED}:{qid}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


# ---------------------------------------------------------------------------
# Generation and memory-efficient last-token attention
# ---------------------------------------------------------------------------
@torch.inference_mode()
def generate_answer(question: str, chunks: list[ContextChunk]) -> tuple[str, float, float]:
    if not chunks:
        return "", np.nan, np.nan
    alignment = llm.align_and_prepare_inputs(make_query(question), chunks)
    inputs = alignment.inputs
    prompt_length = inputs["input_ids"].shape[1]
    outputs = llm.model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True,
        pad_token_id=llm.tokenizer.eos_token_id,
    )
    generated_ids = outputs.sequences[0, prompt_length:]
    answer = llm.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    nlls = []
    for step in range(min(len(outputs.scores), len(generated_ids))):
        logits = outputs.scores[step][0].float()
        token_id = int(generated_ids[step].item())
        nlls.append(-torch.log_softmax(logits, dim=-1)[token_id].item())
    mean_nll = float(np.mean(nlls)) if nlls else np.nan
    ppl = float(math.exp(min(mean_nll, 20.0))) if np.isfinite(mean_nll) else np.nan
    del outputs, alignment, inputs
    return answer, mean_nll, ppl


@torch.inference_mode()
def chunk_attention_scores(
    question: str,
    chunks: list[ContextChunk],
) -> tuple[list[float], float, list[dict[str, Any]]]:
    """Score chunks by Modality-Aware Normalization (Bản 5).
    """
    if not chunks:
        return [], np.nan, []
    alignment = llm.align_and_prepare_inputs(make_query(question), chunks)
    inputs = alignment.inputs
    spans = alignment.batch_chunk_spans[0]
    outputs = llm.model(
        **inputs,
        output_attentions=True,
        use_cache=False,
        return_dict=True,
    )
    last_layer = outputs.attentions[-1]  # [B, H, Q, K]
    token_saliency = last_layer[0].mean(dim=0)[-1]  # final prompt token -> keys
    prompt_length = inputs["input_ids"].shape[1]

    raw_masses = []
    raw_densities = []
    for span in spans:
        if span.start is None or span.end is None:
            raw_masses.append(0.0)
            raw_densities.append(0.0)
            continue
        start = max(0, int(span.start))
        end = min(prompt_length, int(span.end), len(token_saliency))
        if end > start:
            mass = token_saliency[start:end].float().sum().item()
            density = mass / (end - start)
            raw_masses.append(mass)
            raw_densities.append(density)
        else:
            raw_masses.append(0.0)
            raw_densities.append(0.0)


    # BẢN 5: CHIA ROOM + CHUẨN HÓA NỘI BỘ (Modality-Aware Normalization)
    scores = [0.0] * len(chunks)
    modality_indices = {"text": [], "image": [], "table": []}
    for i, chunk in enumerate(chunks):
        mod = getattr(chunk, 'modality', chunk.metadata.get('modality', 'text')) if chunk else 'text'
        if mod not in modality_indices: modality_indices[mod] = []
        modality_indices[mod].append(i)

    for mod, indices in modality_indices.items():
        if not indices: continue
        if mod == "text":
            room_sum = sum(raw_densities[i] for i in indices)
            for i in indices: scores[i] = raw_densities[i] / room_sum if room_sum > 0 else (1.0 / len(indices))
        else:
            room_sum = sum(raw_masses[i] for i in indices)
            for i in indices: scores[i] = raw_masses[i] / room_sum if room_sum > 0 else (1.0 / len(indices))


    # Tính Entropy tổng hợp dựa trên Mass để đo độ bối rối chung của mô hình
    global_raw = np.asarray(raw_masses, dtype=np.float64)
    global_prob = global_raw / global_raw.sum() if global_raw.sum() > 0 else np.ones(len(global_raw)) / max(len(global_raw), 1)
    if len(global_prob) <= 1:
        entropy = 0.0
    else:
        entropy = float(-np.sum(global_prob * np.log(global_prob + 1e-12)) / np.log(len(global_prob)))

    details = []
    for i, (chunk, span, mass) in enumerate(zip(chunks, spans, raw_masses)):
        start = int(span.start) if span.start is not None else None
        end = int(span.end) if span.end is not None else None
        details.append(
            {
                "chunk_id": chunk.id,
                "modality": chunk.modality,
                "is_support": bool(chunk.metadata.get("is_support", False)),
                "span_status": getattr(span.status, "value", str(span.status)),
                "span_start": start,
                "span_end": end,
                "token_count": max(0, end - start) if start is not None and end is not None else 0,
                "attention_mass": float(mass),
                "attention_score": float(scores[i]), # Điểm Modality-Aware
            }
        )
    del outputs, alignment, inputs, last_layer, token_saliency
    return scores, entropy, details


def attention_prune(
    question: str, all_chunks: list[ContextChunk]
) -> tuple[list[ContextChunk], dict[str, Any]]:
    queue = list(all_chunks)
    kept: list[ContextChunk] = []
    iterations = []
    initial_entropy = np.nan
    while queue:
        needed = WINDOW_SIZE - len(kept)
        if needed <= 0:
            scores, _, _ = chunk_attention_scores(question, kept)
            kept = [chunk for index, chunk in enumerate(kept) if index != int(np.argmin(scores))]
            needed = 1
        current = kept + queue[:needed]
        queue = queue[needed:]
        scores, entropy, details = chunk_attention_scores(question, current)
        if not iterations:
            initial_entropy = entropy
        # BẢN 5: CASCADE FILTERING - Áp dụng Threshold cho Attention (Lọc Động)
        selected = []
        for chunk, score in zip(current, scores):
            mod = getattr(chunk, 'modality', chunk.metadata.get('modality', 'text')) if chunk else 'text'
            if mod == "text" and score >= KEEP_THRESHOLD_TEXT:
                selected.append(chunk)
            elif mod == "image" and score >= KEEP_THRESHOLD_IMAGE:
                selected.append(chunk)
            elif mod == "table" and score >= KEEP_THRESHOLD_TABLE:
                selected.append(chunk)
            elif mod not in ("text", "image", "table") and score >= KEEP_THRESHOLD_TEXT:
                selected.append(chunk)

        if not selected and current:
            selected = [current[int(np.argmax(scores))]]
        iterations.append(
            {
                "n_context": len(current),
                "n_kept": len(selected),
                "entropy": entropy,
                "chunks": details,
            }
        )
        kept = selected
    final_scores, final_entropy, final_details = chunk_attention_scores(question, kept)
    return kept, {
        "initial_entropy": float(initial_entropy),
        "final_entropy": float(final_entropy),
        "final_scores": final_scores,
        "final_chunks": final_details,
        "iterations": iterations,
    }


# ---------------------------------------------------------------------------
# Per-question evaluation
# ---------------------------------------------------------------------------
METHODS = ["all", "top10", "attention", "prefix_k", "random_k"]



def blank_method(row: dict[str, Any], method: str) -> None:
    row.update(
        {
            f"{method}_status": "not_run",
            f"{method}_answer": "",
            f"{method}_em": np.nan,
            f"{method}_f1": np.nan,
            f"{method}_ppl": np.nan,
            f"{method}_support_recall": np.nan,
            f"{method}_support_precision": np.nan,
        }
    )


def fill_generation(
    row: dict[str, Any],
    method: str,
    answer: str,
    ppl: float,
    chunks: list[ContextChunk],
    example: BenchmarkExample,
) -> None:
    em, f1 = score_prediction(answer, example.gold_answers)
    recall, precision = support_metrics(chunks, example.support_ids)
    row.update(
        {
            f"{method}_status": "ok",
            f"{method}_answer": answer,
            f"{method}_em": em,
            f"{method}_f1": f1,
            f"{method}_ppl": ppl,
            f"{method}_support_recall": recall,
            f"{method}_support_precision": precision,
        }
    )


def append_oom(dataset: str, example: BenchmarkExample, stage: str, exc: BaseException) -> None:
    record = {
        "dataset": dataset,
        "index": example.index,
        "qid": example.qid,
        "stage": stage,
        "error": str(exc)[:2000],
        "cuda_allocated_gib": torch.cuda.memory_allocated() / 2**30
        if torch.cuda.is_available()
        else 0,
        "cuda_reserved_gib": torch.cuda.memory_reserved() / 2**30
        if torch.cuda.is_available()
        else 0,
    }
    with OOM_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def evaluate_question(example: BenchmarkExample) -> dict[str, Any]:
    chunks = list(example.context_chunks)
    row = {
        "dataset": example.dataset,
        "index": example.index,
        "qid": example.qid,
        "question": example.question,
        "gold_answers": json.dumps(example.gold_answers, ensure_ascii=False),
        "n_retrieved": len(chunks),
        "n_support": len(example.support_ids),
        "n_attention_kept": np.nan,
        "attention_keep_ratio": np.nan,
        "top10_attention_entropy": np.nan,
        "attention_initial_entropy": np.nan,
        "attention_final_entropy": np.nan,
        "attention_delta_entropy": np.nan,
        "attention_entropy_trend": "",
        "attention_iterations": np.nan,
        "attention_window_size": WINDOW_SIZE,
        "attention_score_mode": ATTENTION_SCORE_MODE,
        "top10_attention_trace": "",
        "attention_trace": "",
        "oom_stages": "",
    }
    for method in METHODS:
        blank_method(row, method)
    oom_stages = []

    try:
        answer, _, ppl = generate_answer(example.question, chunks)
        fill_generation(row, "all", answer, ppl, chunks, example)
    except RuntimeError as exc:
        if not is_cuda_oom(exc):
            raise
        row["all_status"] = "oom"
        oom_stages.append("all")
        append_oom(example.dataset, example, "all", exc)
        cuda_cleanup()

    top10_chunks = chunks[:WINDOW_SIZE]
    try:
        answer, _, ppl = generate_answer(example.question, top10_chunks)
        fill_generation(row, "top10", answer, ppl, top10_chunks, example)
        _, row["top10_attention_entropy"], top10_details = chunk_attention_scores(
            example.question, top10_chunks
        )
        row["top10_attention_trace"] = json.dumps(top10_details, ensure_ascii=False)
    except RuntimeError as exc:
        if not is_cuda_oom(exc):
            raise
        row["top10_status"] = "oom"
        oom_stages.append("top10")
        append_oom(example.dataset, example, "top10", exc)
        cuda_cleanup()

    attention_chunks = None
    try:
        sim_filtered_chunks = filter_chunks_by_similarity(example.question, chunks)
        attention_chunks, diagnostics = attention_prune(example.question, sim_filtered_chunks)
        answer, _, ppl = generate_answer(example.question, attention_chunks)
        fill_generation(row, "attention", answer, ppl, attention_chunks, example)

        delta = diagnostics["final_entropy"] - diagnostics["initial_entropy"]
        trend = "decreased" if delta < -1e-5 else ("increased" if delta > 1e-5 else "unchanged")

        row.update(
            {
                "n_attention_kept": len(attention_chunks),
                "attention_keep_ratio": len(attention_chunks) / len(chunks),
                "attention_initial_entropy": diagnostics["initial_entropy"],
                "attention_final_entropy": diagnostics["final_entropy"],
                "attention_delta_entropy": delta,
                "attention_entropy_trend": trend,
                "attention_iterations": len(diagnostics["iterations"]),
                "attention_trace": json.dumps(diagnostics, ensure_ascii=False),
            }
        )
    except RuntimeError as exc:
        if not is_cuda_oom(exc):
            raise
        row["attention_status"] = "oom"
        oom_stages.append("attention")
        append_oom(example.dataset, example, "attention", exc)
        cuda_cleanup()

    if attention_chunks:
        k = len(attention_chunks)
        matched = {
            "prefix_k": chunks[:k],
            "random_k": deterministic_rng(example.qid).sample(chunks, k=min(k, len(chunks))),
        }
        for method, selected in matched.items():
            try:
                answer, _, ppl = generate_answer(example.question, selected)
                fill_generation(row, method, answer, ppl, selected, example)
            except RuntimeError as exc:
                if not is_cuda_oom(exc):
                    raise
                row[f"{method}_status"] = "oom"
                oom_stages.append(method)
                append_oom(example.dataset, example, method, exc)
                cuda_cleanup()

    row["oom_stages"] = ",".join(oom_stages)
    row["status"] = "ok" if row["attention_status"] == "ok" else "partial"
    if np.isfinite(row["attention_em"]) and np.isfinite(row["top10_em"]):
        row["delta_em_attention_vs_top10"] = row["attention_em"] - row["top10_em"]
        row["delta_f1_attention_vs_top10"] = row["attention_f1"] - row["top10_f1"]
    else:
        row["delta_em_attention_vs_top10"] = np.nan
        row["delta_f1_attention_vs_top10"] = np.nan
    # One cleanup per successful question; OOM handlers clean immediately.
    cuda_cleanup()
    return row


# ---------------------------------------------------------------------------
# Run with resumable per-dataset CSV checkpoints
# ---------------------------------------------------------------------------
if not any(RESULT_DIR.glob("*_attention_uq_200q.csv")):
    OOM_LOG.write_text("", encoding="utf-8")
elif not OOM_LOG.exists():
    OOM_LOG.touch()
all_frames = []
for dataset in DATASET_NAMES:
    output_csv = RESULT_DIR / f"{dataset}_attention_uq_200q.csv"
    existing = pd.read_csv(output_csv) if output_csv.is_file() else pd.DataFrame()
    completed_qids = set(existing["qid"].astype(str)) if len(existing) else set()
    rows = existing.to_dict("records") if len(existing) else []
    pending = [example for example in all_examples[dataset] if example.qid not in completed_qids]
    progress = tqdm(pending, desc=f"{dataset} 200Q")
    for count, example in enumerate(progress, start=1):
        row = evaluate_question(example)
        rows.append(row)
        pd.DataFrame(rows).to_csv(output_csv, index=False)
        if count % CHECKPOINT_EVERY == 0:
            ok = pd.DataFrame(rows)
            progress.write(
                f"checkpoint {dataset} {len(rows)}/{N_PER_DATASET}: "
                f"attention EM={pd.to_numeric(ok['attention_em'], errors='coerce').mean():.3f}"
            )
    frame = pd.DataFrame(rows).sort_values("index").reset_index(drop=True)
    frame.to_csv(output_csv, index=False)
    all_frames.append(frame)

combined = pd.concat(all_frames, ignore_index=True)
combined_path = RESULT_DIR / "attention_uq_800q_results.csv"
combined.to_csv(combined_path, index=False)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def safe_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if len(values) else np.nan


def safe_auroc(y_incorrect: pd.Series, uncertainty: pd.Series) -> float:
    try:
        from sklearn.metrics import roc_auc_score

        y = pd.to_numeric(y_incorrect, errors="coerce").to_numpy(float)
        u = pd.to_numeric(uncertainty, errors="coerce").to_numpy(float)
        mask = np.isfinite(y) & np.isfinite(u)
        return float(roc_auc_score(y[mask], u[mask])) if len(np.unique(y[mask])) > 1 else np.nan
    except Exception:
        return np.nan


summary = {
    "configuration": {
        "datasets": DATASET_NAMES,
        "questions_per_dataset": N_PER_DATASET,
        "total_attempted": int(len(combined)),
        "window_size": WINDOW_SIZE,
        "model": MODEL_NAME,
        "dtype": str(model_dtype),
        "quantized": False,
        "attention_extraction": "single full eager forward pass (A100 fast path)",
    },
    "datasets": {},
}
for dataset, frame in combined.groupby("dataset", sort=False):
    dataset_summary = {
        "n_attempted": int(len(frame)),
        "n_attention_completed": int(frame["attention_em"].notna().sum()),
        "n_attention_oom": int(frame["attention_status"].eq("oom").sum()),
        "avg_retrieved": safe_mean(frame["n_retrieved"]),
        "avg_attention_kept": safe_mean(frame["n_attention_kept"]),
        "methods": {},
        "uq": {},
    }
    for method in METHODS:
        dataset_summary["methods"][method] = {
            metric: safe_mean(frame[f"{method}_{metric}"])
            for metric in ("em", "f1", "support_recall", "support_precision")
        }
    dataset_summary["uq"] = {
        "top10_attention_entropy_auroc": safe_auroc(
            1 - frame["top10_em"], frame["top10_attention_entropy"]
        ),
        "top10_ppl_auroc": safe_auroc(1 - frame["top10_em"], frame["top10_ppl"]),
        "attention_entropy_auroc": safe_auroc(
            1 - frame["attention_em"], frame["attention_final_entropy"]
        ),
        "attention_ppl_auroc": safe_auroc(1 - frame["attention_em"], frame["attention_ppl"]),
    }
    summary["datasets"][dataset] = dataset_summary

summary_path = RESULT_DIR / "attention_uq_800q_summary.json"
summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

print("\n800Q summary")
report = []
for dataset, values in summary["datasets"].items():
    for method, metrics in values["methods"].items():
        report.append({"dataset": dataset, "method": method, **metrics})
display(pd.DataFrame(report).round(4))
print(
    "Attention completion:",
    {
        name: f"{values['n_attention_completed']}/{values['n_attempted']}"
        for name, values in summary["datasets"].items()
    },
)

zip_path = Path("/content/attention_uq_800q_fast_a100_results.zip")
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in RESULT_DIR.glob("*"):
        if path.is_file():
            archive.write(path, arcname=path.name)
print(f"Saved downloadable archive: {zip_path}")
try:
    from google.colab import files

    files.download(str(zip_path))
except Exception:
    pass
