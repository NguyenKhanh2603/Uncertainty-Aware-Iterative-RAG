"""Run a paired TATQA smoke test for generator-internal evidence signals.

The script uses calibration questions only.  It selects examples whose labelled
support occurs below top-K in the existing frozen BGE top-30 retrieval log, then
compares top-K distractors against the same context with gold support appended.
No corpus retrieval is performed.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from scipy.stats import spearmanr

from eval.metrics import exact_match, numerical_accuracy, token_f1
from research.internal_state_rag import QwenInternalStateExtractor
from uncertainty_rag.modality.base import ContextChunk
from uncertainty_rag.models.llm_client import (
    AlignmentResult,
    ChunkAlignmentError,
    ChunkSpan,
    ChunkStatus,
    HuggingFaceLocalClient,
)


class ResearchTextClient(HuggingFaceLocalClient):
    """Text-only chunk alignment without modifying the production client."""

    def align_and_prepare_inputs(
        self, query: str, chunks: list[ContextChunk]
    ) -> AlignmentResult:
        if self.is_qwen_vl:
            return super().align_and_prepare_inputs(query, chunks)

        content = query + "\n\nContext:\n"
        marker_rows = []
        for index, chunk in enumerate(chunks):
            start_marker = chr(0xE000 + index * 2)
            end_marker = chr(0xE000 + index * 2 + 1)
            if ord(end_marker) > 0xF8FF:
                raise ChunkAlignmentError("Ran out of private-use alignment markers.")
            content += f"{start_marker}{chunk.content}{end_marker}\n"
            marker_rows.append((index, chunk, start_marker, end_marker))

        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        marker_positions = []
        for index, chunk, start_marker, end_marker in marker_rows:
            start = rendered.find(start_marker)
            end = rendered.find(end_marker)
            if start < 0 or end < 0:
                raise ChunkAlignmentError(f"Markers missing for chunk {chunk.id}.")
            marker_positions.extend(
                [(start, "start", index, chunk), (end, "end", index, chunk)]
            )
        marker_positions.sort(key=lambda item: item[0])

        clean_prompt = ""
        cursor = 0
        character_spans: dict[int, dict[str, int]] = {}
        for position, kind, index, _ in marker_positions:
            clean_prompt += rendered[cursor:position]
            character_spans.setdefault(index, {})[kind] = len(clean_prompt)
            cursor = position + 1
        clean_prompt += rendered[cursor:]

        inputs = self.tokenizer(
            [clean_prompt],
            return_tensors="pt",
            return_offsets_mapping=True,
        ).to(self.model.device)
        offsets = inputs["offset_mapping"][0].tolist()
        spans = []
        for index, chunk, _, _ in marker_rows:
            character_start = character_spans[index]["start"]
            character_end = character_spans[index]["end"]
            token_indices = [
                token_index
                for token_index, (token_start, token_end) in enumerate(offsets)
                if token_start < character_end
                and token_end > character_start
                and token_start != token_end
            ]
            if token_indices:
                start, end = token_indices[0], token_indices[-1] + 1
                status = ChunkStatus.FULL
            else:
                start, end, status = None, None, ChunkStatus.FULLY_TRUNCATED
            spans.append(
                ChunkSpan(index, chunk.id, chunk.modality, start, end, status)
            )
        del inputs["offset_mapping"]
        return AlignmentResult(
            inputs=inputs,
            batch_chunk_spans=[spans],
            prompt_strings=[clean_prompt],
        )


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix == ".gz":
        handle = gzip.open(path, "rt", encoding="utf-8")
    else:
        handle = path.open("rt", encoding="utf-8")
    with handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_rows(path: Path, key: str) -> dict[str, dict[str, Any]]:
    return {str(row[key]): row for row in iter_jsonl(path)}


def load_retrieval(path: Path) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in iter_jsonl(path):
        if row.get("split_role") == "calibration":
            grouped[str(row["qid"])].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: (int(row["rank"]), str(row["chunk_id"])))
    return grouped


def select_examples(
    questions: dict[str, dict[str, Any]],
    retrieval: dict[str, list[dict[str, Any]]],
    *,
    top_k: int,
    limit: int,
) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    selected = []
    for qid, rows in retrieval.items():
        question = questions.get(qid)
        if question is None or len(question.get("gold_answers", [])) != 1:
            continue
        support_ranks = [int(row["rank"]) for row in rows if row["support_label"] == "support"]
        if not support_ranks or min(support_ranks) <= top_k:
            continue
        selected.append((question, rows))
        if len(selected) >= limit:
            break
    if len(selected) < limit:
        raise ValueError(f"Only found {len(selected)} eligible examples; requested {limit}.")
    return selected


def to_chunk(row: dict[str, Any], corpus: dict[str, dict[str, Any]]) -> ContextChunk:
    source = corpus[str(row["chunk_id"])]
    return ContextChunk(
        id=str(source["id"]),
        content=source["content"],
        modality=str(source["modality"]),
        metadata={"rank": int(row["rank"]), "is_support": row["support_label"] == "support"},
    )


@torch.inference_mode()
def generate_answer(
    client: HuggingFaceLocalClient,
    question: str,
    chunks: list[ContextChunk],
    *,
    max_new_tokens: int,
) -> str:
    alignment = client.align_and_prepare_inputs(question, chunks)
    inputs = alignment.inputs
    output = client.model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        use_cache=True,
    )
    generated = output[0, inputs["input_ids"].shape[1] :]
    return client.tokenizer.decode(generated, skip_special_tokens=True).strip()


def support_attention_statistics(trace: Any, support_ids: set[str]) -> tuple[float, float]:
    support_indices = [
        index for index, chunk_id in enumerate(trace.chunk_ids) if chunk_id in support_ids
    ]
    if not support_indices:
        return float("nan"), float("nan")
    total = trace.attention_mass.sum(axis=-1)
    support = trace.attention_mass[:, :, support_indices].sum(axis=-1)
    fractions = np.divide(support, total, out=np.zeros_like(support), where=total > 0)
    return float(fractions.mean()), float(fractions.max())


def safe_spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) < 3 or np.std(left) == 0 or np.std(right) == 0:
        return None
    value = float(spearmanr(left, right).statistic)
    return value if np.isfinite(value) else None


def parse_args() -> argparse.Namespace:
    root = Path("data/conformal_global_run")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--max-answer-tokens", type=int, default=24)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--layers", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n < 1 or args.top_k < 1:
        raise ValueError("n and top-k must be positive.")

    corpus = load_rows(args.corpus, "id")
    questions = load_rows(args.questions, "qid")
    retrieval = load_retrieval(args.retrieval)
    examples = select_examples(questions, retrieval, top_k=args.top_k, limit=args.n)

    client = ResearchTextClient(args.model, device="cuda", load_in_4bit=False)
    layers = [int(value) for value in args.layers.split(",") if value.strip()] or None
    extractor = QwenInternalStateExtractor(client, layers=layers)
    results = []

    for index, (question_row, retrieval_rows) in enumerate(examples, start=1):
        top_rows = retrieval_rows[: args.top_k]
        support_rows = [row for row in retrieval_rows if row["support_label"] == "support"]
        missing_chunks = [to_chunk(row, corpus) for row in top_rows]
        oracle_rows = top_rows + [row for row in support_rows if row not in top_rows]
        oracle_chunks = [to_chunk(row, corpus) for row in oracle_rows]
        support_ids = {str(row["chunk_id"]) for row in support_rows}

        prompt_question = (
            "Answer using only the supplied context. Return only the short answer.\n"
            f"Question: {question_row['question']}"
        )
        gold_answers = [str(value) for value in question_row["gold_answers"]]
        gold = gold_answers[0]

        missing_prediction = generate_answer(
            client,
            prompt_question,
            missing_chunks,
            max_new_tokens=args.max_new_tokens,
        )
        oracle_prediction = generate_answer(
            client,
            prompt_question,
            oracle_chunks,
            max_new_tokens=args.max_new_tokens,
        )
        missing_trace = extractor.extract(
            prompt_question,
            missing_chunks,
            gold,
            max_answer_tokens=args.max_answer_tokens,
        )
        oracle_trace = extractor.extract(
            prompt_question,
            oracle_chunks,
            gold,
            max_answer_tokens=args.max_answer_tokens,
        )

        missing_f1 = token_f1(missing_prediction, gold_answers)
        oracle_f1 = token_f1(oracle_prediction, gold_answers)
        mean_support_attention, max_support_attention = support_attention_statistics(
            oracle_trace, support_ids
        )
        layer_gain = (
            oracle_trace.target_logprob.mean(axis=1)
            - missing_trace.target_logprob.mean(axis=1)
        )
        record = {
            "qid": str(question_row["qid"]),
            "question": str(question_row["question"]),
            "gold": gold,
            "support_ranks": [int(row["rank"]) for row in support_rows],
            "missing_prediction": missing_prediction,
            "oracle_prediction": oracle_prediction,
            "missing_em": exact_match(missing_prediction, gold_answers),
            "oracle_em": exact_match(oracle_prediction, gold_answers),
            "missing_f1": missing_f1,
            "oracle_f1": oracle_f1,
            "missing_numerical_accuracy": numerical_accuracy(missing_prediction, gold_answers),
            "oracle_numerical_accuracy": numerical_accuracy(oracle_prediction, gold_answers),
            "f1_gain": oracle_f1 - missing_f1,
            "layer_ids": oracle_trace.layer_ids,
            "gold_logprob_gain_by_layer": layer_gain.tolist(),
            "mean_gold_logprob_gain": float(layer_gain.mean()),
            "final_gold_logprob_gain": float(layer_gain[-1]),
            "mean_support_attention_fraction": mean_support_attention,
            "max_head_support_attention_fraction": max_support_attention,
        }
        results.append(record)
        print(
            f"[{index}/{len(examples)}] {record['qid']} "
            f"F1 {missing_f1:.3f}->{oracle_f1:.3f}; "
            f"logp gain={record['mean_gold_logprob_gain']:+.3f}; "
            f"top-head support={max_support_attention:.3f}",
            flush=True,
        )

    logprob_gains = [row["mean_gold_logprob_gain"] for row in results]
    f1_gains = [row["f1_gain"] for row in results]
    summary = {
        "model": args.model,
        "split_role": "calibration",
        "n": len(results),
        "top_k_missing_support": args.top_k,
        "layers": extractor.layers,
        "mean_missing_em": float(np.mean([row["missing_em"] for row in results])),
        "mean_oracle_em": float(np.mean([row["oracle_em"] for row in results])),
        "mean_missing_f1": float(np.mean([row["missing_f1"] for row in results])),
        "mean_oracle_f1": float(np.mean([row["oracle_f1"] for row in results])),
        "mean_gold_logprob_gain": float(np.mean(logprob_gains)),
        "positive_gold_logprob_gain_rate": float(np.mean(np.asarray(logprob_gains) > 0)),
        "logprob_gain_f1_gain_spearman": safe_spearman(logprob_gains, f1_gains),
        "mean_support_attention_fraction": float(
            np.mean([row["mean_support_attention_fraction"] for row in results])
        ),
        "mean_max_head_support_attention_fraction": float(
            np.mean([row["max_head_support_attention_fraction"] for row in results])
        ),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, indent=2))


if __name__ == "__main__":
    main()
