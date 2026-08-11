"""Evaluate uncertainty on frozen RAGU WebQuestions contexts via an OpenAI-compatible server.

The runner is intended for a remote vLLM server hosting an LLM such as
``mistralai/Mistral-7B-Instruct-v0.3``.  It does not perform retrieval: each
question is evaluated with the stored top-k Contriever-MSMARCO passages in the
downloaded RAGU artifact.  This keeps the WebQ test inputs fixed across runs.

Example:
    python eval/run_webq_remote.py \
        --base-url http://127.0.0.1:8000/v1 \
        --model mistralai/Mistral-7B-Instruct-v0.3 \
        --api-key "$VLLM_API_KEY"
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

# Make ``src`` importable when this file is launched directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from eval.metrics import exact_match
from uncertainty_rag.core.claim_extractor import ClaimExtractor
from uncertainty_rag.core.sampler import SYSTEM_PROMPT, Sample
from uncertainty_rag.core.semantic_cluster import SemanticClusterer
from uncertainty_rag.core.uncertainty import UncertaintyEstimator
from uncertainty_rag.modality.base import ContextChunk
from uncertainty_rag.modality.text_handler import TextHandler
from uncertainty_rag.models.llm_client import OpenAIClient
from uncertainty_rag.models.nli_model import NLIModel


DEFAULT_DATA = PROJECT_ROOT / "data" / "webq_ragu" / "webq-test-400-seed10.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "webq_remote" / "seed10_predictions.jsonl"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL records while rejecting malformed or empty input."""
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
            records.append(record)
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def load_webq_records(path: Path) -> list[dict[str, Any]]:
    """Load and validate the frozen RAGU WebQuestions schema."""
    records = load_jsonl(path)
    required = {"q_id", "question", "answers", "ctxs"}
    for record in records:
        missing = required.difference(record)
        if missing:
            raise ValueError(f"WebQ record {record.get('q_id', '<unknown>')} is missing {sorted(missing)}")
    return records


def record_to_chunks(record: dict[str, Any], top_k: int) -> list[ContextChunk]:
    """Convert one frozen RAGU record into the project's context representation."""
    chunks = []
    for passage in record["ctxs"][:top_k]:
        passage_id = str(passage["id"])
        title = str(passage.get("title", "")).strip()
        text = str(passage.get("text", "")).strip()
        content = f"{title}\n{text}".strip() if title else text
        chunks.append(
            ContextChunk(
                id=passage_id,
                content=content,
                modality="text",
                metadata={
                    "title": title,
                    "retrieval_score": passage.get("score"),
                    "has_answer": passage.get("hasanswer"),
                },
            )
        )
    if not chunks:
        raise ValueError(f"WebQ record {record['q_id']} has no retrieved passages")
    return chunks


def auroc_for_incorrect_answers(labels_correct: list[int], uncertainty_scores: list[float]) -> float | None:
    """Compute AUROC where a higher score means an answer is more likely incorrect.

    This rank-based implementation has no scikit-learn dependency and gives tied
    scores their average rank.
    """
    if len(labels_correct) != len(uncertainty_scores):
        raise ValueError("labels and scores must have the same length")
    labels_incorrect = [1 - int(label) for label in labels_correct]
    positives = sum(labels_incorrect)
    negatives = len(labels_incorrect) - positives
    if positives == 0 or negatives == 0:
        return None

    ranked = sorted(enumerate(uncertainty_scores), key=lambda item: item[1])
    ranks = [0.0] * len(ranked)
    start = 0
    while start < len(ranked):
        end = start + 1
        while end < len(ranked) and ranked[end][1] == ranked[start][1]:
            end += 1
        average_rank = (start + 1 + end) / 2.0  # ranks are one-indexed
        for position in range(start, end):
            ranks[ranked[position][0]] = average_rank
        start = end

    rank_sum = sum(rank for rank, label in zip(ranks, labels_incorrect) if label)
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def existing_question_ids(path: Path) -> set[str]:
    """Return successful question IDs in an existing JSONL output for --resume."""
    if not path.exists():
        return set()
    return {str(record["q_id"]) for record in load_jsonl(path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run semantic/token uncertainty on the frozen RAGU WebQ seed-10 sample."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-url", default=os.getenv("VLLM_BASE_URL"), help="Server URL ending in /v1")
    parser.add_argument("--model", required=True, help="Model ID exposed by the remote server")
    parser.add_argument("--api-key", default=os.getenv("VLLM_API_KEY", "EMPTY"))
    parser.add_argument("--top-k", type=int, default=5, help="Use the first k frozen retrieved passages")
    parser.add_argument("--num-generations", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-logprobs", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument(
        "--claim-mode",
        choices=("extract", "answer"),
        default="extract",
        help="Use LLM claim extraction or treat each complete sampled answer as one claim",
    )
    parser.add_argument(
        "--no-token-logprobs",
        action="store_true",
        help="Run semantic entropy only for servers that do not expose output logprobs",
    )
    parser.add_argument("--resume", action="store_true", help="Skip question IDs already written to output")
    parser.add_argument("--nli-model", default="cross-encoder/nli-deberta-v3-base")
    args = parser.parse_args()
    if not args.base_url:
        parser.error("--base-url or VLLM_BASE_URL is required")
    if args.top_k < 1 or args.num_generations < 2 or args.max_tokens < 1:
        parser.error("--top-k, --num-generations, and --max-tokens must be positive (generations >= 2)")
    return args


def main() -> None:
    args = parse_args()
    records = load_webq_records(args.data)
    if args.max_examples is not None:
        records = records[: args.max_examples]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = existing_question_ids(args.output) if args.resume else set()
    if not args.resume and args.output.exists():
        args.output.unlink()

    client = OpenAIClient(model=args.model, api_key=args.api_key, base_url=args.base_url)
    handler = TextHandler()
    claim_extractor = ClaimExtractor(client, modality_type="text")
    clusterer = SemanticClusterer(NLIModel(model_name=args.nli_model))
    estimator = UncertaintyEstimator()

    all_results: list[dict[str, Any]] = []
    with args.output.open("a", encoding="utf-8") as output_handle:
        for index, record in enumerate(records, start=1):
            question_id = str(record["q_id"])
            if question_id in completed:
                continue

            chunks = record_to_chunks(record, args.top_k)
            messages = handler.build_prompt_messages(
                query=record["question"], chunks=chunks, system_prompt=SYSTEM_PROMPT
            )
            greedy = client.generate(
                messages=messages,
                n=1,
                temperature=0.0,
                max_tokens=args.max_tokens,
                logprobs=not args.no_token_logprobs,
                top_logprobs=args.top_logprobs,
            )[0]
            generated = client.generate(
                messages=messages,
                n=args.num_generations,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                logprobs=not args.no_token_logprobs,
                top_logprobs=args.top_logprobs,
            )
            samples = [
                Sample(
                    text=result.text,
                    token_logprobs=result.token_logprobs,
                    finish_reason=result.finish_reason,
                )
                for result in generated
            ]
            if args.claim_mode == "extract":
                samples = claim_extractor.extract_all(samples)
            else:
                for sample in samples:
                    sample.claims = [sample.text] if sample.text.strip() else []
                    sample.key_token_logprobs = ClaimExtractor.identify_key_tokens(
                        sample.claims, sample.token_logprobs
                    )

            concepts = clusterer.cluster(samples)
            profile = estimator.compute(samples, concepts)
            normalized_semantic_entropy = profile.se_semantic / math.log2(len(samples))
            result = {
                "q_id": record["q_id"],
                "question": record["question"],
                "gold_answers": record["answers"],
                "answer": greedy.text,
                "correct_em": int(exact_match(greedy.text, record["answers"])),
                "se_semantic": profile.se_semantic,
                "se_semantic_normalized": normalized_semantic_entropy,
                "u_token": profile.u_token if not args.no_token_logprobs else None,
                "num_concepts": profile.num_concepts,
                "num_samples": len(samples),
                "top_k": args.top_k,
                "model": args.model,
                "base_url": args.base_url,
                "claim_mode": args.claim_mode,
            }
            output_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            output_handle.flush()
            all_results.append(result)
            print(
                f"[{index}/{len(records)}] q_id={question_id} em={result['correct_em']} "
                f"se={result['se_semantic_normalized']:.3f} token={result['u_token']}"
            )

    # Include prior resumed results when computing the final aggregate.
    complete_results = load_jsonl(args.output)
    labels = [int(row["correct_em"]) for row in complete_results]
    semantic_scores = [float(row["se_semantic_normalized"]) for row in complete_results]
    summary: dict[str, Any] = {
        "examples": len(complete_results),
        "em": sum(labels) / len(labels) if labels else 0.0,
        "auroc_semantic_incorrect": auroc_for_incorrect_answers(labels, semantic_scores),
    }
    token_rows = [row for row in complete_results if row.get("u_token") is not None]
    if token_rows:
        summary["auroc_token_incorrect"] = auroc_for_incorrect_answers(
            [int(row["correct_em"]) for row in token_rows],
            [float(row["u_token"]) for row in token_rows],
        )
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote per-example results to {args.output}")


if __name__ == "__main__":
    main()
