"""Standalone remote reproduction of RAGU PPL and semantic-entropy baselines.

It contains no imports from related_repos/ragu or uncertainty_rag.  It uses the
frozen WebQ data and sends RAGU's Mistral prompt to any OpenAI-compatible vLLM
server exposing output log probabilities.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI

from metrics import exact_match, ragqa_match
from prompts import STOP_SEQUENCES, make_messages, make_paragraph
from semantic_entropy import DebertaMNLI, compute_entropies


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "data" / "webq_ragu" / "webq-test-400-seed10.jsonl"
DEFAULT_OUTPUT = ROOT / "results" / "webq_paper_baselines" / "mistral7b_seed10.jsonl"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError(f"No records in {path}")
    return rows


def output_token_logprobs(choice: Any) -> list[dict[str, float | str]]:
    content = getattr(getattr(choice, "logprobs", None), "content", None) or []
    return [{"token": str(token.token), "logprob": float(token.logprob)} for token in content]


def generate(client: OpenAI, args: argparse.Namespace, messages: list[dict[str, str]], temperature: float, n: int, seed: int) -> list[tuple[str, list[dict[str, float | str]]]]:
    response = client.chat.completions.create(
        model=args.model,
        messages=messages,
        n=n,
        temperature=temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        logprobs=True,
        top_logprobs=args.top_logprobs,
        stop=STOP_SEQUENCES if args.do_stop else None,
        seed=seed,
        extra_body={"top_k": args.top_k_sampling},
    )
    return [(choice.message.content or "", output_token_logprobs(choice)) for choice in response.choices]


def auroc_incorrect(correct: list[int], uncertainty: list[float]) -> float | None:
    incorrect = [1 - value for value in correct]
    positives = sum(incorrect)
    negatives = len(incorrect) - positives
    if not positives or not negatives:
        return None
    order = sorted(range(len(uncertainty)), key=lambda index: uncertainty[index])
    ranks = [0.0] * len(order)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and uncertainty[order[end]] == uncertainty[order[start]]:
            end += 1
        for position in range(start, end):
            ranks[order[position]] = (start + 1 + end) / 2
        start = end
    rank_sum = sum(rank for rank, label in zip(ranks, incorrect) if label)
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone RAGU WebQ PPL + semantic entropy reproduction")
    parser.add_argument("--base-url", default=os.getenv("VLLM_BASE_URL"), required=os.getenv("VLLM_BASE_URL") is None)
    parser.add_argument("--api-key", default=os.getenv("VLLM_API_KEY", "EMPTY"))
    parser.add_argument("--model", required=True, help="Use mistralai/Mistral-7B-Instruct-v0.3 for paper comparison")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prompt-name", default="chat_directRagQA_REAR3")
    parser.add_argument("--top-k", type=int, default=5, help="Frozen retrieved passages")
    parser.add_argument("--num-generations", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k-sampling", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--top-logprobs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--do-stop", action="store_true", default=True)
    parser.add_argument("--no-do-stop", action="store_false", dest="do_stop")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--entailment-model", default="microsoft/deberta-v2-xlarge-mnli")
    args = parser.parse_args()
    if args.top_k < 1 or args.num_generations < 2:
        parser.error("--top-k must be >=1 and --num-generations must be >=2")
    return args


def main() -> None:
    args = parse_args()
    records = load_jsonl(args.data)
    if args.max_examples:
        records = records[: args.max_examples]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = {str(row["q_id"]) for row in load_jsonl(args.output)} if args.resume and args.output.exists() else set()
    if not args.resume and args.output.exists():
        args.output.unlink()
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    nli = DebertaMNLI(args.entailment_model)

    with args.output.open("a", encoding="utf-8") as destination:
        for position, record in enumerate(records, start=1):
            if str(record["q_id"]) in completed:
                continue
            item = {"instruction": record["question"], "paragraph": make_paragraph(record, args.top_k)}
            messages = make_messages(args.model, args.prompt_name, item)
            greedy_text, greedy_lps = generate(client, args, messages, 0.0, 1, args.seed)[0]
            samples: list[tuple[str, list[float]]] = []
            # RAGU calls vLLM ten times and changes its per-call seed from 0 to 9.
            for sample_seed in range(args.num_generations):
                samples.extend(generate(client, args, messages, args.temperature, 1, sample_seed))
            responses, sample_token_lps = zip(*samples)
            sample_lps = [[float(token["logprob"]) for token in token_lps] for token_lps in sample_token_lps]
            entropy = compute_entropies(list(responses), sample_lps, nli, record["question"], strict_entailment=True)
            golds = [str(answer) for answer in record["answers"]]
            greedy_lps_values = [float(token["logprob"]) for token in greedy_lps]
            nll = -float(np.mean(greedy_lps_values)) if greedy_lps_values else None
            result = {
                "q_id": record["q_id"], "question": record["question"], "gold_answers": golds,
                "answer": greedy_text, "correct_em": exact_match(greedy_text, golds),
                "correct_acc": ragqa_match(greedy_text, golds), "ppl": math.exp(nll) if nll is not None else None,
                "nll": nll, "top_k": args.top_k, "prompt_name": args.prompt_name,
                "model": args.model, "samples": list(responses),
                "sample_token_logprobs": list(sample_token_lps),
                "greedy_token_logprobs": greedy_lps,
                **entropy,
            }
            destination.write(json.dumps(result, ensure_ascii=False) + "\n")
            destination.flush()
            print(f"[{position}/{len(records)}] q_id={record['q_id']} acc={result['correct_acc']} ppl={result['ppl']:.3f} se={result['semantic_entropy']:.3f}")

    rows = load_jsonl(args.output)
    labels = [int(row["correct_acc"]) for row in rows]
    summary = {
        "examples": len(rows), "acc": float(np.mean(labels)),
        "em": float(np.mean([int(row["correct_em"]) for row in rows])),
        "auroc_ppl_incorrect": auroc_incorrect(labels, [float(row["ppl"]) for row in rows]),
        "auroc_regular_entropy_incorrect": auroc_incorrect(labels, [float(row["regular_entropy"]) for row in rows]),
        "auroc_semantic_entropy_incorrect": auroc_incorrect(labels, [float(row["semantic_entropy"]) for row in rows]),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote RAGU-compatible results to {args.output}")


if __name__ == "__main__":
    main()
