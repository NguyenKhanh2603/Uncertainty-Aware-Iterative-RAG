"""Standalone RAGU-style p(True) evaluation on saved WebQ baseline samples.

The script first creates (and caches) 20 labelled WebQ training demonstrations.
It then uses those demonstrations to ask the same Mistral model whether each
saved greedy answer is true. It does not resample the completed 400 evaluation
questions. No RAGU or main-project modules are imported.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI

from metrics import ragqa_match
from p_true import make_p_true_prompt, make_shot, p_true_uncertainty
from prompts import make_messages, make_paragraph
from run_webq_paper_baselines import STOP_SEQUENCES, auroc_incorrect, generate, load_jsonl


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVALUATION = ROOT / "results" / "webq_paper_baselines" / "mistral7b_seed10_with_ours.jsonl"
DEFAULT_TRAIN = ROOT / "data" / "webq_ragu" / "webq-train.jsonl"
DEFAULT_TEST = ROOT / "data" / "webq_ragu" / "webq-test-400-seed10.jsonl"
DEFAULT_SHOTS = ROOT / "results" / "webq_paper_baselines" / "ptrue_train_shots_seed10.jsonl"
DEFAULT_OUTPUT = ROOT / "results" / "webq_paper_baselines" / "mistral7b_seed10_with_ptrue.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone RAGU p(True) reproduction on WebQ")
    parser.add_argument("--input", type=Path, default=DEFAULT_EVALUATION, help="Completed unified evaluation JSONL")
    parser.add_argument("--test-data", type=Path, default=DEFAULT_TEST, help="Frozen contexts for the 400 evaluation rows")
    parser.add_argument("--train-data", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--fewshot-file", type=Path, default=DEFAULT_SHOTS, help="Cached generated 20-shot demonstrations")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-url", default=os.getenv("VLLM_BASE_URL"), required=os.getenv("VLLM_BASE_URL") is None)
    parser.add_argument("--api-key", default=os.getenv("VLLM_API_KEY", "EMPTY"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-name", default="chat_directRagQA_REAR3")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--num-generations", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k-sampling", type=int, default=50)
    parser.add_argument("--top-logprobs", type=int, default=5)
    parser.add_argument("--fewshot-count", type=int, default=20)
    parser.add_argument("--fewshot-seed", type=int, default=10)
    parser.add_argument("--train-proportion", type=float, default=0.5, help="RAGU's documented p(True) training-data proportion")
    parser.add_argument("--ptrue-logprobs", type=int, default=20, help="Must be large enough to contain token A; vLLM commonly caps this at 20")
    parser.add_argument("--max-examples", type=int, help="Smoke-test or limit the evaluation rows")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def generation_args(args: argparse.Namespace) -> argparse.Namespace:
    """Adapt p(True)'s generation flags to the shared RAGU generator helper."""
    return argparse.Namespace(
        model=args.model, top_p=args.top_p, max_tokens=args.max_tokens,
        top_logprobs=args.top_logprobs, do_stop=True, top_k_sampling=args.top_k_sampling,
    )


def create_fewshot_rows(client: OpenAI, args: argparse.Namespace) -> list[dict[str, Any]]:
    train = load_jsonl(args.train_data)
    if not 0 < args.train_proportion <= 1:
        raise ValueError("--train-proportion must be in (0, 1]")
    # RAGU's generate.py takes this deterministic prefix before random sampling.
    train = train[:int(len(train) * args.train_proportion)]
    answerable = [row for row in train if row.get("answers")]
    if len(answerable) < args.fewshot_count:
        raise ValueError("Not enough answerable train examples for p(True) demonstrations")
    selected = random.Random(args.fewshot_seed).sample(answerable, args.fewshot_count)
    local_args = generation_args(args)
    rows = []
    for index, record in enumerate(selected, start=1):
        item = {"instruction": record["question"], "paragraph": make_paragraph(record, args.top_k)}
        messages = make_messages(args.model, args.prompt_name, item)
        greedy, _ = generate(client, local_args, messages, 0.0, 1, args.fewshot_seed)[0]
        samples = []
        for sample_seed in range(args.num_generations):
            samples.extend(text for text, _ in generate(client, local_args, messages, 1.0, 1, sample_seed))
        golds = [str(answer) for answer in record["answers"]]
        rows.append({
            "q_id": record["q_id"], "question": record["question"], "answer": greedy,
            "samples": samples, "correct_acc": ragqa_match(greedy, golds), "gold_answers": golds,
        })
        print(f"[few-shot {index}/{len(selected)}] q_id={record['q_id']} acc={rows[-1]['correct_acc']}")
    return rows


def get_few_shot_prompt(rows: list[dict[str, Any]]) -> str:
    """Match RAGU's resolved A/B demonstration construction."""
    return "\n".join(
        make_shot(row["question"], row["answer"], row["samples"], bool(row["correct_acc"]))
        for row in rows
    )


def logprob_for_a(client: OpenAI, args: argparse.Namespace, prompt: str) -> float:
    """Return raw log p(' A') as RAGU's HuggingFace get_p_true does."""
    response = client.completions.create(
        model=args.model,
        prompt=prompt,
        max_tokens=1,
        temperature=0.0,
        logprobs=args.ptrue_logprobs,
        stop=STOP_SEQUENCES,
    )
    choice = response.choices[0]
    top = choice.logprobs.top_logprobs[0] if choice.logprobs and choice.logprobs.top_logprobs else {}
    # Different vLLM/tokenizer versions render Mistral's leading-space token as
    # either ``" A"`` or SentencePiece's visible marker ``"▁A"``.
    candidates = [
        (token, float(logprob))
        for token, logprob in top.items()
        if token.replace("▁", " ").strip() == "A"
    ]
    if not candidates:
        received = sorted(top)[:10]
        raise ValueError("The completion API did not return token ' A' in top logprobs. Increase --ptrue-logprobs; received " + repr(received))
    # Mistral normally has a single whitespace-prefixed A token. Prefer it if present.
    return next((logprob for token, logprob in candidates if token in {" A", "▁A"}), candidates[0][1])


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.input)
    if args.max_examples is not None:
        rows = rows[:args.max_examples]
    tests = {str(row["q_id"]): row for row in load_jsonl(args.test_data)}
    missing = [str(row["q_id"]) for row in rows if str(row["q_id"]) not in tests]
    if missing:
        raise ValueError("Evaluation contexts missing for q_id " + missing[0])
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    if args.fewshot_file.exists():
        fewshot_rows = load_jsonl(args.fewshot_file)
        if len(fewshot_rows) != args.fewshot_count:
            raise ValueError("Cached few-shot file has a different number of demonstrations")
        print(f"Using cached p(True) demonstrations: {args.fewshot_file}")
    else:
        fewshot_rows = create_fewshot_rows(client, args)
        write_jsonl(args.fewshot_file, fewshot_rows)
        print(f"Wrote p(True) demonstrations to {args.fewshot_file}")
    few_shots = get_few_shot_prompt(fewshot_rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = {str(row["q_id"]) for row in load_jsonl(args.output)} if args.resume and args.output.exists() else set()
    if not args.resume and args.output.exists():
        args.output.unlink()
    with args.output.open("a", encoding="utf-8") as destination:
        for index, row in enumerate(rows, start=1):
            if str(row["q_id"]) in completed:
                continue
            record = tests[str(row["q_id"])]
            prompt = few_shots + "\n\nKnowledge:\n" + make_paragraph(record, args.top_k) + "\n"
            prompt = make_p_true_prompt(row["question"], row["answer"], row["samples"], prompt)
            logprob_a = logprob_for_a(client, args, prompt)
            result = {**row, "p_true_logprob": logprob_a, "p_true_uncertainty": p_true_uncertainty(logprob_a)}
            destination.write(json.dumps(result, ensure_ascii=False) + "\n")
            destination.flush()
            print(f"[{index}/{len(rows)}] q_id={row['q_id']} p_true_u={result['p_true_uncertainty']:.4f}")

    scored = load_jsonl(args.output)
    labels = [int(row["correct_acc"]) for row in scored]
    summary = {
        "examples": len(scored), "acc": float(np.mean(labels)),
        "auroc_ours_semantic_incorrect": auroc_incorrect(labels, [float(row["ours_semantic_uncertainty"]) for row in scored]),
        "auroc_ours_token_incorrect": auroc_incorrect(labels, [float(row["ours_token_uncertainty"]) for row in scored]),
        "auroc_ppl_incorrect": auroc_incorrect(labels, [float(row["ppl"]) for row in scored]),
        "auroc_regular_entropy_incorrect": auroc_incorrect(labels, [float(row["regular_entropy"]) for row in scored]),
        "auroc_semantic_entropy_incorrect": auroc_incorrect(labels, [float(row["semantic_entropy"]) for row in scored]),
        "auroc_p_true_incorrect": auroc_incorrect(labels, [float(row["p_true_uncertainty"]) for row in scored]),
    }
    path = args.output.with_suffix(".summary.json")
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote p(True) comparison rows to {args.output}")


if __name__ == "__main__":
    main()
