"""Evaluate retrieval-posterior uncertainty on the frozen RAGU WebQ split.

The ordinary RAGU baselines repeatedly sample the generator while keeping the
top-5 retrieved passages fixed.  This runner instead samples plausible top-5
passage sets from the top-20 Contriever results, generates answers in each
retrieval state, and measures whether the original greedy answer remains
semantically stable.

The primary score is::

    U_anchor = 1 - P_posterior(semantic answer == greedy semantic answer)

It also reports total posterior answer entropy and its empirical decomposition
into within-retrieval generation entropy and between-retrieval mutual
information.

The file is standalone apart from small helpers in this directory.  Use the
two-stage mode on a 24 GB GPU if vLLM and the NLI model do not comfortably fit
at the same time: first generate and cache answers, stop vLLM, then score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from metrics import ragqa_match
from openai import OpenAI
from prompts import STOP_SEQUENCES, make_messages, make_paragraph
from run_webq_paper_baselines import auroc_incorrect, load_jsonl

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "data" / "webq_ragu" / "webq-test-400-seed10.jsonl"
DEFAULT_CACHE = (
    ROOT / "results" / "webq_retrieval_posterior" / "mistral7b_seed10.generations.jsonl"
)
DEFAULT_OUTPUT = (
    ROOT / "results" / "webq_retrieval_posterior" / "mistral7b_seed10_rpp.jsonl"
)


def write_jsonl_row(handle: Any, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()


def stable_seed(seed: int, q_id: object) -> int:
    """Create a process-independent, OpenAI-compatible non-negative seed."""
    digest = hashlib.sha256(f"{seed}:{q_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    probabilities = np.exp(shifted)
    return probabilities / probabilities.sum()


def retrieval_logits(scores: Iterable[float], temperature: float) -> np.ndarray:
    """Turn retriever scores into scale-invariant posterior logits.

    Retriever score scales differ across models.  Standardizing within the
    top-N pool before applying temperature prevents a hard-coded Contriever
    score scale from silently determining the amount of perturbation.
    """
    array = np.asarray(list(scores), dtype=np.float64)
    if array.size == 0:
        raise ValueError("Cannot construct a retrieval posterior without scores")
    standard_deviation = float(array.std())
    standardized = np.zeros_like(array) if standard_deviation < 1e-12 else (
        (array - float(array.mean())) / standard_deviation
    )
    return standardized / temperature


def gumbel_top_k(logits: np.ndarray, k: int, rng: np.random.Generator) -> list[int]:
    """Sample an unordered set without replacement from Plackett-Luce weights."""
    if not 0 < k <= len(logits):
        raise ValueError(f"k must be in [1, {len(logits)}], received {k}")
    perturbed = logits + rng.gumbel(size=len(logits))
    selected = np.argpartition(perturbed, -k)[-k:]
    # Preserve original retrieval rank in the prompt.  Otherwise passage order
    # becomes an accidental second perturbation.
    return sorted(int(index) for index in selected)


def sample_context_indices(
    scores: list[float],
    num_contexts: int,
    context_size: int,
    temperature: float,
    seed: int,
) -> tuple[list[list[int]], list[float]]:
    logits = retrieval_logits(scores, temperature)
    probabilities = softmax(logits)
    rng = np.random.default_rng(seed)
    draws = [gumbel_top_k(logits, context_size, rng) for _ in range(num_contexts)]
    return draws, probabilities.tolist()


def generate_answers(
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
    n: int,
    temperature: float,
    top_p: float,
    top_k_sampling: int,
    max_tokens: int,
    seed: int,
) -> list[str]:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        n=n,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop=STOP_SEQUENCES,
        seed=seed,
        extra_body={"top_k": top_k_sampling},
    )
    return [choice.message.content or "" for choice in response.choices]


def entropy(cluster_ids: Iterable[int]) -> float:
    ids = list(cluster_ids)
    if not ids:
        return 0.0
    counts = np.asarray(list(Counter(ids).values()), dtype=np.float64)
    probabilities = counts / counts.sum()
    return float(-(probabilities * np.log(probabilities)).sum())


class BatchedStrictNLI:
    """Question-conditioned, strict bidirectional NLI used by RAGU.

    All directed pairs are evaluated in batches, after which RAGU's original
    greedy representative-assignment algorithm is reproduced.  GPU inference
    uses float16 to fit alongside a conservatively configured 7B vLLM server.
    """

    def __init__(
        self,
        model_name: str,
        device: str,
        batch_size: int,
        entailment_index: int | None,
    ) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--nli-device cuda requested but CUDA is unavailable")

        self.torch = torch
        self.device = device
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model_kwargs: dict[str, Any] = {}
        if device == "cuda":
            model_kwargs["torch_dtype"] = torch.float16
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, **model_kwargs
        ).to(device)
        self.model.eval()
        self.entailment_index = (
            entailment_index
            if entailment_index is not None
            else self._infer_entailment_index(model_name)
        )

    def _infer_entailment_index(self, model_name: str) -> int:
        labels = {
            int(index): str(label).lower()
            for index, label in self.model.config.id2label.items()
        }
        matches = [index for index, label in labels.items() if "entail" in label]
        if len(matches) == 1:
            return matches[0]
        lowered = model_name.lower()
        if "microsoft/deberta-v2" in lowered and self.model.config.num_labels == 3:
            return 2
        if "cross-encoder/nli-deberta-v3" in lowered and self.model.config.num_labels == 3:
            return 1
        raise ValueError(
            "Could not infer the entailment label from model config "
            f"{labels}; pass --entailment-index explicitly"
        )

    @property
    def description(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "batch_size": self.batch_size,
            "entailment_index": self.entailment_index,
            "dtype": str(next(self.model.parameters()).dtype),
        }

    @property
    def no_grad(self) -> Any:
        return self.torch.no_grad()

    def semantic_ids(self, question: str, answers: list[str]) -> list[int]:
        conditioned = [f"{question} {answer}" for answer in answers]
        directed_pairs = [
            (left, right)
            for left in range(len(conditioned))
            for right in range(len(conditioned))
            if left != right
        ]
        predictions: dict[tuple[int, int], int] = {}

        with self.no_grad:
            for start in range(0, len(directed_pairs), self.batch_size):
                batch = directed_pairs[start : start + self.batch_size]
                inputs = self.tokenizer(
                    [conditioned[left] for left, _ in batch],
                    [conditioned[right] for _, right in batch],
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True,
                ).to(self.device)
                logits = self.model(**inputs).logits
                classes = self.torch.argmax(logits, dim=-1).cpu().tolist()
                predictions.update(zip(batch, (int(value) for value in classes)))

        # Exact greedy assignment used by RAGU get_semantic_ids().
        semantic_ids = [-1] * len(answers)
        next_id = 0
        for left in range(len(answers)):
            if semantic_ids[left] != -1:
                continue
            semantic_ids[left] = next_id
            for right in range(left + 1, len(answers)):
                equivalent = (
                    predictions[(left, right)] == self.entailment_index
                    and predictions[(right, left)] == self.entailment_index
                )
                if equivalent:
                    semantic_ids[right] = next_id
            next_id += 1
        return semantic_ids


def calculate_rpp_scores(
    semantic_ids: list[int],
    num_contexts: int,
    answers_per_context: int,
) -> dict[str, Any]:
    """Calculate anchor stability and posterior entropy decomposition.

    ``semantic_ids[0]`` is the original greedy top-5 answer.  Remaining IDs
    are ordered by retrieval context and then generation replicate.
    """
    expected = 1 + num_contexts * answers_per_context
    if len(semantic_ids) != expected:
        raise ValueError(f"Expected {expected} semantic IDs, received {len(semantic_ids)}")

    anchor_id = semantic_ids[0]
    posterior_ids = semantic_ids[1:]
    anchor_mass = sum(value == anchor_id for value in posterior_ids) / len(posterior_ids)
    total = entropy(posterior_ids)
    conditional = []
    for context_index in range(num_contexts):
        start = context_index * answers_per_context
        conditional.append(entropy(posterior_ids[start : start + answers_per_context]))
    generation = float(np.mean(conditional))
    retrieval = max(0.0, total - generation)

    return {
        "rpp_anchor_mass": anchor_mass,
        "rpp_anchor_risk": 1.0 - anchor_mass,
        "rpp_total_entropy": total,
        "rpp_generation_entropy": generation,
        "rpp_retrieval_mi": retrieval,
        "rpp_conditional_entropies": conditional,
        "rpp_semantic_ids": semantic_ids,
        "rpp_num_posterior_concepts": len(set(posterior_ids)),
    }


def baseline_map(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    return {str(row["q_id"]): row for row in load_jsonl(path)}


def generate_stage(args: argparse.Namespace) -> None:
    if not args.base_url:
        raise ValueError("--base-url (or VLLM_BASE_URL) is required for the generate stage")
    if not args.model:
        raise ValueError("--model is required for the generate stage")
    records = load_jsonl(args.data)
    if args.max_examples is not None:
        records = records[: args.max_examples]
    baselines = baseline_map(args.baseline_results)

    args.generation_cache.parent.mkdir(parents=True, exist_ok=True)
    completed = (
        {str(row["q_id"]) for row in load_jsonl(args.generation_cache)}
        if args.resume and args.generation_cache.exists()
        else set()
    )
    if not args.resume and args.generation_cache.exists():
        args.generation_cache.unlink()

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    with args.generation_cache.open("a", encoding="utf-8") as destination:
        for position, record in enumerate(records, start=1):
            q_id = str(record["q_id"])
            if q_id in completed:
                continue
            contexts = record.get("ctxs", [])[: args.retrieval_pool_size]
            if len(contexts) < args.context_size:
                raise ValueError(
                    f"q_id={q_id} has {len(contexts)} contexts; need {args.context_size}"
                )
            scores = [float(context["score"]) for context in contexts]
            question = str(record["question"])
            golds = [str(answer) for answer in record["answers"]]
            example_seed = stable_seed(args.seed, q_id)

            baseline = baselines.get(q_id)
            if args.baseline_results is not None and baseline is None:
                raise ValueError(
                    f"q_id={q_id} is missing from --baseline-results {args.baseline_results}"
                )
            if baseline is not None:
                anchor_answer = str(baseline["answer"])
                correct_acc = int(baseline["correct_acc"])
            else:
                anchor_record = {**record, "ctxs": contexts[: args.context_size]}
                anchor_messages = make_messages(
                    args.model,
                    args.prompt_name,
                    {
                        "instruction": question,
                        "paragraph": make_paragraph(anchor_record, args.context_size),
                    },
                )
                anchor_answer = generate_answers(
                    client, args.model, anchor_messages, 1, 0.0, args.top_p,
                    args.top_k_sampling, args.max_tokens, example_seed,
                )[0]
                correct_acc = ragqa_match(anchor_answer, golds)

            draws, probabilities = sample_context_indices(
                scores,
                args.num_contexts,
                args.context_size,
                args.retrieval_temperature,
                example_seed,
            )
            posterior_answers: list[list[str]] = []
            for context_index, indices in enumerate(draws):
                sampled_record = {**record, "ctxs": [contexts[index] for index in indices]}
                messages = make_messages(
                    args.model,
                    args.prompt_name,
                    {
                        "instruction": question,
                        "paragraph": make_paragraph(sampled_record, args.context_size),
                    },
                )
                posterior_answers.append(
                    generate_answers(
                        client,
                        args.model,
                        messages,
                        args.answers_per_context,
                        args.generation_temperature,
                        args.top_p,
                        args.top_k_sampling,
                        args.max_tokens,
                        example_seed + context_index + 1,
                    )
                )

            row = {
                "q_id": record["q_id"],
                "question": question,
                "gold_answers": golds,
                "answer": anchor_answer,
                "correct_acc": correct_acc,
                "rpp_context_indices": draws,
                "rpp_context_ids": [
                    [str(contexts[index].get("id", index)) for index in draw]
                    for draw in draws
                ],
                "rpp_context_hasanswer": [
                    any(bool(contexts[index].get("hasanswer")) for index in draw)
                    for draw in draws
                ],
                "rpp_retrieval_probabilities": probabilities,
                "rpp_posterior_answers": posterior_answers,
                "rpp_unique_context_sets": len({tuple(draw) for draw in draws}),
                "rpp_config": {
                    "retrieval_pool_size": len(contexts),
                    "context_size": args.context_size,
                    "num_contexts": args.num_contexts,
                    "answers_per_context": args.answers_per_context,
                    "retrieval_temperature": args.retrieval_temperature,
                    "generation_temperature": args.generation_temperature,
                    "seed": args.seed,
                    "model": args.model,
                    "prompt_name": args.prompt_name,
                },
            }
            write_jsonl_row(destination, row)
            print(
                f"[generate {position}/{len(records)}] q_id={q_id} "
                f"acc={correct_acc} unique_contexts={row['rpp_unique_context_sets']}"
            )
    print(f"Wrote generation cache to {args.generation_cache}")


def score_stage(args: argparse.Namespace) -> None:
    rows = load_jsonl(args.generation_cache)
    if args.max_examples is not None:
        rows = rows[: args.max_examples]
    baselines = baseline_map(args.baseline_results)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = (
        {str(row["q_id"]) for row in load_jsonl(args.output)}
        if args.resume and args.output.exists()
        else set()
    )
    if not args.resume and args.output.exists():
        args.output.unlink()

    nli = BatchedStrictNLI(
        args.nli_model,
        args.nli_device,
        args.nli_batch_size,
        args.entailment_index,
    )
    print("NLI configuration: " + json.dumps(nli.description))

    with args.output.open("a", encoding="utf-8") as destination:
        for position, generated in enumerate(rows, start=1):
            q_id = str(generated["q_id"])
            if q_id in completed:
                continue
            nested_answers = generated["rpp_posterior_answers"]
            posterior_answers = [answer for group in nested_answers for answer in group]
            all_answers = [generated["answer"], *posterior_answers]
            semantic_ids = nli.semantic_ids(generated["question"], all_answers)
            scores = calculate_rpp_scores(
                semantic_ids,
                len(nested_answers),
                len(nested_answers[0]),
            )
            base = baselines.get(q_id, {})
            result = {**base, **generated, **scores}
            result["rpp_nli_model"] = args.nli_model
            write_jsonl_row(destination, result)
            print(
                f"[score {position}/{len(rows)}] q_id={q_id} "
                f"anchor_risk={scores['rpp_anchor_risk']:.3f} "
                f"H={scores['rpp_total_entropy']:.3f} "
                f"MI={scores['rpp_retrieval_mi']:.3f}"
            )

    target_ids = {str(row["q_id"]) for row in rows}
    scored = [row for row in load_jsonl(args.output) if str(row["q_id"]) in target_ids]
    labels = [int(row["correct_acc"]) for row in scored]
    summary: dict[str, Any] = {
        "examples": len(scored),
        "accuracy": float(np.mean(labels)),
        "auroc_rpp_anchor_risk_incorrect": auroc_incorrect(
            labels, [float(row["rpp_anchor_risk"]) for row in scored]
        ),
        "auroc_rpp_total_entropy_incorrect": auroc_incorrect(
            labels, [float(row["rpp_total_entropy"]) for row in scored]
        ),
        "auroc_rpp_retrieval_mi_incorrect": auroc_incorrect(
            labels, [float(row["rpp_retrieval_mi"]) for row in scored]
        ),
        "auroc_rpp_generation_entropy_incorrect": auroc_incorrect(
            labels, [float(row["rpp_generation_entropy"]) for row in scored]
        ),
        "mean_unique_context_sets": float(
            np.mean([row["rpp_unique_context_sets"] for row in scored])
        ),
        "mean_context_hasanswer": float(
            np.mean(
                [value for row in scored for value in row["rpp_context_hasanswer"]]
            )
        ),
    }
    baseline_fields = {
        "ppl": "auroc_ppl_incorrect",
        "regular_entropy": "auroc_regular_entropy_incorrect",
        "semantic_entropy": "auroc_semantic_entropy_incorrect",
        "ours_token_uncertainty": "auroc_ours_token_incorrect",
        "ours_semantic_uncertainty": "auroc_ours_semantic_incorrect",
        "p_true_uncertainty": "auroc_p_true_incorrect",
    }
    for field, summary_name in baseline_fields.items():
        if scored and all(row.get(field) is not None for row in scored):
            summary[summary_name] = auroc_incorrect(
                labels, [float(row[field]) for row in scored]
            )

    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote retrieval-posterior results to {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieval-Posterior Predictive UQ on frozen RAGU WebQ"
    )
    parser.add_argument("--stage", choices=("generate", "score", "all"), default="all")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--generation-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--baseline-results",
        type=Path,
        help="Optional completed baseline JSONL; reuses its greedy answer and reports its AUROCs",
    )
    parser.add_argument("--base-url", default=os.getenv("VLLM_BASE_URL"))
    parser.add_argument("--api-key", default=os.getenv("VLLM_API_KEY", "EMPTY"))
    parser.add_argument("--model", help="Use mistralai/Mistral-7B-Instruct-v0.3 for comparison")
    parser.add_argument("--prompt-name", default="chat_directRagQA_REAR3")
    parser.add_argument("--retrieval-pool-size", type=int, default=20)
    parser.add_argument("--context-size", type=int, default=5)
    parser.add_argument("--num-contexts", type=int, default=5)
    parser.add_argument("--answers-per-context", type=int, default=2)
    parser.add_argument("--retrieval-temperature", type=float, default=1.0)
    parser.add_argument("--generation-temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k-sampling", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--nli-model", default="microsoft/deberta-v2-xlarge-mnli")
    parser.add_argument("--nli-device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--nli-batch-size", type=int, default=32)
    parser.add_argument("--entailment-index", type=int)
    args = parser.parse_args()

    if args.retrieval_pool_size < args.context_size:
        parser.error("--retrieval-pool-size must be >= --context-size")
    if args.context_size < 1 or args.num_contexts < 1 or args.answers_per_context < 1:
        parser.error("context size, context count, and answers per context must be positive")
    if args.retrieval_temperature <= 0 or args.generation_temperature <= 0:
        parser.error("retrieval and generation temperatures must be positive")
    if args.stage in {"generate", "all"} and not args.model:
        parser.error("--model is required for --stage generate/all")
    if args.stage in {"generate", "all"} and not args.base_url:
        parser.error("--base-url or VLLM_BASE_URL is required for --stage generate/all")
    return args


def main() -> None:
    args = parse_args()
    if args.stage in {"generate", "all"}:
        generate_stage(args)
    if args.stage in {"score", "all"}:
        score_stage(args)


if __name__ == "__main__":
    main()
