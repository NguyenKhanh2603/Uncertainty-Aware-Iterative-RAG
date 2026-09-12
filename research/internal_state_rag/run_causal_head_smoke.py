"""Discover support-focused heads and test them by causal masking on held-out data."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import wilcoxon

from eval.metrics import exact_match, numerical_accuracy, token_f1
from research.internal_state_rag import QwenInternalStateExtractor, mask_attention_heads
from research.internal_state_rag.run_tatqa_smoke import (
    ResearchTextClient,
    generate_answer,
    load_retrieval,
    load_rows,
    to_chunk,
)


def build_oracle_case(
    record: dict[str, Any],
    *,
    questions: dict[str, dict[str, Any]],
    corpus: dict[str, dict[str, Any]],
    retrieval: dict[str, list[dict[str, Any]]],
    top_k: int,
) -> tuple[str, str, list[str], list[Any], set[str]]:
    qid = str(record["qid"])
    question = questions[qid]
    rows = retrieval[qid]
    top_rows = rows[:top_k]
    support_rows = [row for row in rows if row["support_label"] == "support"]
    oracle_rows = top_rows + [row for row in support_rows if row not in top_rows]
    chunks = [to_chunk(row, corpus) for row in oracle_rows]
    prompt = (
        "Answer using only the supplied context. Return only the short answer.\n"
        f"Question: {question['question']}"
    )
    gold_answers = [str(value) for value in question["gold_answers"]]
    return prompt, gold_answers[0], gold_answers, chunks, {
        str(row["chunk_id"]) for row in support_rows
    }


def support_fraction(trace: Any, support_ids: set[str]) -> np.ndarray:
    support_indices = [
        index for index, chunk_id in enumerate(trace.chunk_ids) if chunk_id in support_ids
    ]
    total = trace.attention_mass.sum(axis=-1)
    support = trace.attention_mass[:, :, support_indices].sum(axis=-1)
    return np.divide(support, total, out=np.zeros_like(support), where=total > 0)


def ranked_heads(scores: np.ndarray, layer_ids: list[int]) -> list[dict[str, Any]]:
    ranking = []
    for layer_offset, layer in enumerate(layer_ids):
        for head in range(scores.shape[1]):
            ranking.append(
                {
                    "layer": int(layer),
                    "head": int(head),
                    "support_fraction": float(scores[layer_offset, head]),
                }
            )
    return sorted(ranking, key=lambda row: (-row["support_fraction"], row["layer"], row["head"]))


def layer_matched_controls(
    ranking: list[dict[str, Any]],
    top_heads: list[tuple[int, int]],
    *,
    num_heads: int,
    seed: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Choose random and bottom controls with the same per-layer counts as top heads."""

    score = {
        (int(row["layer"]), int(row["head"])): float(row["support_fraction"])
        for row in ranking
    }
    top_set = set(top_heads)
    counts = Counter(layer for layer, _ in top_heads)
    generator = random.Random(seed)
    random_heads = []
    bottom_heads = []
    for layer, count in sorted(counts.items()):
        candidates = [(layer, head) for head in range(num_heads) if (layer, head) not in top_set]
        if len(candidates) < count:
            raise ValueError(f"Layer {layer} has too few non-top heads for a matched control.")
        random_heads.extend(generator.sample(candidates, count))
        bottom_heads.extend(sorted(candidates, key=lambda head: score[head])[:count])
    return random_heads, bottom_heads


def parse_args() -> argparse.Namespace:
    root = Path("data/conformal_global_run")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--source-results",
        type=Path,
        default=Path("research/internal_state_rag/results/tatqa_smoke_n39_seed17.json"),
    )
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
    parser.add_argument("--discovery-n", type=int, default=20)
    parser.add_argument("--evaluation-n", type=int, default=19)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--max-answer-tokens", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--layers", default="")
    return parser.parse_args()


def paired_summary(rows: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    logp_delta = np.asarray([row[f"{condition}_gold_logprob_delta"] for row in rows])
    numerical_delta = np.asarray(
        [
            row[f"{condition}_numerical_accuracy"]
            - row["baseline_numerical_accuracy"]
            for row in rows
        ]
    )
    f1_delta = np.asarray([row[f"{condition}_f1"] - row["baseline_f1"] for row in rows])
    return {
        "mean_gold_logprob_delta": float(logp_delta.mean()),
        "mean_numerical_accuracy_delta": float(numerical_delta.mean()),
        "mean_f1_delta": float(f1_delta.mean()),
        "improved_f1": int((f1_delta > 0).sum()),
        "same_f1": int((f1_delta == 0).sum()),
        "worse_f1": int((f1_delta < 0).sum()),
    }


def main() -> None:
    args = parse_args()
    source = json.loads(args.source_results.read_text(encoding="utf-8"))
    source_rows = list(source["results"])
    if args.discovery_n + args.evaluation_n > len(source_rows):
        raise ValueError("Discovery and evaluation sizes exceed the source result count.")
    random.Random(args.seed).shuffle(source_rows)
    discovery_rows = source_rows[: args.discovery_n]
    evaluation_rows = source_rows[
        args.discovery_n : args.discovery_n + args.evaluation_n
    ]

    corpus = load_rows(args.corpus, "id")
    questions = load_rows(args.questions, "qid")
    retrieval = load_retrieval(args.retrieval)
    client = ResearchTextClient(args.model, device="cuda", load_in_4bit=False)
    layers = [int(value) for value in args.layers.split(",") if value.strip()] or None
    extractor = QwenInternalStateExtractor(client, layers=layers)

    discovery_scores = None
    for index, row in enumerate(discovery_rows, start=1):
        prompt, gold, _, chunks, support_ids = build_oracle_case(
            row,
            questions=questions,
            corpus=corpus,
            retrieval=retrieval,
            top_k=args.top_k,
        )
        trace = extractor.extract(
            prompt, chunks, gold, max_answer_tokens=args.max_answer_tokens
        )
        fractions = support_fraction(trace, support_ids)
        discovery_scores = fractions if discovery_scores is None else discovery_scores + fractions
        print(f"[discover {index}/{len(discovery_rows)}] {row['qid']}", flush=True)
    assert discovery_scores is not None
    discovery_scores /= len(discovery_rows)
    ranking = ranked_heads(discovery_scores, extractor.layers)
    top_heads = [(row["layer"], row["head"]) for row in ranking[: args.n_heads]]
    random_heads, bottom_heads = layer_matched_controls(
        ranking,
        top_heads,
        num_heads=int(client.model.config.num_attention_heads),
        seed=args.seed + 1,
    )
    print("top heads", top_heads, flush=True)
    print("random heads", random_heads, flush=True)

    evaluated = []
    conditions = {"top_mask": top_heads, "random_mask": random_heads, "bottom_mask": bottom_heads}
    for index, row in enumerate(evaluation_rows, start=1):
        prompt, gold, gold_answers, chunks, _ = build_oracle_case(
            row,
            questions=questions,
            corpus=corpus,
            retrieval=retrieval,
            top_k=args.top_k,
        )
        baseline_trace = extractor.extract(
            prompt, chunks, gold, max_answer_tokens=args.max_answer_tokens
        )
        baseline_logprob = float(baseline_trace.target_logprob[-1].mean())
        record = {
            "qid": row["qid"],
            "question": row["question"],
            "gold": gold,
            "baseline_prediction": row["oracle_prediction"],
            "baseline_f1": float(row["oracle_f1"]),
            "baseline_em": float(row["oracle_em"]),
            "baseline_numerical_accuracy": float(row["oracle_numerical_accuracy"]),
            "baseline_gold_logprob": baseline_logprob,
        }
        for name, heads in conditions.items():
            with mask_attention_heads(client.model, heads):
                prediction = generate_answer(
                    client,
                    prompt,
                    chunks,
                    max_new_tokens=args.max_new_tokens,
                )
                trace = extractor.extract(
                    prompt,
                    chunks,
                    gold,
                    max_answer_tokens=args.max_answer_tokens,
                )
            masked_logprob = float(trace.target_logprob[-1].mean())
            record.update(
                {
                    f"{name}_prediction": prediction,
                    f"{name}_f1": token_f1(prediction, gold_answers),
                    f"{name}_em": exact_match(prediction, gold_answers),
                    f"{name}_numerical_accuracy": numerical_accuracy(prediction, gold_answers),
                    f"{name}_gold_logprob": masked_logprob,
                    f"{name}_gold_logprob_delta": masked_logprob - baseline_logprob,
                }
            )
        evaluated.append(record)
        print(
            f"[evaluate {index}/{len(evaluation_rows)}] {row['qid']} "
            f"dlogp top={record['top_mask_gold_logprob_delta']:+.3f} "
            f"random={record['random_mask_gold_logprob_delta']:+.3f} "
            f"bottom={record['bottom_mask_gold_logprob_delta']:+.3f}",
            flush=True,
        )

    top_delta = [row["top_mask_gold_logprob_delta"] for row in evaluated]
    random_delta = [row["random_mask_gold_logprob_delta"] for row in evaluated]
    comparison = np.asarray(top_delta) - np.asarray(random_delta)
    try:
        comparison_p = float(wilcoxon(comparison).pvalue)
    except ValueError:
        comparison_p = None
    output = {
        "model": args.model,
        "split_role": "calibration",
        "seed": args.seed,
        "discovery_n": len(discovery_rows),
        "evaluation_n": len(evaluation_rows),
        "n_heads_masked": args.n_heads,
        "layers": extractor.layers,
        "top_heads": [list(head) for head in top_heads],
        "random_heads": [list(head) for head in random_heads],
        "bottom_heads": [list(head) for head in bottom_heads],
        "head_ranking": ranking,
        "top_mask_summary": paired_summary(evaluated, "top_mask"),
        "random_mask_summary": paired_summary(evaluated, "random_mask"),
        "bottom_mask_summary": paired_summary(evaluated, "bottom_mask"),
        "top_minus_random_gold_logprob_delta": float(comparison.mean()),
        "top_vs_random_wilcoxon_p": comparison_p,
        "results": evaluated,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    compact = {
        key: value
        for key, value in output.items()
        if key not in {"results", "head_ranking"}
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
