"""Run answer-conditioned causal, hidden-state, and head-screening pilots.

The existing pairwise features ask Qwen2-VL whether an isolated candidate looks
useful.  This pilot asks a different question: how much does the model's
teacher-forced gold-answer trajectory change when one retrieved chunk is
removed?  It records three signals on disjoint development discovery/evaluation
queries:

* ``gold_logprob_drop``: answer-conditioned leave-one-chunk-out value;
* ``hidden_delta_*``: residual-state change caused by that removal; and
* attention-only head screening followed by causal masking of the screened
  heads, with random and bottom-head controls.

The run is deliberately a development pilot.  The discovery split is used only
to choose heads; the evaluation split is used only for the causal mask check.
No model or probe weights are trained.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from research.internal_state_rag.causal import mask_attention_heads
from research.internal_state_rag.contrastive_saliency import _memory_efficient_attention
from research.internal_state_rag.directer_plausibility import teacher_forced_final_logits
from research.internal_state_rag.run_causal_value_labels import atomic_write
from research.internal_state_rag.run_tatqa_smoke import ResearchTextClient
from research.internal_state_rag.signals import QwenInternalStateExtractor
from uncertainty_rag.modality.base import ContextChunk


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_keyed(path: Path, key: str) -> dict[str, dict[str, Any]]:
    return {str(row[key]): row for row in iter_jsonl(path)}


def load_retrieval(path: Path, role: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in iter_jsonl(path):
        if str(row.get("split_role")) == role:
            grouped[str(row["qid"])].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: (int(row["rank"]), str(row["chunk_id"])))
    return grouped


def to_chunk(row: dict[str, Any], corpus: dict[str, dict[str, Any]]) -> ContextChunk:
    source = corpus[str(row["chunk_id"])]
    return ContextChunk(
        id=str(source["id"]),
        content=source["content"],
        modality=str(source.get("modality", row.get("modality", "text"))),
        metadata={
            "rank": int(row["rank"]),
            "is_support": row.get("support_label") == "support",
        },
    )


def load_feature_scores(path: Path | None) -> dict[tuple[str, str], float]:
    """Map feature-manifest reranker scores back to raw retrieval rows."""

    if path is None:
        return {}
    values = np.load(path, allow_pickle=False)
    qids = values["qids"].astype(str)
    chunk_ids = values["chunk_ids"].astype(str)
    scores = values["reranker_scores"].astype(np.float64)
    return {
        (str(qids[q_index]), str(chunk_ids[q_index, c_index])): float(
            scores[q_index, c_index]
        )
        for q_index in range(len(qids))
        for c_index in range(chunk_ids.shape[1])
        if np.isfinite(scores[q_index, c_index])
    }


def answer_logprobs(logits: torch.Tensor, answer_ids: Sequence[int]) -> np.ndarray:
    if logits.ndim != 2 or logits.shape[0] != len(answer_ids):
        raise ValueError("logits must have shape [answer_tokens, vocabulary]")
    target = torch.as_tensor(answer_ids, dtype=torch.long, device=logits.device)[:, None]
    return (
        torch.log_softmax(logits.float(), dim=-1)
        .gather(-1, target)
        .squeeze(-1)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )


def row_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.sum(left * right, axis=1, dtype=np.float64)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator > 0,
    )


def head_screen_score(trace: Any, support_mask: np.ndarray) -> np.ndarray:
    """Return per-layer/head support-vs-distractor attention contrast."""

    mass = np.asarray(trace.attention_mass, dtype=np.float64)
    if mass.ndim != 3 or mass.shape[-1] != len(support_mask):
        raise ValueError("attention mass and support mask have incompatible shapes")
    support = np.flatnonzero(support_mask)
    distractors = np.flatnonzero(~support_mask)
    score = np.full(mass.shape[:2], np.nan, dtype=np.float64)
    if not len(support) or not len(distractors):
        return score
    total = mass.sum(axis=-1, keepdims=True)
    normalized = np.divide(mass, total, out=np.zeros_like(mass), where=total > 0)
    score[:] = normalized[:, :, support].mean(axis=-1) - normalized[:, :, distractors].mean(
        axis=-1
    )
    return score


def candidate_attention_fraction(trace: Any) -> np.ndarray:
    """Attention mass per candidate, normalized independently per head."""

    mass = np.asarray(trace.attention_mass, dtype=np.float64)
    total = mass.sum(axis=-1, keepdims=True)
    normalized = np.divide(mass, total, out=np.zeros_like(mass), where=total > 0)
    return normalized.mean(axis=(0, 1))


def layer_matched_controls(
    ranking: list[dict[str, Any]],
    top_heads: list[tuple[int, int]],
    *,
    num_heads: int,
    seed: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Build random and bottom-head controls with the same layer counts."""

    score = {
        (int(row["layer"]), int(row["head"])): float(row["score"])
        for row in ranking
    }
    top_set = set(top_heads)
    counts: dict[int, int] = defaultdict(int)
    for layer, _ in top_heads:
        counts[int(layer)] += 1
    generator = random.Random(seed)
    random_heads: list[tuple[int, int]] = []
    bottom_heads: list[tuple[int, int]] = []
    for layer, count in sorted(counts.items()):
        available = [(layer, head) for head in range(num_heads) if (layer, head) not in top_set]
        if len(available) < count:
            raise ValueError(f"Layer {layer} has too few control heads")
        random_heads.extend(generator.sample(available, count))
        bottom_heads.extend(sorted(available, key=lambda item: (score[item], item[1]))[:count])
    return random_heads, bottom_heads


def rank_metrics(records: list[dict[str, Any]], score_key: str) -> dict[str, Any]:
    """Compute query AP/MRR on candidates for which a signal is available."""

    aps: list[float] = []
    mrr: list[float] = []
    top1: list[float] = []
    scored_queries = 0
    scored_candidates = 0
    for record in records:
        candidates = [
            row
            for row in record["chunks"]
            if row.get("is_support") is not None
            and isinstance(row.get(score_key), (int, float))
            and math.isfinite(float(row[score_key]))
        ]
        if not candidates:
            continue
        scored_queries += 1
        scored_candidates += len(candidates)
        ordered = sorted(candidates, key=lambda row: (-float(row[score_key]), int(row["rank"])))
        positions = [index + 1 for index, row in enumerate(ordered) if row["is_support"]]
        if not positions:
            continue
        aps.append(float(np.mean([(index + 1) / position for index, position in enumerate(positions)])))
        mrr.append(float(1.0 / positions[0]))
        top1.append(float(positions[0] == 1))
    return {
        "mean_query_ap": float(np.mean(aps)) if aps else None,
        "mrr": float(np.mean(mrr)) if mrr else None,
        "top1_support_rate": float(np.mean(top1)) if top1 else None,
        "scored_queries": scored_queries,
        "scored_candidates": scored_candidates,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--feature-npz", type=Path, default=None)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--discovery-n", type=int, default=4)
    parser.add_argument("--loo-candidates", type=int, default=30)
    parser.add_argument("--hidden-candidates", type=int, default=8)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--max-answer-tokens", type=int, default=8)
    parser.add_argument("--layers", default="3,7,11,15,19,23,27")
    parser.add_argument("--seed", type=int, default=20260914)
    return parser.parse_args()


def load_existing(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        return list(json.loads(path.read_text(encoding="utf-8")).get("queries", []))
    except (OSError, json.JSONDecodeError):
        return []


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.feature_manifest.read_text(encoding="utf-8"))
    full_plan = [str(value) for value in manifest["plan"]]
    if args.start < 0 or args.n < 1 or args.start + args.n > len(full_plan):
        raise ValueError("Requested slice lies outside the feature manifest plan")
    plan = full_plan[args.start : args.start + args.n]
    discovery_n = max(0, min(int(args.discovery_n), len(plan)))
    discovery_qids = set(plan[:discovery_n])
    evaluation_qids = set(plan[discovery_n:])
    input_role = str(manifest["input_role"])
    top_l = int(manifest["top_l"])
    layers = [int(value) for value in args.layers.split(",") if value.strip()]

    questions = load_keyed(args.questions, "qid")
    corpus = load_keyed(args.corpus, "id")
    retrieval = load_retrieval(args.retrieval, input_role)
    reranker_scores = load_feature_scores(args.feature_npz)
    completed = load_existing(args.output)
    completed_by_qid = {str(row["qid"]): row for row in completed}
    unexpected = set(completed_by_qid).difference(plan)
    if unexpected:
        raise ValueError(f"Output contains qids outside requested plan: {sorted(unexpected)}")

    client = ResearchTextClient(args.model, device="cuda", load_in_4bit=False)
    extractor = QwenInternalStateExtractor(client, layers=layers)
    started = time.monotonic()
    loo_limit = top_l if args.loo_candidates <= 0 else min(top_l, int(args.loo_candidates))
    hidden_limit = min(loo_limit, max(0, int(args.hidden_candidates)))

    def payload(status: str, *, head_screening: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "status": status,
            "method": "qwen2vl_answer_conditioned_causal_hidden_head_pilot",
            "model": args.model,
            "feature_manifest": str(args.feature_manifest),
            "input_role": input_role,
            "top_l": top_l,
            "start": args.start,
            "requested_queries": len(plan),
            "discovery_queries": len(discovery_qids),
            "evaluation_queries": len(evaluation_qids),
            "loo_candidates_per_query": loo_limit,
            "hidden_candidates_per_query": hidden_limit,
            "max_answer_tokens": args.max_answer_tokens,
            "layers": extractor.layers,
            "completed_queries": len(completed_by_qid),
            "head_screening": head_screening,
            "queries": [completed_by_qid[qid] for qid in plan if qid in completed_by_qid],
        }

    with _memory_efficient_attention(client.model):
        for plan_index, qid in enumerate(plan, start=1):
            if qid in completed_by_qid:
                print(f"[{plan_index}/{len(plan)}] resume-skip {qid}", flush=True)
                continue
            question = questions.get(qid)
            candidates = retrieval.get(qid, [])[:top_l]
            if question is None or len(question.get("gold_answers", [])) == 0:
                print(f"[{plan_index}/{len(plan)}] skip {qid}: no gold answer", flush=True)
                continue
            if len(candidates) != top_l:
                raise ValueError(f"{qid} has {len(candidates)} candidates; expected {top_l}")
            chunks = [to_chunk(row, corpus) for row in candidates]
            prompt = (
                "Answer using only the supplied context. Return only the short answer.\n"
                f"Question: {question['question']}"
            )
            answer = str(question["gold_answers"][0])
            answer_ids = client.tokenizer.encode(answer, add_special_tokens=False)[
                : args.max_answer_tokens
            ]
            if not answer_ids:
                print(f"[{plan_index}/{len(plan)}] skip {qid}: empty answer tokens", flush=True)
                continue

            query_started = time.monotonic()
            full_trace = extractor.extract(
                prompt, chunks, answer, max_answer_tokens=args.max_answer_tokens
            )
            full_logprobs = np.asarray(full_trace.target_logprob[-1], dtype=np.float64)
            support_mask = np.asarray(
                [row.get("support_label") == "support" for row in candidates], dtype=bool
            )
            head_scores = head_screen_score(full_trace, support_mask)
            attention_fractions = candidate_attention_fraction(full_trace)

            hidden_indices = set(range(hidden_limit))
            chunk_rows: list[dict[str, Any]] = []
            for candidate_index, candidate in enumerate(candidates):
                row: dict[str, Any] = {
                    "chunk_id": str(candidate["chunk_id"]),
                    "rank": int(candidate["rank"]),
                    "modality": str(candidate.get("modality", "text")),
                    "is_support": bool(support_mask[candidate_index]),
                    "cosine_score": float(candidate.get("cosine_score", np.nan)),
                    "jina_reranker_score": float(
                        reranker_scores.get((qid, str(candidate["chunk_id"])), np.nan)
                    ),
                    "attention_fraction": float(attention_fractions[candidate_index]),
                    "loo_computed": candidate_index < loo_limit,
                    "hidden_delta_computed": candidate_index in hidden_indices,
                }
                if candidate_index < loo_limit:
                    ablated_chunks = chunks[:candidate_index] + chunks[candidate_index + 1 :]
                    if candidate_index in hidden_indices:
                        ablated_trace = extractor.extract(
                            prompt,
                            ablated_chunks,
                            answer,
                            max_answer_tokens=args.max_answer_tokens,
                        )
                        ablated_logprobs = np.asarray(
                            ablated_trace.target_logprob[-1], dtype=np.float64
                        )
                        residual_delta = full_trace.residual_mean - ablated_trace.residual_mean
                        cosine = row_cosine(full_trace.residual_mean, ablated_trace.residual_mean)
                        row["hidden_delta_l2_by_layer"] = np.linalg.norm(
                            residual_delta, axis=1
                        ).astype(float).tolist()
                        row["hidden_delta_cosine_distance_by_layer"] = (1.0 - cosine).astype(
                            float
                        ).tolist()
                        row["hidden_delta_l2_mean"] = float(
                            np.linalg.norm(residual_delta, axis=1).mean()
                        )
                        row["hidden_delta_cosine_distance_mean"] = float((1.0 - cosine).mean())
                        row["ablated_mean_target_logprob_by_layer"] = (
                            ablated_trace.target_logprob.mean(axis=1).astype(float).tolist()
                        )
                    else:
                        ablated_logits = teacher_forced_final_logits(
                            extractor,
                            prompt,
                            ablated_chunks,
                            answer,
                            max_answer_tokens=args.max_answer_tokens,
                        )
                        ablated_logprobs = answer_logprobs(ablated_logits, answer_ids)
                    token_drops = full_logprobs - ablated_logprobs
                    row["ablated_mean_gold_logprob"] = float(ablated_logprobs.mean())
                    row["gold_logprob_drop"] = float(token_drops.mean())
                    row["token_logprob_drops"] = token_drops.astype(float).tolist()
                else:
                    row["ablated_mean_gold_logprob"] = None
                    row["gold_logprob_drop"] = None
                    row["token_logprob_drops"] = None
                    for key in (
                        "hidden_delta_l2_by_layer",
                        "hidden_delta_cosine_distance_by_layer",
                        "hidden_delta_l2_mean",
                        "hidden_delta_cosine_distance_mean",
                        "ablated_mean_target_logprob_by_layer",
                    ):
                        row[key] = None
                chunk_rows.append(row)

            record = {
                "qid": qid,
                "query_split": "discovery" if qid in discovery_qids else "evaluation",
                "question": str(question["question"]),
                "gold_answer": answer,
                "answer_token_ids": [int(value) for value in answer_ids],
                "full_mean_gold_logprob": float(full_logprobs.mean()),
                "full_token_logprobs": full_logprobs.astype(float).tolist(),
                "full_mean_target_logprob_by_layer": full_trace.target_logprob.mean(axis=1).astype(float).tolist(),
                "full_mean_target_margin_by_layer": full_trace.target_margin.mean(axis=1).astype(float).tolist(),
                "full_mean_entropy_by_layer": full_trace.entropy.mean(axis=1).astype(float).tolist(),
                "full_residual_norm_by_layer": np.linalg.norm(
                    full_trace.residual_mean, axis=1
                ).astype(float).tolist(),
                "head_screen_score_by_layer_head": head_scores.astype(float).tolist(),
                "chunks": chunk_rows,
                "elapsed_seconds": round(time.monotonic() - query_started, 3),
            }
            completed_by_qid[qid] = record
            atomic_write(args.output, payload("running"))
            print(
                f"[{plan_index}/{len(plan)}] {qid} done {record['elapsed_seconds']:.1f}s "
                f"loo={loo_limit} hidden={hidden_limit}",
                flush=True,
            )

    usable_records = [completed_by_qid[qid] for qid in plan if qid in completed_by_qid]
    discovery_records = [row for row in usable_records if row["qid"] in discovery_qids]
    evaluation_records = [row for row in usable_records if row["qid"] in evaluation_qids]
    score_arrays = []
    for record in discovery_records:
        values = np.asarray(record["head_screen_score_by_layer_head"], dtype=np.float64)
        score_arrays.append(values)
    if score_arrays:
        aggregate_head_scores = np.nanmean(np.stack(score_arrays, axis=0), axis=0)
    else:
        aggregate_head_scores = np.full((len(extractor.layers), int(client.model.config.num_attention_heads)), np.nan)
    ranking = [
        {
            "layer": int(layer),
            "head": int(head),
            "score": float(aggregate_head_scores[layer_index, head]),
        }
        for layer_index, layer in enumerate(extractor.layers)
        for head in range(int(client.model.config.num_attention_heads))
        if np.isfinite(aggregate_head_scores[layer_index, head])
    ]
    ranking.sort(key=lambda row: (-row["score"], row["layer"], row["head"]))
    top_heads = [(row["layer"], row["head"]) for row in ranking[: max(0, int(args.n_heads))]]
    random_heads, bottom_heads = layer_matched_controls(
        ranking,
        top_heads,
        num_heads=int(client.model.config.num_attention_heads),
        seed=args.seed,
    ) if top_heads else ([], [])

    conditions = {
        "top_mask": top_heads,
        "random_mask": random_heads,
        "bottom_mask": bottom_heads,
    }
    for record in evaluation_records:
        qid = str(record["qid"])
        question = questions[qid]
        candidates = retrieval[qid][:top_l]
        chunks = [to_chunk(row, corpus) for row in candidates]
        prompt = (
            "Answer using only the supplied context. Return only the short answer.\n"
            f"Question: {question['question']}"
        )
        answer = str(question["gold_answers"][0])
        condition_results: dict[str, Any] = {}
        for name, heads in conditions.items():
            if not heads:
                condition_results[name] = {"heads": [], "masked_mean_gold_logprob": None, "gold_logprob_delta": None}
                continue
            with mask_attention_heads(client.model, heads):
                masked_logits = teacher_forced_final_logits(
                    extractor,
                    prompt,
                    chunks,
                    answer,
                    max_answer_tokens=args.max_answer_tokens,
                )
            masked_lp = answer_logprobs(masked_logits, record["answer_token_ids"])
            condition_results[name] = {
                "heads": [[int(layer), int(head)] for layer, head in heads],
                "masked_mean_gold_logprob": float(masked_lp.mean()),
                "gold_logprob_delta": float(masked_lp.mean() - record["full_mean_gold_logprob"]),
                "token_logprob_delta": (masked_lp - np.asarray(record["full_token_logprobs"])).astype(float).tolist(),
            }
        record["causal_head_masking"] = condition_results
        atomic_write(
            args.output,
            payload(
                "running",
                head_screening={
                    "layers": extractor.layers,
                    "num_heads": int(client.model.config.num_attention_heads),
                    "discovery_queries_used": len(discovery_records),
                    "ranking": ranking,
                    "top_heads": [[int(layer), int(head)] for layer, head in top_heads],
                    "random_heads": [[int(layer), int(head)] for layer, head in random_heads],
                    "bottom_heads": [[int(layer), int(head)] for layer, head in bottom_heads],
                },
            ),
        )
        print(f"[mask] {qid} top={condition_results['top_mask']['gold_logprob_delta']}", flush=True)

    ranking_metrics = {}
    for key in (
        "cosine_score",
        "jina_reranker_score",
        "attention_fraction",
        "gold_logprob_drop",
        "hidden_delta_l2_mean",
        "hidden_delta_cosine_distance_mean",
    ):
        ranking_metrics[key] = {
            "discovery": rank_metrics(discovery_records, key),
            "evaluation": rank_metrics(evaluation_records, key),
        }
    causal_summary: dict[str, Any] = {}
    for name in conditions:
        values = [
            float(row["causal_head_masking"][name]["gold_logprob_delta"])
            for row in evaluation_records
            if row.get("causal_head_masking", {}).get(name, {}).get("gold_logprob_delta") is not None
        ]
        causal_summary[name] = {
            "n": len(values),
            "mean_gold_logprob_delta": float(np.mean(values)) if values else None,
            "median_gold_logprob_delta": float(np.median(values)) if values else None,
            "positive_delta_rate": float(np.mean(np.asarray(values) > 0)) if values else None,
        }

    head_screening = {
        "layers": extractor.layers,
        "num_heads": int(client.model.config.num_attention_heads),
        "discovery_queries_used": len(discovery_records),
        "ranking": ranking,
        "top_heads": [[int(layer), int(head)] for layer, head in top_heads],
        "random_heads": [[int(layer), int(head)] for layer, head in random_heads],
        "bottom_heads": [[int(layer), int(head)] for layer, head in bottom_heads],
    }
    output = payload("complete", head_screening=head_screening)
    output.update(
        {
            "elapsed_seconds_this_invocation": round(time.monotonic() - started, 3),
            "ranking_metrics": ranking_metrics,
            "causal_head_masking_summary": causal_summary,
        }
    )
    atomic_write(args.output, output)
    printable = {key: value for key, value in output.items() if key != "queries"}
    print(json.dumps(printable, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
