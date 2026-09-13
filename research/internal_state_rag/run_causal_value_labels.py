"""Measure each retrieved chunk's causal value for the gold answer.

For every query, this script scores the gold answer under the full Top-L context
and under L leave-one-chunk-out contexts.  A positive ``gold_logprob_drop``
means removing the chunk lowered the gold answer likelihood.  The output is
incremental and can be resumed after interruption.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from research.internal_state_rag.contrastive_saliency import _memory_efficient_attention
from research.internal_state_rag.directer_plausibility import teacher_forced_final_logits
from research.internal_state_rag.run_tatqa_smoke import (
    ResearchTextClient,
    load_retrieval,
    load_rows,
    to_chunk,
)
from research.internal_state_rag.signals import QwenInternalStateExtractor


def parse_args() -> argparse.Namespace:
    root = Path("data/conformal_global_run")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--max-answer-tokens", type=int, default=16)
    return parser.parse_args()


def answer_logprobs(logits: torch.Tensor, answer_ids: list[int]) -> np.ndarray:
    """Return target-token log probabilities from teacher-forced logits."""

    if logits.shape[0] != len(answer_ids):
        raise ValueError("logit and answer-token lengths differ")
    targets = torch.as_tensor(answer_ids, dtype=torch.long)[:, None]
    return (
        torch.log_softmax(logits.float(), dim=-1)
        .gather(-1, targets)
        .squeeze(-1)
        .cpu()
        .numpy()
    )


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def load_existing(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("queries", []))


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.feature_manifest.read_text(encoding="utf-8"))
    full_plan = [str(qid) for qid in manifest["plan"]]
    if args.start < 0 or args.n < 1 or args.start + args.n > len(full_plan):
        raise ValueError("Requested slice lies outside the feature manifest plan")
    plan = full_plan[args.start : args.start + args.n]
    input_role = str(manifest["input_role"])
    top_l = int(manifest["top_l"])

    questions = load_rows(args.questions, "qid")
    corpus = load_rows(args.corpus, "id")
    retrieval = load_retrieval(args.retrieval, split_role=input_role)
    completed = load_existing(args.output)
    completed_qids = {str(row["qid"]) for row in completed}
    unexpected = completed_qids.difference(plan)
    if unexpected:
        raise ValueError(f"Output contains qids outside requested plan: {unexpected}")

    client = ResearchTextClient(args.model, device="cuda", load_in_4bit=False)
    extractor = QwenInternalStateExtractor(
        client, layers=[int(client.model.config.num_hidden_layers) - 1]
    )
    started = time.monotonic()

    def payload(status: str) -> dict[str, Any]:
        return {
            "status": status,
            "method": "full_top_l_leave_one_chunk_out_gold_log_likelihood",
            "model": args.model,
            "feature_manifest": str(args.feature_manifest),
            "input_role": input_role,
            "source_split": str(manifest["source_split"]),
            "top_l": top_l,
            "start": args.start,
            "requested_queries": args.n,
            "max_answer_tokens": args.max_answer_tokens,
            "completed_queries": len(completed),
            "queries": completed,
        }

    with _memory_efficient_attention(client.model):
        for plan_index, qid in enumerate(plan, start=1):
            if qid in completed_qids:
                print(f"[{plan_index}/{len(plan)}] resume-skip {qid}", flush=True)
                continue
            question = questions[qid]
            candidates = retrieval[qid][:top_l]
            if len(candidates) != top_l:
                raise ValueError(f"{qid} has {len(candidates)} candidates, expected {top_l}")
            chunks = [to_chunk(row, corpus) for row in candidates]
            answer = str(question["gold_answers"][0])
            answer_ids = client.tokenizer.encode(answer, add_special_tokens=False)[
                : args.max_answer_tokens
            ]
            prompt = (
                "Answer using only the supplied context. Return only the short answer.\n"
                f"Question: {question['question']}"
            )

            query_started = time.monotonic()
            full_logits = teacher_forced_final_logits(
                extractor,
                prompt,
                chunks,
                answer,
                max_answer_tokens=args.max_answer_tokens,
            )
            full_token_logprobs = answer_logprobs(full_logits, answer_ids)
            full_mean = float(full_token_logprobs.mean())
            chunk_rows = []
            for chunk_index, (candidate, chunk) in enumerate(
                zip(candidates, chunks, strict=True), start=1
            ):
                ablated_chunks = chunks[: chunk_index - 1] + chunks[chunk_index:]
                ablated_logits = teacher_forced_final_logits(
                    extractor,
                    prompt,
                    ablated_chunks,
                    answer,
                    max_answer_tokens=args.max_answer_tokens,
                )
                ablated_token_logprobs = answer_logprobs(ablated_logits, answer_ids)
                token_drops = full_token_logprobs - ablated_token_logprobs
                chunk_rows.append(
                    {
                        "chunk_id": str(chunk.id),
                        "rank": int(candidate["rank"]),
                        "is_support": candidate["support_label"] == "support",
                        "bge_score": float(candidate["selection_score"]),
                        "ablated_mean_gold_logprob": float(ablated_token_logprobs.mean()),
                        "gold_logprob_drop": float(token_drops.mean()),
                        "token_logprob_drops": token_drops.astype(float).tolist(),
                    }
                )
                if chunk_index % 5 == 0 or chunk_index == top_l:
                    print(
                        f"[{plan_index}/{len(plan)}] {qid} ablations={chunk_index}/{top_l}",
                        flush=True,
                    )

            elapsed = time.monotonic() - query_started
            completed.append(
                {
                    "qid": qid,
                    "question": str(question["question"]),
                    "gold_answer": answer,
                    "answer_token_ids": answer_ids,
                    "full_mean_gold_logprob": full_mean,
                    "full_token_logprobs": full_token_logprobs.astype(float).tolist(),
                    "elapsed_seconds": elapsed,
                    "chunks": chunk_rows,
                }
            )
            completed_qids.add(qid)
            atomic_write(args.output, payload("running"))
            support_drops = [
                row["gold_logprob_drop"] for row in chunk_rows if row["is_support"]
            ]
            print(
                f"[{plan_index}/{len(plan)}] {qid} done {elapsed:.1f}s "
                f"support_drop_max={max(support_drops):.4g}",
                flush=True,
            )

    output = payload("complete")
    output["elapsed_seconds_this_invocation"] = time.monotonic() - started
    atomic_write(args.output, output)
    print(
        json.dumps(
            {
                key: value
                for key, value in output.items()
                if key not in {"queries"}
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
