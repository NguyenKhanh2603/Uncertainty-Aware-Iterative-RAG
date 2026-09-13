"""Extract full-Top-L leave-one-out causal labels with batched interventions."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from research.internal_state_rag.contrastive_saliency import _memory_efficient_attention
from research.internal_state_rag.run_causal_value_labels import atomic_write, load_existing
from research.internal_state_rag.run_tatqa_smoke import (
    ResearchTextClient,
    load_retrieval,
    load_rows,
    to_chunk,
)


def parse_args() -> argparse.Namespace:
    root = Path("data/conformal_global_run")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--plan", type=Path, required=True)
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
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-answer-tokens", type=int, default=12)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


@torch.inference_mode()
def batched_gold_logprobs(
    client: ResearchTextClient,
    prompt: str,
    contexts: list[list[Any]],
    answer_ids: list[int],
    *,
    batch_size: int,
) -> np.ndarray:
    """Score one gold answer under many contexts in memory-bounded batches."""

    sequences = []
    answer_positions = []
    answer_prefix = torch.as_tensor(answer_ids[:-1], dtype=torch.long)
    for chunks in contexts:
        alignment = client.align_and_prepare_inputs(prompt, chunks)
        prompt_ids = alignment.inputs["input_ids"][0].detach().cpu()
        sequences.append(torch.cat([prompt_ids, answer_prefix]))
        answer_positions.append(
            torch.arange(len(prompt_ids) - 1, len(prompt_ids) - 1 + len(answer_ids))
        )

    outputs = []
    pad_id = int(client.tokenizer.pad_token_id or client.tokenizer.eos_token_id)
    target_ids = torch.as_tensor(answer_ids, device=client.model.device, dtype=torch.long)
    for start in range(0, len(sequences), batch_size):
        batch_sequences = sequences[start : start + batch_size]
        batch_positions = answer_positions[start : start + batch_size]
        max_length = max(len(value) for value in batch_sequences)
        input_ids = torch.full(
            (len(batch_sequences), max_length), pad_id, dtype=torch.long
        )
        attention_mask = torch.zeros_like(input_ids)
        for row, sequence in enumerate(batch_sequences):
            input_ids[row, : len(sequence)] = sequence
            attention_mask[row, : len(sequence)] = 1
        input_ids = input_ids.to(client.model.device)
        attention_mask = attention_mask.to(client.model.device)
        positions = torch.stack(batch_positions).to(client.model.device)
        hidden = client.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        ).last_hidden_state
        batch_indices = torch.arange(len(batch_sequences), device=hidden.device)[:, None]
        answer_hidden = hidden[batch_indices, positions]
        logits = client.model.lm_head(answer_hidden).float()
        logprobs = torch.log_softmax(logits, dim=-1).gather(
            -1, target_ids[None, :, None].expand(len(batch_sequences), -1, 1)
        )
        outputs.append(logprobs.squeeze(-1).cpu().numpy())
        del hidden, answer_hidden, logits, input_ids, attention_mask
    return np.concatenate(outputs, axis=0)


def main() -> None:
    args = parse_args()
    plan_payload = json.loads(args.plan.read_text(encoding="utf-8"))
    plan = [str(qid) for qid in plan_payload["causal_teacher_qids"]]
    if args.limit > 0:
        plan = plan[: args.limit]
    input_role = str(plan_payload["input_role"])
    top_l = int(plan_payload["top_l"])
    questions = load_rows(args.questions, "qid")
    corpus = load_rows(args.corpus, "id")
    retrieval = load_retrieval(args.retrieval, split_role=input_role)
    completed = load_existing(args.output)
    completed_qids = {str(row["qid"]) for row in completed}
    unexpected = completed_qids.difference(plan)
    if unexpected:
        raise ValueError(f"Output contains qids outside requested plan: {unexpected}")
    client = ResearchTextClient(args.model, device="cuda", load_in_4bit=False)
    started = time.monotonic()

    def payload(status: str) -> dict[str, Any]:
        return {
            "status": status,
            "method": "batched_full_top_l_leave_one_chunk_out_gold_log_likelihood",
            "model": args.model,
            "plan": str(args.plan),
            "input_role": input_role,
            "source_split": str(plan_payload["source_split"]),
            "top_l": top_l,
            "requested_queries": len(plan),
            "batch_size": args.batch_size,
            "max_answer_tokens": args.max_answer_tokens,
            "completed_queries": len(completed),
            "queries": completed,
        }

    with _memory_efficient_attention(client.model):
        for query_index, qid in enumerate(plan, start=1):
            if qid in completed_qids:
                print(f"[{query_index}/{len(plan)}] resume-skip {qid}", flush=True)
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
            if not answer_ids:
                raise ValueError(f"Empty gold answer for {qid}")
            prompt = (
                "Answer using only the supplied context. Return only the short answer.\n"
                f"Question: {question['question']}"
            )
            contexts = [chunks]
            contexts.extend(chunks[:index] + chunks[index + 1 :] for index in range(top_l))
            query_started = time.monotonic()
            logprobs = batched_gold_logprobs(
                client,
                prompt,
                contexts,
                answer_ids,
                batch_size=args.batch_size,
            )
            full_token_logprobs = logprobs[0]
            chunk_rows = []
            for index, candidate in enumerate(candidates):
                drops = full_token_logprobs - logprobs[index + 1]
                chunk_rows.append(
                    {
                        "chunk_id": str(candidate["chunk_id"]),
                        "rank": int(candidate["rank"]),
                        "is_support": candidate["support_label"] == "support",
                        "bge_score": float(candidate["selection_score"]),
                        "ablated_mean_gold_logprob": float(logprobs[index + 1].mean()),
                        "gold_logprob_drop": float(drops.mean()),
                        "token_logprob_drops": drops.astype(float).tolist(),
                    }
                )
            elapsed = time.monotonic() - query_started
            support_drops = [
                row["gold_logprob_drop"] for row in chunk_rows if row["is_support"]
            ]
            completed.append(
                {
                    "qid": qid,
                    "question": str(question["question"]),
                    "gold_answer": answer,
                    "answer_token_ids": answer_ids,
                    "full_mean_gold_logprob": float(full_token_logprobs.mean()),
                    "full_token_logprobs": full_token_logprobs.astype(float).tolist(),
                    "elapsed_seconds": elapsed,
                    "chunks": chunk_rows,
                }
            )
            completed_qids.add(qid)
            atomic_write(args.output, payload("running"))
            maximum = max(support_drops) if support_drops else float("nan")
            print(
                f"[{query_index}/{len(plan)}] {qid} done {elapsed:.1f}s "
                f"support_drop_max={maximum:.4g}",
                flush=True,
            )

    output = payload("complete")
    output["elapsed_seconds_this_invocation"] = time.monotonic() - started
    atomic_write(args.output, output)
    print(json.dumps({k: v for k, v in output.items() if k != "queries"}, indent=2))


if __name__ == "__main__":
    main()
