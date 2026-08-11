"""Score this project's UQ signals from a completed RAGU-baseline JSONL file.

This intentionally does not regenerate QA answers.  Its input must be produced
by run_webq_paper_baselines.py after that runner has saved sample token logs.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI

from ours_uncertainty import ClaimExtractor, ClaimNLI, Sample, cluster_samples, semantic_entropy, token_uncertainty
from run_webq_paper_baselines import auroc_incorrect, load_jsonl


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "results" / "webq_paper_baselines" / "mistral7b_seed10.jsonl"
DEFAULT_OUTPUT = ROOT / "results" / "webq_paper_baselines" / "mistral7b_seed10_with_ours.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run our claim-level UQ on saved RAGU samples")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-url", default=os.getenv("VLLM_BASE_URL"), required=os.getenv("VLLM_BASE_URL") is None)
    parser.add_argument("--api-key", default=os.getenv("VLLM_API_KEY", "EMPTY"))
    parser.add_argument("--claim-model", required=True, help="LLM used for our JSON claim extraction")
    parser.add_argument("--claim-mode", choices=("extract", "answer"), default="extract")
    parser.add_argument("--nli-model", default="cross-encoder/nli-deberta-v3-base")
    parser.add_argument("--entailment-threshold", type=float, default=0.5)
    parser.add_argument("--nli-batch-size", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.input)
    if any("sample_token_logprobs" not in row for row in rows):
        raise ValueError("Input lacks sample_token_logprobs. Re-run run_webq_paper_baselines.py with the patched version; old output cannot recover token-U.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = {str(row["q_id"]) for row in load_jsonl(args.output)} if args.resume and args.output.exists() else set()
    if not args.resume and args.output.exists():
        args.output.unlink()
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    extractor = ClaimExtractor(client, args.claim_model)
    nli = ClaimNLI(args.nli_model, args.entailment_threshold, args.nli_batch_size)

    with args.output.open("a", encoding="utf-8") as destination:
        for position, row in enumerate(rows, start=1):
            if str(row["q_id"]) in completed:
                continue
            samples = []
            for text, token_logprobs in zip(row["samples"], row["sample_token_logprobs"]):
                claims = extractor.extract(text) if args.claim_mode == "extract" else ([text] if text.strip() else [])
                samples.append(Sample(text=text, token_logprobs=token_logprobs, claims=claims))
            clusters = cluster_samples(samples, nli)
            ours_semantic = semantic_entropy(clusters, len(samples))
            result = {
                **row,
                "ours_claim_mode": args.claim_mode,
                "ours_semantic_uncertainty": ours_semantic,
                "ours_semantic_uncertainty_normalized": ours_semantic / np.log2(len(samples)),
                "ours_token_uncertainty": token_uncertainty(samples),
                "ours_num_concepts": len(clusters),
                "ours_concept_sizes": [len(cluster) for cluster in clusters],
            }
            destination.write(json.dumps(result, ensure_ascii=False) + "\n")
            destination.flush()
            print(f"[{position}/{len(rows)}] q_id={row['q_id']} concepts={len(clusters)} ours_se={ours_semantic:.3f} ours_token={result['ours_token_uncertainty']:.3f}")

    scored = load_jsonl(args.output)
    labels = [int(row["correct_acc"]) for row in scored]
    summary = {
        "examples": len(scored), "acc": float(np.mean(labels)),
        "auroc_ours_semantic_incorrect": auroc_incorrect(labels, [float(row["ours_semantic_uncertainty"]) for row in scored]),
        "auroc_ours_token_incorrect": auroc_incorrect(labels, [float(row["ours_token_uncertainty"]) for row in scored]),
        "auroc_ppl_incorrect": auroc_incorrect(labels, [float(row["ppl"]) for row in scored]),
        "auroc_regular_entropy_incorrect": auroc_incorrect(labels, [float(row["regular_entropy"]) for row in scored]),
        "auroc_semantic_entropy_incorrect": auroc_incorrect(labels, [float(row["semantic_entropy"]) for row in scored]),
    }
    path = args.output.with_suffix(".summary.json")
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote unified comparison rows to {args.output}")


if __name__ == "__main__":
    main()
