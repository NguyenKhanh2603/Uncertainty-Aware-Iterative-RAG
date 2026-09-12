"""Run paired analysis for downstream answers produced after conformal pruning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

COMPARISONS = {
    "fusion_a05_minus_bge_a05": ("fusion_a05", "bge_a05"),
    "fusion_a10_minus_bge_a10": ("fusion_a10", "bge_a10"),
    "bge_a05_minus_full_top30": ("bge_a05", "full_top30"),
    "fusion_a05_minus_full_top30": ("fusion_a05", "full_top30"),
    "bge_a10_minus_full_top30": ("bge_a10", "full_top30"),
    "fusion_a10_minus_full_top30": ("fusion_a10", "full_top30"),
}


def paired_bootstrap(
    delta: np.ndarray, *, seed: int, n_resamples: int
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(delta), size=(n_resamples, len(delta)))
    bootstrapped = delta[indices].mean(axis=1)
    tolerance = 1e-12
    return {
        "mean_delta": float(delta.mean()),
        "ci95": np.quantile(bootstrapped, [0.025, 0.975]).tolist(),
        "bootstrap_probability_delta_gt_zero": float(np.mean(bootstrapped > 0)),
        "wins": int(np.sum(delta > tolerance)),
        "ties": int(np.sum(np.abs(delta) <= tolerance)),
        "losses": int(np.sum(delta < -tolerance)),
    }


def analyze(
    records: list[dict[str, Any]], *, seed: int, n_resamples: int
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    bootstrap_index = 0
    for stratum in ("all", "easy", "hard"):
        selected = [
            row for row in records if stratum == "all" or row["stratum"] == stratum
        ]
        comparisons = {}
        for comparison, (left, right) in COMPARISONS.items():
            comparisons[comparison] = {}
            for metric in ("em", "f1", "numerical_accuracy"):
                delta = np.asarray(
                    [
                        row["metrics"][left][metric]
                        - row["metrics"][right][metric]
                        for row in selected
                    ]
                )
                comparisons[comparison][metric] = paired_bootstrap(
                    delta,
                    seed=seed + bootstrap_index,
                    n_resamples=n_resamples,
                )
                bootstrap_index += 1
            comparisons[comparison]["prediction_change_rate"] = float(
                np.mean(
                    [
                        row["predictions"][left] != row["predictions"][right]
                        for row in selected
                    ]
                )
            )
            comparisons[comparison]["mean_chunk_delta"] = float(
                np.mean(
                    [
                        row["contexts"][left]["n_chunks"]
                        - row["contexts"][right]["n_chunks"]
                        for row in selected
                    ]
                )
            )
            comparisons[comparison]["mean_token_delta"] = float(
                np.mean(
                    [
                        row["contexts"][left]["n_tokens"]
                        - row["contexts"][right]["n_tokens"]
                        for row in selected
                    ]
                )
            )
        output[stratum] = {"n_queries": len(selected), "comparisons": comparisons}
    return output


def parse_args() -> argparse.Namespace:
    root = Path("research/internal_state_rag/results/full_top30_n420_seed101")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, default=root / "pruned_answers.jsonl")
    parser.add_argument(
        "--output", type=Path, default=root / "pruned_answer_analysis.json"
    )
    parser.add_argument("--seed", type=int, default=8128)
    parser.add_argument("--n-resamples", type=int, default=20_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = [
        json.loads(line)
        for line in args.rows.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    output = {
        "n_queries": len(records),
        "seed": args.seed,
        "n_resamples": args.n_resamples,
        "paired_analysis": analyze(
            records, seed=args.seed, n_resamples=args.n_resamples
        ),
    }
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
