"""Freeze disjoint causal-teacher and conformal-calibration query roles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-causal", type=int, default=128)
    parser.add_argument("--seed", type=int, default=809)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = np.load(args.features)
    qids = np.asarray([str(value) for value in data["qids"].tolist()])
    labels = data["labels"].astype(bool)
    retrievable = np.flatnonzero(labels.any(axis=1))
    if args.n_causal > len(retrievable):
        raise ValueError("Requested more causal queries than retrievable queries")
    selected_indices = np.random.default_rng(args.seed).choice(
        retrievable, size=args.n_causal, replace=False
    )
    selected = qids[selected_indices].tolist()
    selected_set = set(selected)
    remaining = [qid for qid in qids.tolist() if qid not in selected_set]
    remaining_indices = [index for index, qid in enumerate(qids) if qid not in selected_set]
    payload = {
        "status": "frozen",
        "method": "disjoint_causal_teacher_and_conformal_plan",
        "source_features": str(args.features),
        "input_role": "calibration",
        "source_split": "train",
        "top_l": int(labels.shape[1]),
        "seed": args.seed,
        "causal_teacher_queries": len(selected),
        "causal_teacher_retrievable_queries": int(
            labels[selected_indices].any(axis=1).sum()
        ),
        "conformal_queries_total": len(remaining),
        "conformal_queries_retrievable": int(
            labels[np.asarray(remaining_indices)].any(axis=1).sum()
        ),
        "overlap": len(selected_set & set(remaining)),
        "causal_teacher_qids": selected,
        "conformal_qids": remaining,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if not k.endswith("qids")}, indent=2))


if __name__ == "__main__":
    main()
