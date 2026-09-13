"""Freeze a reproducible query plan from one role in a retrieval log."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from research.internal_state_rag.run_tatqa_smoke import iter_jsonl


def load_exclusions(paths: list[Path]) -> set[str]:
    excluded: set[str] = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        excluded.update(str(qid) for qid in payload["plan"])
    return excluded


def qid_sha256(qids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(qids)).encode("utf-8")).hexdigest()


def make_plan(
    available: list[str], *, n: int, seed: int, excluded: set[str]
) -> list[str]:
    candidates = sorted(set(available) - excluded)
    if n == 0:
        return candidates
    if n < 0 or n > len(candidates):
        raise ValueError(f"Requested n={n}, but only {len(candidates)} are available")
    return random.Random(seed).sample(candidates, n)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--input-role", required=True)
    parser.add_argument("--source-split", required=True)
    parser.add_argument("--n", type=int, default=0, help="Zero selects every available query.")
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--exclude-plan", type=Path, action="append", default=[])
    args = parser.parse_args()

    role_qids = [
        str(row["qid"])
        for row in iter_jsonl(args.retrieval)
        if row.get("split_role") == args.input_role
    ]
    excluded = load_exclusions(args.exclude_plan)
    plan = make_plan(role_qids, n=args.n, seed=args.seed, excluded=excluded)
    payload = {
        "status": "planned",
        "dataset": args.dataset,
        "input_role": args.input_role,
        "source_split": args.source_split,
        "n_queries": len(plan),
        "top_l": 30,
        "seed": args.seed,
        "excluded_queries": len(excluded),
        "qid_sha256": qid_sha256(plan),
        "retrieval": str(args.retrieval),
        "plan": plan,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "plan"}, indent=2))


if __name__ == "__main__":
    main()
