"""Make two disjoint additional three-way plans for seed-variance evaluation."""
from __future__ import annotations

import argparse
import gzip
import json
import random
from pathlib import Path


CONFIG = {
    "hotpotqa": {"probe_role": "development", "probe_split": "train", "cal_split": "train", "test_split": "validation"},
    "mmqa": {"probe_role": "development", "probe_split": "train", "cal_split": "train", "test_split": "dev"},
    "tatqa": {"probe_role": "development", "probe_split": "train", "cal_split": "train", "test_split": "dev"},
}


def retrieval_qids(path: Path) -> dict[str, list[str]]:
    values: dict[str, set[str]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            values.setdefault(str(row["split_role"]), set()).add(str(row["qid"]))
    return {role: sorted(qids) for role, qids in values.items()}


def old_plan(root: Path, dataset: str, part: str) -> set[str]:
    path = root / dataset / part / "manifest.json"
    return set(json.loads(path.read_text(encoding="utf-8"))["plan"])


def manifest(role: str, source_split: str, plan: list[str], seed: int) -> dict[str, object]:
    return {
        "status": "complete",
        "method": "three_seed_disjoint_feature_plan",
        "input_role": role,
        "source_split": source_split,
        "top_l": 30,
        "seed": seed,
        "n_queries": len(plan),
        "plan": plan,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, default=Path("research/internal_state_rag/results/qwen2vl_7b_jina4/fusion_features"))
    parser.add_argument("--retrieval-root", type=Path, default=Path("research/internal_state_rag/results/qwen2vl_jina4"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", default="2027,2028")
    parser.add_argument("--n", type=int, default=100)
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",")]
    if len(seeds) != 2 or args.n < 1:
        raise ValueError("This runner expects exactly two positive additional seeds")
    for dataset, config in CONFIG.items():
        qids = retrieval_qids(args.retrieval_root / f"{dataset}_jina_v4_top30.jsonl.gz")
        pools = {
            "probe_train": [qid for qid in qids[config["probe_role"]] if qid not in old_plan(args.feature_root, dataset, "probe_train")],
            "calibration": [qid for qid in qids["calibration"] if qid not in old_plan(args.feature_root, dataset, "calibration")],
            "test": [qid for qid in qids["test"] if qid not in old_plan(args.feature_root, dataset, "test")],
        }
        selected: dict[int, dict[str, list[str]]] = {}
        for part, pool in pools.items():
            if len(pool) < args.n * len(seeds):
                raise ValueError(f"{dataset}/{part}: only {len(pool)} qids available")
            sampled = random.Random(10_000 + sum(ord(ch) for ch in dataset) + len(part)).sample(pool, args.n * len(seeds))
            for index, seed in enumerate(seeds):
                selected.setdefault(seed, {})[part] = sorted(sampled[index * args.n : (index + 1) * args.n])
        for seed, parts in selected.items():
            specifications = {
                "probe_train": (config["probe_role"], config["probe_split"]),
                "calibration": ("calibration", config["cal_split"]),
                "test": ("test", config["test_split"]),
            }
            for part, plan in parts.items():
                role, source_split = specifications[part]
                output = args.output_root / f"seed_{seed}" / dataset / part / "manifest.json"
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps(manifest(role, source_split, plan, seed), indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "seeds": seeds, "datasets": list(CONFIG), "queries_per_part": args.n}, indent=2))


if __name__ == "__main__":
    main()
