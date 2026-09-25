"""Create deterministic full-test manifests for the internal-fusion column.

The manifests are derived only from the frozen retrieval logs.  They keep the
same sorted test qids and Top-30 candidates used by the cosine comparison.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path


LOGS = {
    "hotpotqa": Path("research/internal_state_rag/results/official_seed42_hotpotqa_cosine_top30.jsonl.gz"),
    "mmqa": Path("research/internal_state_rag/results/qwen2vl_jina4/mmqa_jina_v4_top30.jsonl.gz"),
    "tatqa": Path("research/internal_state_rag/results/qwen2vl_jina4/tatqa_jina_v4_top30.jsonl.gz"),
    "webqa": Path("research/internal_state_rag/results/qwen2vl_jina4/webqa_jina_v4_top30.jsonl.gz"),
}
EXPECTED_TEST_QUERIES = {"hotpotqa": 1000, "mmqa": 1000, "tatqa": 1000, "webqa": 250}


def test_qids(path: Path) -> list[str]:
    rows: dict[str, int] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("split_role")) != "test":
                continue
            qid = str(row["qid"])
            rows[qid] = rows.get(qid, 0) + 1
    if not rows or any(count != 30 for count in rows.values()):
        raise ValueError(f"{path}: expected exactly 30 test candidates per query")
    return sorted(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", default="hotpotqa,mmqa,tatqa,webqa")
    args = parser.parse_args()
    names = [name.strip() for name in args.datasets.split(",") if name.strip()]
    if not names or any(name not in LOGS for name in names):
        raise ValueError(f"datasets must be a non-empty subset of {tuple(LOGS)}")
    for dataset in names:
        plan = test_qids(LOGS[dataset])
        if len(plan) != EXPECTED_TEST_QUERIES[dataset]:
            raise ValueError(
                f"{dataset}: expected {EXPECTED_TEST_QUERIES[dataset]} test qids, got {len(plan)}"
            )
        target = args.output_dir / dataset / "manifest.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                {
                    "protocol": "frozen_top30_test_role_for_query_level_internal_fusion",
                    "dataset": dataset,
                    "input_role": "test",
                    "source_split": "frozen_test_role",
                    "top_l": 30,
                    "plan": plan,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(target)


if __name__ == "__main__":
    main()
