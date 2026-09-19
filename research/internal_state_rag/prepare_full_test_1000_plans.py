"""Write deterministic manifests for every frozen Top-30 test query."""
from __future__ import annotations
import gzip, json
from pathlib import Path

ROOT = Path("research/internal_state_rag/results/qwen2vl_jina4")
OUT = Path("research/internal_state_rag/results/qwen2vl_7b_jina4/full_test_1000_plans")

for dataset in ("hotpotqa", "mmqa", "tatqa"):
    qids = set()
    with gzip.open(ROOT / f"{dataset}_jina_v4_top30.jsonl.gz", "rt") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("split_role") == "test":
                qids.add(str(row["qid"]))
    plan = sorted(qids)
    if len(plan) != 1000:
        raise ValueError(f"{dataset}: expected 1000 frozen test qids, found {len(plan)}")
    target = OUT / dataset / "manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"protocol": "fixed_frozen_jina_v4_top30_test_role", "dataset": dataset, "input_role": "test", "top_l": 30, "plan": plan}, indent=2) + "\n")
    print(target)
