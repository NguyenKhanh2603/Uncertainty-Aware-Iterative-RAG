"""Materialize the inverse 1,000-calibration / 100-held-out-test protocol.

The prior four-dataset cosine report used 100 calibration queries and up to
1,000 test queries.  This script inverts the evaluation scale without using a
query twice: each dataset gets its full 1,000-query non-test pool for
calibration and the first 100 lexicographically sorted frozen test qids for
held-out evaluation.  The source roles are preserved in the manifests so the
selection experiment is reproducible and auditable.
"""
from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


SOURCES = {
    "hotpotqa": Path("research/internal_state_rag/results/official_seed42_hotpotqa_cosine_top30.jsonl.gz"),
    "mmqa": Path("research/internal_state_rag/results/qwen2vl_jina4/mmqa_jina_v4_top30.jsonl.gz"),
    "tatqa": Path("research/internal_state_rag/results/qwen2vl_jina4/tatqa_jina_v4_top30.jsonl.gz"),
    "webqa": Path("research/internal_state_rag/results/qwen2vl_jina4/webqa_jina_v4_top30.jsonl.gz"),
}
# HotpotQA's official-seed log exposes exactly 1,000 calibration rows.  The
# remaining logs divide their 1,000 non-test rows between calibration and
# development; both are available before any held-out test qid is touched.
CALIBRATION_ROLES = {
    "hotpotqa": ("calibration",),
    "mmqa": ("calibration", "development"),
    "tatqa": ("calibration", "development"),
    "webqa": ("calibration", "development"),
}


def qids_by_role(path: Path) -> dict[str, dict[str, int]]:
    """Return per-role qid row counts, validating one Top-30 group per qid."""

    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row: dict[str, Any] = json.loads(line)
                counts[str(row["split_role"])][str(row["qid"])] += 1
    result = {role: dict(values) for role, values in counts.items()}
    if any(count != 30 for values in result.values() for count in values.values()):
        raise ValueError(f"{path}: each planned qid must have exactly 30 candidates")
    return result


def manifest(
    *,
    dataset: str,
    role: str,
    qids: list[str],
    source_roles: tuple[str, ...],
    source: Path,
) -> dict[str, Any]:
    return {
        "protocol": "inverse_1000_calibration_100_heldout_test_2026_09_26",
        "dataset": dataset,
        "role": role,
        "n_queries": len(qids),
        "source_retrieval": str(source),
        "source_roles": list(source_roles),
        "selection": "all_source_qids" if role == "calibration" else "first_100_lexicographically_sorted_test_qids",
        "top_l": 30,
        "plan": qids,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", default=",".join(SOURCES))
    args = parser.parse_args()
    datasets = tuple(value.strip() for value in args.datasets.split(",") if value.strip())
    if not datasets or set(datasets).difference(SOURCES):
        raise ValueError(f"--datasets must be a non-empty subset of {tuple(SOURCES)}")

    audit: dict[str, Any] = {
        "protocol": "inverse_1000_calibration_100_heldout_test_2026_09_26",
        "top_l": 30,
        "datasets": {},
    }
    for dataset in datasets:
        source = SOURCES[dataset]
        counts = qids_by_role(source)
        calibration_roles = CALIBRATION_ROLES[dataset]
        calibration_qids = sorted(
            qid for role in calibration_roles for qid in counts.get(role, {})
        )
        test_qids = sorted(counts.get("test", {}))[:100]
        if len(calibration_qids) != 1000:
            raise ValueError(f"{dataset}: expected 1,000 calibration qids, got {len(calibration_qids)}")
        if len(test_qids) != 100:
            raise ValueError(f"{dataset}: expected at least 100 test qids, got {len(test_qids)}")
        overlap = set(calibration_qids).intersection(test_qids)
        if overlap:
            raise ValueError(f"{dataset}: calibration/test overlap: {sorted(overlap)[:3]}")
        target = args.output_dir / dataset
        target.mkdir(parents=True, exist_ok=True)
        (target / "calibration_manifest.json").write_text(
            json.dumps(
                manifest(
                    dataset=dataset,
                    role="calibration",
                    qids=calibration_qids,
                    source_roles=calibration_roles,
                    source=source,
                ),
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (target / "test_manifest.json").write_text(
            json.dumps(
                manifest(
                    dataset=dataset,
                    role="test",
                    qids=test_qids,
                    source_roles=("test",),
                    source=source,
                ),
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        audit["datasets"][dataset] = {
            "calibration_queries": len(calibration_qids),
            "test_queries": len(test_qids),
            "overlap_count": 0,
            "source_role_counts": {role: len(qids) for role, qids in counts.items()},
        }
        print(dataset, len(calibration_qids), len(test_qids))
    (args.output_dir / "SPLIT_INTEGRITY.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
