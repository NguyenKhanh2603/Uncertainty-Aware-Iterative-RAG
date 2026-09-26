"""Prepare disjoint Qwen-7B internal-signal inputs for the inverse protocol.

The parent experiment fixes 1,000 calibration and 100 held-out test queries
per dataset.  This script partitions the 1,000 calibration queries into a
probe-training role and a disjoint conformal-calibration role, preserving the
already frozen test role.  It also materializes the matching 100-query test
feature files by subsetting previously extracted Qwen2-VL-7B features after
validating both qids and candidate order.

No support labels are used to choose the split.  The parent calibration qids,
retrieval roles, and the HotpotQA random seed are the only inputs.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


DATASETS = ("hotpotqa", "mmqa", "tatqa", "webqa")
TEST_FEATURE_SOURCES = {
    "hotpotqa": Path(
        "research/internal_state_rag/results/official_seed42_hotpot_features/test"
    ),
    "mmqa": Path(
        "research/internal_state_rag/results/qwen2vl_7b_jina4/"
        "full_test_internal_fusion_2026_09_25_v2/features/mmqa"
    ),
    "tatqa": Path(
        "research/internal_state_rag/results/qwen2vl_7b_jina4/"
        "full_test_internal_fusion_2026_09_25_v2/features/tatqa"
    ),
    "webqa": Path(
        "research/internal_state_rag/results/qwen2vl_7b_jina4/"
        "full_test_internal_fusion_2026_09_25_v2/features/webqa"
    ),
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_or_validate_json(path: Path, value: dict[str, Any]) -> None:
    if path.is_file():
        if read_json(path) != value:
            raise RuntimeError(
                f"Existing generated plan differs from the requested protocol: {path}"
            )
        return
    atomic_json(path, value)


def grouped_retrieval(path: Path) -> dict[str, dict[str, list[dict[str, Any]]]]:
    result: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                result[str(row["split_role"])][str(row["qid"])].append(row)
    for role in result.values():
        for rows in role.values():
            rows.sort(key=lambda row: (int(row["rank"]), str(row["chunk_id"])))
    return result


def validate_top_l(
    dataset: str,
    retrieval: dict[str, dict[str, list[dict[str, Any]]]],
    qids: list[str],
    role: str,
) -> None:
    missing = [qid for qid in qids if qid not in retrieval.get(role, {})]
    if missing:
        raise ValueError(f"{dataset}/{role}: missing planned qids: {missing[:3]}")
    malformed = [qid for qid in qids if len(retrieval[role][qid]) != 30]
    if malformed:
        raise ValueError(f"{dataset}/{role}: non-Top-30 qids: {malformed[:3]}")


def hotpot_partition(qids: list[str]) -> tuple[list[str], list[str]]:
    """Split a single source role without inspecting labels or scores."""

    ranked = sorted(
        qids,
        key=lambda qid: hashlib.sha256(f"20260926:{qid}".encode("utf-8")).hexdigest(),
    )
    probe = sorted(ranked[:500])
    calibration = sorted(ranked[500:])
    if len(probe) != 500 or len(calibration) != 500:
        raise ValueError("HotpotQA must yield exactly 500 probe and 500 calibration qids")
    return probe, calibration


def plan_payload(
    *,
    dataset: str,
    role: str,
    input_role: str,
    qids: list[str],
    parent: dict[str, Any],
    selection: str,
) -> dict[str, Any]:
    return {
        "protocol": "inverse_1000_calibration_100_test_internal_signal_2026_09_26",
        "dataset": dataset,
        "role": role,
        "input_role": input_role,
        "source_split": "parent_inverse_1000_calibration_100_test",
        "source_retrieval": parent["source_retrieval"],
        "top_l": 30,
        "n_queries": len(qids),
        "selection": selection,
        "plan": qids,
    }


def validate_existing_test_features(
    *,
    dataset: str,
    source_dir: Path,
    test_qids: list[str],
    expected_chunk_ids: list[list[str]],
    target: Path,
) -> dict[str, Any]:
    source_npz = source_dir / "features.npz"
    source_manifest = source_dir / "manifest.json"
    if not source_npz.is_file() or not source_manifest.is_file():
        raise FileNotFoundError(f"{dataset}: missing Qwen-7B test feature source at {source_dir}")
    source_info = read_json(source_manifest)
    if "Qwen2-VL-7B-Instruct" not in str(source_info.get("model", "")):
        raise ValueError(f"{dataset}: test feature source is not Qwen2-VL-7B: {source_dir}")
    with np.load(source_npz) as source:
        source_qids = source["qids"].astype(str).tolist()
        positions = {qid: index for index, qid in enumerate(source_qids)}
        missing = [qid for qid in test_qids if qid not in positions]
        if missing:
            raise ValueError(f"{dataset}: cached test features miss qids: {missing[:3]}")
        indices = np.asarray([positions[qid] for qid in test_qids], dtype=np.intp)
        observed_chunks = source["chunk_ids"][indices].astype(str).tolist()
        if observed_chunks != expected_chunk_ids:
            raise ValueError(f"{dataset}: cached test feature candidate order differs from retrieval")
        if target.is_file():
            with np.load(target) as existing:
                if existing["qids"].astype(str).tolist() != test_qids:
                    raise RuntimeError(f"{dataset}: existing generated test feature qids differ")
                if existing["chunk_ids"].astype(str).tolist() != expected_chunk_ids:
                    raise RuntimeError(f"{dataset}: existing generated test feature candidates differ")
        else:
            values = {
                name: (array[indices] if array.ndim and array.shape[0] == len(source_qids) else array)
                for name, array in ((name, source[name]) for name in source.files)
            }
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                dir=target.parent, prefix=f".{target.stem}.", suffix=".npz", delete=False
            ) as handle:
                temporary = Path(handle.name)
            try:
                np.savez_compressed(temporary, **values)
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
    return {
        "method": "validated_subset_of_preexisting_qwen2vl_7b_test_features",
        "source_dir": str(source_dir),
        "source_manifest": source_info,
        "n_queries": len(test_qids),
        "qids": test_qids,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parent-plan-root",
        type=Path,
        default=Path(
            "research/internal_state_rag/results/"
            "all_datasets_cosine_six_methods_1000cal_100test_2026_09_26/splits"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    audit: dict[str, Any] = {
        "protocol": "inverse_1000_calibration_100_test_internal_signal_2026_09_26",
        "partition_policy": {
            "hotpotqa": "SHA256(seed=20260926,qid) balanced 500/500 within calibration role",
            "mmqa": "development role (504) probe train; calibration role (496) conformal calibration",
            "tatqa": "development role (516) probe train; calibration role (484) conformal calibration",
            "webqa": "development role (465) probe train; calibration role (535) conformal calibration",
        },
        "datasets": {},
    }
    for dataset in DATASETS:
        parent_dir = args.parent_plan_root / dataset
        parent_calibration = read_json(parent_dir / "calibration_manifest.json")
        parent_test = read_json(parent_dir / "test_manifest.json")
        calibration_qids = [str(qid) for qid in parent_calibration["plan"]]
        test_qids = [str(qid) for qid in parent_test["plan"]]
        if len(calibration_qids) != 1000 or len(test_qids) != 100:
            raise ValueError(f"{dataset}: parent plans must be exactly 1,000 calibration / 100 test")
        if len(set(calibration_qids)) != 1000 or len(set(test_qids)) != 100:
            raise ValueError(f"{dataset}: parent plan has duplicate qids")
        if set(calibration_qids) & set(test_qids):
            raise ValueError(f"{dataset}: parent calibration and test qids overlap")
        retrieval = grouped_retrieval(Path(parent_calibration["source_retrieval"]))

        if dataset == "hotpotqa":
            probe_qids, conformal_qids = hotpot_partition(calibration_qids)
            probe_role = conformal_role = "calibration"
            probe_selection = "hash_ranked_500_of_parent_calibration_qids_seed_20260926"
            conformal_selection = "remaining_hash_ranked_500_of_parent_calibration_qids_seed_20260926"
        else:
            by_role = {
                role: sorted(
                    qid for qid in calibration_qids if qid in retrieval.get(role, {})
                )
                for role in ("calibration", "development")
            }
            if set(by_role["calibration"]) & set(by_role["development"]):
                raise ValueError(f"{dataset}: source roles overlap")
            if set(by_role["calibration"]) | set(by_role["development"]) != set(calibration_qids):
                raise ValueError(f"{dataset}: parent calibration qids do not match frozen source roles")
            probe_qids, conformal_qids = by_role["development"], by_role["calibration"]
            probe_role, conformal_role = "development", "calibration"
            probe_selection = "all_parent_calibration_qids_with_source_role_development"
            conformal_selection = "all_parent_calibration_qids_with_source_role_calibration"

        if set(probe_qids) & set(conformal_qids):
            raise ValueError(f"{dataset}: probe/conformal partition overlap")
        if set(probe_qids) | set(conformal_qids) != set(calibration_qids):
            raise ValueError(f"{dataset}: partition does not reconstruct the 1,000 calibration qids")
        for qids, role in (
            (probe_qids, probe_role),
            (conformal_qids, conformal_role),
            (test_qids, "test"),
        ):
            validate_top_l(dataset, retrieval, qids, role)

        split_dir = args.output_dir / "splits" / dataset
        probe_plan = plan_payload(
            dataset=dataset,
            role="probe_train",
            input_role=probe_role,
            qids=probe_qids,
            parent=parent_calibration,
            selection=probe_selection,
        )
        conformal_plan = plan_payload(
            dataset=dataset,
            role="conformal_calibration",
            input_role=conformal_role,
            qids=conformal_qids,
            parent=parent_calibration,
            selection=conformal_selection,
        )
        test_plan = plan_payload(
            dataset=dataset,
            role="heldout_test",
            input_role="test",
            qids=test_qids,
            parent=parent_test,
            selection="parent_inverse_heldout_test_100_qids",
        )
        write_or_validate_json(split_dir / "probe_train_manifest.json", probe_plan)
        write_or_validate_json(split_dir / "conformal_calibration_manifest.json", conformal_plan)
        write_or_validate_json(split_dir / "test_manifest.json", test_plan)

        expected_chunk_ids = [
            [str(row["chunk_id"]) for row in retrieval["test"][qid]] for qid in test_qids
        ]
        test_feature_manifest = validate_existing_test_features(
            dataset=dataset,
            source_dir=TEST_FEATURE_SOURCES[dataset],
            test_qids=test_qids,
            expected_chunk_ids=expected_chunk_ids,
            target=args.output_dir / "features" / dataset / "test" / "features.npz",
        )
        write_or_validate_json(
            args.output_dir / "features" / dataset / "test" / "source_manifest.json",
            test_feature_manifest,
        )
        audit["datasets"][dataset] = {
            "parent_calibration_queries": len(calibration_qids),
            "probe_train_queries": len(probe_qids),
            "conformal_calibration_queries": len(conformal_qids),
            "heldout_test_queries": len(test_qids),
            "top_l": 30,
            "overlaps": {
                "probe_train__conformal_calibration": len(set(probe_qids) & set(conformal_qids)),
                "probe_train__heldout_test": len(set(probe_qids) & set(test_qids)),
                "conformal_calibration__heldout_test": len(set(conformal_qids) & set(test_qids)),
            },
            "source_role_counts": dict(Counter((probe_role,) * len(probe_qids) + (conformal_role,) * len(conformal_qids))),
            "test_feature_source": str(TEST_FEATURE_SOURCES[dataset]),
        }
        print(
            f"{dataset}: probe={len(probe_qids)} calibration={len(conformal_qids)} test={len(test_qids)}",
            flush=True,
        )
    if any(any(value != 0 for value in row["overlaps"].values()) for row in audit["datasets"].values()):
        raise AssertionError("Internal-signal splits overlap")
    write_or_validate_json(args.output_dir / "splits" / "SPLIT_INTEGRITY.json", audit)


if __name__ == "__main__":
    main()
