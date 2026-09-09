"""Build Section 2.3 false-match reference banks from a scored top-L log."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from uncertainty_rag.core.conformal_retrieval import (
    ConformalDataError,
    build_reference_bank_artifact,
    parse_condition_fields,
    parse_rank_bins,
)


def open_text(path: Path, mode: str):
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with open_text(path, "rt") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ConformalDataError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ConformalDataError(f"Expected an object at {path}:{line_number}")
            yield row


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_artifact(path: Path, artifact: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    opener = gzip.open if path.suffix == ".gz" else Path.open
    if path.suffix == ".gz":
        handle_context = opener(temporary, "wt", encoding="utf-8")
    else:
        handle_context = opener(temporary, "w", encoding="utf-8")
    with handle_context as handle:
        json.dump(artifact, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build conformal false-match banks from frozen top-L retrieval logs."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        nargs="+",
        help="One or more JSONL/JSONL.GZ retrieval logs built by the same frozen pipeline",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON artifact; use a .gz suffix for compression",
    )
    parser.add_argument("--rank-bins", default="1-3,4-10,11-20")
    parser.add_argument(
        "--conditioning",
        default="dataset,query_type,modality,rank_bin",
        help=(
            "Comma-separated bank fields. Use dataset,modality for the first "
            "modality-aware smoke experiment."
        ),
    )
    parser.add_argument("--min-bank-size", type=int, default=1000)
    parser.add_argument(
        "--allow-small-banks",
        action="store_true",
        help="Create an explicitly non-paper-ready smoke artifact instead of failing",
    )
    args = parser.parse_args()

    missing_inputs = [path for path in args.input if not path.is_file()]
    if missing_inputs:
        parser.error(f"Input file does not exist: {missing_inputs[0]}")

    artifact = build_reference_bank_artifact(
        (row for path in args.input for row in read_jsonl(path)),
        rank_bins=parse_rank_bins(args.rank_bins),
        condition_fields=parse_condition_fields(args.conditioning),
        min_bank_size=args.min_bank_size,
        allow_small_banks=args.allow_small_banks,
    )
    artifact["source"] = {
        "files": [
            {"path": path.name, "sha256": sha256(path)} for path in args.input
        ],
    }
    write_artifact(args.output, artifact)

    summary = artifact["summary"]
    print(
        f"Wrote {len(artifact['banks'])} banks to {args.output} | "
        f"rows={summary['input_rows']} | "
        f"underpowered={summary['underpowered_banks']} | "
        f"paper_ready={artifact['is_paper_ready']}"
    )


if __name__ == "__main__":
    main()
