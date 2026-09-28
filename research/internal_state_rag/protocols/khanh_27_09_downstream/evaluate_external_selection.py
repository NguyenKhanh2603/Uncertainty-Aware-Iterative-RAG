#!/usr/bin/env python3
"""Run the fixed Khánh-27-09 Qwen downstream protocol for one external selector.

The input is one JSONL record per held-out qid:
    {"qid": "...", "selected_chunk_ids": ["chunk-id-1", "chunk-id-2"]}

Candidates must be a subset of the frozen Jina-v4 pool.  Context is always
reordered by the frozen retrieval rank, so a selector cannot change the
presentation order.  The program validates all 100 test qids before it loads
Qwen; ``--dry-run`` performs only this validation.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SPLIT_ROOT = (
    REPO_ROOT
    / "research/internal_state_rag/results/"
    "full_context_selection_splits_khanh_27_09_2026_09_27/splits"
)
DEFAULT_RETRIEVAL_ROOT = (
    REPO_ROOT / "research/internal_state_rag/results/zip_calibration_split_20_09/retrieval"
)
DEFAULT_DATA_ROOT = REPO_ROOT / "data/zip_calibration_split_20_09"
DEFAULT_WEBQA_ROOT = Path("/dev/shm/uncertainty_rag_webqa_stage_20260922")
DATASETS = ("hotpotqa", "mmqa", "tatqa", "webqa")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--selection-jsonl", type=Path, required=True)
    parser.add_argument("--method", required=True, help="Name written into the output artifact.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("Qwen/Qwen2-VL-7B-Instruct"))
    parser.add_argument("--model-revision", default="eed13092ef92e448dd6875b2a00151bd3f7db0ac")
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--retrieval-root", type=Path, default=DEFAULT_RETRIEVAL_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--webqa-bundle-root",
        type=Path,
        default=Path(os.environ.get("WEBQA_BUNDLE_ROOT", DEFAULT_WEBQA_ROOT)),
    )
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--min-pixels", type=int, default=3136)
    parser.add_argument("--max-pixels", type=int, default=200704)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number}: each JSONL row must be an object")
                yield value


def read_plan(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("plan")
    if not isinstance(values, list):
        raise ValueError(f"{path}: expected a JSON object containing a list under 'plan'")
    return [str(qid) for qid in values]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_selection(path: Path, expected_qids: list[str]) -> dict[str, tuple[str, ...]]:
    expected = set(expected_qids)
    selected: dict[str, tuple[str, ...]] = {}
    for row in iter_jsonl(path):
        qid = str(row.get("qid", ""))
        chunk_ids = row.get("selected_chunk_ids")
        if not qid:
            raise ValueError(f"{path}: selection row is missing qid")
        if qid in selected:
            raise ValueError(f"{path}: duplicate qid {qid}")
        if qid not in expected:
            raise ValueError(f"{path}: qid {qid} is not in the fixed held-out split")
        if not isinstance(chunk_ids, list) or any(not isinstance(value, str) for value in chunk_ids):
            raise ValueError(f"{path}: {qid} must contain a string list under selected_chunk_ids")
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError(f"{path}: {qid} repeats a chunk id")
        selected[qid] = tuple(chunk_ids)
    missing = [qid for qid in expected_qids if qid not in selected]
    if missing:
        raise ValueError(f"{path}: missing {len(missing)} held-out qids; first is {missing[0]}")
    return selected


def load_frozen_test_rows(path: Path, qids: list[str]) -> dict[str, list[dict[str, Any]]]:
    wanted = set(qids)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in iter_jsonl(path):
        qid = str(row.get("qid", ""))
        if qid in wanted:
            grouped[qid].append(row)
    missing = [qid for qid in qids if qid not in grouped]
    if missing:
        raise ValueError(f"{path}: frozen retrieval is missing test qid {missing[0]}")
    for qid, rows in grouped.items():
        rows.sort(key=lambda row: (int(row["rank"]), str(row["chunk_id"])))
        ranks = [int(row["rank"]) for row in rows]
        if not rows or ranks != list(range(1, len(rows) + 1)) or len(rows) > 30:
            raise ValueError(f"{path}: invalid frozen candidate pool for {qid}")
    return grouped


def source_paths(dataset: str, data_root: Path, webqa_root: Path) -> tuple[Path, Path, Path]:
    bundle_root = webqa_root if dataset == "webqa" else data_root
    return (
        bundle_root / dataset / "questions.jsonl",
        bundle_root / dataset / "corpus.jsonl",
        bundle_root,
    )


def keyed(path: Path, key: str) -> dict[str, dict[str, Any]]:
    return {str(row[key]): row for row in iter_jsonl(path)}


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.max_new_tokens < 1 or args.min_pixels < 1 or args.max_pixels < args.min_pixels:
        raise ValueError("Invalid generation or pixel limits")
    calibration = read_plan(args.split_root / args.dataset / "calibration_manifest.json")
    test = read_plan(args.split_root / args.dataset / "test_manifest.json")
    if len(calibration) != 1000 or len(test) != 100 or set(calibration).intersection(test):
        raise ValueError("Expected the fixed 1,000-calibration / 100-test disjoint Khánh split")

    selection = load_selection(args.selection_jsonl, test)
    retrieval_path = args.retrieval_root / f"{args.dataset}_jina_v4_candidates.jsonl.gz"
    frozen_rows = load_frozen_test_rows(retrieval_path, test)
    ordered_rows: dict[str, list[dict[str, Any]]] = {}
    for qid in test:
        by_id = {str(row["chunk_id"]): row for row in frozen_rows[qid]}
        unknown = set(selection[qid]).difference(by_id)
        if unknown:
            raise ValueError(f"{args.dataset}/{qid}: selected chunk not in frozen pool: {sorted(unknown)[0]}")
        # The protocol fixes the context order to the frozen Jina retrieval order.
        selected = set(selection[qid])
        ordered_rows[qid] = [row for row in frozen_rows[qid] if str(row["chunk_id"]) in selected]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "protocol": "khanh_27_09_qwen2vl_direct_answer_v1",
        "dataset": args.dataset,
        "method": args.method,
        "selection_sha256": sha256(args.selection_jsonl),
        "selection_file": str(args.selection_jsonl),
        "split_root": str(args.split_root),
        "retrieval_file": str(retrieval_path),
        "model": str(args.model),
        "model_revision": args.model_revision,
        "decoding": "greedy",
        "max_new_tokens": args.max_new_tokens,
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
        "calibration_queries": len(calibration),
        "test_queries": len(test),
    }
    config_path = args.output_dir / "RUN_CONFIG.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous != run_config:
            raise ValueError(f"{config_path} differs from this invocation; choose a new output directory")
    else:
        atomic_json(config_path, run_config)

    validation = {
        "status": "validated",
        "dataset": args.dataset,
        "test_queries": len(test),
        "mean_selected_chunks": sum(len(ordered_rows[qid]) for qid in test) / len(test),
        "empty_context_queries": sum(not ordered_rows[qid] for qid in test),
    }
    if args.dry_run:
        atomic_json(args.output_dir / "VALIDATION.json", validation)
        print(json.dumps(validation, ensure_ascii=False))
        return

    # Delay CUDA/model imports until every split and selection invariant passed.
    sys.path.insert(0, str(REPO_ROOT))
    from eval.metrics import exact_match, numerical_accuracy, token_f1
    from research.internal_state_rag.run_downstream_qa_conformal_baselines import (
        QwenDirectAnswerGenerator,
        chunks,
    )

    questions_path, corpus_path, bundle_root = source_paths(
        args.dataset, args.data_root, args.webqa_bundle_root
    )
    if not questions_path.is_file() or not corpus_path.is_file():
        raise FileNotFoundError(f"Missing questions/corpus: {questions_path}, {corpus_path}")
    questions = keyed(questions_path, "qid")
    corpus = keyed(corpus_path, "id")
    missing_questions = [qid for qid in test if qid not in questions]
    if missing_questions:
        raise ValueError(f"Questions are missing held-out qid {missing_questions[0]}")

    prediction_path = args.output_dir / "predictions.jsonl"
    completed = {str(row["qid"]): row for row in iter_jsonl(prediction_path)} if prediction_path.exists() else {}
    unexpected = set(completed).difference(test)
    if unexpected:
        raise ValueError(f"{prediction_path} contains qid outside this split: {sorted(unexpected)[0]}")
    generator = QwenDirectAnswerGenerator(
        args.model,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        revision=args.model_revision,
    )
    with prediction_path.open("a", encoding="utf-8") as handle:
        for index, qid in enumerate(test, start=1):
            if qid in completed:
                print(f"[{args.dataset} {index}/100] resume {qid}", flush=True)
                continue
            rows = frozen_rows[qid]
            selected_ids = {str(row["chunk_id"]) for row in ordered_rows[qid]}
            keep_mask = [str(row["chunk_id"]) in selected_ids for row in rows]
            context = chunks(rows, keep_mask, corpus, bundle_root)
            question = questions[qid]
            gold = [str(answer) for answer in question["gold_answers"]]
            prompt = "Answer using only the supplied context. Return only the short answer.\nQuestion: " + str(question["question"])
            prediction = generator.generate(prompt, context, max_new_tokens=args.max_new_tokens)
            record = {
                "qid": qid,
                "gold_answers": gold,
                "prediction": prediction,
                "metrics": {
                    "em": exact_match(prediction, gold),
                    "f1": token_f1(prediction, gold),
                    "numerical_accuracy": numerical_accuracy(prediction, gold),
                },
                "context": {"n_chunks": len(context), "chunk_ids": [chunk.id for chunk in context]},
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            completed[qid] = record
            print(f"[{args.dataset} {index}/100] {qid}", flush=True)

    records = [completed[qid] for qid in test]
    summary = {
        **validation,
        "status": "complete",
        "metrics": {
            metric: fmean(float(row["metrics"][metric]) for row in records)
            for metric in ("em", "f1", "numerical_accuracy")
        },
    }
    atomic_json(args.output_dir / "SUMMARY.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
