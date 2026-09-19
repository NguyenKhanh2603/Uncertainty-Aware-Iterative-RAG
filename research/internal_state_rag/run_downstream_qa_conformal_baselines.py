"""Evaluate frozen retrieval selectors with greedy downstream QA.

This is a controlled component comparison: every selector receives the same
Jina-v4 Top-30 candidates and Qwen2-VL-7B generates one short answer from its
selected context.  It evaluates the project BY cosine selector and the
retrieval components of CCE, CONFLARE, and TRAQ.  It does *not* claim to be
TRAQ's complete answer-set conformal procedure.

The JSONL prediction file is append-only and resumable by query id.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from eval.metrics import exact_match, numerical_accuracy, token_f1
from research.internal_state_rag.run_tatqa_smoke import ResearchTextClient, generate_answer
from uncertainty_rag.core.conformal_selection import benjamini_yekutieli
from uncertainty_rag.modality.base import ContextChunk


DATASETS = ("hotpotqa", "mmqa", "tatqa", "webqa")
METHODS = (
    "full_top30",
    "by_cosine",
    "ecir_cce_embedding",
    "conflare_retrieval_adapter",
    "traq_retrieval_bonferroni_adapter",
)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def keyed(path: Path, key: str) -> dict[str, dict[str, Any]]:
    return {str(row[key]): row for row in iter_jsonl(path)}


def grouped_retrieval(path: Path) -> dict[str, dict[str, list[dict[str, Any]]]]:
    result: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in iter_jsonl(path):
        result[str(row.get("split_role"))][str(row["qid"])].append(row)
    for by_qid in result.values():
        for rows in by_qid.values():
            rows.sort(key=lambda row: (int(row["rank"]), str(row["chunk_id"])))
    return result


def finite_support_threshold(scores: np.ndarray, alpha: float) -> float:
    ordered = np.sort(scores)
    # Same finite-sample convention as the CCE adapter runner.
    return float(ordered[min(len(ordered) - 1, int(math.floor((len(ordered) + 1) * alpha)))])


def by_masks(calibration: list[list[dict[str, Any]]], test: list[list[dict[str, Any]]], alpha: float) -> list[np.ndarray]:
    bank = np.sort(
        np.asarray(
            [float(row["cosine_score"]) for rows in calibration for row in rows if row["support_label"] != "support"],
            dtype=float,
        )
    )
    if not len(bank):
        raise ValueError("BY calibration false-score bank is empty")
    masks = []
    for rows in test:
        p_values = [
            (1.0 + len(bank) - np.searchsorted(bank, float(row["cosine_score"]), side="left")) / (len(bank) + 1.0)
            for row in rows
        ]
        mask = np.zeros(len(rows), dtype=bool)
        mask[list(benjamini_yekutieli(p_values, alpha).rejected_indices)] = True
        masks.append(mask)
    return masks


def selector_masks(calibration: list[list[dict[str, Any]]], test: list[list[dict[str, Any]]], alpha: float) -> tuple[dict[str, list[np.ndarray]], dict[str, float | None]]:
    support = np.asarray(
        [float(row["cosine_score"]) for rows in calibration for row in rows if row["support_label"] == "support"],
        dtype=float,
    )
    if not len(support):
        raise ValueError("Calibration support-score bank is empty")
    cce = finite_support_threshold(support, alpha)
    conflare = float(np.percentile(support, alpha * 100))
    traq = float(np.quantile(support, alpha / 2, method="lower"))
    all_true = [np.ones(len(rows), dtype=bool) for rows in test]
    return {
        "full_top30": all_true,
        "by_cosine": by_masks(calibration, test, alpha),
        "ecir_cce_embedding": [np.asarray([float(r["cosine_score"]) >= cce for r in rows]) for rows in test],
        "conflare_retrieval_adapter": [np.asarray([float(r["cosine_score"]) >= conflare for r in rows]) for rows in test],
        "traq_retrieval_bonferroni_adapter": [np.asarray([float(r["cosine_score"]) >= traq for r in rows]) for rows in test],
    }, {"by_cosine": None, "ecir_cce_embedding": cce, "conflare_retrieval_adapter": conflare, "traq_retrieval_bonferroni_adapter": traq}


def image_path(value: str, bundle_root: Path) -> str:
    candidate = Path(value)
    if candidate.is_file() or value.startswith(("http://", "https://", "data:")):
        return value
    resolved = bundle_root / candidate
    if resolved.is_file():
        return str(resolved)
    # The mmqa source bundle stores images one directory above role_split.
    alternative = bundle_root.parent / "official_bundle_1000_more" / candidate
    if alternative.is_file():
        return str(alternative)
    raise FileNotFoundError(f"Missing image context: {value}")


def chunks(rows: list[dict[str, Any]], mask: np.ndarray, corpus: dict[str, dict[str, Any]], bundle_root: Path) -> list[ContextChunk]:
    selected = []
    for row, keep in zip(rows, mask):
        if not keep:
            continue
        source = corpus[str(row["chunk_id"])]
        modality = str(source.get("modality", row.get("modality", "text")))
        content = str(source["content"])
        if modality == "image":
            content = image_path(content, bundle_root)
        selected.append(ContextChunk(id=str(row["chunk_id"]), content=content, modality=modality, metadata={"rank": int(row["rank"])}))
    return selected


def metrics(prediction: str, gold: list[str]) -> dict[str, float]:
    return {"em": exact_match(prediction, gold), "f1": token_f1(prediction, gold), "numerical_accuracy": numerical_accuracy(prediction, gold)}


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"n_queries": len(rows), "methods": {}}
    for method in METHODS:
        result["methods"][method] = {
            metric: float(np.mean([row["metrics"][method][metric] for row in rows]))
            for metric in ("em", "f1", "numerical_accuracy")
        }
        result["methods"][method].update({
            "mean_chunks": float(np.mean([row["contexts"][method]["n_chunks"] for row in rows])),
            "empty_rate": float(np.mean([row["contexts"][method]["n_chunks"] == 0 for row in rows])),
        })
    return result


def paths(dataset: str, root: Path) -> tuple[Path, Path, Path, Path]:
    bundle = root / f"official_bundle_role_split_{dataset}"
    retrieval = root / "retrieval" / f"{dataset}_top30_global_retrieval.jsonl.gz"
    return bundle / dataset / "questions.jsonl", bundle / dataset / "corpus.jsonl", retrieval, bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/conformal_global_run"))
    parser.add_argument("--feature-root", type=Path, default=Path("research/internal_state_rag/results/qwen2vl_7b_jina4/fusion_features"))
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=0, help="0 evaluates every frozen test query")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chosen = [item.strip() for item in args.datasets.split(",") if item.strip()]
    if not 0 < args.alpha < 1:
        raise ValueError("alpha must be in (0,1)")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_manifest = {"status": "running", "model": str(args.model), "alpha": args.alpha, "max_new_tokens": args.max_new_tokens, "datasets": {}}
    client: ResearchTextClient | None = None
    for dataset in chosen:
        if dataset not in DATASETS:
            raise ValueError(f"Unknown dataset: {dataset}")
        questions_path, corpus_path, retrieval_path, bundle_root = paths(dataset, args.data_root)
        if not (questions_path.is_file() and corpus_path.is_file() and retrieval_path.is_file()):
            run_manifest["datasets"][dataset] = {"status": "unavailable_source_files", "questions": str(questions_path), "corpus": str(corpus_path), "retrieval": str(retrieval_path)}
            continue
        plan = json.loads((args.feature_root / dataset / "test" / "manifest.json").read_text(encoding="utf-8"))["plan"]
        if args.limit:
            plan = plan[: args.limit]
        questions = keyed(questions_path, "qid")
        corpus = keyed(corpus_path, "id")
        retrieval = grouped_retrieval(retrieval_path)
        calibration = retrieval["calibration"]
        test_lookup = retrieval["test"]
        test = [test_lookup[str(qid)] for qid in plan]
        if any(len(rows) != 30 for rows in test):
            raise ValueError(f"{dataset}: frozen test plan does not have Top-30 rows")
        masks, thresholds = selector_masks(list(calibration.values()), test, args.alpha)
        output = args.output_dir / f"{dataset}_predictions.jsonl"
        completed = {str(row["qid"]): row for row in iter_jsonl(output)} if output.exists() else {}
        if client is None:
            client = ResearchTextClient(str(args.model), device="cuda", load_in_4bit=False)
        for index, (qid, rows) in enumerate(zip(plan, test), start=1):
            qid = str(qid)
            if qid in completed:
                print(f"[{dataset} {index}/{len(plan)}] resume {qid}", flush=True)
                continue
            question = questions[qid]
            gold = [str(answer) for answer in question["gold_answers"]]
            prompt = "Answer using only the supplied context. Return only the short answer.\nQuestion: " + str(question["question"])
            predictions, contexts = {}, {}
            for method in METHODS:
                selected = chunks(rows, masks[method][index - 1], corpus, bundle_root)
                prediction = generate_answer(client, prompt, selected, max_new_tokens=args.max_new_tokens)
                predictions[method] = prediction
                contexts[method] = {"n_chunks": len(selected), "chunk_ids": [chunk.id for chunk in selected]}
            record = {"qid": qid, "gold_answers": gold, "predictions": predictions, "metrics": {name: metrics(prediction, gold) for name, prediction in predictions.items()}, "contexts": contexts}
            append_jsonl(output, record)
            completed[qid] = record
            print(f"[{dataset} {index}/{len(plan)}] {qid} em=" + " ".join(f"{name}:{record['metrics'][name]['em']:.0f}" for name in METHODS), flush=True)
        ordered = [completed[str(qid)] for qid in plan if str(qid) in completed]
        summary = {"status": "complete" if len(ordered) == len(plan) else "partial", "dataset": dataset, "alpha": args.alpha, "thresholds": thresholds, "frozen_test_queries": len(plan), "results": summarize(ordered)}
        (args.output_dir / f"{dataset}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        run_manifest["datasets"][dataset] = {"status": summary["status"], "summary": str(args.output_dir / f"{dataset}_summary.json")}
    run_manifest["status"] = "complete"
    (args.output_dir / "manifest.json").write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
