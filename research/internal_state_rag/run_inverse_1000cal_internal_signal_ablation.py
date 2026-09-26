"""Evaluate matched query-level cosine and Qwen internal-signal ablations.

Inputs are produced by ``prepare_inverse_1000cal_internal_signal_plans.py``:
the 1,000 parent calibration qids are split into a probe-training role and a
disjoint conformal-calibration role, while the 100 held-out qids are untouched.
The feature analysis contributes five masks at alpha 0.10: pure cosine,
LM-head-only, hidden-probe-only, internal-only fusion, and cosine+internal
fusion.  This runner validates their qids and candidate order before running
Qwen downstream QA.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from eval.metrics import exact_match, numerical_accuracy, token_f1
from research.internal_state_rag.run_all_datasets_cosine_six_methods import (
    CONFIGS,
    validate_downstream_inputs,
)
from research.internal_state_rag.run_downstream_qa_conformal_baselines import (
    QwenDirectAnswerGenerator,
    append_jsonl,
    chunks,
    iter_jsonl,
)
from research.internal_state_rag.run_hotpotqa_cosine_six_methods import selection_summary


METHODS = {
    "query_level_cosine_matched_alpha_0.10": "mask_jina_v4_cosine_alpha_0.1",
    "query_level_lm_head_alpha_0.10": "mask_lm_head_only_alpha_0.1",
    "query_level_hidden_probe_alpha_0.10": "mask_hidden_probe_only_alpha_0.1",
    "query_level_lm_head_hidden_alpha_0.10": "mask_internal_lm_hidden_alpha_0.1",
    "query_level_cosine_lm_head_hidden_alpha_0.10": "mask_cosine_internal_alpha_0.1",
}
DISPLAY = {
    "query_level_cosine_matched_alpha_0.10": "Query-level cosine (matched 500-ish calibration)",
    "query_level_lm_head_alpha_0.10": "LM-head only",
    "query_level_hidden_probe_alpha_0.10": "Hidden-state probe only",
    "query_level_lm_head_hidden_alpha_0.10": "LM-head + hidden probe",
    "query_level_cosine_lm_head_hidden_alpha_0.10": "Cosine + LM-head + hidden probe",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def grouped_test_rows(path: Path, test_qids: Sequence[str]) -> list[list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if str(row.get("split_role")) == "test":
                    grouped[str(row["qid"])].append(row)
    rows = []
    for qid in test_qids:
        candidates = grouped.get(qid, [])
        candidates.sort(key=lambda row: (int(row["rank"]), str(row["chunk_id"])))
        if len(candidates) != 30:
            raise ValueError(f"{qid}: expected exactly 30 frozen test candidates")
        rows.append(candidates)
    return rows


def metric_summary(records: Sequence[dict[str, Any]]) -> dict[str, dict[str, float]]:
    return {
        method: {
            name: float(np.mean([row["metrics"][method][name] for row in records]))
            for name in ("em", "f1", "numerical_accuracy")
        }
        for method in METHODS
    }


def make_report(output: Path, state: dict[str, dict[str, Any]]) -> None:
    lines = [
        "# Qwen-7B internal-signal ablation: 1,000 calibration / 100 held-out test",
        "",
        "Each dataset partitions the same 1,000 parent calibration qids into a disjoint probe-training role and conformal-calibration role. All rows use the same frozen Top-30 cosine candidates and the same 100 held-out test qids. LM-head is the direct Yes-vs-No logit signal; the hidden probe is an L2 logistic relevance probe selected with grouped OOF AP on probe-training qids. Every row uses an all-support query-level split-conformal threshold at α=0.10 and a deterministic Top-1 fallback.",
        "",
    ]
    for dataset, result in state.items():
        split = result["split"]
        lines.extend(
            [
                f"## {dataset} ({result['status']}; probe={split['probe_train_queries']}, calibration={split['conformal_calibration_queries']}, test={split['heldout_test_queries']})",
                "",
                "| Method | Chunks | Precision | Support recall | Empty | Any support | All support | EM | F1 | Numeric |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        downstream = result.get("downstream", {})
        for method in METHODS:
            selection = result["selection"][method]
            qa = downstream.get(method)
            metrics = (
                f"{qa['em']:.3f} | {qa['f1']:.3f} | {qa['numerical_accuracy']:.3f}"
                if qa
                else "pending | pending | pending"
            )
            lines.append(
                f"| {DISPLAY[method]} | {selection['mean_chunks']:.2f} | "
                f"{selection['precision']:.1%} | {selection['support_recall']:.1%} | "
                f"{selection['empty_rate']:.1%} | {selection['query_any_support']:.1%} | "
                f"{selection['query_all_support']:.1%} | {metrics} |"
            )
        analysis = result["analysis"]
        lines.extend(
            [
                "",
                f"Selected hidden-probe layer/C: `{analysis['probe']['selected_layer']}` / `{analysis['probe']['selected_c']}`.",
                "",
                "```json",
                json.dumps(analysis["selected_weights"], indent=2),
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## Audit",
            "",
            "The qid partition and zero-overlap checks are recorded in `splits/SPLIT_INTEGRITY.json`. `analysis/{dataset}/fusion_predictions.npz` contains the five exact masks and score arrays used here; `*_downstream_predictions.jsonl` contains the selected contexts and answers for each held-out qid.",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--datasets", default=",".join(CONFIGS))
    parser.add_argument("--selection-only", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--min-pixels", type=int, default=3136)
    parser.add_argument("--max-pixels", type=int, default=200704)
    args = parser.parse_args()
    names = [name.strip() for name in args.datasets.split(",") if name.strip()]
    if not names or any(name not in CONFIGS for name in names):
        raise ValueError(f"datasets must be a non-empty subset of {tuple(CONFIGS)}")
    if not args.selection_only and args.model is None:
        raise ValueError("--model is required unless --selection-only is set")

    state: dict[str, dict[str, Any]] = {}
    pending: list[tuple[str, list[str], list[list[dict[str, Any]]], dict[str, list[np.ndarray]]]] = []
    for dataset in names:
        config = CONFIGS[dataset]
        split_dir = args.output_dir / "splits" / dataset
        probe = read_json(split_dir / "probe_train_manifest.json")
        calibration = read_json(split_dir / "conformal_calibration_manifest.json")
        test_plan = read_json(split_dir / "test_manifest.json")
        probe_qids = [str(qid) for qid in probe["plan"]]
        calibration_qids = [str(qid) for qid in calibration["plan"]]
        test_qids = [str(qid) for qid in test_plan["plan"]]
        if len(test_qids) != 100:
            raise ValueError(f"{dataset}: expected exactly 100 held-out test qids")
        if any(
            left & right
            for left, right in (
                (set(probe_qids), set(calibration_qids)),
                (set(probe_qids), set(test_qids)),
                (set(calibration_qids), set(test_qids)),
            )
        ):
            raise ValueError(f"{dataset}: train/calibration/test qids overlap")
        test = grouped_test_rows(config.retrieval, test_qids)
        artifact_path = args.output_dir / "analysis" / dataset / "fusion_predictions.npz"
        analysis_path = args.output_dir / "analysis" / dataset / "fusion_analysis.json"
        if not artifact_path.is_file() or not analysis_path.is_file():
            raise FileNotFoundError(f"{dataset}: run internal fusion analysis before downstream evaluation")
        analysis = read_json(analysis_path)
        with np.load(artifact_path) as artifact:
            expected_qids = np.asarray(test_qids, dtype=str)
            expected_chunks = np.asarray(
                [[str(row["chunk_id"]) for row in rows] for rows in test], dtype=str
            )
            if not np.array_equal(artifact["qids"].astype(str), expected_qids):
                raise ValueError(f"{dataset}: internal artifact qids differ from held-out plan")
            if not np.array_equal(artifact["chunk_ids"].astype(str), expected_chunks):
                raise ValueError(f"{dataset}: internal artifact candidates differ from retrieval")
            absent = [key for key in METHODS.values() if key not in artifact.files]
            if absent:
                raise ValueError(f"{dataset}: internal artifact misses masks: {absent}")
            masks = {name: list(artifact[key].astype(bool)) for name, key in METHODS.items()}
        state[dataset] = {
            "status": "selection_complete" if args.selection_only else "running",
            "split": {
                "probe_train_queries": len(probe_qids),
                "conformal_calibration_queries": len(calibration_qids),
                "heldout_test_queries": len(test_qids),
                "top_l": 30,
            },
            "analysis": analysis,
            "selection": selection_summary(test, masks),
        }
        pending.append((dataset, test_qids, test, masks))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    make_report(args.output_dir / "REPORT.md", state)
    (args.output_dir / "selection_summary.json").write_text(
        json.dumps(state, indent=2) + "\n", encoding="utf-8"
    )
    if args.selection_only:
        return

    resolved = {
        dataset: validate_downstream_inputs(CONFIGS[dataset], test_qids, test)
        for dataset, test_qids, test, _masks in pending
    }
    generator = QwenDirectAnswerGenerator(
        args.model, min_pixels=args.min_pixels, max_pixels=args.max_pixels
    )
    for dataset, test_qids, test, masks in pending:
        questions, corpus = resolved[dataset]
        output = args.output_dir / f"{dataset}_downstream_predictions.jsonl"
        done = {str(row["qid"]): row for row in iter_jsonl(output)} if output.is_file() else {}
        for number, (qid, rows) in enumerate(zip(test_qids, test, strict=True), start=1):
            if qid in done:
                continue
            question = questions[qid]
            gold = [str(value) for value in question["gold_answers"]]
            prompt = (
                "Answer using only the supplied context. Return only the short answer.\nQuestion: "
                + str(question["question"])
            )
            answer_cache: dict[tuple[str, ...], str] = {}
            predictions: dict[str, str] = {}
            contexts: dict[str, dict[str, Any]] = {}
            for method, mask in masks.items():
                selected = chunks(rows, mask[number - 1], corpus, CONFIGS[dataset].bundle_root)
                key = tuple(chunk.id for chunk in selected)
                answer = answer_cache.get(key)
                if answer is None:
                    answer = generator.generate(prompt, selected, max_new_tokens=args.max_new_tokens)
                    answer_cache[key] = answer
                predictions[method] = answer
                contexts[method] = {"n_chunks": len(selected), "chunk_ids": list(key)}
            record = {
                "qid": qid,
                "gold_answers": gold,
                "predictions": predictions,
                "contexts": contexts,
                "metrics": {
                    method: {
                        "em": exact_match(answer, gold),
                        "f1": token_f1(answer, gold),
                        "numerical_accuracy": numerical_accuracy(answer, gold),
                    }
                    for method, answer in predictions.items()
                },
            }
            append_jsonl(output, record)
            done[qid] = record
            print(f"[{dataset} {number}/{len(test_qids)}] {qid}", flush=True)
        records = [done[qid] for qid in test_qids]
        state[dataset]["status"] = "complete"
        state[dataset]["downstream"] = metric_summary(records)
        (args.output_dir / f"{dataset}_summary.json").write_text(
            json.dumps(state[dataset], indent=2) + "\n", encoding="utf-8"
        )
        make_report(args.output_dir / "REPORT.md", state)
        (args.output_dir / "summary.json").write_text(
            json.dumps({"status": "running", "datasets": state}, indent=2) + "\n",
            encoding="utf-8",
        )
    (args.output_dir / "summary.json").write_text(
        json.dumps({"status": "complete", "datasets": state}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
