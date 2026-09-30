"""Evaluate CCE, CONFLARE, and TRAQ on all eligible official test queries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from eval.metrics import exact_match, numerical_accuracy, token_f1
from research.internal_state_rag.run_all_datasets_cosine_six_methods import (
    DatasetConfig,
    validate_downstream_inputs,
)
from research.internal_state_rag.run_downstream_qa_conformal_baselines import (
    QwenDirectAnswerGenerator,
    append_jsonl,
    chunks,
    iter_jsonl,
)
from research.internal_state_rag.run_hotpotqa_cosine_six_methods import grouped, iter_retrieval
from research.internal_state_rag.run_literature_protocol_1000cal import selection_metrics

METHODS = (
    "cce_conformal_embedding_jina_alpha_0.10",
    "conflare_source_question_jina_alpha_0.10",
    "traq_retrieval_bonferroni_jina_alpha_0.10",
)
DISPLAY = {
    METHODS[0]: "CCE Conformal-Embedding (α=.10)",
    METHODS[1]: "CONFLARE source-question (α=.10)",
    METHODS[2]: "TRAQ retrieval Bonferroni (α=.10; α_R=.05)",
}
DATASETS = ("hotpotqa", "mmqa", "tatqa", "webqa")
DEFAULT_OUTPUT = Path(
    "research/internal_state_rag/results/full_context_selection_eligible_full_2026_09_30"
)
DEFAULT_SPLITS = Path("research/internal_state_rag/splits/official_evaluable_test_2026_09_30")
DEFAULT_BUNDLE = Path("data/official_evaluable_test_2026_09_30")
DEFAULT_RETRIEVAL = DEFAULT_OUTPUT / "retrieval"
CALIBRATION_SUMMARY = Path(
    "research/internal_state_rag/results/full_context_selection_splits_khanh_27_09_2026_09_27/summary.json"
)


def qids(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = [str(value) for value in payload["plan"]]
    if len(values) != int(payload["count"]) or len(values) != len(set(values)):
        raise ValueError(f"Invalid qid manifest: {path}")
    return values


def load_test(retrieval_path: Path, manifest: Path) -> tuple[list[str], list[list[dict[str, Any]]]]:
    ordered_qids = qids(manifest)
    retrieval = grouped(iter_retrieval(retrieval_path))
    all_queries: dict[str, list[dict[str, Any]]] = {}
    for by_qid in retrieval.values():
        for qid, rows in by_qid.items():
            if qid in all_queries:
                raise ValueError(f"QID occurs in multiple roles: {qid}")
            all_queries[qid] = rows
    if missing := [qid for qid in ordered_qids if qid not in all_queries]:
        raise ValueError(f"Retrieval log misses {len(missing)} planned qids; first={missing[0]}")
    test = [all_queries[qid] for qid in ordered_qids]
    for qid, rows in zip(ordered_qids, test, strict=True):
        ranks = [int(row["rank"]) for row in rows]
        if not rows or ranks != list(range(1, len(rows) + 1)) or len(rows) > 30:
            raise ValueError(f"Invalid candidate ranks for {qid}")
    return ordered_qids, test


def masks_from_thresholds(
    test: Sequence[Sequence[dict[str, Any]]], thresholds: dict[str, Any]
) -> dict[str, list[np.ndarray]]:
    result: dict[str, list[np.ndarray]] = {}
    for method in METHODS:
        cutoff = float(thresholds[method]["similarity_threshold"])
        strict = method.startswith("conflare_")
        result[method] = [
            np.asarray(
                [
                    float(row["cosine_score"]) > cutoff
                    if strict
                    else float(row["cosine_score"]) >= cutoff
                    for row in rows
                ],
                dtype=bool,
            )
            for rows in test
        ]
    return result


def downstream_metrics(records: Sequence[dict[str, Any]]) -> dict[str, dict[str, float]]:
    return {
        method: {
            metric: float(np.mean([row["metrics"][method][metric] for row in records]))
            for metric in ("em", "f1", "numerical_accuracy")
        }
        for method in METHODS
    }


def write_report(output: Path, state: dict[str, dict[str, Any]]) -> None:
    lines = [
        "# Three post-retrieval baselines — official eligible full test (2026-09-30)",
        "",
        "The experiment uses the complete official splits with public answer and support "
        "labels: HotpotQA distractor/validation, MMQA dev, TAT-QA dev, and WebQA "
        "validation. The three selectors share the same Jina-v4 ranked candidate pool "
        "(dataset-provided candidates, Top-L≤30), the same 1,000-query calibration "
        "thresholds from `splits_khanh_27_09`, and greedy Qwen2-VL-7B generation with "
        "at most 24 new tokens.",
        "",
        "- Split definition: "
        "[`SPLIT_SUMMARY.json`](../../splits/official_evaluable_test_2026_09_30/"
        "SPLIT_SUMMARY.json)",
        "- Runner: [`run_eligible_full_three_baselines.py`](../../"
        "run_eligible_full_three_baselines.py)",
        "- Retrieval launcher: [`run_eligible_full_three_baselines_2026_09_30.sh`](../../../../"
        "scripts/run_eligible_full_three_baselines_2026_09_30.sh)",
        "",
    ]
    for dataset in DATASETS:
        if dataset not in state:
            continue
        result = state[dataset]
        lines.extend(
            [
                f"## {dataset} ({result['status']}; calibration=1,000, "
                f"test={result['test_queries']:,})",
                "",
                "| Family | Method | Chunks | Precision | Recall | Empty | Any support | "
                "All support | EM | F1 | Numeric |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for method in METHODS:
            selected = result["selection"][method]
            qa = result.get("downstream", {}).get(method)
            downstream = (
                f"{qa['em']:.3f} | {qa['f1']:.3f} | {qa['numerical_accuracy']:.3f}"
                if qa
                else "pending | pending | pending"
            )
            family = (
                "CCE"
                if method.startswith("cce_")
                else "CONFLARE"
                if method.startswith("conflare_")
                else "TRAQ"
            )
            lines.append(
                f"| {family} | {DISPLAY[method]} | {selected['mean_chunks']:.2f} | "
                f"{selected['precision']:.1%} | {selected['support_recall']:.1%} | "
                f"{selected['empty_rate']:.1%} | {selected['query_any_support_conditional']:.1%} | "
                f"{selected['query_all_support_conditional']:.1%} | {downstream} |"
            )
        lines.extend(
            [
                "",
                f"Retrieval ceiling: {result['retrieval_query_coverage_ceiling']:.1%} "
                f"({result['retrievable_queries']:,}/{result['test_queries']:,} queries "
                "contain at least one labelled support in the frozen candidate pool). "
                "Any-support and all-support are conditional on those retrievable queries.",
                "",
            ]
        )
    lines.extend(
        [
            "## Metric definitions",
            "",
            "- **Chunks:** mean selected chunks per query.",
            "- **Precision:** selected labelled-support chunks divided by all selected chunks.",
            "- **Recall:** selected labelled-support chunks divided by labelled supports "
            "present in the candidate pool.",
            "- **Empty:** fraction of queries for which the selector keeps no chunk.",
            "- **Any support / All support:** fraction of retrievable queries retaining at "
            "least one / every labelled support.",
            "- **EM / F1 / Numeric:** exact match, token F1, and numerical answer accuracy "
            "from the shared deterministic downstream generator.",
            "",
            "## Selection rules",
            "",
            "CCE pools every labelled support chunk from the 1,000 calibration queries, "
            "takes the finite-sample 90th percentile of nonconformity `1 − cosine`, and "
            "keeps test chunks whose cosine reaches the resulting cutoff.",
            "",
            "CONFLARE contributes one calibration value per query: the highest-scoring "
            "labelled support. It takes the 90th percentile of cosine distance and keeps "
            "test chunks with distance strictly below that cutoff.",
            "",
            "TRAQ contributes the best labelled-support cosine from each calibration query, "
            "assigns half of α=.10 to retrieval, and keeps test chunks at or above the "
            "lower 5% retrieval quantile.",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--retrieval-root", type=Path, default=DEFAULT_RETRIEVAL)
    parser.add_argument("--calibration-summary", type=Path, default=CALIBRATION_SUMMARY)
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--selection-only", action="store_true")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--model-revision")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--min-pixels", type=int, default=3136)
    parser.add_argument("--max-pixels", type=int, default=200704)
    args = parser.parse_args()
    selected_datasets = [value.strip() for value in args.datasets.split(",") if value.strip()]
    if not selected_datasets or set(selected_datasets) - set(DATASETS):
        parser.error(f"--datasets must be a subset of {DATASETS}")
    if not args.selection_only and args.model is None:
        parser.error("--model is required for downstream evaluation")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "selection_summary.json"
    state = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    calibrated = json.loads(args.calibration_summary.read_text())["datasets"]
    pending = []
    for dataset in selected_datasets:
        retrieval_path = args.retrieval_root / f"{dataset}_jina_v4_candidates.jsonl.gz"
        manifest = args.split_root / dataset / "test_manifest.json"
        test_qids, test = load_test(retrieval_path, manifest)
        thresholds = {method: calibrated[dataset]["thresholds"][method] for method in METHODS}
        masks = masks_from_thresholds(test, thresholds)
        selected = selection_metrics(test, masks)
        first = selected[METHODS[0]]
        state[dataset] = {
            "status": "selection_complete"
            if args.selection_only
            else state.get(dataset, {}).get("status", "running"),
            "calibration_queries": 1000,
            "test_queries": len(test_qids),
            "candidate_pool": {
                "top_l_cap": 30,
                "test_min": min(map(len, test)),
                "test_max": max(map(len, test)),
                "test_mean": float(np.mean(list(map(len, test)))),
            },
            "thresholds": thresholds,
            "selection": selected,
            "retrievable_queries": first["retrievable_queries"],
            "retrieval_query_coverage_ceiling": first["retrieval_query_coverage_ceiling"],
            **(
                {"downstream": state[dataset]["downstream"]}
                if dataset in state and "downstream" in state[dataset]
                else {}
            ),
        }
        config = DatasetConfig(
            dataset,
            retrieval_path,
            Path(),
            None,
            args.bundle_dir / dataset / "questions.jsonl",
            args.bundle_dir / dataset / "corpus.jsonl",
            args.bundle_dir,
        )
        pending.append((config, test_qids, test, masks))
    summary_path.write_text(json.dumps(state, indent=2) + "\n")
    write_report(args.output_dir / "REPORT.md", state)
    if args.selection_only:
        return

    resolved = {
        config.name: validate_downstream_inputs(config, test_qids, test)
        for config, test_qids, test, _ in pending
    }
    generator: QwenDirectAnswerGenerator | None = None
    for config, test_qids, test, masks in pending:
        questions, corpus = resolved[config.name]
        prediction_path = args.output_dir / f"{config.name}_downstream_predictions.jsonl"
        completed = (
            {str(row["qid"]): row for row in iter_jsonl(prediction_path)}
            if prediction_path.is_file()
            else {}
        )
        for index, (qid, rows) in enumerate(zip(test_qids, test, strict=True), start=1):
            item = questions[qid]
            gold = [str(answer) for answer in item["gold_answers"]]
            prompt = (
                "Answer using only the supplied context. Return only the short answer.\nQuestion: "
                + str(item["question"])
            )
            selected_by_method = {
                method: chunks(rows, masks[method][index - 1], corpus, config.bundle_root)
                for method in METHODS
            }
            expected_chunk_ids = {
                method: tuple(chunk.id for chunk in selected_by_method[method])
                for method in METHODS
            }
            existing = completed.get(qid)
            if existing is not None and all(
                tuple(existing.get("contexts", {}).get(method, {}).get("chunk_ids", ()))
                == expected_chunk_ids[method]
                for method in METHODS
            ):
                continue

            if generator is None:
                generator = QwenDirectAnswerGenerator(
                    args.model,
                    min_pixels=args.min_pixels,
                    max_pixels=args.max_pixels,
                    revision=args.model_revision,
                )
            cache: dict[tuple[str, ...], str] = {}
            predictions: dict[str, str] = {}
            contexts: dict[str, dict[str, Any]] = {}
            for method in METHODS:
                selected = selected_by_method[method]
                key = expected_chunk_ids[method]
                answer = cache.get(key)
                if answer is None:
                    answer = generator.generate(
                        prompt, selected, max_new_tokens=args.max_new_tokens
                    )
                    cache[key] = answer
                predictions[method] = answer
                contexts[method] = {"n_chunks": len(selected), "chunk_ids": list(key)}
            record = {
                "qid": qid,
                "gold_answers": gold,
                "predictions": predictions,
                "contexts": contexts,
                "metrics": {
                    method: {
                        "em": exact_match(predictions[method], gold),
                        "f1": token_f1(predictions[method], gold),
                        "numerical_accuracy": numerical_accuracy(predictions[method], gold),
                    }
                    for method in METHODS
                },
            }
            append_jsonl(prediction_path, record)
            completed[qid] = record
            print(f"[{config.name} {index}/{len(test_qids)}] {qid}", flush=True)
        records = [completed[qid] for qid in test_qids]
        state[config.name]["status"] = "complete"
        state[config.name]["downstream"] = downstream_metrics(records)
        (args.output_dir / f"{config.name}_summary.json").write_text(
            json.dumps(state[config.name], indent=2) + "\n"
        )
        summary_path.write_text(json.dumps(state, indent=2) + "\n")
        write_report(args.output_dir / "REPORT.md", state)
    (args.output_dir / "summary.json").write_text(
        json.dumps({"status": "complete", "datasets": state}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
