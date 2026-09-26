"""Compare CCE, CONFLARE, and TRAQ retrieval constructions on one fixed split.

This runner is deliberately narrower than claiming a reproduction of all
published end-to-end systems.  All methods receive one frozen Jina Top-30
candidate pool and the same 1,000 calibration / 100 held-out qid manifests.
That makes chunk selection directly comparable.

* CCE follows Conformal-Embedding: one score for every labelled relevant
  (question, chunk) pair.
* CONFLARE follows its ``QuestionEvaluation``/``ConformalRetrievalQA``
  construction: one score for the first known relevant chunk for a calibration
  question.  Since the benchmark provides human questions and support ids, the
  highest-scoring labelled support is the deterministic analogue of that
  first relevant retrieval; no synthetic question generation is substituted.
* TRAQ follows its retrieval component and Bonferroni allocation: one best
  true-retrieval score per calibration question and alpha_R = alpha / 2.

The full TRAQ answer prediction set additionally needs multi-sample generation,
semantic clustering, a generator-calibration score, and a second allocation
stage.  It has an answer-set coverage target rather than a chunk-pruning
target, so it is recorded as out of scope for this selection/downstream table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from eval.metrics import exact_match, numerical_accuracy, token_f1
from research.internal_state_rag.run_all_datasets_cosine_six_methods import (
    CONFIGS,
    DatasetConfig,
    load_dataset,
    query_level_cosine_mask,
    validate_downstream_inputs,
)
from research.internal_state_rag.run_downstream_qa_conformal_baselines import (
    QwenDirectAnswerGenerator,
    append_jsonl,
    chunks,
    iter_jsonl,
)


METHODS = (
    "fixed_top10",
    "cce_conformal_embedding_jina_alpha_0.10",
    "conflare_source_question_jina_alpha_0.10",
    "traq_retrieval_bonferroni_jina_alpha_0.10",
    "query_level_all_support_cosine_alpha_0.10",
)
DISPLAY = {
    "fixed_top10": "Fixed Top-10",
    "cce_conformal_embedding_jina_alpha_0.10": "CCE Conformal-Embedding (Jina adaptation)",
    "conflare_source_question_jina_alpha_0.10": "CONFLARE source-question (Jina adaptation)",
    "traq_retrieval_bonferroni_jina_alpha_0.10": "TRAQ retrieval, Bonferroni (Jina adaptation)",
    "query_level_all_support_cosine_alpha_0.10": "Query-level all-support cosine",
}


def support_scores_by_query(rows: Sequence[Sequence[dict[str, Any]]]) -> list[np.ndarray]:
    """Return positive candidate scores separately for every query."""

    return [
        np.asarray(
            [float(row["cosine_score"]) for row in query if row["support_label"] == "support"],
            dtype=float,
        )
        for query in rows
    ]


def cce_embedding_threshold(per_query: Sequence[np.ndarray], alpha: float) -> tuple[float, dict[str, Any]]:
    """Apply CCE's positive-pair nonconformity quantile on raw Jina cosine."""

    positive = np.concatenate([scores for scores in per_query if len(scores)])
    if not len(positive):
        raise ValueError("CCE calibration has no retrieved positive pairs")
    nonconformity = 1.0 - positive
    # CCE retains A(q,c) <= tau. ``higher`` is the conservative empirical
    # quantile corresponding to the finite calibration bank.
    tau = float(np.quantile(nonconformity, 1.0 - alpha, method="higher"))
    return 1.0 - tau, {
        "positive_pairs": int(len(positive)),
        "nonconformity_threshold": tau,
        "similarity_threshold": 1.0 - tau,
        "calibration_unit": "labelled_relevant_question_chunk_pair",
    }


def conflare_threshold(per_query: Sequence[np.ndarray], alpha: float) -> tuple[float, dict[str, Any]]:
    """Mirror CONFLARE's one-record-per-question distance calibration."""

    # The official evaluator walks a ranked corpus and stops at the first
    # relevant chunk.  With frozen labels, that is the support with maximum
    # similarity / minimum cosine distance for each retrievable query.
    first_relevant_similarity = np.asarray(
        [scores.max() for scores in per_query if len(scores)], dtype=float
    )
    if not len(first_relevant_similarity):
        raise ValueError("CONFLARE calibration has no retrievable questions")
    distances = 1.0 - first_relevant_similarity
    distance_threshold = float(np.percentile(distances, 100.0 * (1.0 - alpha)))
    return 1.0 - distance_threshold, {
        "calibration_records": int(len(first_relevant_similarity)),
        "distance_threshold": distance_threshold,
        "similarity_threshold": 1.0 - distance_threshold,
        "calibration_unit": "one_first_known_relevant_chunk_per_question",
        "question_source": "benchmark_human_question_with_labelled_support",
    }


def traq_retrieval_threshold(
    per_query: Sequence[np.ndarray], alpha: float
) -> tuple[float, dict[str, Any]]:
    """Mirror TRAQ's Bonferroni retrieval-score threshold.

    TRAQ's retrieval component calibrates a single best true-retrieval score
    per question.  Its Bonferroni baseline splits total alpha equally between
    retrieval and answer prediction sets, hence alpha_R = alpha / 2.
    """

    best_true = np.asarray([scores.max() for scores in per_query if len(scores)], dtype=float)
    if not len(best_true):
        raise ValueError("TRAQ calibration has no retrievable questions")
    alpha_retrieval = alpha / 2.0
    threshold = float(np.quantile(best_true, alpha_retrieval, method="lower"))
    return threshold, {
        "retrievable_calibration_queries": int(len(best_true)),
        "similarity_threshold": threshold,
        "total_alpha": alpha,
        "alpha_retrieval": alpha_retrieval,
        "alpha_answer": alpha_retrieval,
        "calibration_unit": "one_best_true_retrieval_score_per_question",
        "scope": "retrieval_component_only; no_answer_prediction_set",
    }


def score_mask(rows: Sequence[Sequence[dict[str, Any]]], threshold: float, *, strict: bool) -> list[np.ndarray]:
    compare = (lambda score: score > threshold) if strict else (lambda score: score >= threshold)
    return [
        np.asarray([compare(float(item["cosine_score"])) for item in query], dtype=bool)
        for query in rows
    ]


def build_masks(
    calibration: list[list[dict[str, Any]]], test: list[list[dict[str, Any]]], alpha: float
) -> tuple[dict[str, list[np.ndarray]], dict[str, Any]]:
    per_query = support_scores_by_query(calibration)
    cce, cce_audit = cce_embedding_threshold(per_query, alpha)
    conflare, conflare_audit = conflare_threshold(per_query, alpha)
    traq, traq_audit = traq_retrieval_threshold(per_query, alpha)
    query_masks, query_audit = query_level_cosine_mask(calibration, test, alpha)
    masks = {
        "fixed_top10": [
            np.asarray([int(row["rank"]) <= 10 for row in rows], dtype=bool) for rows in test
        ],
        "cce_conformal_embedding_jina_alpha_0.10": score_mask(test, cce, strict=False),
        # Official CONFLARE accepts a document only when distance < threshold.
        "conflare_source_question_jina_alpha_0.10": score_mask(test, conflare, strict=True),
        "traq_retrieval_bonferroni_jina_alpha_0.10": score_mask(test, traq, strict=False),
        "query_level_all_support_cosine_alpha_0.10": query_masks,
    }
    return masks, {
        "cce_conformal_embedding_jina_alpha_0.10": cce_audit,
        "conflare_source_question_jina_alpha_0.10": conflare_audit,
        "traq_retrieval_bonferroni_jina_alpha_0.10": traq_audit,
        "query_level_all_support_cosine_alpha_0.10": query_audit,
        "calibration_queries": len(calibration),
        "calibration_retrievable_queries": int(sum(bool(len(scores)) for scores in per_query)),
    }


def selection_metrics(
    test: Sequence[Sequence[dict[str, Any]]], masks: dict[str, list[np.ndarray]]
) -> dict[str, dict[str, float]]:
    """Evaluate evidence retention without treating missing retrieval as success."""

    result: dict[str, dict[str, float]] = {}
    for method, method_masks in masks.items():
        selected_total = selected_supports = support_total = empty = 0
        retrievable = any_support = all_support = 0
        for rows, mask in zip(test, method_masks, strict=True):
            labels = np.asarray([row["support_label"] == "support" for row in rows], dtype=bool)
            chosen_supports = labels & mask
            selected_total += int(mask.sum())
            selected_supports += int(chosen_supports.sum())
            support_total += int(labels.sum())
            empty += int(not mask.any())
            if labels.any():
                retrievable += 1
                any_support += int(chosen_supports.any())
                all_support += int(chosen_supports.sum() == labels.sum())
        result[method] = {
            "mean_chunks": selected_total / len(test),
            "precision": selected_supports / selected_total if selected_total else 0.0,
            "support_recall": selected_supports / support_total if support_total else 0.0,
            "empty_rate": empty / len(test),
            "retrievable_queries": retrievable,
            "retrieval_query_coverage_ceiling": retrievable / len(test),
            "query_any_support_conditional": any_support / retrievable if retrievable else 0.0,
            "query_all_support_conditional": all_support / retrievable if retrievable else 0.0,
            "selected_chunks": selected_total,
            "selected_supports": selected_supports,
            "reserve_supports": support_total,
        }
    return result


def metrics(records: Sequence[dict[str, Any]]) -> dict[str, dict[str, float]]:
    return {
        method: {
            name: float(np.mean([record["metrics"][method][name] for record in records]))
            for name in ("em", "f1", "numerical_accuracy")
        }
        for method in METHODS
    }


def write_report(output: Path, state: dict[str, dict[str, Any]]) -> None:
    lines = [
        "# Literature-protocol comparison: 1,000 calibration / 100 test",
        "",
        "Every row uses the same frozen Jina Top-30 candidates and the same disjoint qid manifests. CCE, CONFLARE, and TRAQ follow their distinct published **retrieval calibration units**. They are adapted to benchmark human questions and frozen labelled supports; the original paper models, source corpora, and end-to-end generators are not substituted or claimed. TRAQ's row is its retrieval component with Bonferroni allocation, not the full answer-prediction-set method.",
        "",
    ]
    for dataset, result in state.items():
        lines.extend(
            [
                f"## {dataset} ({result['status']}; calibration={result['calibration_queries']}, test={result['test_queries']})",
                "",
                "| Method | Chunks | Precision | Support recall | Empty | Any support* | All support* | EM | F1 | Numeric |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for method in METHODS:
            selected = result["selection"][method]
            qa = result.get("downstream", {}).get(method)
            values = (
                f"{qa['em']:.3f} | {qa['f1']:.3f} | {qa['numerical_accuracy']:.3f}"
                if qa
                else "pending | pending | pending"
            )
            lines.append(
                f"| {DISPLAY[method]} | {selected['mean_chunks']:.2f} | "
                f"{selected['precision']:.1%} | {selected['support_recall']:.1%} | "
                f"{selected['empty_rate']:.1%} | {selected['query_any_support_conditional']:.1%} | "
                f"{selected['query_all_support_conditional']:.1%} | {values} |"
            )
        lines.extend(
            [
                "",
                "*Any/all-support coverage is conditional on at least one labelled support being present in the frozen Top-30. The retrieval ceiling is recorded in the JSON below; missing retrieval is never counted as coverage.*",
                "",
                "```json",
                json.dumps(result["thresholds"], indent=2),
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## Scope and provenance",
            "",
            "- CCE source: `baselines/conformal-context-engineering` (commit `91732d6058267f180ba9f47873d743288b2625af`).",
            "- CONFLARE source: `baselines/conflare` (commit `ce081a45fb452704daa87f3b37f601b4accc7a82`).",
            "- TRAQ source: `baselines/TRAQ` (commit `e9b66b5b7fd8fcd0b3c3dbfebc5a3391ca79aa55`).",
            "- Exact qid manifests are copied under `splits/`; `SPLIT_INTEGRITY.json` verifies calibration/test disjointness.",
            "- Downstream EM/F1 is a shared deterministic Qwen direct-answer diagnostic. It is not TRAQ answer-set coverage and must not be presented as TRAQ's end-to-end guarantee.",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--plan-root",
        type=Path,
        default=Path(
            "research/internal_state_rag/results/"
            "all_datasets_cosine_six_methods_1000cal_100test_2026_09_26/splits"
        ),
    )
    parser.add_argument("--datasets", default=",".join(CONFIGS))
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--selection-only", action="store_true")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--min-pixels", type=int, default=3136)
    parser.add_argument("--max-pixels", type=int, default=200704)
    args = parser.parse_args()
    if not 0 < args.alpha < 1:
        raise ValueError("--alpha must be in (0, 1)")
    names = [name.strip() for name in args.datasets.split(",") if name.strip()]
    if not names or any(name not in CONFIGS for name in names):
        raise ValueError(f"--datasets must be a non-empty subset of {tuple(CONFIGS)}")
    if not args.selection_only and args.model is None:
        raise ValueError("--model is required unless --selection-only is set")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_root = args.output_dir / "splits"
    split_root.mkdir(exist_ok=True)
    state: dict[str, dict[str, Any]] = {}
    pending: list[tuple[DatasetConfig, list[str], list[list[dict[str, Any]]], dict[str, list[np.ndarray]]]] = []
    split_audit: dict[str, Any] = {"protocol": "literature_retrieval_constructions_on_inverse_1000cal_100test", "datasets": {}}

    for name in names:
        config = CONFIGS[name]
        calibration_plan = args.plan_root / name / "calibration_manifest.json"
        test_plan = args.plan_root / name / "test_manifest.json"
        calibration_qids, calibration, test_qids, test = load_dataset(
            config, calibration_plan=calibration_plan, test_plan=test_plan
        )
        if len(calibration_qids) != 1000 or len(test_qids) != 100:
            raise ValueError(f"{name}: require exactly 1,000 calibration and 100 test qids")
        if set(calibration_qids) & set(test_qids):
            raise ValueError(f"{name}: calibration/test overlap")
        for source, target in ((calibration_plan, split_root / name / "calibration_manifest.json"), (test_plan, split_root / name / "test_manifest.json")):
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.read_text(encoding="utf-8") != source.read_text(encoding="utf-8"):
                raise RuntimeError(f"{name}: existing copied manifest differs: {target}")
            if not target.exists():
                target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        masks, thresholds = build_masks(calibration, test, args.alpha)
        state[name] = {
            "status": "selection_complete" if args.selection_only else "running",
            "calibration_queries": len(calibration_qids),
            "test_queries": len(test_qids),
            "thresholds": thresholds,
            "selection": selection_metrics(test, masks),
        }
        split_audit["datasets"][name] = {
            "calibration_queries": len(calibration_qids),
            "test_queries": len(test_qids),
            "overlap": 0,
            "frozen_top_l": 30,
        }
        pending.append((config, test_qids, test, masks))

    (split_root / "SPLIT_INTEGRITY.json").write_text(
        json.dumps(split_audit, indent=2) + "\n", encoding="utf-8"
    )
    write_report(args.output_dir / "REPORT.md", state)
    (args.output_dir / "selection_summary.json").write_text(
        json.dumps(state, indent=2) + "\n", encoding="utf-8"
    )
    if args.selection_only:
        return

    resolved = {
        config.name: validate_downstream_inputs(config, test_qids, test)
        for config, test_qids, test, _masks in pending
    }
    generator = QwenDirectAnswerGenerator(
        args.model, min_pixels=args.min_pixels, max_pixels=args.max_pixels
    )
    for config, test_qids, test, masks in pending:
        questions, corpus = resolved[config.name]
        prediction_path = args.output_dir / f"{config.name}_downstream_predictions.jsonl"
        completed = {str(row["qid"]): row for row in iter_jsonl(prediction_path)} if prediction_path.exists() else {}
        for index, (qid, rows) in enumerate(zip(test_qids, test, strict=True), start=1):
            if qid in completed:
                continue
            item = questions[qid]
            gold = [str(answer) for answer in item["gold_answers"]]
            prompt = "Answer using only the supplied context. Return only the short answer.\nQuestion: " + str(item["question"])
            cache: dict[tuple[str, ...], str] = {}
            predictions: dict[str, str] = {}
            contexts: dict[str, dict[str, Any]] = {}
            for method in METHODS:
                selected = chunks(rows, masks[method][index - 1], corpus, config.bundle_root)
                key = tuple(chunk.id for chunk in selected)
                answer = cache.get(key)
                if answer is None:
                    answer = generator.generate(prompt, selected, max_new_tokens=args.max_new_tokens)
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
                        "em": exact_match(answer, gold),
                        "f1": token_f1(answer, gold),
                        "numerical_accuracy": numerical_accuracy(answer, gold),
                    }
                    for method, answer in predictions.items()
                },
            }
            append_jsonl(prediction_path, record)
            completed[qid] = record
            print(f"[{config.name} {index}/{len(test_qids)}] {qid}", flush=True)
        records = [completed[qid] for qid in test_qids]
        state[config.name]["status"] = "complete"
        state[config.name]["downstream"] = metrics(records)
        (args.output_dir / f"{config.name}_summary.json").write_text(
            json.dumps(state[config.name], indent=2) + "\n", encoding="utf-8"
        )
        write_report(args.output_dir / "REPORT.md", state)
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
