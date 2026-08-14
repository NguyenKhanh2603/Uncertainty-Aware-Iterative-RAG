"""Evaluate answer quality, evidence retrieval, and active-retrieval efficiency."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from common import exact_match, mean, read_jsonl, token_f1


def metrics(rows: list[dict]) -> dict:
    em = []
    f1 = []
    initial_recall = []
    cumulative_recall = []
    cumulative_precision = []
    all_evidence = []
    total_retrieval_calls = []
    active_retrieval_calls = []
    active_trigger_rates = []
    additional_trigger_rates = []
    draft_min_probabilities = []
    final_answers = []
    generation_steps = []
    examples_with_active_retrieval = []
    forced_finalizations = []
    finalization_active_retrievals = []
    clean_outputs = []
    exact_token_alignments = []
    extraction_pattern_matches = []
    reasoning_stop_reasons = Counter()
    termination_reasons = Counter()

    for row in rows:
        em.append(exact_match(row["answer"], row["gold_answer"]))
        f1.append(token_f1(row["answer"], row["gold_answer"]))
        gold_urls = {evidence["url"] for evidence in row.get("gold_evidence", [])}
        # v2 records mandatory input-query retrieval separately from active
        # look-ahead retrieval. Fall back to the legacy trace layout so old
        # outputs remain inspectable (but should not be compared as FLARE).
        initial_items = row.get("initial_retrieved")
        if initial_items is None:
            initial_items = row.get("trace", [{}])[0].get("retrieved", [])
        initial_urls = {item["url"] for item in initial_items}
        retrieved_urls = {
            item["url"]
            for step in row.get("trace", [])
            for item in step.get("retrieved", [])
        }
        retrieved_urls |= initial_urls
        if gold_urls:
            initial_recall.append(len(gold_urls & initial_urls) / len(gold_urls))
            cumulative_recall.append(len(gold_urls & retrieved_urls) / len(gold_urls))
            cumulative_precision.append(
                len(gold_urls & retrieved_urls) / len(retrieved_urls) if retrieved_urls else 0.0
            )
            all_evidence.append(float(gold_urls <= retrieved_urls))
        total_retrieval_calls.append(float(row.get("retrieval_calls", 0)))
        steps = row.get("trace", [])
        active_calls = float(row.get(
            "active_retrieval_calls",
            sum(bool(step.get("active_retrieval_triggered", False)) for step in steps),
        ))
        active_retrieval_calls.append(active_calls)
        examples_with_active_retrieval.append(float(active_calls > 0))
        generation_steps.append(float(len(steps)))
        final_answers.append(float(row.get("finished_with_final_answer", False)))
        forced_finalizations.append(float(row.get("forced_finalization_used", False)))
        finalization_active_retrievals.append(float(
            row.get("finalization_active_retrieval_triggered", False)
        ))
        reasoning_stop_reasons[row.get("reasoning_stop_reason", "legacy_or_unknown")] += 1
        termination_reasons[row.get("termination_reason", "legacy_or_unknown")] += 1
        extraction_debug = row.get("answer_extraction_debug", {})
        clean_outputs.append(float(not extraction_debug.get(
            "contains_tokenizer_artifacts",
            "▁" in row.get("generation", "") or "</s>" in row.get("generation", ""),
        )))
        exact_token_alignments.append(float(extraction_debug.get(
            "all_token_alignments_exact", False
        )))
        extraction_pattern_matches.append(float(extraction_debug.get(
            "matched_final_answer_pattern", row.get("finished_with_final_answer", False)
        )))
        active_trigger_rates.append(
            mean(float(step.get("active_retrieval_triggered", step.get("retrieval_triggered", False)))
                 for step in steps) if steps else 0.0
        )
        later_steps = steps[1:]
        additional_trigger_rates.append(
            mean(float(step.get("active_retrieval_triggered", step.get("retrieval_triggered", False)))
                 for step in later_steps)
            if later_steps else 0.0
        )
        draft_min_probabilities.extend(
            float(step["draft_min_probability"])
            for step in steps if step.get("draft_min_probability") is not None
        )

    return {
        "examples": len(rows),
        "answer_em": mean(em),
        "answer_token_f1": mean(f1),
        "final_answer_rate": mean(final_answers),
        "forced_finalization_rate": mean(forced_finalizations),
        "finalization_active_retrieval_rate": mean(finalization_active_retrievals),
        "clean_output_rate": mean(clean_outputs),
        "all_token_alignments_exact_rate": mean(exact_token_alignments),
        "answer_extraction_pattern_match_rate": mean(extraction_pattern_matches),
        "reasoning_stop_reasons": dict(sorted(reasoning_stop_reasons.items())),
        "termination_reasons": dict(sorted(termination_reasons.items())),
        "avg_generation_steps": mean(generation_steps),
        "initial_evidence_recall": mean(initial_recall),
        "cumulative_evidence_recall": mean(cumulative_recall),
        "cumulative_evidence_precision": mean(cumulative_precision),
        "all_evidence_success": mean(all_evidence),
        "avg_retrieval_calls": mean(total_retrieval_calls),
        "avg_active_retrieval_calls": mean(active_retrieval_calls),
        "examples_with_active_retrieval_rate": mean(examples_with_active_retrieval),
        "active_step_trigger_rate": mean(active_trigger_rates),
        "additional_step_trigger_rate": mean(additional_trigger_rates),
        "mean_draft_min_probability": mean(draft_min_probabilities),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    rows = read_jsonl(args.input)
    by_type: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_type[row.get("question_type", "unknown")].append(row)
    result = {
        "implementation_versions": sorted({
            row.get("implementation_version", "legacy_or_unknown") for row in rows
        }),
        "overall": metrics(rows),
        "by_question_type": {name: metrics(items) for name, items in sorted(by_type.items())},
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
