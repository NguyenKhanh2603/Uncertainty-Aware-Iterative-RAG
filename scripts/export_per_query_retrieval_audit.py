"""Export query- and chunk-level audits for conformal and attention runs."""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping
from zipfile import ZIP_DEFLATED, ZipFile

from tqdm import tqdm

from uncertainty_rag.core.conformal_selection import (
    backfill_rejected_indices,
    benjamini_yekutieli,
)


def metric_counts(
    candidates: list[Mapping[str, Any]], selected: set[int]
) -> dict[str, int | float | None]:
    support_total = sum(item["support_label"] == "support" for item in candidates)
    true_positive = sum(candidates[index]["support_label"] == "support" for index in selected)
    false_positive = sum(candidates[index]["support_label"] == "false" for index in selected)
    selected_labelled = true_positive + false_positive
    return {
        "selected": len(selected),
        "tp": true_positive,
        "fp": false_positive,
        "fn": support_total - true_positive,
        "support_total": support_total,
        "precision": true_positive / selected_labelled if selected_labelled else None,
        "recall": true_positive / support_total if support_total else None,
    }


def selected_policies(
    decisions: list[Mapping[str, Any]], *, alpha: float, max_context: int
) -> tuple[dict[str, set[int]], set[int], dict[str, Any]]:
    retrieval_order = sorted(
        range(len(decisions)),
        key=lambda index: (int(decisions[index]["rank"]), str(decisions[index]["chunk_id"])),
    )
    top_k = set(retrieval_order[:max_context])
    by = benjamini_yekutieli([float(item["p_value"]) for item in decisions], alpha)
    with_backfill, without_backfill = backfill_rejected_indices(
        [int(item["rank"]) for item in decisions],
        [str(item["chunk_id"]) for item in decisions],
        by.rejected_indices,
        max_context,
    )
    return (
        {
            "fixed_top_k": top_k,
            "conformal_no_backfill": set(without_backfill),
            "conformal_backfill": set(with_backfill),
        },
        set(by.rejected_indices),
        {
            "by_rejections": len(by.rejected_indices),
            "by_cutoff_rank": by.cutoff_rank,
            "by_cutoff_p_value": by.cutoff_p_value,
            "by_harmonic_number": by.harmonic_number,
        },
    )


def iter_gzip_jsonl_from_zip(archive: ZipFile, entry: str) -> Iterable[dict[str, Any]]:
    with archive.open(entry) as compressed:
        with gzip.GzipFile(fileobj=compressed) as binary:
            with io.TextIOWrapper(binary, encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        yield json.loads(line)


def iter_jsonl_path(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_retrieval_metadata(
    results_zip: Path | None,
    inputs: list[Path] | None,
    split_role: str,
) -> dict[tuple[str, str], Any]:
    queries: dict[tuple[str, str], dict[str, Any]] = {}
    sources: list[Iterable[dict[str, Any]]] = []
    if results_zip is not None:
        archive = ZipFile(results_zip)
        entries = sorted(
            name for name in archive.namelist() if name.endswith("_retrieval.jsonl.gz")
        )
        sources.extend(iter_gzip_jsonl_from_zip(archive, entry) for entry in entries)
    else:
        archive = None
        sources.extend(iter_jsonl_path(path) for path in inputs or [])
    try:
        for source in sources:
            for row in source:
                if row["split_role"] != split_role:
                    continue
                key = (str(row["dataset"]), str(row["qid"]))
                query = queries.setdefault(
                    key,
                    {
                        "question": str(row["query_text"]),
                        "source_split": str(row["source_split"]),
                        "query_type": str(row["query_type"]),
                        "retrieved_l": int(row["retrieved_l"]),
                        "candidate_pool_size": int(row["candidate_pool_size"]),
                        "chunks": {},
                    },
                )
                query["chunks"][str(row["chunk_id"])] = row
    finally:
        if archive is not None:
            archive.close()
    return queries


def load_selection(path: Path, strategy: str) -> list[dict[str, Any]]:
    rows = []
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            wrapper = json.loads(line)
            if strategy not in wrapper["strategies"]:
                raise KeyError(f"Missing selection strategy {strategy!r}")
            rows.append(wrapper["strategies"][strategy])
    return rows


def load_optional_corpus(
    bundle_dir: Path | None, needed_ids: Mapping[str, set[str]]
) -> dict[tuple[str, str], dict[str, str]]:
    content: dict[tuple[str, str], dict[str, str]] = {}
    if bundle_dir is None:
        return content
    for dataset, chunk_ids in needed_ids.items():
        corpus_path = bundle_dir / dataset / "corpus.jsonl"
        if not corpus_path.is_file():
            continue
        with corpus_path.open(encoding="utf-8") as handle:
            for line in tqdm(handle, desc=f"Join {dataset} corpus", unit="chunk"):
                if not line.strip():
                    continue
                row = json.loads(line)
                chunk_id = str(row["id"])
                if chunk_id in chunk_ids:
                    content[(dataset, chunk_id)] = {
                        "content": str(row.get("content", "")),
                        "caption": str(row.get("caption", "")),
                    }
    return content


def json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def write_csv(path: Path, rows: list[Mapping[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_nested_jsonl(
    path: Path,
    query_rows: list[Mapping[str, Any]],
    chunk_rows: list[Mapping[str, Any]],
) -> None:
    chunks_by_query: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for chunk in chunk_rows:
        chunks_by_query[(str(chunk["dataset"]), str(chunk["qid"]))].append(chunk)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for query in query_rows:
            key = (str(query["dataset"]), str(query["qid"]))
            handle.write(
                json.dumps(
                    {"query": query, "chunks": chunks_by_query[key]},
                    ensure_ascii=False,
                )
                + "\n"
            )


def export_conformal(
    *,
    results_zip: Path | None,
    inputs: list[Path] | None,
    selection_path: Path,
    output_dir: Path,
    official_bundle_dir: Path | None,
    split_role: str,
    strategy: str,
    alpha: float,
    max_context: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    retrieval = load_retrieval_metadata(results_zip, inputs, split_role)
    selections = load_selection(selection_path, strategy)
    needed_ids: dict[str, set[str]] = defaultdict(set)
    for base in selections:
        needed_ids[str(base["dataset"])].update(str(item["chunk_id"]) for item in base["decisions"])
    content = load_optional_corpus(official_bundle_dir, needed_ids)
    query_rows: list[dict[str, Any]] = []
    chunk_rows: list[dict[str, Any]] = []

    for base in tqdm(selections, desc="Audit conformal", unit="query"):
        dataset = str(base["dataset"])
        qid = str(base["qid"])
        query = retrieval[(dataset, qid)]
        decisions = base["decisions"]
        policies, by_rejected, by_summary = selected_policies(
            decisions, alpha=alpha, max_context=max_context
        )
        metrics = {
            method: metric_counts(decisions, selected) for method, selected in policies.items()
        }
        row: dict[str, Any] = {
            "pipeline": "conformal",
            "dataset": dataset,
            "qid": qid,
            "question": query["question"],
            "source_split": query["source_split"],
            "split_role": split_role,
            "query_type": query["query_type"],
            "strategy": strategy,
            "alpha": alpha,
            "max_context": max_context,
            "candidate_pool_size": query["candidate_pool_size"],
            "retrieved_l": query["retrieved_l"],
            **by_summary,
        }
        for method, result in metrics.items():
            for field, value in result.items():
                row[f"{method}_{field}"] = value
            selected_order = sorted(policies[method], key=lambda index: decisions[index]["rank"])
            row[f"{method}_ranks"] = json_cell(
                [int(decisions[index]["rank"]) for index in selected_order]
            )
            row[f"{method}_chunk_ids"] = json_cell(
                [str(decisions[index]["chunk_id"]) for index in selected_order]
            )
        query_rows.append(row)

        for index, decision in enumerate(decisions):
            chunk_id = str(decision["chunk_id"])
            retrieval_row = query["chunks"][chunk_id]
            payload = content.get((dataset, chunk_id), {})
            chunk_rows.append(
                {
                    "pipeline": "conformal",
                    "dataset": dataset,
                    "qid": qid,
                    "question": query["question"],
                    "chunk_id": chunk_id,
                    "source_doc_id": decision["source_doc_id"],
                    "rank": int(decision["rank"]),
                    "modality_rank": retrieval_row.get("modality_rank"),
                    "modality": decision["modality"],
                    "content": payload.get("content", ""),
                    "caption": payload.get("caption", ""),
                    "content_status": "joined" if payload else "not_packaged_in_results_zip",
                    "support_label": decision["support_label"],
                    "is_gold_support": decision["support_label"] == "support",
                    "cosine_score": decision["cosine_score"],
                    "p_value": decision["p_value"],
                    "bank_condition": json_cell(decision["condition"]),
                    "bank_size": decision["bank_size"],
                    "bank_status": decision["bank_status"],
                    "by_rejected_current": index in by_rejected,
                    "fixed_top_k_selected": index in policies["fixed_top_k"],
                    "conformal_no_backfill_selected": index in policies["conformal_no_backfill"],
                    "conformal_backfill_selected": index in policies["conformal_backfill"],
                    "is_backfill": (
                        index in policies["conformal_backfill"]
                        and int(decision["rank"]) > max_context
                    ),
                    "query_fixed_top_k_precision": metrics["fixed_top_k"]["precision"],
                    "query_fixed_top_k_recall": metrics["fixed_top_k"]["recall"],
                    "query_conformal_precision": metrics["conformal_backfill"]["precision"],
                    "query_conformal_recall": metrics["conformal_backfill"]["recall"],
                }
            )

    query_fields = list(query_rows[0])
    chunk_fields = list(chunk_rows[0])
    write_csv(output_dir / "conformal_query_metrics.csv", query_rows, query_fields)
    write_csv(output_dir / "conformal_chunk_audit.csv", chunk_rows, chunk_fields)
    write_nested_jsonl(output_dir / "conformal_per_query_audit.jsonl.gz", query_rows, chunk_rows)
    return query_rows, chunk_rows


def parse_json_field(value: Any, default: Any) -> Any:
    if value is None:
        return default
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return default
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return default


def load_attention_questions(bundle_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    questions = {}
    for path in sorted(bundle_dir.glob("*/questions.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    questions[(str(row["dataset"]), str(row["qid"]))] = row
    return questions


def attention_observations(trace: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    observations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for iteration_index, iteration in enumerate(trace.get("iterations", []), start=1):
        for chunk in iteration.get("chunks", []):
            observations[str(chunk["chunk_id"])].append(
                {
                    "iteration": iteration_index,
                    "attention_score": chunk.get("attention_score"),
                    "attention_mass": chunk.get("attention_mass"),
                    "token_count": chunk.get("token_count"),
                    "span_status": chunk.get("span_status"),
                }
            )
    return observations


def export_attention(
    *, attention_zip: Path, attention_bundle_dir: Path, output_dir: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    questions = load_attention_questions(attention_bundle_dir)
    with ZipFile(attention_zip) as archive:
        raw = archive.read("attention_uq_800q_results.csv").decode("utf-8-sig")
    result_rows = list(csv.DictReader(io.StringIO(raw)))
    query_rows: list[dict[str, Any]] = []
    chunk_rows: list[dict[str, Any]] = []

    for result in tqdm(result_rows, desc="Audit attention", unit="query"):
        dataset = str(result["dataset"])
        qid = str(result["qid"])
        question_row = questions[(dataset, qid)]
        chunks = question_row["chunks"]
        top10_indices = set(range(min(10, len(chunks))))
        trace = parse_json_field(result.get("attention_trace"), {})
        final_chunks = {str(item["chunk_id"]): item for item in trace.get("final_chunks", [])}
        final_ids = set(final_chunks)
        observations = attention_observations(trace)
        attention_indices = {
            index for index, chunk in enumerate(chunks) if str(chunk["id"]) in final_ids
        }
        normalized_candidates = [
            {"support_label": "support" if chunk["is_support"] else "false"} for chunk in chunks
        ]
        metrics = {
            "attention_top10": metric_counts(normalized_candidates, top10_indices),
            "attention_final": metric_counts(normalized_candidates, attention_indices),
        }
        query_rows.append(
            {
                "pipeline": "attention_total_mass",
                "dataset": dataset,
                "qid": qid,
                "question": result["question"],
                "gold_answers": result["gold_answers"],
                "status": result["status"],
                "attention_status": result["attention_status"],
                "oom_stages": result["oom_stages"],
                "retrieved_l": len(chunks),
                "support_total": sum(bool(chunk["is_support"]) for chunk in chunks),
                "attention_iterations": result["attention_iterations"],
                "attention_initial_entropy": result["attention_initial_entropy"],
                "attention_final_entropy": result["attention_final_entropy"],
                "top10_answer": result["top10_answer"],
                "top10_em": result["top10_em"],
                "top10_f1": result["top10_f1"],
                "attention_answer": result["attention_answer"],
                "attention_em": result["attention_em"],
                "attention_f1": result["attention_f1"],
                **{
                    f"{method}_{field}": value
                    for method, values in metrics.items()
                    for field, value in values.items()
                },
                "attention_final_ranks": json_cell(
                    [index + 1 for index in sorted(attention_indices)]
                ),
                "attention_final_chunk_ids": json_cell(
                    [str(chunks[index]["id"]) for index in sorted(attention_indices)]
                ),
            }
        )

        for index, chunk in enumerate(chunks):
            chunk_id = str(chunk["id"])
            final = final_chunks.get(chunk_id, {})
            chunk_rows.append(
                {
                    "pipeline": "attention_total_mass",
                    "dataset": dataset,
                    "qid": qid,
                    "question": result["question"],
                    "gold_answers": result["gold_answers"],
                    "chunk_id": chunk_id,
                    "rank": index + 1,
                    "modality": chunk["modality"],
                    "content": chunk["content"],
                    "is_gold_support": bool(chunk["is_support"]),
                    "top10_selected": index in top10_indices,
                    "attention_final_selected": chunk_id in final_ids,
                    "final_attention_score": final.get("attention_score"),
                    "final_attention_mass": final.get("attention_mass"),
                    "final_token_count": final.get("token_count"),
                    "attention_observations": json_cell(observations.get(chunk_id, [])),
                    "query_top10_precision": metrics["attention_top10"]["precision"],
                    "query_top10_recall": metrics["attention_top10"]["recall"],
                    "query_attention_precision": metrics["attention_final"]["precision"],
                    "query_attention_recall": metrics["attention_final"]["recall"],
                    "attention_answer": result["attention_answer"],
                    "attention_em": result["attention_em"],
                    "attention_f1": result["attention_f1"],
                    "attention_status": result["attention_status"],
                }
            )

    write_csv(output_dir / "attention_query_metrics.csv", query_rows, list(query_rows[0]))
    write_csv(output_dir / "attention_chunk_audit.csv", chunk_rows, list(chunk_rows[0]))
    write_nested_jsonl(output_dir / "attention_per_query_audit.jsonl.gz", query_rows, chunk_rows)
    return query_rows, chunk_rows


def numeric(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def summarize_method(rows: list[Mapping[str, Any]], prefix: str) -> dict[str, int | float | None]:
    selected = sum(int(row[f"{prefix}_selected"]) for row in rows)
    true_positive = sum(int(row[f"{prefix}_tp"]) for row in rows)
    false_positive = sum(int(row[f"{prefix}_fp"]) for row in rows)
    supports = sum(int(row[f"{prefix}_support_total"]) for row in rows)
    precisions = [numeric(row[f"{prefix}_precision"]) for row in rows]
    recalls = [numeric(row[f"{prefix}_recall"]) for row in rows]
    precisions = [value for value in precisions if value is not None]
    recalls = [value for value in recalls if value is not None]
    return {
        "queries": len(rows),
        "average_selected_chunks": selected / len(rows),
        "empty_context_rate": sum(int(row[f"{prefix}_selected"]) == 0 for row in rows) / len(rows),
        "micro_precision": true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else None,
        "micro_recall": true_positive / supports if supports else None,
        "macro_query_precision_nonempty": sum(precisions) / len(precisions) if precisions else None,
        "macro_query_recall": sum(recalls) / len(recalls) if recalls else None,
        "queries_with_any_selected_support_rate": sum(int(row[f"{prefix}_tp"]) > 0 for row in rows)
        / len(rows),
    }


def dataset_summaries(
    conformal: list[dict[str, Any]], attention: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    output = []
    definitions = [
        ("conformal", conformal, "fixed_top_k", "fixed_top_k"),
        ("conformal", conformal, "conformal_no_backfill", "conformal_no_backfill"),
        ("conformal", conformal, "conformal_backfill", "conformal_backfill"),
        ("attention_total_mass", attention, "attention_top10", "attention_top10"),
        ("attention_total_mass", attention, "attention_final", "attention_final_all_attempted"),
        (
            "attention_total_mass",
            [row for row in attention if row["attention_status"] == "ok"],
            "attention_final",
            "attention_final_completed_only",
        ),
    ]
    for pipeline, rows, prefix, method in definitions:
        if not rows:
            continue
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row["dataset"])].append(row)
        grouped["ALL"] = rows
        for dataset, members in sorted(grouped.items()):
            output.append(
                {
                    "pipeline": pipeline,
                    "method": method,
                    "dataset": dataset,
                    **summarize_method(members, prefix),
                }
            )
    return output


def overlap_rows(
    conformal: list[dict[str, Any]], attention: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    attention_by_qid = {(str(row["dataset"]), str(row["qid"])): row for row in attention}
    rows = []
    for conformal_row in conformal:
        key = (str(conformal_row["dataset"]), str(conformal_row["qid"]))
        attention_row = attention_by_qid.get(key)
        if attention_row is None:
            continue
        rows.append(
            {
                "dataset": key[0],
                "question": conformal_row["question"],
                "conformal_qid": conformal_row["qid"],
                "attention_qid": attention_row["qid"],
                "same_qid": True,
                "conformal_precision": conformal_row["conformal_backfill_precision"],
                "conformal_recall": conformal_row["conformal_backfill_recall"],
                "attention_precision": attention_row["attention_final_precision"],
                "attention_recall": attention_row["attention_final_recall"],
                "attention_answer": attention_row["attention_answer"],
                "attention_em": attention_row["attention_em"],
                "attention_f1": attention_row["attention_f1"],
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    conformal_source = parser.add_mutually_exclusive_group(required=True)
    conformal_source.add_argument("--conformal-results-zip", type=Path)
    conformal_source.add_argument("--conformal-input", type=Path, nargs="+")
    parser.add_argument("--conformal-selection", type=Path, required=True)
    parser.add_argument("--official-bundle-dir", type=Path)
    parser.add_argument("--attention-results-zip", type=Path)
    parser.add_argument("--attention-bundle-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-role", default="development")
    parser.add_argument("--strategy", default="modality_aware")
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--max-context", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.attention_results_zip is None) != (args.attention_bundle_dir is None):
        raise ValueError(
            "--attention-results-zip and --attention-bundle-dir must be supplied together"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    conformal_queries, conformal_chunks = export_conformal(
        results_zip=args.conformal_results_zip,
        inputs=args.conformal_input,
        selection_path=args.conformal_selection,
        output_dir=args.output_dir,
        official_bundle_dir=args.official_bundle_dir,
        split_role=args.split_role,
        strategy=args.strategy,
        alpha=args.alpha,
        max_context=args.max_context,
    )
    if args.attention_results_zip is not None:
        attention_queries, attention_chunks = export_attention(
            attention_zip=args.attention_results_zip,
            attention_bundle_dir=args.attention_bundle_dir,
            output_dir=args.output_dir,
        )
    else:
        attention_queries, attention_chunks = [], []
    summaries = dataset_summaries(conformal_queries, attention_queries)
    write_csv(args.output_dir / "dataset_method_summary.csv", summaries, list(summaries[0]))
    overlaps = overlap_rows(conformal_queries, attention_queries)
    overlap_fields = list(overlaps[0]) if overlaps else ["dataset", "qid"]
    write_csv(args.output_dir / "cross_run_question_overlap.csv", overlaps, overlap_fields)

    manifest = {
        "audit_version": 1,
        "conformal": {
            "queries": len(conformal_queries),
            "chunks": len(conformal_chunks),
            "split_role": args.split_role,
            "strategy": args.strategy,
            "alpha": args.alpha,
            "max_context": args.max_context,
            "chunks_with_content": sum(bool(row["content"]) for row in conformal_chunks),
            "chunks_without_content": sum(not bool(row["content"]) for row in conformal_chunks),
        },
        "attention": (
            {
                "queries": len(attention_queries),
                "chunks": len(attention_chunks),
                "content_joined": True,
            }
            if attention_queries
            else None
        ),
        "cross_run_question_overlap": len(overlaps),
        "paired_comparison_valid": False,
        "paired_comparison_warning": (
            "The conformal and attention runs use different query samples and candidate pools. "
            "Only exactly matching dataset/qid pairs are written to the overlap CSV."
        ),
        "metric_definitions": {
            "query_precision": "selected support chunks / selected labelled chunks",
            "query_recall": "selected support chunks / support chunks in that run's reserve",
            "micro": "sum TP divided by summed denominator",
            "macro": "mean of per-query values; precision excludes empty contexts",
        },
    }
    (args.output_dir / "audit_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    archive_path = Path(
        shutil.make_archive(str(args.output_dir), "zip", root_dir=args.output_dir, base_dir=".")
    )
    # Repack deterministically enough for ordinary inspection and avoid nesting an old archive.
    if not archive_path.is_file():
        with ZipFile(archive_path, "w", ZIP_DEFLATED):
            pass
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"Audit directory: {args.output_dir}")
    print(f"Audit ZIP: {archive_path}")


if __name__ == "__main__":
    main()
