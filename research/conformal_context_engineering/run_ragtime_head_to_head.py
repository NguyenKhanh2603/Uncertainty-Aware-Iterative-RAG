"""Compare three conformal context-selection targets on public RAGTIME English.

This is a *document-unit proxy*, not a reproduction of Chakraborty et al.
(2026): their released repository contains prompts but not their topic split,
retrieval runs, Llama-3.3 relevance labels, Qwen3-Embedding-8B vectors, or
GPT-4o scores.  We use official English document qrels only for evaluation and
one shared TF-IDF retrieval/scoring pipeline, so target definitions can be
compared without granting a method a better retriever.

Methods:
  cce_positive: their positive-only split-conformal threshold.  Its target is
      marginal retention of a relevant candidate document.
  query_all_support: a threshold calibrated on each query's lowest positive
      score.  Its target is retention of all retrieved relevant documents.
  candidate_by: the repository's legacy false-bank p-values plus BY procedure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from uncertainty_rag.core.conformal_selection import benjamini_yekutieli


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--topics", type=Path, required=True)
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--negative-documents", type=int, default=25_000)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--alphas", default="0.05,0.1,0.2")
    parser.add_argument("--seed", type=int, default=20260917)
    return parser.parse_args()


def load_topics(path: Path) -> dict[str, dict[str, str]]:
    topics = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            qid = str(row["topic_id"])
            topics[qid] = {
                "title": str(row.get("title", "")),
                "text": "\n".join(
                    str(row.get(field, ""))
                    for field in ("title", "background", "problem_statement")
                ),
            }
    return topics


def load_english_qrels(path: Path, topics: dict[str, dict[str, str]]) -> dict[str, set[str]]:
    labels: dict[str, set[str]] = defaultdict(set)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            qid, language, docid, grade = line.split()
            if language == "eng" and qid in topics and int(grade) > 0:
                labels[qid].add(docid)
    return labels


def stable_calibration_group(title: str, seed: int) -> bool:
    digest = hashlib.sha256(f"{seed}\0{title}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < 0.5


def sample_corpus(
    corpus_path: Path,
    positive_ids: set[str],
    negative_documents: int,
    seed: int,
) -> tuple[list[str], list[str], dict[str, int]]:
    """Keep every qrel-positive document and deterministic reservoir negatives."""

    rng = np.random.default_rng(seed)
    positives: list[tuple[str, str]] = []
    reservoir: list[tuple[str, str]] = []
    negative_seen = 0
    total = 0
    with corpus_path.open(encoding="utf-8") as handle:
        for line in handle:
            total += 1
            row = json.loads(line)
            docid, text = str(row["id"]), str(row["text"])
            if docid in positive_ids:
                positives.append((docid, text))
                continue
            negative_seen += 1
            if len(reservoir) < negative_documents:
                reservoir.append((docid, text))
            else:
                replacement = int(rng.integers(negative_seen))
                if replacement < negative_documents:
                    reservoir[replacement] = (docid, text)
    documents = positives + reservoir
    document_ids, texts = zip(*documents, strict=True)
    return list(document_ids), list(texts), {
        "corpus_documents_scanned": total,
        "qrel_positive_documents": len(positives),
        "sampled_negative_documents": len(reservoir),
    }


def finite_score_threshold(positive_scores: np.ndarray, alpha: float) -> float:
    """Score threshold for at-least 1-alpha marginal positive retention."""

    if not len(positive_scores):
        raise ValueError("Calibration has no retrieved relevant documents")
    nonconformity = np.sort(1.0 - positive_scores)
    rank = min(len(nonconformity) - 1, int(np.ceil((len(nonconformity) + 1) * (1 - alpha))) - 1)
    return float(1.0 - nonconformity[rank])


def metrics(labels: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    retrievable = labels.any(axis=1)
    retained = (labels & mask).sum(axis=1)
    supports = labels.sum(axis=1)
    selected = int(mask.sum())
    conditional_support = int(supports[retrievable].sum())
    conditional_retained = int(retained[retrievable].sum())
    return {
        "retrievable_queries": int(retrievable.sum()),
        "retrieval_ceiling": float(retrievable.mean()),
        "mean_candidates_kept": float(mask.sum(axis=1).mean()),
        "context_reduction": float(1.0 - mask.mean()),
        "empty_rate": float((mask.sum(axis=1) == 0).mean()),
        "chunk_precision": float((labels & mask).sum() / selected) if selected else 0.0,
        "micro_relevant_recall_conditional": float(conditional_retained / conditional_support) if conditional_support else 0.0,
        "query_any_support_conditional": float((retained[retrievable] > 0).mean()) if retrievable.any() else 0.0,
        "query_all_support_conditional": float((retained[retrievable] == supports[retrievable]).mean()) if retrievable.any() else 0.0,
    }


def main() -> None:
    args = arguments()
    if args.top_k < 1 or args.negative_documents < 1:
        raise ValueError("--top-k and --negative-documents must be positive")
    alphas = tuple(float(value) for value in args.alphas.split(","))
    if any(not 0 < alpha < 1 for alpha in alphas):
        raise ValueError("Every alpha must lie in (0, 1)")

    topics = load_topics(args.topics)
    qrels = load_english_qrels(args.qrels, topics)
    eligible_topics = {qid: topic for qid, topic in topics.items() if qrels.get(qid)}
    calibration_qids = sorted(
        qid for qid, topic in eligible_topics.items() if stable_calibration_group(topic["title"], args.seed)
    )
    test_qids = sorted(set(eligible_topics) - set(calibration_qids))
    if not calibration_qids or not test_qids:
        raise RuntimeError("Title-group split unexpectedly created an empty role")

    all_positive_ids = set().union(*(qrels[qid] for qid in eligible_topics))
    document_ids, texts, sample_audit = sample_corpus(
        args.corpus, all_positive_ids, args.negative_documents, args.seed
    )
    vectorizer = TfidfVectorizer(
        stop_words="english", ngram_range=(1, 2), min_df=2, max_features=150_000, dtype=np.float32
    )
    document_vectors = vectorizer.fit_transform(texts)
    ordered_qids = calibration_qids + test_qids
    query_vectors = vectorizer.transform([eligible_topics[qid]["text"] for qid in ordered_qids])
    all_scores = (query_vectors @ document_vectors.T).toarray().astype(np.float32)
    top_indices = np.argpartition(all_scores, -args.top_k, axis=1)[:, -args.top_k:]
    top_scores = np.take_along_axis(all_scores, top_indices, axis=1)
    order = np.argsort(-top_scores, axis=1, kind="stable")
    top_indices = np.take_along_axis(top_indices, order, axis=1)
    top_scores = np.take_along_axis(top_scores, order, axis=1)
    selected_ids = np.asarray(document_ids, dtype=object)[top_indices]
    labels = np.asarray(
        [[docid in qrels[qid] for docid in row] for qid, row in zip(ordered_qids, selected_ids, strict=True)],
        dtype=bool,
    )
    calibration_rows = np.arange(len(calibration_qids))
    test_rows = np.arange(len(calibration_qids), len(ordered_qids))
    cal_scores, test_scores = top_scores[calibration_rows], top_scores[test_rows]
    cal_labels, test_labels = labels[calibration_rows], labels[test_rows]
    results: dict[str, dict[str, dict[str, float | int]]] = {}
    for alpha in alphas:
        cce_tau = finite_score_threshold(cal_scores[cal_labels], alpha)
        query_tau = finite_score_threshold(
            np.asarray([row[label].min() for row, label in zip(cal_scores, cal_labels, strict=True) if label.any()]),
            alpha,
        )
        results[f"{alpha:g}"] = {
            "cce_positive": {"threshold": cce_tau, **metrics(test_labels, test_scores >= cce_tau)},
            "query_all_support": {"threshold": query_tau, **metrics(test_labels, test_scores >= query_tau)},
            "candidate_by": {},
        }
        # BY needs a false bank from calibration but applies decisions to test scores.
        bank = np.sort(cal_scores[~cal_labels])
        by_test = np.zeros_like(test_scores, dtype=bool)
        for i, row in enumerate(test_scores):
            p_values = (1.0 + len(bank) - np.searchsorted(bank, row, side="left")) / (len(bank) + 1.0)
            by_test[i, benjamini_yekutieli(p_values.tolist(), alpha).rejected_indices] = True
        results[f"{alpha:g}"]["candidate_by"] = metrics(test_labels, by_test)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "status": "complete_document_unit_proxy",
        "not_an_exact_paper_reproduction": [
            "uses official document qrels rather than released-or-unreleased Llama-3.3 snippet labels",
            "uses TF-IDF cosine rather than Qwen3-Embedding-8B",
            "does not run GPT-4o Conformal-LLM or downstream ARGUE generation",
            "candidate pool is qrel-positive documents plus deterministic sampled corpus negatives",
        ],
        "dataset": "trec-ragtime/ragtime1 English",
        "split": {
            "unit": "topic title; duplicate request variants stay in one role",
            "calibration_queries": len(calibration_qids),
            "test_queries": len(test_qids),
        },
        "candidate_pool": {"top_k": args.top_k, **sample_audit},
        "results": results,
    }
    (args.output_dir / "ragtime_head_to_head_summary.json").write_text(
        json.dumps(artifact, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        args.output_dir / "ragtime_head_to_head_predictions.npz",
        qids=np.asarray(ordered_qids), scores=top_scores, labels=labels, selected_ids=selected_ids,
        calibration_count=len(calibration_qids),
    )
    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
