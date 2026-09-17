"""Compare CCE, query-level nugget support, and BY on CoverageBench NeuCLIR.

CoverageBench releases fixed Qwen3-Embedding-8B candidate scores and
nugget-to-document qrels for 19 NeuCLIR 2024 report-generation topics.  Since a
nugget can have multiple interchangeable supporting documents, the query-level
event is: retain at least one retrieved document for every retrievable nugget.
This is the report-generation analogue of all-support coverage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from uncertainty_rag.core.conformal_selection import benjamini_yekutieli


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rankings", type=Path, required=True)
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--nuggets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-l", type=int, default=30)
    parser.add_argument("--alphas", default="0.05,0.1,0.2")
    parser.add_argument("--seed", type=int, default=20260917)
    return parser.parse_args()


def split_role(qid: str, seed: int) -> str:
    value = int.from_bytes(hashlib.sha256(f"{seed}\0{qid}".encode()).digest()[:8], "big") / 2**64
    return "calibration" if value < 0.5 else "test"


def threshold(scores: np.ndarray, alpha: float) -> float:
    if not len(scores):
        raise ValueError("No calibration scores")
    nonconformity = np.sort(1.0 - scores)
    index = min(len(scores) - 1, int(np.ceil((len(scores) + 1) * (1 - alpha))) - 1)
    return float(1.0 - nonconformity[index])


def metrics(
    labels: np.ndarray,
    nugget_support: list[list[set[int]]],
    mask: np.ndarray,
) -> dict[str, float | int]:
    retrieved = labels.any(axis=1)
    retained = labels & mask
    supported_nuggets = []
    all_nuggets = []
    for row, groups in zip(mask, nugget_support, strict=True):
        eligible = [positions for positions in groups if positions]
        all_nuggets.append(len(eligible))
        supported_nuggets.append(sum(bool(set(np.flatnonzero(row)) & positions) for positions in eligible))
    selected = int(mask.sum())
    return {
        "retrievable_queries": int(retrieved.sum()),
        "retrieval_query_ceiling": float(retrieved.mean()),
        "mean_documents_kept": float(mask.sum(axis=1).mean()),
        "context_reduction": float(1.0 - mask.mean()),
        "empty_rate": float((mask.sum(axis=1) == 0).mean()),
        "document_precision": float(retained.sum() / selected) if selected else 0.0,
        "document_recall_conditional": float(retained[retrieved].sum() / labels[retrieved].sum()) if labels[retrieved].sum() else 0.0,
        "nugget_micro_coverage": float(sum(supported_nuggets) / sum(all_nuggets)) if sum(all_nuggets) else 0.0,
        "query_all_nugget_coverage": float(np.mean([a == b for a, b in zip(supported_nuggets, all_nuggets, strict=True)])),
    }


def main() -> None:
    args = parse_args()
    if args.top_l < 1:
        raise ValueError("--top-l must be positive")
    alphas = tuple(float(value) for value in args.alphas.split(","))
    rankings = json.loads(args.rankings.read_text())
    qrel_docs: dict[str, set[str]] = defaultdict(set)
    with args.qrels.open() as handle:
        for line in handle:
            qid, _, docid, relevance = line.split()
            if int(relevance) > 0:
                qrel_docs[qid].add(docid)
    raw_nuggets = json.loads(args.nuggets.read_text())
    qids = sorted(set(rankings) & set(raw_nuggets))
    ordered = sorted(qids, key=lambda qid: (split_role(qid, args.seed), qid))
    calibration_count = sum(split_role(qid, args.seed) == "calibration" for qid in ordered)
    if not calibration_count or calibration_count == len(ordered):
        raise RuntimeError("Empty calibration or test role")
    candidate_ids, candidate_scores = [], []
    labels, nugget_support = [], []
    for qid in ordered:
        ranked = sorted(rankings[qid].items(), key=lambda item: (-float(item[1]), item[0]))[: args.top_l]
        docs = [docid for docid, _ in ranked]
        positions = {docid: index for index, docid in enumerate(docs)}
        candidate_ids.append(docs)
        candidate_scores.append([float(score) for _, score in ranked])
        labels.append([docid in qrel_docs[qid] for docid in docs])
        groups = []
        for nugget in raw_nuggets[qid]["nugget_bank"].values():
            supporting_ids = {
                reference["doc_id"]
                for answer in nugget.get("answers", {}).values()
                for reference in answer.get("references", [])
            }
            groups.append({positions[docid] for docid in supporting_ids & positions.keys()})
        nugget_support.append(groups)
    scores = np.asarray(candidate_scores, dtype=np.float64)
    labels_array = np.asarray(labels, dtype=bool)
    cal_scores, test_scores = scores[:calibration_count], scores[calibration_count:]
    cal_labels, test_labels = labels_array[:calibration_count], labels_array[calibration_count:]
    test_nuggets = nugget_support[calibration_count:]
    result = {}
    for alpha in alphas:
        cce_tau = threshold(cal_scores[cal_labels], alpha)
        critical = []
        for row, groups in zip(cal_scores, nugget_support[:calibration_count], strict=True):
            maxima = [row[list(positions)].max() for positions in groups if positions]
            if maxima:
                critical.append(min(maxima))
        query_tau = threshold(np.asarray(critical), alpha)
        false_bank = np.sort(cal_scores[~cal_labels])
        by = np.zeros_like(test_scores, dtype=bool)
        for index, row in enumerate(test_scores):
            p_values = (1 + len(false_bank) - np.searchsorted(false_bank, row, side="left")) / (len(false_bank) + 1)
            by[index, benjamini_yekutieli(p_values.tolist(), alpha).rejected_indices] = True
        result[f"{alpha:g}"] = {
            "cce_positive": {"threshold": cce_tau, **metrics(test_labels, test_nuggets, test_scores >= cce_tau)},
            "query_all_nugget": {"threshold": query_tau, **metrics(test_labels, test_nuggets, test_scores >= query_tau)},
            "candidate_by": metrics(test_labels, test_nuggets, by),
        }
    artifact = {
        "status": "complete_score_only_comparison",
        "dataset": "CoverageBench NeuCLIR 2024 report-generation topics",
        "score": "released Qwen3-Embedding-8B initial retrieval score",
        "not_downstream_f1": "No document text or generated reports are included; this measures evidence selection and nugget coverage.",
        "split": {"calibration_queries": calibration_count, "test_queries": len(ordered) - calibration_count, "seed": args.seed},
        "top_l": args.top_l,
        "results": result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
