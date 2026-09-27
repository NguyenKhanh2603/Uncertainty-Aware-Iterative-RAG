import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts')

from uncertainty_rag.core.conformal_retrieval import RetrievalCandidate
from run_two_stage_conformal import iter_rows, iter_query_groups, MetricsAccumulator

def top5_select(query_rows):
    candidates = [RetrievalCandidate.from_mapping(row) for row in query_rows]
    retrieval_order = sorted(range(len(candidates)), key=lambda i: (candidates[i].rank, candidates[i].chunk_id))
    return set(retrieval_order[:5])

def main():
    inputs = [
        r'E:\Downloads\filtered_splits_conformal_data\hotpotqa_filtered_top30.jsonl.gz',
        r'E:\Downloads\filtered_splits_conformal_data\mmqa_filtered_top30.jsonl.gz',
        r'E:\Downloads\filtered_splits_conformal_data\tatqa_filtered_top30.jsonl.gz',
        r'E:\Downloads\filtered_splits_conformal_data\webqa_filtered_top30.jsonl.gz'
    ]
    def all_rows():
        for path in inputs:
            yield from iter_rows(Path(path))

    datasets_order = ['hotpotqa', 'mmqa', 'tatqa', 'webqa']
    results = {ds: MetricsAccumulator() for ds in datasets_order}

    for query_rows in iter_query_groups(all_rows(), 'test'):
        candidates = [RetrievalCandidate.from_mapping(row) for row in query_rows]
        dataset = candidates[0].dataset
        sel = top5_select(query_rows)
        results[dataset].update(candidates, sel)

    for ds in datasets_order:
        stats = results[ds].finish()
        c = stats['average_selected_chunks']
        p = stats['micro_evidence_precision'] or 0
        r = stats['conditional_reserve_support_recall'] or 0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0
        print(f"| **Fixed Top-5** | {c:.2f} | {p*100:.1f}% | {r*100:.1f}% | {f1*100:.1f}% |")

if __name__ == '__main__':
    main()
