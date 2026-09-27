import json
import os
import sys
from collections import defaultdict
from itertools import groupby
import gzip
from pathlib import Path

# Add src and scripts to pythonpath
sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts')

from uncertainty_rag.core.conformal_retrieval import RetrievalCandidate
from uncertainty_rag.core.conformal_selection import benjamini_hochberg
from run_two_stage_conformal import iter_rows, iter_query_groups, SupportBankIndex, MetricsAccumulator

def top10_stage2_select(query_rows, support_bank, alpha2, max_context=10):
    candidates = [RetrievalCandidate.from_mapping(row) for row in query_rows]
    retrieval_order = sorted(range(len(candidates)), key=lambda i: (candidates[i].rank, candidates[i].chunk_id))
    stage1 = set(retrieval_order[:max_context])
    if not stage1 or alpha2 <= 0:
        return stage1
    stage1_list = sorted(stage1)
    p_support = []
    for idx in stage1_list:
        p = support_bank.p_value(candidates[idx].cosine_score, candidates[idx].dataset, candidates[idx].modality)
        p_support.append(p)
    bh2 = benjamini_hochberg(p_support, alpha2)
    rejected_in_stage1 = {stage1_list[i] for i in bh2.rejected_indices}
    return stage1 - rejected_in_stage1

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

    support_bank = SupportBankIndex(conditioning='dataset')
    for row in all_rows():
        support_bank.add(row)
    support_bank.finalize()

    alpha2_values = [0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    datasets_order = ['hotpotqa', 'mmqa', 'tatqa', 'webqa']
    
    results = {}
    for a2 in alpha2_values:
        key = f'top10_a2={a2}'
        results[key] = {ds: MetricsAccumulator() for ds in datasets_order + ['ALL']}

    for query_rows in iter_query_groups(all_rows(), 'test'):
        candidates = [RetrievalCandidate.from_mapping(row) for row in query_rows]
        dataset = candidates[0].dataset
        for a2 in alpha2_values:
            sel = top10_stage2_select(query_rows, support_bank, alpha2=a2, max_context=10)
            key = f'top10_a2={a2}'
            results[key][dataset].update(candidates, sel)
            results[key]['ALL'].update(candidates, sel)

    out = {}
    for a2 in alpha2_values:
        key = f'top10_a2={a2}'
        out[key] = {ds: results[key][ds].finish() for ds in datasets_order + ['ALL']}
    
    with open('top10_stage2_results.json', 'w') as f:
        json.dump(out, f)
    print('DONE')

if __name__ == '__main__':
    main()
