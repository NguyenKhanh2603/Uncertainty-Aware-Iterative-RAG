import json
import os
import sys
import math
from collections import defaultdict
from pathlib import Path
import gzip

sys.path.insert(0, 'src')
sys.path.insert(0, os.path.abspath('scripts'))

from run_two_stage_conformal import iter_rows, iter_query_groups, MetricsAccumulator
from uncertainty_rag.core.conformal_retrieval import RetrievalCandidate

def finite_tau(s, a):
    x = sorted(s)
    idx = min(len(x)-1, int(math.floor((len(x)+1)*a)))
    return float(x[idx])

def simple_percentile(s, p):
    x = sorted(s)
    k = (len(x) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(x[int(k)])
    d0 = x[int(f)] * (c - k)
    d1 = x[int(c)] * (k - f)
    return float(d0 + d1)

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

    support_scores = defaultdict(list)
    for row in all_rows():
        if row['split'] == 'calibration' and row['label'] == 'Support':
            support_scores[row['dataset']].append(row['cosine_score'])

    thresholds = {}
    for ds, scores in support_scores.items():
        positive = sorted(scores)
        alpha = 0.1
        cce = finite_tau(positive, alpha)
        conflare = simple_percentile(positive, alpha*100)
        traq = simple_percentile(positive, (alpha/2)*100) # approximate lower
        thresholds[ds] = {'cce': cce, 'conflare': conflare, 'traq': traq}

    datasets_order = ['hotpotqa', 'mmqa', 'tatqa', 'webqa']
    results = {
        'cce': {ds: MetricsAccumulator() for ds in datasets_order},
        'conflare': {ds: MetricsAccumulator() for ds in datasets_order},
        'traq': {ds: MetricsAccumulator() for ds in datasets_order}
    }

    for query_rows in iter_query_groups(all_rows(), 'test'):
        candidates = [RetrievalCandidate.from_mapping(row) for row in query_rows]
        dataset = candidates[0].dataset
        
        cce_sel = {i for i, c in enumerate(candidates) if c.cosine_score >= thresholds[dataset]['cce']}
        results['cce'][dataset].update(candidates, cce_sel)
        
        conflare_sel = {i for i, c in enumerate(candidates) if c.cosine_score >= thresholds[dataset]['conflare']}
        results['conflare'][dataset].update(candidates, conflare_sel)
        
        traq_sel = {i for i, c in enumerate(candidates) if c.cosine_score >= thresholds[dataset]['traq']}
        results['traq'][dataset].update(candidates, traq_sel)

    for method in ['cce', 'conflare', 'traq']:
        print(f"--- {method.upper()} ---")
        for ds in datasets_order:
            stats = results[method][ds].finish()
            c = stats['average_selected_chunks']
            p = stats['micro_evidence_precision'] or 0
            r = stats['conditional_reserve_support_recall'] or 0
            print(f"{ds}: Kept={c:.2f}, P={p*100:.1f}%, R={r*100:.1f}%")

if __name__ == '__main__':
    main()
