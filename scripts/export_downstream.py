import json
import os
import sys
import math
import gzip
from collections import defaultdict
from pathlib import Path
from itertools import groupby

sys.path.insert(0, 'src')

from uncertainty_rag.core.conformal_retrieval import RetrievalCandidate

def iter_rows(p):
    with gzip.open(p, 'rt', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            yield json.loads(line)

def iter_query_groups(rows, split):
    for qid, group in groupby(rows, key=lambda x: x.get('qid') or x.get('query_id')):
        group = list(group)
        if group[0].get('split_role') == split:
            yield group

def finite_tau(s, a):
    if not s: return 1.0
    x = sorted(s)
    idx = min(len(x)-1, int(math.floor((len(x)+1)*a)))
    return float(x[idx])

def simple_percentile(s, p):
    if not s: return 1.0
    x = sorted(s)
    k = (len(x) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(x[int(k)])
    d0 = x[int(f)] * (c - k)
    d1 = x[int(c)] * (k - f)
    return float(d0 + d1)

import bisect

def build_banks(rows_iter):
    false_scores = defaultdict(lambda: defaultdict(list))
    support_scores = defaultdict(list)
    for row in rows_iter:
        if row.get('split_role') == 'calibration':
            ds = row['dataset']
            mod = row['modality']
            sc = float(row['cosine_score'])
            if str(row.get('support_label')).lower() == 'support':
                support_scores[ds].append(sc)
            else:
                false_scores[ds][mod].append(sc)
                
    for ds in false_scores:
        for mod in false_scores[ds]:
            false_scores[ds][mod].sort()
    for ds in support_scores:
        support_scores[ds].sort()
            
    return false_scores, support_scores

def stage1_bh(candidates, false_scores, alpha1):
    n = len(candidates)
    if n == 0: return set()
    pvals = []
    for i, c in enumerate(candidates):
        bank = false_scores[c.dataset][c.modality]
        b_size = len(bank)
        if b_size == 0:
            pvals.append((i, 1.0))
            continue
        idx = bisect.bisect_left(bank, c.cosine_score)
        count_ge = b_size - idx
        p = (1.0 + count_ge) / (b_size + 1.0)
        pvals.append((i, p))
    pvals.sort(key=lambda x: x[1])
    k_max = -1
    for k, (i, p) in enumerate(pvals, 1):
        if p <= (k / n) * alpha1:
            k_max = k
    if k_max == -1: return set()
    return {i for i, p in pvals[:k_max]}

def stage2_bh(candidates, S1_indices, support_scores, alpha2):
    M = len(S1_indices)
    if M == 0: return set()
    S1_list = list(S1_indices)
    pvals = []
    for i in S1_list:
        c = candidates[i]
        bank = support_scores[c.dataset]
        b_size = len(bank)
        if b_size == 0:
            pvals.append((i, 1.0))
            continue
        idx = bisect.bisect_right(bank, c.cosine_score)
        count_le = idx
        p = (1.0 + count_le) / (b_size + 1.0)
        pvals.append((i, p))
    pvals.sort(key=lambda x: x[1])
    k_max = -1
    for k, (i, p) in enumerate(pvals, 1):
        if p <= (k / M) * alpha2:
            k_max = k
    if k_max == -1: return set(S1_list) # None rejected -> all survive
    rejected = {i for i, p in pvals[:k_max]}
    return set(S1_list) - rejected

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

    false_scores, support_scores = build_banks(all_rows())

    exported_data = []

    for query_rows in iter_query_groups(all_rows(), 'test'):
        # For simplicity, extract query level info from first row
        first = query_rows[0]
        dataset = first['dataset']
        qid = first.get('qid') or first.get('query_id')
        query_text = first.get('query_text', '')

        candidates = [RetrievalCandidate.from_mapping(row) for row in query_rows]
        
        methods_chunks = {}

        def get_chunk_dicts(indices, max_len=10):
            sorted_indices = sorted(list(indices), key=lambda x: candidates[x].rank)
            return [
                {
                    "chunk_id": candidates[i].chunk_id,
                    "modality": candidates[i].modality,
                    "content": query_rows[i].get('content', '') 
                } for i in sorted_indices[:max_len]
            ]

        # Stage 1 only
        for a1 in [0.1, 0.9, 0.99]:
            s1 = stage1_bh(candidates, false_scores, a1)
            methods_chunks[f"stage1_a{a1}"] = get_chunk_dicts(s1)

        # Two stage (a1=0.99)
        s1_099 = stage1_bh(candidates, false_scores, 0.99)
        for a2 in [0.15, 0.2, 0.3, 0.4]:
            s2 = stage2_bh(candidates, s1_099, support_scores, a2)
            methods_chunks[f"twostage_a1_0.99_a2_{a2}"] = get_chunk_dicts(s2)

        # Top 10 + Stage 2
        # Fixed top 10 naive
        retrieval_order = sorted(range(len(candidates)), key=lambda i: (candidates[i].rank, candidates[i].chunk_id))
        top10 = set(retrieval_order[:10])
        for a2 in [0.15, 0.2, 0.3, 0.4]:
            t10_s2 = stage2_bh(candidates, top10, support_scores, a2)
            methods_chunks[f"top10_stage2_a2_{a2}"] = get_chunk_dicts(t10_s2)
            
        # Top 10 base
        methods_chunks["fixed_top10"] = get_chunk_dicts(top10)

        # Top 5 base
        top5 = set(retrieval_order[:5])
        methods_chunks["fixed_top5"] = get_chunk_dicts(top5)

        exported_data.append({
            "dataset": dataset,
            "qid": qid,
            "query_text": query_text,
            "methods": methods_chunks
        })

    out_path = r'E:\Downloads\conformal prediction\Conformal\Uncertainty-Aware-Iterative-RAG\downstream_export.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(exported_data, f, ensure_ascii=False, indent=2)
    print(f"Exported {len(exported_data)} queries to {out_path}")

if __name__ == '__main__':
    main()
