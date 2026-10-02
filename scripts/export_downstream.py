import json
import gzip
from collections import defaultdict
import bisect
from pathlib import Path
import math

class RetrievalCandidate:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
    @classmethod
    def from_mapping(cls, m):
        return cls(
            dataset=m['dataset'],
            qid=str(m.get('qid') or m.get('query_id')),
            split_role=m.get('split_role'),
            chunk_id=m['chunk_id'],
            modality=m['modality'],
            rank=int(m['rank']),
            cosine_score=float(m['cosine_score']),
            support_label=str(m.get('support_label')).lower(),
            raw_dict=m
        )
    def to_dict(self):
        return self.raw_dict

def iter_rows(path: Path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            yield json.loads(line)

def iter_query_groups(rows_iter, target_split):
    current_qid = None
    group = []
    for row in rows_iter:
        if row.get('split_role') != target_split: continue
        qid = str(row.get('qid') or row.get('query_id'))
        if current_qid is None: current_qid = qid
        if qid != current_qid:
            yield group
            group = []
            current_qid = qid
        group.append(row)
    if group: yield group

def get_p_val_s1(score, bank):
    if not bank: return 1.0
    idx = bisect.bisect_left(bank, score)
    n_greater_equal = len(bank) - idx
    return (1.0 + n_greater_equal) / (len(bank) + 1.0)

def get_p_val_s2(score, bank):
    if not bank: return 1.0
    idx = bisect.bisect_right(bank, score)
    return (1.0 + idx) / (len(bank) + 1.0)

def stage1_bh(candidates, false_bank_func, alpha1):
    M = len(candidates)
    if M == 0: return []
    pvals = []
    for i, c in enumerate(candidates):
        bank = false_bank_func(c)
        p = get_p_val_s1(c.cosine_score, bank)
        pvals.append((i, p))
    pvals.sort(key=lambda x: x[1])
    k_max = -1
    for k, (i, p) in enumerate(pvals, 1):
        if p <= (k / M) * alpha1:
            k_max = k
    if k_max == -1: return []
    return [i for i, p in pvals[:k_max]]

def stage2_bh(candidates, S1_list, support_bank_func, alpha2):
    M = len(candidates)
    if M == 0 or not S1_list: return set()
    pvals = []
    for i in S1_list:
        c = candidates[i]
        bank = support_bank_func(c)
        p = get_p_val_s2(c.cosine_score, bank)
        pvals.append((i, p))
    pvals.sort(key=lambda x: x[1])
    k_max = -1
    for k, (i, p) in enumerate(pvals, 1):
        if p <= (k / M) * alpha2:
            k_max = k
    if k_max == -1: return set(S1_list)
    rejected = {i for i, p in pvals[:k_max]}
    return set(S1_list) - rejected

def conformal_backfill(accepted_indices, max_k, retrieval_order):
    accepted = set(accepted_indices)
    original_top = set(retrieval_order[:max_k])
    accepted_top_k = [i for i in retrieval_order if i in accepted and i in original_top]
    free_slots = max_k - len(accepted_top_k)
    accepted_reserve = [i for i in retrieval_order if i in accepted and i not in original_top]
    return accepted_top_k + accepted_reserve[:free_slots]

def finite_threshold(scores, alpha):
    n = len(scores)
    if n == 0: return 1.0
    order = min(n, math.ceil((n + 1) * (1 - alpha)))
    ordered_desc = sorted(scores, reverse=True)
    return float(ordered_desc[order - 1])

def percentile_threshold(scores, p):
    if not scores: return 1.0
    ordered = sorted(scores)
    k = (len(ordered) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c: return float(ordered[int(k)])
    return float(ordered[int(f)] * (c - k) + ordered[int(c)] * (k - f))

def quantile_lower(scores, q):
    if not scores: return 1.0
    ordered = sorted(scores)
    k = (len(ordered) - 1) * q
    return float(ordered[math.floor(k)])

def main():
    inputs = [
        r'E:\Downloads\precomputed_cosine_cal1000_official_test_2026_09_30\portable_bundle\rowwise\hotpotqa_top30_retrieval.jsonl.gz',
        r'E:\Downloads\precomputed_cosine_cal1000_official_test_2026_09_30\portable_bundle\rowwise\mmqa_top30_retrieval.jsonl.gz',
        r'E:\Downloads\precomputed_cosine_cal1000_official_test_2026_09_30\portable_bundle\rowwise\tatqa_top30_retrieval.jsonl.gz',
        r'E:\Downloads\precomputed_cosine_cal1000_official_test_2026_09_30\portable_bundle\rowwise\webqa_top30_retrieval.jsonl.gz'
    ]

    def all_rows():
        for path in inputs:
            yield from iter_rows(Path(path))

    false_scores_mod = defaultdict(lambda: defaultdict(list))
    support_scores_ds = defaultdict(list)

    print("Building banks...")
    for row in all_rows():
        if row.get('split_role') == 'calibration':
            ds = row['dataset']
            mod = row['modality']
            sc = float(row['cosine_score'])
            if str(row.get('support_label')).lower() == 'support':
                support_scores_ds[ds].append(sc)
            else:
                false_scores_mod[ds][mod].append(sc)

    for ds in false_scores_mod:
        for mod in false_scores_mod[ds]:
            false_scores_mod[ds][mod].sort()
        support_scores_ds[ds].sort()

    # thresholds for static baselines
    static_thresholds = {}
    for ds, scores in support_scores_ds.items():
        static_thresholds[ds] = {
            'CCE': finite_threshold(scores, 0.10),
            'CONFLARE': percentile_threshold(scores, 10),
            'TRAQ': quantile_lower(scores, 0.05)
        }

    f_func = lambda c: false_scores_mod[c.dataset][c.modality]
    s_func = lambda c: support_scores_ds[c.dataset]

    out_file = r'E:\Downloads\conformal prediction\Conformal\Uncertainty-Aware-Iterative-RAG\downstream_export.json'
    print("Exporting to", out_file)
    
    with open(out_file, 'w', encoding='utf-8') as fout:
        c_idx = 0
        for query_rows in iter_query_groups(all_rows(), 'test'):
            c_idx += 1
            if c_idx % 1000 == 0: print(f"Exported {c_idx} queries...")
            
            candidates = [RetrievalCandidate.from_mapping(r) for r in query_rows]
            if not candidates: continue
            dataset = candidates[0].dataset
            qid = candidates[0].qid
            
            retrieval_order = sorted(range(len(candidates)), key=lambda i: (candidates[i].rank, candidates[i].chunk_id))
            
            methods_chunks = {}
            def get_dicts(indices):
                # Return chunks mapped directly, NO TRUNCATION to 10
                return [candidates[i].to_dict() for i in indices]

            # Fixed Top K
            for k in [3, 5, 7, 10, 20, 30]:
                methods_chunks[f"fixed_top{k}"] = get_dicts(retrieval_order[:k])
            
            # Static baselines
            for b_name in ['CCE', 'CONFLARE', 'TRAQ']:
                t = static_thresholds[dataset][b_name]
                kept = [i for i, c in enumerate(candidates) if c.cosine_score >= t]
                methods_chunks[b_name] = get_dicts(kept)

            # Stage 1
            for a1 in [0.1, 0.9, 0.99]:
                s1 = stage1_bh(candidates, f_func, a1)
                methods_chunks[f"stage1_a{a1}"] = get_dicts(s1)
            
            s1_099 = stage1_bh(candidates, f_func, 0.99)
            
            # Two Stage
            for a2 in [0.15, 0.2, 0.3, 0.4]:
                s2 = stage2_bh(candidates, s1_099, s_func, a2)
                methods_chunks[f"twostage_a1_0.99_a2_{a2}"] = get_dicts(s2)
                
            # Top K + Stage 2 & Conformal Backfill
            for k in [3, 5, 7, 10, 20, 30]:
                top_k = set(retrieval_order[:k])
                for a2 in [0.15, 0.2]:
                    # Fixed Top K + Stage 2
                    t_k_s2 = stage2_bh(candidates, top_k, s_func, a2)
                    methods_chunks[f"top{k}_stage2_a2_{a2}"] = get_dicts(t_k_s2)

                    # Top K + Stage 2 + Conformal Backfill
                    s2_all = stage2_bh(candidates, list(range(len(candidates))), s_func, a2)
                    s2_bf_k = conformal_backfill(s2_all, k, retrieval_order)
                    methods_chunks[f"top{k}_s2_bf_a2_{a2}"] = get_dicts(s2_bf_k)
            
            out_obj = {
                "dataset": dataset,
                "qid": qid,
                "query_text": candidates[0].raw_dict.get('query_text', ''),
                "methods": methods_chunks
            }
            fout.write(json.dumps(out_obj) + "\n")
    print("Done exporting!")

if __name__ == '__main__':
    main()
