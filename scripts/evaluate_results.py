import json
from collections import defaultdict
import gzip
from pathlib import Path

def main():
    jsonl_files = [
        r'E:\Downloads\precomputed_cosine_cal1000_official_test_2026_09_30\portable_bundle\rowwise\hotpotqa_top30_retrieval.jsonl.gz',
        r'E:\Downloads\precomputed_cosine_cal1000_official_test_2026_09_30\portable_bundle\rowwise\mmqa_top30_retrieval.jsonl.gz',
        r'E:\Downloads\precomputed_cosine_cal1000_official_test_2026_09_30\portable_bundle\rowwise\tatqa_top30_retrieval.jsonl.gz',
        r'E:\Downloads\precomputed_cosine_cal1000_official_test_2026_09_30\portable_bundle\rowwise\webqa_top30_retrieval.jsonl.gz'
    ]

    total_supports_test = defaultdict(int)
    print("Reading total supports from test set...")
    for path in jsonl_files:
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                r = json.loads(line)
                if r.get('split_role') == 'test' and str(r.get('support_label')).lower() == 'support':
                    qid = str(r.get('qid') or r.get('query_id'))
                    total_supports_test[qid] += 1

    print("Evaluating downstream_export.json...")
    metrics = defaultdict(lambda: defaultdict(lambda: {'kept':0, 'supports':0, 'total_supports':0, 'queries':0}))
    
    with open(r'E:\Downloads\conformal prediction\Conformal\Uncertainty-Aware-Iterative-RAG\downstream_export.json', 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if not line.strip(): continue
            row = json.loads(line)
            ds = row['dataset']
            qid = row['qid']
            methods_chunks = row['methods']
            
            for m, chunks in methods_chunks.items():
                kept = len(chunks)
                supp = sum(1 for c in chunks if str(c.get('support_label')).lower() == 'support')
                metrics[m][ds]['kept'] += kept
                metrics[m][ds]['supports'] += supp
                metrics[m][ds]['queries'] += 1
                metrics[m][ds]['total_supports'] += total_supports_test[qid]

    print(f"\n{'Method':<25} {'HOTPOTQA':<25} {'MMQA':<25} {'TATQA':<25} {'WEBQA':<25}")
    for m in sorted(metrics.keys()):
        line = f"{m[:24]:<25}"
        for ds in ['hotpotqa', 'mmqa', 'tatqa', 'webqa']:
            if ds not in metrics[m]: continue
            mets = metrics[m][ds]
            n_queries = mets['queries']
            if n_queries == 0:
                line += f" {'-':<24}"
                continue
            k = mets['kept'] / n_queries
            p = (mets['supports'] / mets['kept'] * 100) if mets['kept'] > 0 else 0
            r = (mets['supports'] / mets['total_supports'] * 100) if mets['total_supports'] > 0 else 0
            line += f" {k:.2f} | {p:.1f}% | {r:.1f}%   "
        print(line)

if __name__ == '__main__':
    main()
