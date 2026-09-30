import json
import gzip
from collections import defaultdict
from pathlib import Path

# Load Ground Truth
inputs = [
    r'E:\Downloads\filtered_splits_conformal_data\hotpotqa_filtered_top30.jsonl.gz',
    r'E:\Downloads\filtered_splits_conformal_data\mmqa_filtered_top30.jsonl.gz',
    r'E:\Downloads\filtered_splits_conformal_data\tatqa_filtered_top30.jsonl.gz',
    r'E:\Downloads\filtered_splits_conformal_data\webqa_filtered_top30.jsonl.gz'
]

gt = defaultdict(lambda: defaultdict(set))
for path in inputs:
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            row = json.loads(line)
            if row.get('split_role') == 'test':
                ds = row['dataset']
                qid = row.get('qid') or row.get('query_id')
                if str(row.get('support_label')).lower() == 'support':
                    gt[ds][qid].add(row['chunk_id'])

# Load Selections
with open(r'E:\Downloads\conformal prediction\Conformal\Uncertainty-Aware-Iterative-RAG\downstream_export.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Evaluate
results = defaultdict(lambda: defaultdict(lambda: {'tp': 0, 'selected': 0, 'total_recalls': [], 'kept': []}))

for row in data:
    ds = row['dataset']
    qid = row['qid']
    gold = gt[ds][qid]
    
    for method, chunks in row['methods'].items():
        selected = {c['chunk_id'] for c in chunks}
        tp = len(selected.intersection(gold))
        
        results[method][ds]['tp'] += tp
        results[method][ds]['selected'] += len(selected)
        results[method][ds]['kept'].append(len(selected))
        
        if len(gold) > 0:
            results[method][ds]['total_recalls'].append(tp / len(gold))

# Print Table
datasets = ['hotpotqa', 'mmqa', 'tatqa', 'webqa']
print(f'{"Method":<25}', end='')
for ds in datasets:
    print(f'{ds.upper():<25}', end='')
print()

methods_to_print = [
    'fixed_top5', 'fixed_top10',
    'stage1_a0.1', 'stage1_a0.9', 'stage1_a0.99',
    'twostage_a1_0.99_a2_0.15', 'twostage_a1_0.99_a2_0.2', 'twostage_a1_0.99_a2_0.3',
    'top10_stage2_a2_0.15', 'top10_stage2_a2_0.2', 'top10_stage2_a2_0.3', 'top10_stage2_a2_0.4'
]

for method in methods_to_print:
    print(f'{method:<25}', end='')
    for ds in datasets:
        stats = results[method][ds]
        avg_kept = sum(stats['kept']) / len(stats['kept']) if stats['kept'] else 0
        micro_prec = stats['tp'] / stats['selected'] if stats['selected'] > 0 else 0
        macro_rec = sum(stats['total_recalls']) / len(stats['total_recalls']) if stats['total_recalls'] else 0
        print(f'{avg_kept:.2f} | {micro_prec*100:4.1f}% | {macro_rec*100:4.1f}%   ', end='')
    print()
