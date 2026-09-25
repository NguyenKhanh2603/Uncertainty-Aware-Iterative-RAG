import csv
from collections import defaultdict


def simulate_bh(file_path, alpha, max_context=10, filter_file=None):
    # Load target query IDs if filter_file is provided
    target_qids = set()
    if filter_file:
        with open(filter_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                target_qids.add(row['QueryID'])

    # Group by query
    queries = defaultdict(list)
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset = row['Dataset']
            qid = row['QueryID']
            if filter_file and qid not in target_qids:
                continue
            queries[(dataset, qid)].append(row)
            
    dataset_metrics = defaultdict(lambda: {"total_true": 0, "kept": 0, "true_kept": 0, "false_kept": 0, "queries": 0, "empty": 0, "top_k_kept": 0, "top_k_true_kept": 0})
    
    for (dataset, qid), chunks in queries.items():
        dataset_metrics[dataset]["queries"] += 1
        
        # Count total true in top L for this query
        total_true = sum(1 for c in chunks if c['SupportLabel'] == 'support')
        dataset_metrics[dataset]["total_true"] += total_true
        
        # Calculate Top-K metrics
        chunks_sorted_by_rank = sorted(chunks, key=lambda x: int(x['Rank']))
        top_k_chunks = chunks_sorted_by_rank[:max_context]
        dataset_metrics[dataset]["top_k_kept"] += len(top_k_chunks)
        dataset_metrics[dataset]["top_k_true_kept"] += sum(1 for c in top_k_chunks if c['SupportLabel'] == 'support')
        
        # Apply BH
        p_values = sorted([float(c['PValue']) for c in chunks])
        L = len(p_values)
        
        k_hat = 0
        for i, p in enumerate(p_values, 1):
            if p <= alpha * i / L:
                k_hat = i
                
        # Limit to max_context
        k_hat = min(k_hat, max_context)
        
        if k_hat == 0:
            dataset_metrics[dataset]["empty"] += 1
            continue
            
        # We need to sort original chunks by p-value (or rank if p-values are tied)
        chunks_sorted = sorted(chunks, key=lambda x: (float(x['PValue']), int(x['Rank'])))
        selected = chunks_sorted[:k_hat]
        
        dataset_metrics[dataset]["kept"] += len(selected)
        dataset_metrics[dataset]["true_kept"] += sum(1 for c in selected if c['SupportLabel'] == 'support')
        dataset_metrics[dataset]["false_kept"] += sum(1 for c in selected if c['SupportLabel'] == 'false')
        
    print(f"\n--- Results for {file_path} at alpha = {alpha} (Max Context = {max_context}) ---")
    print(f"{'Dataset':<10} | {'Kept/Q':<6} | {'Precision':<9} | {'Recall':<9} | {'F1':<9} | {'Empty':<6}")
    for ds in sorted(dataset_metrics.keys()):
        m = dataset_metrics[ds]
        kept_per_q = m["kept"] / m["queries"] if m["queries"] else 0
        prec = m["true_kept"] / m["kept"] if m["kept"] else 0
        rec = m["true_kept"] / m["total_true"] if m["total_true"] else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        empty_rate = m["empty"] / m["queries"] if m["queries"] else 0
        
        print(f"{ds:<10} | {kept_per_q:<6.2f} | {prec:>7.1%} | {rec:>7.1%} | {f1:>7.1%} | {empty_rate:>5.1%}")

    print(f"\n--- Top-K (K={max_context}) Baseline ---")
    print(f"{'Dataset':<10} | {'Kept/Q':<6} | {'Precision':<9} | {'Recall':<9} | {'F1':<9} | {'Empty':<6}")
    for ds in sorted(dataset_metrics.keys()):
        m = dataset_metrics[ds]
        tk_kept = m["top_k_kept"] / m["queries"] if m["queries"] else 0
        tk_prec = m["top_k_true_kept"] / m["top_k_kept"] if m["top_k_kept"] else 0
        tk_rec = m["top_k_true_kept"] / m["total_true"] if m["total_true"] else 0
        tk_f1 = 2 * tk_prec * tk_rec / (tk_prec + tk_rec) if (tk_prec + tk_rec) else 0
        print(f"{ds:<10} | {tk_kept:<6.2f} | {tk_prec:>7.1%} | {tk_rec:>7.1%} | {tk_f1:>7.1%} | {'0.0%':>6}")

if __name__ == "__main__":
    pvalues_file = r"E:\Downloads\filtered_manifest_conformal_data\inference_pvalues_test.csv"
    
    print("\n--- Fixed Top-10 and BH (ctx=10) ---")
    for alpha in [0.10, 0.90, 0.99]:
        print(f"\n================ ALPHA = {alpha}, CTX = 10 ================")
        simulate_bh(pvalues_file, alpha, max_context=10, filter_file=None)
        
    print("\n--- Fixed Top-20 and BH (ctx=20) ---")
    for alpha in [0.99]:
        print(f"\n================ ALPHA = {alpha}, CTX = 20 ================")
        simulate_bh(pvalues_file, alpha, max_context=20, filter_file=None)
