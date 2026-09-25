import csv
from collections import defaultdict

def analyze_pvalues(file_path):
    print(f"\n--- Analyzing {file_path} ---")
    true_chunks = []
    false_chunks = []
    
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            p_val = float(row['PValue'])
            rank = int(row['Rank'])
            if row['SupportLabel'] == 'support':
                true_chunks.append((p_val, rank))
            elif row['SupportLabel'] == 'false':
                false_chunks.append((p_val, rank))
                
    total_true = len(true_chunks)
    total_false = len(false_chunks)
    
    print(f"Total True chunks: {total_true}")
    print(f"Total False chunks: {total_false}")
    
    cutoffs = [0.001, 0.0034, 0.005, 0.01, 0.02, 0.03, 0.05, 0.10, 0.20]
    print("\nGlobal Fixed Threshold Simulation:")
    print(f"{'Cutoff':<10} | {'True Passed':<12} | {'False Passed':<12} | {'Recall':<8} | {'Precision':<10}")
    for cutoff in cutoffs:
        true_pass = sum(1 for p, r in true_chunks if p <= cutoff)
        false_pass = sum(1 for p, r in false_chunks if p <= cutoff)
        recall = true_pass / total_true if total_true else 0
        precision = true_pass / (true_pass + false_pass) if (true_pass + false_pass) else 0
        print(f"{cutoff:<10} | {true_pass:<12} | {false_pass:<12} | {recall:.2%} | {precision:.2%}")

    print("\nTrue Chunks Median P-value by Rank:")
    rank_pvals = defaultdict(list)
    for p, r in true_chunks:
        rank_pvals[r].append(p)
        
    for rank in sorted(rank_pvals.keys())[:10]:
        pvals = sorted(rank_pvals[rank])
        median_p = pvals[len(pvals)//2] if pvals else 0
        print(f"Rank {rank}: Median P-Value = {median_p:.5f}")

if __name__ == "__main__":
    analyze_pvalues(r"E:\Downloads\conformal_backfill_2_3_results\inference_pvalues.csv")
    analyze_pvalues(r"E:\Downloads\conformal_backfill_2_3_results\inference_pvalues_1.csv")
