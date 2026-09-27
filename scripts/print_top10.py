import json
with open('top10_stage2_results.json') as f:
    data = json.load(f)

ds_list = ['hotpotqa', 'mmqa', 'tatqa', 'webqa']
alphas = [0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

for ds in ds_list:
    print(f"\n--- {ds.upper()} ---")
    for a in alphas:
        k = f"top10_a2={a}"
        c = data[k][ds]['average_selected_chunks']
        p = data[k][ds]['micro_evidence_precision'] or 0
        r = data[k][ds]['conditional_reserve_support_recall'] or 0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0
        print(f"| **Ours (Top-10 + Stage 2: $\\alpha_2={a}$)** | {c:.2f} | {p*100:.1f}% | {r*100:.1f}% | {f1*100:.1f}% |")
