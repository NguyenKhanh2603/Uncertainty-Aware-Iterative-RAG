import json

def format_percentage(val):
    if val is None: return "N/A"
    return f"{val * 100:.1f}%"

def main():
    summary_path = r"E:\Downloads\filtered_manifest_conformal_data\benchmark_with_pooled\backfill_benchmark_summary.json"
    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    datasets = ["hotpotqa", "mmqa", "tatqa", "webqa"]
    
    with open("temp_output.md", "w", encoding="utf-8") as out:
        for ds in datasets:
            out.write(f"### Dataset: {ds}\n")
            out.write("| Method | Kept Chunks | Precision | Recall | Empty Rate |\n")
            out.write("| :--- | :--- | :--- | :--- | :--- |\n")
            
            base = data["strategies"]["dataset_pooled"]["0.1"][ds]["methods"]["fixed_top_k"]
            out.write(f"| Fixed Top-10 | {base['average_selected_chunks']:.2f} | {format_percentage(base['micro_evidence_precision'])} | {format_percentage(base['conditional_reserve_support_recall'])} | {format_percentage(base['empty_context_rate'])} |\n")
            
            for alpha in ["0.1", "0.9", "0.99"]:
                bh = data["strategies"]["dataset_pooled"][alpha][ds]["methods"]["conformal_backfill"]
                out.write(f"| BH cosine (α={alpha}, ctx=10) | {bh['average_selected_chunks']:.2f} | {format_percentage(bh['micro_evidence_precision'])} | {format_percentage(bh['conditional_reserve_support_recall'])} | {format_percentage(bh['empty_context_rate'])} |\n")
                
            for alpha in ["0.1", "0.9", "0.99"]:
                mod = data["strategies"]["modality_aware"][alpha][ds]["methods"]["conformal_backfill"]
                out.write(f"| Conformal BH (modality_aware, α={alpha}) | {mod['average_selected_chunks']:.2f} | {format_percentage(mod['micro_evidence_precision'])} | {format_percentage(mod['conditional_reserve_support_recall'])} | {format_percentage(mod['empty_context_rate'])} |\n")
            out.write("\n")

if __name__ == "__main__":
    main()
