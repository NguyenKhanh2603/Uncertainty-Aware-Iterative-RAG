import json

def format_percentage(val):
    if val is None: return "N/A"
    return f"{val * 100:.1f}%"

def get_f1(prec, rec):
    if prec is None or rec is None or prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)

def main():
    path_10 = r"E:\Downloads\filtered_splits_conformal_data\benchmark_modality_aware\backfill_benchmark_summary.json"
    path_20 = r"E:\Downloads\filtered_splits_conformal_data\benchmark_modality_aware_ctx20\backfill_benchmark_summary.json"
    
    with open(path_10, "r", encoding="utf-8") as f:
        data10 = json.load(f)
    with open(path_20, "r", encoding="utf-8") as f:
        data20 = json.load(f)
        
    datasets = {
        "hotpotqa": [
            ("ECIR CCE", "N/A", "29.1%", "90.0%", "N/A"),
            ("CONFLARE", "N/A", "29.2%", "90.0%", "N/A"),
            ("TRAQ retrieval", "N/A", "25.0%", "93.0%", "N/A"),
        ],
        "mmqa": [
            ("ECIR CCE", "N/A", "12.4%", "92.9%", "N/A"),
            ("CONFLARE", "N/A", "12.3%", "92.3%", "N/A"),
            ("TRAQ retrieval", "N/A", "10.9%", "97.4%", "N/A"),
        ],
        "tatqa": [
            ("ECIR CCE", "N/A", "29.0%", "85.1%", "N/A"),
            ("CONFLARE", "N/A", "28.8%", "84.3%", "N/A"),
            ("TRAQ retrieval", "N/A", "25.9%", "89.6%", "N/A"),
        ],
        "webqa": [
            ("ECIR CCE", "N/A", "12.0%", "81.7%", "N/A"),
            ("CONFLARE", "N/A", "12.0%", "81.7%", "N/A"),
            ("TRAQ retrieval", "N/A", "10.7%", "89.7%", "N/A"),
        ]
    }
    
    with open("E:\\Downloads\\conformal prediction\\Conformal\\Uncertainty-Aware-Iterative-RAG\\baseline_comparison_report_1000calib_100test_full.md", "w", encoding="utf-8") as out:
        out.write("# Báo cáo So sánh Tổng thể (1000 Calibration, 100 Test)\n\n")
        out.write("Báo cáo này tổng hợp kết quả của các baseline (ECIR CCE, CONFLARE, TRAQ) cùng với Fixed Top-10, Top-20 và phương pháp Conformal BH (cả dataset_pooled lẫn modality_aware).\n\n")
        
        for ds, baselines in datasets.items():
            out.write(f"### Dataset: {ds.upper()}\n")
            out.write("| Method | Kept Chunks | Precision | Recall | Empty Rate |\n")
            out.write("| :--- | :--- | :--- | :--- | :--- |\n")
            
            # Print Fixed Top-10
            base10 = data10["strategies"]["dataset_pooled"]["0.1"][ds]["methods"]["fixed_top_k"]
            out.write(f"| Fixed Top-10 | {base10['average_selected_chunks']:.2f} | {format_percentage(base10['micro_evidence_precision'])} | {format_percentage(base10['conditional_reserve_support_recall'])} | {format_percentage(base10['empty_context_rate'])} |\n")
            
            # Print Fixed Top-20
            base20 = data20["strategies"]["dataset_pooled"]["0.99"][ds]["methods"]["fixed_top_k"]
            out.write(f"| Fixed Top-20 | {base20['average_selected_chunks']:.2f} | {format_percentage(base20['micro_evidence_precision'])} | {format_percentage(base20['conditional_reserve_support_recall'])} | {format_percentage(base20['empty_context_rate'])} |\n")
            
            # Print Baselines
            for b in baselines:
                out.write(f"| {b[0]} | {b[1]} | {b[2]} | {b[3]} | {b[4]} |\n")
            
            # dataset_pooled (BH cosine) ctx=10
            for alpha in ["0.1", "0.9", "0.99"]:
                bh = data10["strategies"]["dataset_pooled"][alpha][ds]["methods"]["conformal_backfill"]
                out.write(f"| BH cosine (α={alpha}, ctx=10) | {bh['average_selected_chunks']:.2f} | {format_percentage(bh['micro_evidence_precision'])} | {format_percentage(bh['conditional_reserve_support_recall'])} | {format_percentage(bh['empty_context_rate'])} |\n")
                
            # dataset_pooled (BH cosine) ctx=20
            bh20 = data20["strategies"]["dataset_pooled"]["0.99"][ds]["methods"]["conformal_backfill"]
            out.write(f"| BH cosine (α=0.99, ctx=20) | {bh20['average_selected_chunks']:.2f} | {format_percentage(bh20['micro_evidence_precision'])} | {format_percentage(bh20['conditional_reserve_support_recall'])} | {format_percentage(bh20['empty_context_rate'])} |\n")
            
            # modality_aware ctx=10
            for alpha in ["0.1", "0.9", "0.99"]:
                mod = data10["strategies"]["modality_aware"][alpha][ds]["methods"]["conformal_backfill"]
                out.write(f"| Conformal BH (modality_aware, α={alpha}, ctx=10) | {mod['average_selected_chunks']:.2f} | {format_percentage(mod['micro_evidence_precision'])} | {format_percentage(mod['conditional_reserve_support_recall'])} | {format_percentage(mod['empty_context_rate'])} |\n")
                
            # modality_aware ctx=20
            mod20 = data20["strategies"]["modality_aware"]["0.99"][ds]["methods"]["conformal_backfill"]
            out.write(f"| Conformal BH (modality_aware, α=0.99, ctx=20) | {mod20['average_selected_chunks']:.2f} | {format_percentage(mod20['micro_evidence_precision'])} | {format_percentage(mod20['conditional_reserve_support_recall'])} | {format_percentage(mod20['empty_context_rate'])} |\n")
            
            out.write("\n")

if __name__ == "__main__":
    main()
