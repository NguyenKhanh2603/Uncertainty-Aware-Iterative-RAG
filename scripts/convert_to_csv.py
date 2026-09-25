import gzip
import json
import csv

def main():
    files = [
        r"E:\Downloads\filtered_manifest_conformal_data\test_selection\selection_decisions.jsonl.gz"
    ]
    
    out_csv = r"E:\Downloads\filtered_manifest_conformal_data\inference_pvalues_test.csv"
    
    with open(out_csv, "w", encoding="utf-8", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["Dataset", "QueryID", "Rank", "ChunkID", "PValue", "SupportLabel"])
        
        for f in files:
            with gzip.open(f, "rt", encoding="utf-8") as f_in:
                for line in f_in:
                    if not line.strip(): continue
                    data = json.loads(line)
                    strat = data.get("strategies", {}).get("modality_aware", {})
                    dataset = strat.get("dataset")
                    qid = strat.get("qid")
                    decisions = strat.get("decisions", [])
                    for d in decisions:
                        writer.writerow([
                            dataset,
                            qid,
                            d.get("rank"),
                            d.get("chunk_id"),
                            d.get("p_value"),
                            d.get("support_label")
                        ])

if __name__ == "__main__":
    main()
