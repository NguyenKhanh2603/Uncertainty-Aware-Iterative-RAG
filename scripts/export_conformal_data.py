import gzip
import json
import csv
import argparse
from pathlib import Path
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(description="Extract Conformal empirical data for algorithm redesign.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Path to the conformal_backfill_2_3_results directory")
    parser.add_argument("--selection-subdir", type=str, default="selection", help="Subdirectory containing selection_decisions.jsonl.gz")
    parser.add_argument("--out-inference-name", type=str, default="inference_pvalues.csv", help="Name of the output inference CSV file")
    args = parser.parse_args()

    output_dir = args.output_dir
    datasets = ["mmqa", "webqa", "hotpotqa", "tatqa"]
    top_l = 30

    print("Generating false_rank_distribution.csv...")
    rank_dist = defaultdict(lambda: defaultdict(int)) # (dataset, modality) -> {rank: count}

    for dataset in datasets:
        retrieval_file = output_dir / f"{dataset}_top{top_l}_retrieval.jsonl.gz"
        if not retrieval_file.exists():
            print(f"Warning: {retrieval_file} not found, skipping...")
            continue
            
        with gzip.open(retrieval_file, "rt", encoding="utf-8") as f:
            for line in f:
                if line.startswith("["):
                    continue # skip [PERFORMANCE]
                record = json.loads(line)
                # Chỉ đếm các False chunks từ tập calibration để làm chuẩn Reference Bank
                if record.get("split_role") == "calibration":
                    if record.get("support_label") == "false":
                        rank = record.get("rank")
                        modality = record.get("modality", "text")
                        if rank is not None:
                            rank_dist[(dataset, modality)][rank] += 1

    out_csv = output_dir / "false_rank_distribution.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset", "Modality", "Rank", "False_Count"])
        for (dataset, modality), ranks in sorted(rank_dist.items()):
            for rank, count in sorted(ranks.items()):
                writer.writerow([dataset, modality, rank, count])
    print(f"Saved {out_csv}")

    print("Generating inference_pvalues.csv...")
    selection_file = output_dir / args.selection_subdir / "selection_decisions.jsonl.gz"
    out_inference = output_dir / args.out_inference_name
    
    if not selection_file.exists():
        print(f"Error: {selection_file} not found!")
        return

    with gzip.open(selection_file, "rt", encoding="utf-8") as f, \
         open(out_inference, "w", newline="", encoding="utf-8") as out_f:
        
        writer = csv.writer(out_f)
        writer.writerow(["Dataset", "QueryID", "ChunkID", "Rank", "SupportLabel", "CosineScore", "PValue", "Numerator", "Denominator"])
        
        for line in f:
            record = json.loads(line)
            if "strategies" not in record or "modality_aware" not in record["strategies"]:
                continue
            strategy_record = record["strategies"]["modality_aware"]
            dataset = strategy_record["dataset"]
            qid = strategy_record["qid"]
            for decision in strategy_record.get("decisions", []):
                # Khôi phục tử số và mẫu số từ P-value đã lưu
                denominator = decision["bank_size"] + 1
                numerator = round(decision["p_value"] * denominator)
                writer.writerow([
                    dataset, 
                    qid, 
                    decision["chunk_id"], 
                    decision["rank"], 
                    decision["support_label"],
                    decision["cosine_score"],
                    decision["p_value"], 
                    numerator, 
                    denominator
                ])
    print(f"Saved {out_inference}")

if __name__ == "__main__":
    main()
