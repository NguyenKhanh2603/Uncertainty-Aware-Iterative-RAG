import csv
import gzip
import json
import os
from collections import defaultdict

def main():
    report_csv = r"E:\Downloads\report_100_calibration_100_test_question_ids_2026-09-19\report_100_calibration_100_test_question_ids_2026-09-19.csv"
    
    # Map (dataset, split) -> set of QueryIDs
    target_qids = defaultdict(set)
    with open(report_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            target_qids[(row['Dataset'], row['SplitRole'])].add(row['QueryID'])
            
    input_dir = r"E:\Downloads\bacao\Uncertainty-Aware-Iterative-RAG-results-qwen2vl-jina4-four-datasets-2026-09-14\research\internal_state_rag\results\qwen2vl_jina4"
    output_dir = r"E:\Downloads\filtered_conformal_data"
    os.makedirs(output_dir, exist_ok=True)
    
    datasets = ["hotpotqa", "mmqa", "tatqa", "webqa"]
    
    for ds in datasets:
        in_file = os.path.join(input_dir, f"{ds}_jina_v4_top30.jsonl.gz")
        out_file = os.path.join(output_dir, f"{ds}_filtered_top30.jsonl.gz")
        
        calib_ids = target_qids[(ds, "calibration")]
        test_ids = target_qids[(ds, "test")]
        
        print(f"Filtering {ds}... Targets: {len(calib_ids)} calib, {len(test_ids)} test.")
        
        kept_calib = set()
        kept_test = set()
        
        with gzip.open(in_file, "rt", encoding="utf-8") as f_in, gzip.open(out_file, "wt", encoding="utf-8") as f_out:
            for line in f_in:
                if not line.strip(): continue
                row = json.loads(line)
                qid = row.get("qid")
                # Force the split_role to be what we want if it matches
                if qid in calib_ids:
                    row["split_role"] = "calibration"
                    f_out.write(json.dumps(row) + "\n")
                    kept_calib.add(qid)
                elif qid in test_ids:
                    row["split_role"] = "test"
                    f_out.write(json.dumps(row) + "\n")
                    kept_test.add(qid)
                    
        print(f"  -> Kept {len(kept_calib)} calib and {len(kept_test)} test queries.")

if __name__ == "__main__":
    main()
