import gzip
import json
import os
from collections import defaultdict

def main():
    manifests = {
        ("tatqa", "test"): r"E:\splits\tatqa\test_manifest.json",
        ("mmqa", "test"): r"E:\splits\mmqa\test_manifest.json",
        ("hotpotqa", "test"): r"E:\splits\hotpotqa\test_manifest.json",
        ("webqa", "test"): r"E:\splits\webqa\test_manifest.json",
        ("tatqa", "calibration"): r"E:\splits\tatqa\calibration_manifest.json",
        ("webqa", "calibration"): r"E:\splits\webqa\calibration_manifest.json",
        ("mmqa", "calibration"): r"E:\splits\mmqa\calibration_manifest.json",
        ("hotpotqa", "calibration"): r"E:\splits\hotpotqa\calibration_manifest.json"
    }
    
    # Map (dataset, split) -> set of QueryIDs
    target_qids = defaultdict(set)
    for (ds, split), mf in manifests.items():
        with open(mf, "r", encoding="utf-8") as f:
            data = json.load(f)
            for qid in data.get("plan", []):
                target_qids[(ds, split)].add(qid)
            print(f"Loaded {len(data.get('plan', []))} queries for {ds} {split}")
            
    input_dir = r"E:\Downloads\conformal_backfill_2_3_results"
    output_dir = r"E:\Downloads\filtered_splits_conformal_data"
    os.makedirs(output_dir, exist_ok=True)
    
    datasets = ["hotpotqa", "mmqa", "tatqa", "webqa"]
    
    for ds in datasets:
        in_file = os.path.join(input_dir, f"{ds}_top30_retrieval.jsonl.gz")
        out_file = os.path.join(output_dir, f"{ds}_filtered_top30.jsonl.gz")
        
        calib_ids = target_qids[(ds, "calibration")]
        test_ids = target_qids[(ds, "test")]
        
        print(f"\nFiltering {ds}... Targets: {len(calib_ids)} calib, {len(test_ids)} test.")
        
        kept_calib = set()
        kept_test = set()
        
        with gzip.open(in_file, "rt", encoding="utf-8") as f_in, gzip.open(out_file, "wt", encoding="utf-8") as f_out:
            for line in f_in:
                if not line.strip(): continue
                row = json.loads(line)
                qid = row.get("qid")
                
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
