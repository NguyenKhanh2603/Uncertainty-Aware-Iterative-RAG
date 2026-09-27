import json
import gzip

with open(r'E:\splits\hotpotqa\calibration_manifest.json', 'r') as f:
    splits_qids = set(json.load(f))

data_qids = set()
with gzip.open(r'E:\Downloads\filtered_splits_conformal_data\hotpotqa_filtered_top30.jsonl.gz', 'rt') as f:
    for line in f:
        row = json.loads(line)
        if row.get('split_role') == 'calibration':
            data_qids.add(row.get('qid'))

print(f"Splits calib count: {len(splits_qids)}")
print(f"Data calib count: {len(data_qids)}")
print(f"Intersection: {len(splits_qids.intersection(data_qids))}")
