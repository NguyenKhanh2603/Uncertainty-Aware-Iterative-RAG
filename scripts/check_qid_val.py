import json
import gzip

with gzip.open(r'E:\Downloads\filtered_splits_conformal_data\hotpotqa_filtered_top30.jsonl.gz', 'rt') as f:
    for line in f:
        row = json.loads(line)
        print(row.get('qid'))
        break
