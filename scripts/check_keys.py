import gzip
import json
with gzip.open(r'E:\Downloads\filtered_splits_conformal_data\hotpotqa_filtered_top30.jsonl.gz', 'rt') as f:
    for i, line in enumerate(f):
        if i == 0:
            print(list(json.loads(line).keys()))
            break
