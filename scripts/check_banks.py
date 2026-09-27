import gzip, json

data = json.load(gzip.open(r"E:\Downloads\filtered_splits_conformal_data\pooled_reference_banks.json.gz", "rt", encoding="utf-8"))
for b in data["banks"]:
    print(f"  {b['condition']}: n_false={b['n_false_scores']}")
print(f"min_bank_size: {data['min_bank_size']}")
print(f"underpowered: {data.get('underpowered', [])}")
