import json
data = json.load(open(r'E:\splits\hotpotqa\test_manifest.json', encoding='utf-8'))
if isinstance(data, dict):
    print(data.keys())
print(open(r'E:\splits\hotpotqa\test_manifest.json', encoding='utf-8').read()[:500])
