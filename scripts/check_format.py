import json
with open(r'E:\splits\hotpotqa\calibration_manifest.json', 'r') as f:
    data = json.load(f)
    print(type(data))
    if isinstance(data, list):
        print(len(data))
        if len(data) > 0:
            print(data[0])
    elif isinstance(data, dict):
        print(list(data.keys())[:5])
