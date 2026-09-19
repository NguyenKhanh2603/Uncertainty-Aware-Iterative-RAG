"""Create fixed 100-calibration/1000-official-dev-test feature plans."""
import gzip,json,random
from pathlib import Path
r=Path('research/internal_state_rag/results/official_seed42_hotpotqa_cosine_top30.jsonl.gz')
roles={}
with gzip.open(r,'rt') as f:
    for line in f:
        x=json.loads(line); roles.setdefault(x['split_role'],set()).add(str(x['qid']))
cal=sorted(roles['calibration']); random.Random(42).shuffle(cal); cal=cal[:100]
test=sorted(roles['test'])
out=Path('research/internal_state_rag/results/official_seed42_hotpotqa_plans')
for role,plan in [('calibration',cal),('test',test)]:
    p=out/role/'manifest.json';p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps({'dataset':'hotpotqa','input_role':role,'top_l':30,'seed':42,'plan':plan},indent=2)+'\n')
print(len(cal),len(test))
