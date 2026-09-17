"""Compare CCE and query-nugget conformal targets on common English chunks.

Nugget qrels are document-level, so chunk relevance is inherited from its parent
for this *selection proxy*. It is not an ARGUE F1 experiment.
"""
from __future__ import annotations
import argparse, hashlib, json
from collections import defaultdict
from pathlib import Path
import numpy as np
from uncertainty_rag.core.conformal_selection import benjamini_yekutieli

def args():
 p=argparse.ArgumentParser(); p.add_argument('--scores',type=Path,required=True); p.add_argument('--qrels',type=Path,required=True); p.add_argument('--nuggets',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--top-l',type=int,default=100); p.add_argument('--alphas',default='0.05,0.1,0.2'); p.add_argument('--seed',type=int,default=20260917); return p.parse_args()
def cal(q,s): return int.from_bytes(hashlib.sha256(f'{s}\0{q}'.encode()).digest()[:8],'big')/2**64<.5
def tau(x,a):
 x=np.sort(1-x); return float(1-x[min(len(x)-1,int(np.ceil((len(x)+1)*(1-a)))-1)])
def met(labels, groups, mask):
 kept=mask.sum(); retrieved=labels.any(1); retained=labels&mask
 coverage=[sum(bool(set(np.flatnonzero(row))&g) for g in gs if g) for row,gs in zip(mask,groups)]
 totals=[sum(bool(g) for g in gs) for gs in groups]
 return {'mean_chunks_kept':float(mask.sum(1).mean()),'context_reduction':float(1-mask.mean()),'empty_rate':float((mask.sum(1)==0).mean()),'inherited_chunk_precision':float(retained.sum()/kept) if kept else 0.,'inherited_chunk_recall':float(retained[retrieved].sum()/labels[retrieved].sum()) if labels[retrieved].sum() else 0.,'nugget_micro_coverage':float(sum(coverage)/sum(totals)) if sum(totals) else 0.,'query_all_nugget_coverage':float(np.mean([x==y for x,y in zip(coverage,totals)]))}
def main():
 a=args(); raw=json.load(open(a.scores))['scores']; qrel=defaultdict(set)
 for line in open(a.qrels):
  q,_,d,r=line.split();
  if int(r)>0:qrel[q].add(d)
 nuggets=json.load(open(a.nuggets)); qids=sorted(raw,key=lambda q:(not cal(q,a.seed),q)); ncal=sum(cal(q,a.seed) for q in qids)
 rows=[raw[q][:a.top_l] for q in qids]; scores=np.array([[x['score'] for x in r] for r in rows]); labels=np.array([[x['docid'] in qrel[q] for x in r] for q,r in zip(qids,rows)])
 groups=[]
 for q,r in zip(qids,rows):
  g=[]
  for nug in nuggets[q]['nugget_bank'].values():
   docs={ref['doc_id'] for ans in nug.get('answers',{}).values() for ref in ans.get('references',[])}
   g.append({i for i,x in enumerate(r) if x['docid'] in docs})
  groups.append(g)
 result={}
 for alpha in map(float,a.alphas.split(',')):
  ct=tau(scores[:ncal][labels[:ncal]],alpha); crit=[min(scores[i,list(g)].max() for g in gs if g) for i,gs in enumerate(groups[:ncal]) if any(gs)]; qt=tau(np.array(crit),alpha); bank=np.sort(scores[:ncal][~labels[:ncal]]); by=np.zeros_like(scores[ncal:],bool)
  for i,row in enumerate(scores[ncal:]):
   pv=(1+len(bank)-np.searchsorted(bank,row,'left'))/(len(bank)+1); by[i,benjamini_yekutieli(pv.tolist(),alpha).rejected_indices]=True
  result[str(alpha)]={'cce_positive_inherited_labels':{'threshold':ct,**met(labels[ncal:],groups[ncal:],scores[ncal:]>=ct)},'query_all_nugget':{'threshold':qt,**met(labels[ncal:],groups[ncal:],scores[ncal:]>=qt)},'candidate_by':met(labels[ncal:],groups[ncal:],by)}
 out={'status':'selection_proxy_not_ARGUE_F1','score':'Qwen3-Embedding-8B on extracted English sentence-boundary chunks','labels':'public nugget-to-document qrels inherited by chunks','split':{'calibration_topics':ncal,'test_topics':len(qids)-ncal},'top_l':a.top_l,'results':result}; a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
