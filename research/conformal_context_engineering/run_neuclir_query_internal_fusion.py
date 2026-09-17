"""Train frozen-state fusion on train topics; calibrate and test query-level pruning."""
from __future__ import annotations
import argparse,hashlib,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def role(q): return int.from_bytes(hashlib.sha256(('neuclir-fusion\0'+q).encode()).digest()[:8],'big')
def z(x):
 s=x.std(); return (x-x.mean())/(s if s else 1.)
def threshold(x,a):
 x=np.sort(1-x); return float(1-x[min(len(x)-1,int(np.ceil((len(x)+1)*(1-a)))-1)])
def main():
 p=argparse.ArgumentParser();p.add_argument('--features',type=Path,required=True);p.add_argument('--qrels',type=Path,required=True);p.add_argument('--nuggets',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--alpha',type=float,default=.1);a=p.parse_args()
 d=np.load(a.features); qids=np.unique(d['qids']); ordered=sorted(qids,key=role); train,cal,test=set(ordered[:5]),set(ordered[5:10]),set(ordered[10:])
 pos=defaultdict(set)
 for line in open(a.qrels):
  q,_,doc,r=line.split();
  if int(r)>0:pos[q].add(doc)
 labels=np.array([doc in pos[q] for q,doc in zip(d['qids'],d['docids'])])
 # Probe learns only on five train topics. Cosine and LM logit remain explicit features.
 base=np.column_stack([d['cosine'],d['lm_yes_no'],d['hidden'].astype(np.float32)])
 tr=np.isin(d['qids'],list(train)); scaler=StandardScaler(); Xtr=scaler.fit_transform(base[tr]); probe=LogisticRegression(C=.01,max_iter=1000,class_weight='balanced').fit(Xtr,labels[tr])
 raw=probe.decision_function(scaler.transform(base)); scores=np.empty(len(raw),np.float32)
 for q in qids:
  m=d['qids']==q; scores[m]=z(d['cosine'][m])+.2*z(d['lm_yes_no'][m])+.75*z(raw[m])
 scores=1/(1+np.exp(-scores))
 nuggets=json.load(open(a.nuggets)); critical=[]
 for q in cal:
  m=np.where(d['qids']==q)[0]; bydoc=defaultdict(list)
  for i in m: bydoc[d['docids'][i]].append(i)
  vals=[]
  for nug in nuggets[q]['nugget_bank'].values():
   docs={r['doc_id'] for ans in nug.get('answers',{}).values() for r in ans.get('references',[])}; ids=[i for doc in docs for i in bydoc.get(doc,[])]
   if ids: vals.append(scores[ids].max())
  if vals: critical.append(min(vals))
 tau=threshold(np.array(critical),a.alpha); rows=[]
 for q in test:
  m=np.where(d['qids']==q)[0]; keep=m[scores[m]>=tau]; docs=set(d['docids'][keep]); eligible=covered=0
  for nug in nuggets[q]['nugget_bank'].values():
   support={r['doc_id'] for ans in nug.get('answers',{}).values() for r in ans.get('references',[])}
   if support & set(d['docids'][m]): eligible+=1; covered+=bool(support&docs)
  rows.append((len(keep),eligible,covered))
 out={'status':'pilot_frozen_internal_fusion_query_level','splits':{'train_topics':sorted(train),'calibration_topics':sorted(cal),'test_topics':sorted(test)},'alpha':a.alpha,'threshold':tau,'fusion':'z(Qwen3 cosine)+0.20 z(Qwen2.5 LM Yes-No)+0.75 z(logistic probe(final hidden))','mean_chunks_kept':float(np.mean([r[0] for r in rows])),'context_reduction':float(1-np.mean([r[0] for r in rows])/100),'nugget_micro_coverage':float(sum(r[2] for r in rows)/sum(r[1] for r in rows)),'query_all_nugget_coverage':float(np.mean([r[1]==r[2] for r in rows])),'note':'Pilot only: five train and five calibration topics are too small for a formal generalization claim.'};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
