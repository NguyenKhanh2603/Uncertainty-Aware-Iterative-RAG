"""Fair same-split CCE, cosine query-level, and internal-fusion comparison."""
from __future__ import annotations
import argparse,hashlib,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
def key(q):return int.from_bytes(hashlib.sha256(('neuclir-fusion\0'+q).encode()).digest()[:8],'big')
def z(x):return (x-x.mean())/(x.std() or 1)
def tau(x,a):
 x=np.sort(1-x);return float(1-x[min(len(x)-1,int(np.ceil((len(x)+1)*(1-a)))-1)])
def main():
 p=argparse.ArgumentParser();p.add_argument('--features',type=Path,required=True);p.add_argument('--qrels',type=Path,required=True);p.add_argument('--nuggets',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--alpha',type=float,default=.1);a=p.parse_args();d=np.load(a.features);qs=np.unique(d['qids']);o=sorted(qs,key=key);tr,ca,te=set(o[:5]),set(o[5:10]),set(o[10:]); rel=defaultdict(set)
 for line in open(a.qrels):
  q,_,doc,r=line.split();
  if int(r)>0:rel[q].add(doc)
 labels=np.array([doc in rel[q] for q,doc in zip(d['qids'],d['docids'])]); X=np.column_stack([d['cosine'],d['lm_yes_no'],d['hidden'].astype('float32')]);mtr=np.isin(d['qids'],list(tr));sc=StandardScaler().fit(X[mtr]);pr=LogisticRegression(C=.01,max_iter=1000,class_weight='balanced').fit(sc.transform(X[mtr]),labels[mtr]); raw=pr.decision_function(sc.transform(X)); fused=np.empty(len(raw))
 for q in qs:
  m=d['qids']==q;fused[m]=1/(1+np.exp(-(z(d['cosine'][m])+.2*z(d['lm_yes_no'][m])+.75*z(raw[m]))))
 nuggets=json.load(open(a.nuggets)); arrays={'cce_qwen3_cosine':d['cosine'],'query_qwen3_cosine':d['cosine'],'query_lm_head':1/(1+np.exp(-d['lm_yes_no'])),'query_internal_fusion':fused};out={}
 for name,s in arrays.items():
  if name.startswith('cce'): t=tau(s[np.isin(d['qids'],list(ca))&labels],a.alpha)
  else:
   crit=[]
   for q in ca:
    m=np.where(d['qids']==q)[0]
    for n in nuggets[q]['nugget_bank'].values():
     docs={r['doc_id'] for ans in n.get('answers',{}).values() for r in ans.get('references',[])};ids=[i for i in m if d['docids'][i] in docs]
     if ids:crit.append(s[ids].max())
   t=tau(np.array(crit),a.alpha)
  kept=[];eligible=[];covered=[]
  for q in te:
   m=np.where(d['qids']==q)[0]; docs=set(d['docids'][m][s[m]>=t]);kept.append(sum(s[m]>=t)); es=cs=0
   for n in nuggets[q]['nugget_bank'].values():
    sup={r['doc_id'] for ans in n.get('answers',{}).values() for r in ans.get('references',[])}
    if sup&set(d['docids'][m]):es+=1;cs+=bool(sup&docs)
   eligible.append(es);covered.append(cs)
  out[name]={'threshold':float(t),'mean_chunks_kept':float(np.mean(kept)),'context_reduction':float(1-np.mean(kept)/100),'nugget_micro_coverage':float(sum(covered)/sum(eligible)),'query_all_nugget_coverage':float(np.mean(np.array(covered)==np.array(eligible)))}
 r={'status':'same_split_selection_proxy','alpha':a.alpha,'splits':{'train':sorted(tr),'calibration':sorted(ca),'test':sorted(te)},'results':out,'note':'Document nugget labels inherited by chunks; this is not report F1.'};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
if __name__=='__main__':main()
