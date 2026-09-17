"""Sweep conservative internal-fusion weights on fixed NeuCLIR topic roles."""
from __future__ import annotations
import argparse,hashlib,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
def h(q):return int.from_bytes(hashlib.sha256(('neuclir-fusion\0'+q).encode()).digest()[:8],'big')
def z(x):return (x-x.mean())/(x.std() or 1)
def tau(x,a):
 x=np.sort(1-x);return float(1-x[min(len(x)-1,int(np.ceil((len(x)+1)*(1-a)))-1)])
def main():
 p=argparse.ArgumentParser();p.add_argument('--features',type=Path,required=True);p.add_argument('--qrels',type=Path,required=True);p.add_argument('--nuggets',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();d=np.load(a.features);q=np.unique(d['qids']);o=sorted(q,key=h);tr,ca,te=set(o[:5]),set(o[5:10]),set(o[10:]);rel=defaultdict(set)
 for l in open(a.qrels):
  qi,_,doc,r=l.split();
  if int(r)>0:rel[qi].add(doc)
 y=np.array([doc in rel[qi] for qi,doc in zip(d['qids'],d['docids'])]);X=np.column_stack([d['cosine'],d['lm_yes_no'],d['hidden'].astype('float32')]);mt=np.isin(d['qids'],list(tr));s=StandardScaler().fit(X[mt]);p0=LogisticRegression(C=.01,max_iter=1000,class_weight='balanced').fit(s.transform(X[mt]),y[mt]);raw=p0.decision_function(s.transform(X));n=json.load(open(a.nuggets));out={}
 for w in [0,.05,.1,.2,.4,.75]:
  score=np.empty(len(raw))
  for qi in q:
   m=d['qids']==qi;score[m]=1/(1+np.exp(-(z(d['cosine'][m])+.2*z(d['lm_yes_no'][m])+w*z(raw[m]))))
  crit=[]
  for qi in ca:
   m=np.where(d['qids']==qi)[0]
   for ng in n[qi]['nugget_bank'].values():
    sup={r['doc_id'] for ans in ng.get('answers',{}).values() for r in ans.get('references',[])};ids=[i for i in m if d['docids'][i] in sup]
    if ids:crit.append(score[ids].max())
  t=tau(np.array(crit),.1);ks=[];cov=[];tot=[]
  for qi in te:
   m=np.where(d['qids']==qi)[0];docs=set(d['docids'][m][score[m]>=t]);ks.append(sum(score[m]>=t));c=e=0
   for ng in n[qi]['nugget_bank'].values():
    sup={r['doc_id'] for ans in ng.get('answers',{}).values() for r in ans.get('references',[])}
    if sup&set(d['docids'][m]):e+=1;c+=bool(sup&docs)
   cov.append(c);tot.append(e)
  out[str(w)]={'threshold':t,'mean_chunks_kept':float(np.mean(ks)),'nugget_micro_coverage':float(sum(cov)/sum(tot)),'query_all_nugget_coverage':float(np.mean(np.array(cov)==np.array(tot)))}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps({'alpha':.1,'weights':out},indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
