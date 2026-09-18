"""Same-split BY ablation: cosine, LM head, hidden probe, and fusion."""
from __future__ import annotations
import argparse, hashlib, json
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from uncertainty_rag.core.conformal_selection import benjamini_yekutieli

def h(q): return int.from_bytes(hashlib.sha256(('neuclir-fusion\0'+q).encode()).digest()[:8],'big')
def z(x): return (x-x.mean())/(x.std() or 1.)
def main():
 p=argparse.ArgumentParser();p.add_argument('--features',type=Path,required=True);p.add_argument('--qrels',type=Path,required=True);p.add_argument('--nuggets',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--top-l',type=int,default=30);p.add_argument('--alphas',default='0.1,0.2');a=p.parse_args();d=np.load(a.features);qs=np.unique(d['qids']);ordered=sorted(qs,key=h);tr,ca,te=set(ordered[:5]),set(ordered[5:10]),set(ordered[10:]);rel=defaultdict(set)
 for line in open(a.qrels):
  q,_,doc,r=line.split()
  if int(r)>0:rel[q].add(doc)
 # Restrict every topic to the same Qwen3 Top-L order stored in the feature artifact.
 keep=np.zeros(len(d['qids']),bool)
 for q in qs: keep[np.where(d['qids']==q)[0][:a.top_l]]=True
 d={k:v[keep] for k,v in d.items() if getattr(v, 'ndim', 0) > 0}; y=np.array([doc in rel[q] for q,doc in zip(d['qids'],d['docids'])])
 X=np.column_stack([d['cosine'],d['lm_yes_no'],d['hidden'].astype('float32')]); mt=np.isin(d['qids'],list(tr));scale=StandardScaler().fit(X[mt]);probe=LogisticRegression(C=.01,max_iter=1000,class_weight='balanced').fit(scale.transform(X[mt]),y[mt]);raw=probe.decision_function(scale.transform(X));scores={}
 for name,vals in [('cosine',d['cosine']),('cosine_plus_lm',None),('cosine_plus_hidden',None),('cosine_plus_lm_plus_hidden',None)]:
  s=np.empty(len(raw))
  for q in qs:
   m=d['qids']==q
   if name=='cosine': s[m]=z(d['cosine'][m])
   elif name=='cosine_plus_lm': s[m]=z(d['cosine'][m])+.2*z(d['lm_yes_no'][m])
   elif name=='cosine_plus_hidden': s[m]=z(d['cosine'][m])+.1*z(raw[m])
   else:s[m]=z(d['cosine'][m])+.2*z(d['lm_yes_no'][m])+.1*z(raw[m])
  scores[name]=s
 nuggets=json.load(open(a.nuggets));out={}
 for alpha in map(float,a.alphas.split(',')):
  result={}
  for name,s in scores.items():
   bank=np.sort(s[np.isin(d['qids'],list(ca)) & ~y]); selected=np.zeros(len(s),bool)
   for q in te:
    ix=np.where(d['qids']==q)[0]; pv=(1+len(bank)-np.searchsorted(bank,s[ix],side='left'))/(len(bank)+1); selected[ix[list(benjamini_yekutieli(pv.tolist(),alpha).rejected_indices)]]=True
   test=np.isin(d['qids'],list(te)); nsel=selected[test].sum(); tp=(selected[test]&y[test]).sum(); groups=[]
   for q in te:
    ix=np.where(d['qids']==q)[0]; docs=set(d['docids'][ix[selected[ix]]]); eligible=covered=0
    for ng in nuggets[q]['nugget_bank'].values():
     sup={r['doc_id'] for ans in ng.get('answers',{}).values() for r in ans.get('references',[])}
     if sup&set(d['docids'][ix]): eligible+=1;covered+=bool(sup&docs)
    groups.append((eligible,covered,selected[ix].sum()))
   result[name]={'mean_chunks_kept':float(selected[test].sum()/len(te)),'empty_rate':float(np.mean([x[2]==0 for x in groups])),'inherited_chunk_precision':float(tp/nsel) if nsel else 0.,'inherited_chunk_recall':float(tp/y[test].sum()) if y[test].sum() else 0.,'nugget_micro_coverage':float(sum(x[1] for x in groups)/sum(x[0] for x in groups)),'query_all_nugget_coverage':float(np.mean([x[0]==x[1] for x in groups]))}
  out[str(alpha)]=result
 r={'status':'same_split_by_selection_proxy','top_l':a.top_l,'splits':{'train':sorted(tr),'calibration':sorted(ca),'test':sorted(te)},'scores':{'cosine':'Qwen3 cosine','lm':'Qwen2.5 Yes-No logit','hidden':'trained logistic probe over Qwen2.5 final hidden state'},'results':out,'note':'Public document nugget labels are inherited by chunks; BY tests candidate false-score banks.'};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
if __name__=='__main__':main()
