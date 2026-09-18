"""Candidate-wise BY ablation on cached Qwen-7B features for four datasets."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from uncertainty_rag.core.conformal_selection import benjamini_yekutieli

def qz(x): return (x-x.mean(1,keepdims=True))/(x.std(1,keepdims=True)+1e-8)
def metric(labels,mask):
 sel=mask.sum();ret=labels&mask;avail=labels.any(1)
 return {'mean_chunks_kept':float(mask.sum(1).mean()),'empty_rate':float((mask.sum(1)==0).mean()),'precision':float(ret.sum()/sel) if sel else 0.,'conditional_support_recall':float(ret[avail].sum()/labels[avail].sum()) if labels[avail].sum() else 0.,'query_any_support':float((ret[avail].sum(1)>0).mean()) if avail.any() else 0.}
def scores(train,parts,layer,c):
 x=train['features'][:,:,layer,:].astype('float32').reshape(-1,256); y=train['labels'].ravel(); m=LogisticRegression(C=c,max_iter=1000,class_weight='balanced').fit(x,y);out=[]
 for p in parts:
  hidden=m.predict_proba(p['features'][:,:,layer,:].astype('float32').reshape(-1,256))[:,1].reshape(p['labels'].shape); cos=qz(p['cosine_scores'].astype(float)); lm=qz(p['lm_relevance_scores'].astype(float)); hid=qz(hidden);out.append({'cosine':cos,'cosine_plus_lm':cos+.2*lm,'cosine_plus_hidden':cos+.1*hid,'cosine_plus_lm_plus_hidden':cos+.2*lm+.1*hid})
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--alphas',default='0.1,0.2');a=p.parse_args();result={}
 for ds in ('hotpotqa','mmqa','tatqa','webqa'):
  base=a.root/ds;tr=np.load(base/'probe_train/features.npz');cal=np.load(base/'calibration/features.npz');te=np.load(base/'test/features.npz');summary=json.load(open(a.root.parent/f'{ds}_fusion_ablation_100_aligned.json'));layers=tr['layer_ids'].astype(int).tolist();layer=layers.index(summary['probe']['selected_layer']);c=summary['probe']['selected_c'];_,sc,st=scores(tr,[tr,cal,te],layer,c); labels_cal=cal['labels'].astype(bool);labels=te['labels'].astype(bool); dsout={}
  for alpha in map(float,a.alphas.split(',')):
   rows={}
   for name in st:
    bank=np.sort(sc[name][~labels_cal]);mask=np.zeros_like(labels,bool)
    for i,row in enumerate(st[name]):
     pv=(1+len(bank)-np.searchsorted(bank,row,side='left'))/(len(bank)+1);mask[i,list(benjamini_yekutieli(pv.tolist(),alpha).rejected_indices)]=True
    rows[name]=metric(labels,mask)
   dsout[str(alpha)]=rows
  result[ds]={'selected_layer':summary['probe']['selected_layer'],'selected_c':c,'test_queries':len(labels),'results':dsout}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps({'status':'complete','protocol':'disjoint cached probe-train / calibration false-bank / test; pooled false bank; Top-30 BY','results':result},indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
