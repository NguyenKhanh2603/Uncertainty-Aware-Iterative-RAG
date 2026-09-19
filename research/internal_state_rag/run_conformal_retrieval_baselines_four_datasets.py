"""Run CCE, CONFLARE, and TRAQ retrieval adapters on the cached four datasets.

All methods share frozen Jina-v4 Top-30 candidate scores and the project's
probe-train/calibration/test split.  CONFLARE's synthetic-LLM calibration is
replaced by calibration gold-support chunks; this is needed for a controlled
same-data selection benchmark and is reported as an adapter, not reproduction.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

def metric(y,m):
 avail=y.any(1);r=y&m;sel=m.sum()
 return {'mean_chunks_kept':float(m.sum(1).mean()),'context_reduction':float(1-m.mean()),'empty_rate':float((m.sum(1)==0).mean()),'precision':float(r.sum()/sel) if sel else 0.,'conditional_support_recall':float(r[avail].sum()/y[avail].sum()) if y[avail].sum() else 0.,'query_any_support':float((r[avail].sum(1)>0).mean()) if avail.any() else 0.,'query_all_support':float((r[avail].sum(1)==y[avail].sum(1)).mean()) if avail.any() else 0.}
def finite_tau(s,a):
 x=np.sort(s);return float(x[min(len(x)-1,int(np.floor((len(x)+1)*a)) )])
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--alphas',default='0.05,0.1,0.2');a=p.parse_args();out={}
 for ds in ('hotpotqa','mmqa','tatqa','webqa'):
  cal=np.load(a.root/ds/'calibration/features.npz');te=np.load(a.root/ds/'test/features.npz');score_cal=cal['cosine_scores'].astype(float);score=te['cosine_scores'].astype(float);positive=score_cal[cal['labels'].astype(bool)];rows={}
  for alpha in map(float,a.alphas.split(',')):
   cce=finite_tau(positive,alpha)
   # CONFLARE source: np.percentile(distances, 100 - error_rate*100).
   # distance = 1-cosine, so threshold equals the ordinary alpha score percentile.
   conflare=float(np.percentile(positive,alpha*100))
   # TRAQ allocates alpha/2 to retrieval and uses np.quantile(scores, alpha/2, lower).
   traq=float(np.quantile(positive,alpha/2,method='lower'))
   rows[str(alpha)]={'ecir_cce_embedding':{'threshold':cce,**metric(te['labels'].astype(bool),score>=cce)},'conflare_retrieval_adapter':{'threshold':conflare,**metric(te['labels'].astype(bool),score>=conflare)},'traq_retrieval_bonferroni_adapter':{'threshold':traq,**metric(te['labels'].astype(bool),score>=traq)}}
  out[ds]={'calibration_positive_chunks':int(len(positive)),'test_queries':int(len(te['labels'])),'results':rows}
 artifact={'status':'complete_retrieval_component_adapter_benchmark','shared_score':'cached Jina v4 cosine; Top-L=30','methods':{'ecir_cce_embedding':'finite-sample positive-chunk split conformal','conflare_retrieval_adapter':'official percentile distance threshold; gold calibration support replaces its synthetic LLM questions','traq_retrieval_bonferroni_adapter':'official TRAQ retrieval quantile at alpha/2; generation component cannot be compared without generated answer sets'},'results':out};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(artifact,indent=2)+'\n');print(json.dumps(artifact,indent=2))
if __name__=='__main__':main()
