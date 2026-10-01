#!/usr/bin/env python3
"""Run the three post-retrieval cosine selectors without encoding or GPU."""
import argparse
from pathlib import Path
import gzip
import json
from common import METHODS, load_dataset, read_json, thresholds_for, masks_for, support_metrics, write_json

HERE = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True,help='Extracted portable_bundle folder')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=HERE/'config.json')
    parser.add_argument('--recalibrate',action='store_true',help='Fit new thresholds from the bundled calibration rows')
    args = parser.parse_args()
    config = read_json(args.config); alpha = float(config['alpha'])
    if not 0<alpha<1: raise ValueError('alpha must be in (0,1)')
    if not args.recalibrate and alpha != .1:
        raise ValueError('Frozen thresholds are alpha=.10; use --recalibrate for other alpha')
    args.output.mkdir(parents=True,exist_ok=True)
    summary = {'protocol':'post_retrieval_shared_cosine','alpha':alpha,'threshold_mode':'recalibrated' if args.recalibrate else 'frozen','datasets':{}}
    frozen = read_json(HERE/'frozen_thresholds.json')['datasets']
    for dataset in config['datasets']:
        cal_ids,cal,test_ids,test = load_dataset(args.bundle,dataset)
        thresholds = thresholds_for(cal,alpha) if args.recalibrate else frozen[dataset]
        masks = masks_for(test,thresholds)
        output = args.output/f'{dataset}_selection.jsonl.gz'
        with gzip.open(output,'wt',encoding='utf-8') as handle:
            for i,(qid,rows) in enumerate(zip(test_ids,test,strict=True)):
                record = {'dataset':dataset,'qid':qid,'methods':{m:[str(r['chunk_id']) for r,keep in zip(rows,masks[m][i],strict=True) if keep] for m in METHODS}}
                handle.write(json.dumps(record,separators=(',',':'))+'\n')
        summary['datasets'][dataset] = {'calibration_queries':len(cal_ids),'test_queries':len(test_ids),'thresholds':thresholds,'selection':support_metrics(test,masks),'selection_file':output.name}
        print(dataset,len(cal_ids),len(test_ids),'selection complete',flush=True)
    write_json(args.output/'selection_summary.json',summary)

if __name__=='__main__': main()
