#!/usr/bin/env python3
"""Compare isolated selectors and selected contexts with the completed experiment."""
import argparse
import ast
import json
from pathlib import Path
from typing import Any, Sequence
import numpy as np
from common import METHODS, load_dataset, masks_for, thresholds_for, read_json, iter_rows, write_json

HERE = Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-root',type=Path,required=True)
    p.add_argument('--selection-output',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    result_root=a.repo_root/'research/internal_state_rag/results'
    bundle=result_root/'precomputed_cosine_cal1000_official_test_2026_09_30/portable_bundle'
    original_path=a.repo_root/'research/internal_state_rag/run_literature_protocol_1000cal.py'
    original=original_path.read_text()
    tree=ast.parse(original)
    # Execute only the original calibration kernels, avoiding GPU or research imports.
    ns={'np':np,'Any':Any,'Sequence':Sequence}
    wanted={'support_scores_by_query','cce_embedding_threshold','conflare_threshold','traq_retrieval_threshold'}
    for node in tree.body:
        if isinstance(node,ast.FunctionDef) and node.name in wanted:
            exec(compile(ast.Module(body=[node],type_ignores=[]),str(original_path),'exec'),ns)
    names={'cce_conformal_embedding_jina_alpha_0.10':'cce_embedding_threshold','conflare_source_question_jina_alpha_0.10':'conflare_threshold','traq_retrieval_bonferroni_jina_alpha_0.10':'traq_retrieval_threshold'}
    completed=read_json(result_root/'precomputed_cosine_cal1000_official_test_2026_09_30/three_baselines/summary.json')['datasets']
    replay=read_json(a.selection_output/'selection_summary.json')['datasets']
    frozen=read_json(HERE/'frozen_thresholds.json')['datasets']
    audit={}
    for ds in completed:
        cal_ids,cal,test_ids,test=load_dataset(bundle,ds)
        current=thresholds_for(cal,.1)
        positive=ns['support_scores_by_query'](cal)
        for method,func in names.items():
            expected=ns[func](positive,.1)[1]
            assert current[method]==expected,(ds,method,'calibration-kernel mismatch')
            assert replay[ds]['selection'][method]==completed[ds]['selection'][method],(ds,method,'support-metric mismatch')
        decisions={r['qid']:r for r in iter_rows(a.selection_output/f'{ds}_selection.jsonl.gz')}
        predictions={r['qid']:r for r in iter_rows(result_root/'full_context_selection_eligible_full_2026_09_30'/f'{ds}_downstream_predictions.jsonl')}
        contexts=0
        for qid in test_ids:
            for method in METHODS:
                assert decisions[qid]['methods'][method]==predictions[qid]['contexts'][method]['chunk_ids'],(ds,qid,method,'context mismatch')
                contexts+=1
        audit[ds]={'calibration_queries':len(cal_ids),'test_queries':len(test_ids),'contexts_verified':contexts,'all_support_metrics_identical':True,'recalibrated_kernels_match_original':True,'selected_contexts_match_completed_downstream':True}
    report={'status':'passed','original_code_commit':'fa6dd68','queries_verified':sum(r['test_queries'] for r in audit.values()),'contexts_verified':sum(r['contexts_verified'] for r in audit.values()),'datasets':audit}
    write_json(a.output,report)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
