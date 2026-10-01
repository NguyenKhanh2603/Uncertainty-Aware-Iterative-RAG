"""Portable loaders and metrics for frozen, already-retrieved candidate rows."""
from __future__ import annotations
import gzip
import json
import math
from pathlib import Path
import numpy as np
from cce import cce_embedding_threshold, select as cce_select
from conflare import conflare_threshold, select as conflare_select
from traq import traq_retrieval_threshold, select as traq_select

METHODS = {
    'cce_conformal_embedding_jina_alpha_0.10': (cce_embedding_threshold, cce_select),
    'conflare_source_question_jina_alpha_0.10': (conflare_threshold, conflare_select),
    'traq_retrieval_bonferroni_jina_alpha_0.10': (traq_retrieval_threshold, traq_select),
}

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def iter_rows(path):
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def plan(path):
    data = read_json(path)
    values = list(map(str, data['plan']))
    if not values or len(set(values)) != len(values) or len(values) != data.get('count', len(values)):
        raise ValueError(f'Invalid qid plan: {path}')
    return values

def load_dataset(bundle, dataset):
    bundle = Path(bundle)
    cal_ids = plan(bundle/'splits'/dataset/'calibration_manifest.json')
    test_ids = plan(bundle/'splits'/dataset/'test_manifest.json')
    if set(cal_ids) & set(test_ids):
        raise ValueError(f'{dataset}: calibration/test overlap')
    rows_by_qid = {}
    fingerprint = None
    for row in iter_rows(bundle/'rowwise'/f'{dataset}_top30_retrieval.jsonl.gz'):
        if row['dataset'] != dataset:
            raise ValueError('Unexpected dataset in rowwise file')
        key = tuple(row[k] for k in ('retriever_id','corpus_revision','preprocess_hash','top_l','query_type_rule_id'))
        if fingerprint is None:
            fingerprint = key
        if key != fingerprint:
            raise ValueError(f'{dataset}: mixed pipeline fingerprints')
        if not math.isfinite(float(row['cosine_score'])):
            raise ValueError('Non-finite cosine score')
        rows_by_qid.setdefault(str(row['qid']), []).append(row)
    result = []
    for ids,role in ((cal_ids,'calibration'),(test_ids,'test')):
        queries = []
        for qid in ids:
            rows = rows_by_qid[qid]
            if any(r['split_role'] != role for r in rows):
                raise ValueError(f'{dataset}/{qid}: wrong split role')
            if [int(r['rank']) for r in rows] != list(range(1,len(rows)+1)) or len(rows)>30:
                raise ValueError(f'{dataset}/{qid}: invalid ranks')
            if len({r['chunk_id'] for r in rows}) != len(rows):
                raise ValueError('Duplicate candidate IDs')
            queries.append(rows)
        result.append(queries)
    return cal_ids, result[0], test_ids, result[1]

def thresholds_for(calibration, alpha):
    positive = [np.asarray([float(r['cosine_score']) for r in q if r['support_label']=='support']) for q in calibration]
    return {m: fit(positive, alpha)[1] for m,(fit,_) in METHODS.items()}

def masks_for(test, thresholds):
    return {m: [select([r['cosine_score'] for r in q], thresholds[m]['similarity_threshold']) for q in test] for m,(_,select) in METHODS.items()}

def support_metrics(test, masks):
    out = {}
    for method,query_masks in masks.items():
        kept = true = total_true = empty = retrievable = any_support = all_support = 0
        for rows,mask in zip(test,query_masks,strict=True):
            labels = np.array([r['support_label']=='support' for r in rows],dtype=bool)
            retained = labels & mask
            kept += int(mask.sum()); true += int(retained.sum()); total_true += int(labels.sum())
            empty += int(not mask.any())
            if labels.any():
                retrievable += 1
                any_support += int(retained.any())
                all_support += int(retained.sum()==labels.sum())
        out[method] = dict(mean_chunks=kept/len(test),precision=true/kept if kept else 0.,support_recall=true/total_true if total_true else 0.,empty_rate=empty/len(test),retrievable_queries=retrievable,retrieval_query_coverage_ceiling=retrievable/len(test),query_any_support_conditional=any_support/retrievable if retrievable else 0.,query_all_support_conditional=all_support/retrievable if retrievable else 0.,selected_chunks=kept,selected_supports=true,reserve_supports=total_true)
    return out

def write_json(path,data):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    temporary.replace(path)
