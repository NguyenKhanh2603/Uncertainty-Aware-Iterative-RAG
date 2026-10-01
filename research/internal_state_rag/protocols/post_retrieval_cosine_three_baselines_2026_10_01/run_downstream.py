#!/usr/bin/env python3
"""Generate answers for the three saved selection lists using one shared protocol."""
import argparse
import hashlib
import json
from pathlib import Path
from common import METHODS, iter_rows, read_json, write_json
from answer_metrics import exact_match, token_f1, numerical_accuracy

HERE = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection-dir',type=Path,required=True)
    parser.add_argument('--data-bundle',type=Path,required=True,help='Local questions.jsonl, corpus.jsonl and actual image assets')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=HERE/'config.json')
    parser.add_argument('--model',help='Optional local Qwen model directory')
    args = parser.parse_args()
    config = read_json(args.config); generation = config['generation']
    model = args.model or generation['model']
    args.output.mkdir(parents=True,exist_ok=True)
    client = None; summaries = {}
    for ds in config['datasets']:
        selections = list(iter_rows(args.selection_dir/f'{ds}_selection.jsonl.gz'))
        questions = {str(r['qid']):r for r in iter_rows(args.data_bundle/ds/'questions.jsonl')}
        corpus = {str(r['id']):r for r in iter_rows(args.data_bundle/ds/'corpus.jsonl')}
        target = args.output/f'{ds}_predictions.jsonl'
        completed = {str(r['qid']):r for r in iter_rows(target)} if target.exists() else {}
        for index,row in enumerate(selections,start=1):
            qid = row['qid']; question = questions[qid]; gold = list(map(str,question['gold_answers']))
            identity = hashlib.sha256(json.dumps({'generation':generation,'model':model,'selection':row,'question':question['question'],'gold':gold},sort_keys=True).encode()).hexdigest()
            if qid in completed:
                if completed[qid].get('identity') != identity:
                    raise ValueError(f'Checkpoint mismatch for {ds}/{qid}; use a new output folder')
                continue
            from generator import QwenDirectAnswerGenerator, ContextChunk
            if client is None:
                client = QwenDirectAnswerGenerator(Path(model),min_pixels=generation['min_pixels'],max_pixels=generation['max_pixels'],revision=generation['revision'])
            prompt = 'Answer using only the supplied context. Return only the short answer.\nQuestion: '+str(question['question'])
            predictions = {}; cache = {}
            for method in METHODS:
                ids = tuple(row['methods'][method]); selected = []
                for chunk_id in ids:
                    source = corpus[chunk_id]; content = str(source['content']); modality = source.get('modality','text')
                    if modality=='image' and not content.startswith(('http://','https://')):
                        image = (args.data_bundle/content).resolve()
                        if not image.is_file() or not image.stat().st_size: raise FileNotFoundError(image)
                        content = str(image)
                    selected.append(ContextChunk(chunk_id,content,modality))
                if ids not in cache:
                    cache[ids] = client.generate(prompt,selected,max_new_tokens=generation['max_new_tokens'])
                predictions[method] = cache[ids]
            record = {'dataset':ds,'qid':qid,'identity':identity,'gold_answers':gold,'predictions':predictions,'contexts':row['methods'],'metrics':{m:{'em':exact_match(p,gold),'f1':token_f1(p,gold),'numerical_accuracy':numerical_accuracy(p,gold)} for m,p in predictions.items()}}
            with target.open('a',encoding='utf-8') as handle:
                handle.write(json.dumps(record,ensure_ascii=False)+'\n'); handle.flush()
            completed[qid] = record
            print(f'[{ds} {index}/{len(selections)}] {qid}',flush=True)
        ordered = [completed[r['qid']] for r in selections]
        summaries[ds] = {m:{k:sum(r['metrics'][m][k] for r in ordered)/len(ordered) for k in ('em','f1','numerical_accuracy')} for m in METHODS}
        write_json(args.output/'summary.json',{'status':'complete' if len(summaries)==len(config['datasets']) else 'partial','generation':generation,'datasets':summaries})

if __name__=='__main__': main()
