"""Extract frozen Qwen2.5 LM-head and final-state signals for NeuCLIR chunks."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
 p=argparse.ArgumentParser(); p.add_argument('--scores',type=Path,required=True); p.add_argument('--chunks',type=Path,required=True); p.add_argument('--topics',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--model',default='Qwen/Qwen2.5-7B-Instruct'); p.add_argument('--top-chunks',type=int,default=100); p.add_argument('--batch-size',type=int,default=4); p.add_argument('--max-length',type=int,default=768); a=p.parse_args()
 if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
 text={r['chunk_id']:r for r in map(json.loads,a.chunks.read_text().splitlines())}
 topics={x.split('\t',1)[0]:x.split('\t',1)[1] for x in a.topics.read_text().splitlines()}
 ranking=json.load(open(a.scores))['scores']; rows=[]
 for qid, ranked in ranking.items():
  for row in ranked[:a.top_chunks]: rows.append((qid,row['chunk_id'],row['docid'],float(row['score'])))
 tok=AutoTokenizer.from_pretrained(a.model, padding_side='right'); tok.pad_token=tok.eos_token
 model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True).to('cuda').eval()
 yes=tok.encode(' Yes',add_special_tokens=False)[0]; no=tok.encode(' No',add_special_tokens=False)[0]
 logits=[]; states=[]
 for begin in range(0,len(rows),a.batch_size):
  batch=rows[begin:begin+a.batch_size]; prompts=[f'Question: {topics[q]}\n\nPassage: {text[c]["text"]}\n\nIs this passage useful evidence for answering the question? Answer:' for q,c,_,_ in batch]
  enc=tok(prompts,padding=True,truncation=True,max_length=a.max_length,return_tensors='pt').to('cuda')
  with torch.inference_mode(): out=model(**enc,output_hidden_states=True,return_dict=True)
  pos=enc.attention_mask.sum(1)-1; idx=torch.arange(len(batch),device='cuda'); logits.extend((out.logits[idx,pos,yes]-out.logits[idx,pos,no]).float().cpu().tolist()); states.append(out.hidden_states[-1][idx,pos].float().cpu().numpy())
  print(f'{min(begin+a.batch_size,len(rows))}/{len(rows)}',flush=True)
 a.output.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(a.output,qids=np.array([x[0] for x in rows]),chunk_ids=np.array([x[1] for x in rows]),docids=np.array([x[2] for x in rows]),cosine=np.array([x[3] for x in rows],np.float32),lm_yes_no=np.array(logits,np.float32),hidden=np.concatenate(states).astype(np.float16),model=a.model)
if __name__=='__main__': main()
