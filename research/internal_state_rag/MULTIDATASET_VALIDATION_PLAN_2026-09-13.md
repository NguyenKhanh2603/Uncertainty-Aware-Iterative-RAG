# Multi-dataset validation plan

## Claim being tested

The TAT-QA result supports a narrow hypothesis: a frozen decoder LLM's
question--chunk LM-head and hidden state contain relevance information that is
complementary to a cross-encoder reranker, and query-level all-support conformal
calibration can turn the improved ranking into fewer retained chunks at similar
support recall.

Hidden-state probing, LLM reranking, and adaptive retrieval are established
research directions. The potentially publishable unit is therefore not
"hidden states contain relevance." It is the combined chunk-selection protocol,
its recall guarantee target, its efficiency frontier, and the distinction
between semantic relevance and context-conditional causal usage.

## Dataset 2: HotpotQA

- Official distractor/train: 1,000 query development bank.
- Official distractor/validation: 1,000 untouched test queries.
- Global corpus: 19,265 passages.
- Frozen dense Top-30 already exists; BGE reranker-v2-m3 will only rerank this
  reserve and will not change the retrieval ceiling.
- The train bank will be split once into 500 probe-training and 500 conformal
  calibration queries. Official validation remains test-only.
- Qwen2.5-3B-Instruct, prompt, layers, random projection, Top-30, and alpha sweep
  remain identical to TAT-QA.

Primary comparisons:

1. BGE reranker alone.
2. Yes-vs-No LM-head alone.
3. TAT-QA-trained hidden probe transferred without HotpotQA labels.
4. HotpotQA-trained hidden probe.
5. BGE + LM-head + hidden probe, with fusion weights selected by group OOF on
   the 500 probe-training queries.

The finding generalizes only if internal fusion improves held-out ranking and/or
retains fewer chunks at matched conformal recall on HotpotQA. Dataset-specific
weight tuning cannot by itself establish a universal probe; the zero-shot
TAT-QA-to-Hotpot transfer row addresses that stronger question.

## Dataset 3

MMQA is locally available, but its current retrieval log contains only 1,000
official-train queries divided into development/calibration roles and includes
image chunks. Qwen2.5-3B-Instruct cannot inspect those images. It may be used for
a clearly labelled text/table exploratory transfer test, but it will not be
mixed into the paper-grade HotpotQA result. A full multimodal claim requires a
frozen VLM and a separate official held-out MMQA bundle.
