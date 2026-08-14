# FLARE on MultiHop-RAG

This standalone harness benchmarks Direct FLARE against the complete shared
MultiHop-RAG news corpus. It does not import or modify `src/uncertainty_rag`.

For a ZIP-based Google Colab setup performed entirely in the terminal, see
[`../../COLAB_TERMINAL_SETUP.md`](../../COLAB_TERMINAL_SETUP.md).

## What is installed

- Official FLARE source: `related_repos/FLARE`
- Official MultiHop-RAG source: `related_repos/MultiHop-RAG`
- Full dataset and 609-document corpus: `data/multihop_rag`
- Deterministic 256-word BM25 chunks: `artifacts/corpus_chunks.jsonl`

The archived official FLARE code targets `text-davinci-003` and Elasticsearch.
`run_flare.py` is an OpenAI-compatible adaptation for the MultiHop-RAG corpus.
It implements mandatory initial retrieval, sentence look-ahead, token-confidence
triggering, low-confidence masking, retrieval, and sentence regeneration.

## Environment

The prepared environment is `.venv` inside this directory. In PowerShell:

```powershell
$py = ".\benchmarks\flare_multihoprag\.venv\Scripts\python.exe"
& $py benchmarks\flare_multihoprag\validate_setup.py
& $py -m unittest discover -s benchmarks\flare_multihoprag -p test_benchmark.py
```

Evaluate the local retriever over all 2,255 answerable queries before making
any model calls:

```powershell
& $py benchmarks\flare_multihoprag\evaluate_retriever.py
```

This writes `artifacts/bm25_metrics.json` with evidence Recall@k,
all-evidence success, and MRR. It searches all 4,976 chunks for every query.

## Run a smoke benchmark

For an OpenAI-compatible vLLM endpoint:

```powershell
$env:OPENAI_BASE_URL = "http://SERVER:8000/v1"
$env:OPENAI_API_KEY = "EMPTY"
$py = ".\benchmarks\flare_multihoprag\.venv\Scripts\python.exe"
& $py benchmarks\flare_multihoprag\run_flare.py `
  --model "mistralai/Mistral-7B-Instruct-v0.3" `
  --theta 0.8 `
  --beta 0.4 `
  --top-k 2 `
  --look-ahead-tokens 64 `
  --max-generation-tokens 256 `
  --min-reasoning-steps 2 `
  --concurrency 5 `
  --max-examples 10 `
  --output benchmarks\flare_multihoprag\results\smoke_flare_v4.jsonl
```

The server must support Chat Completions output log-probabilities. Start with
10 examples because FLARE can make several model calls per question.

For the OpenAI API, omit `OPENAI_BASE_URL`, set `OPENAI_API_KEY`, and provide a
model that supports chat output log-probabilities.

Resume safely with the same output path and `--resume`.
Do not resume into output produced by a superseded adapter; v4 rejects mixed
implementation versions.
With concurrency, one failed question does not cancel its siblings. Successful
rows are flushed to the output, failures go to `<output stem>.errors.jsonl`, and
`--resume` retries only questions absent from the main output.

## Evaluate

```powershell
& $py benchmarks\flare_multihoprag\evaluate.py `
  --input benchmarks\flare_multihoprag\results\smoke_flare_v4.jsonl `
  --output benchmarks\flare_multihoprag\results\smoke_flare_v4.metrics.json
```

Reported metrics include answer EM/F1, initial and cumulative gold-evidence
recall, evidence precision, all-evidence success, total/active retrieval calls,
active sentence trigger rate, completion rate, and draft confidence. Valid new
outputs report only `direct_flare_v4` under `implementation_versions`. v4 also
records forced-finalization usage, reasoning/termination reasons, clean-output
rate, token-alignment health, and answer-extraction health.

## Paper defaults

The runner defaults to the Direct FLARE thresholds used for 2WikiMultiHopQA:

- retrieval trigger `theta = 0.8`
- query masking `beta = 0.4`
- look-ahead length `64` tokens, truncated to its first sentence
- maximum complete generation length `256` tokens
- `top_k = 2` for paper parity with 2WikiMultiHopQA
- two intermediate reasoning sentences before allowing the final answer; this
  is the explicit chat-model substitute for the paper's task-specific CoT
  exemplars, not a threshold from the original FLARE configuration

Also report a separate `top_k = 4` ablation for MultiHop-RAG, whose questions
may name up to four evidence documents. Tune thresholds on a held-out subset
before reporting final results. Exclude
the 301 `null_query` examples during initial retrieval experiments; pass
`--include-null` for a later abstention evaluation.

For a controlled retrieval-policy comparison using the same runner, model,
prompt, and seed:

- `--theta 0`: initial-retrieval-only RAG
- `--theta 0.8`: paper-setting active FLARE
- `--theta 1`: retrieve on virtually every predicted sentence

Use a separate output file for every condition.
