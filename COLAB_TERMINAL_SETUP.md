# Colab terminal setup: FLARE on MultiHop-RAG

This guide uses the Colab **terminal**, not notebook cells. It assumes you
upload `flare_multihoprag_colab.zip` to `/content` through Colab's Files panel.
No Git clone is needed in Colab.

The bundle contains the complete MultiHop-RAG corpus and questions, the
prepared BM25 chunks, the standalone FLARE runner, the unchanged project
source, and snapshots of the official FLARE and MultiHop-RAG repositories.

## 1. Upload and extract the ZIP

Open **Terminal** from the Colab left sidebar and run:

```bash
mkdir -p /content/flare_multihoprag
unzip -q /content/flare_multihoprag_colab.zip -d /content/flare_multihoprag
cd /content/flare_multihoprag
```

Confirm that the important files arrived:

```bash
test -f /content/flare_multihoprag/data/multihop_rag/MultiHopRAG.json
test -f /content/flare_multihoprag/data/multihop_rag/corpus.json
test -f /content/flare_multihoprag/benchmarks/flare_multihoprag/run_flare.py
echo "Bundle layout OK"
```

## 2. Install uv and create `/content/.venv`

Install `uv` for the current Colab session:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

Create the environment with an explicit base interpreter. Colab images normally
provide `/usr/bin/python3`; `command -v python3` prints the path if that changes.

```bash
uv venv /content/.venv --python "$(command -v python3)"
```

Install the benchmark dependencies. The explicit `--python` is intentional: it
does not depend on the current directory, environment activation, or uv's
project discovery.

```bash
uv pip install \
  --python /content/.venv/bin/python \
  -r /content/flare_multihoprag/benchmarks/flare_multihoprag/requirements.txt
```

Install vLLM into the same explicit environment. `--torch-backend=auto` asks
uv to select a wheel compatible with the Colab CUDA/PyTorch platform.

```bash
uv pip install \
  --python /content/.venv/bin/python \
  vllm \
  --torch-backend=auto
```

You do not need to activate the environment. Use
`/content/.venv/bin/python` explicitly in every command below.

## 3. Validate the uploaded data and runner

```bash
cd /content/flare_multihoprag

/content/.venv/bin/python \
  /content/flare_multihoprag/benchmarks/flare_multihoprag/validate_setup.py

/content/.venv/bin/python -m unittest discover \
  -s /content/flare_multihoprag/benchmarks/flare_multihoprag \
  -p 'test_benchmark.py'
```

Expected dataset summary:

- 2,556 questions
- 609 full corpus documents
- 4,976 prepared BM25 chunks
- 2,255 answerable questions and 301 null queries

## 4. Optional retrieval-only sanity benchmark

This makes no LLM/API calls. It evaluates BM25 against gold evidence over the
whole corpus and may take a few minutes on a Colab CPU.

```bash
/content/.venv/bin/python \
  /content/flare_multihoprag/benchmarks/flare_multihoprag/evaluate_retriever.py \
  --output /content/flare_multihoprag/benchmarks/flare_multihoprag/artifacts/colab_bm25_metrics.json
```

## 5. Start Mistral 7B in native BF16 on a Colab L4

In Colab, select **Runtime > Change runtime type > L4 GPU** before starting.
Confirm that the terminal sees the 24 GB L4:

```bash
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
```

Use the official instruction-tuned checkpoint in native BF16. There is no
quantization in this benchmark configuration.

```bash
export FLARE_MODEL="mistralai/Mistral-7B-Instruct-v0.3"
export OPENAI_BASE_URL="http://127.0.0.1:8000/v1"
export OPENAI_API_KEY="EMPTY"
export HF_HOME="/content/huggingface"
```

Start the vLLM server in the background:

```bash
nohup /content/.venv/bin/vllm serve "$FLARE_MODEL" \
  --host 127.0.0.1 \
  --port 8000 \
  --api-key "$OPENAI_API_KEY" \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --max-num-seqs 5 \
  --gpu-memory-utilization 0.90 \
  > /content/vllm-mistral.log 2>&1 &

echo $! > /content/vllm-mistral.pid
```

The first launch downloads roughly 15 GB of model weights and can take several
minutes. Watch startup with:

```bash
tail -f /content/vllm-mistral.log
```

Press `Ctrl-C` after the log says the server is running. This stops `tail`, not
the background vLLM server.

Confirm the server and served model:

```bash
curl -sS http://127.0.0.1:8000/v1/models | /content/.venv/bin/python -m json.tool
```

Finally, verify the feature FLARE specifically depends on: output token
log-probabilities.

```bash
/content/.venv/bin/python \
  /content/flare_multihoprag/benchmarks/flare_multihoprag/verify_endpoint.py \
  --base-url "$OPENAI_BASE_URL" \
  --api-key "$OPENAI_API_KEY" \
  --model "$FLARE_MODEL"
```

Do not start the FLARE run unless this prints both `Endpoint OK` and
`Token log-probabilities OK`.

The 8,192-token cap is a serving-memory guardrail, not quantization. The FLARE
prompt uses two 256-word chunks by default and fits comfortably inside this context. It
leaves useful L4 memory for the KV cache and vLLM runtime while keeping the
official Mistral weights in BF16.

## 6. Run a 10-question FLARE smoke test

```bash
mkdir -p /content/flare_multihoprag/benchmarks/flare_multihoprag/results

/content/.venv/bin/python \
  /content/flare_multihoprag/benchmarks/flare_multihoprag/run_flare.py \
  --model "$FLARE_MODEL" \
  --theta 0.8 \
  --beta 0.4 \
  --top-k 2 \
  --look-ahead-tokens 64 \
  --max-generation-tokens 256 \
  --min-reasoning-steps 2 \
  --concurrency 5 \
  --max-examples 10 \
  --output /content/flare_multihoprag/benchmarks/flare_multihoprag/results/smoke_flare_v4.jsonl
```

If the job is interrupted, run the same command with `--resume` appended.
Do not reuse `smoke.jsonl`, `smoke_flare_v2.jsonl`, or any v3 output: they were
produced by superseded adapters and are rejected by the v4 resume check.

Completed questions are flushed individually. If one concurrent worker fails,
the other workers finish; failures are recorded in `smoke_flare_v4.errors.jsonl`.
After updating the runner, repeat the same command with `--resume` to run only
questions missing from the main result JSONL.

## 7. Evaluate generated answers and active retrieval

```bash
/content/.venv/bin/python \
  /content/flare_multihoprag/benchmarks/flare_multihoprag/evaluate.py \
  --input /content/flare_multihoprag/benchmarks/flare_multihoprag/results/smoke_flare_v4.jsonl \
  --output /content/flare_multihoprag/benchmarks/flare_multihoprag/results/smoke_flare_v4.metrics.json

cat /content/flare_multihoprag/benchmarks/flare_multihoprag/results/smoke_flare_v4.metrics.json
```

The report includes answer exact match/F1, initial and cumulative evidence
recall, evidence precision, all-evidence success, total and active retrieval
calls, and active sentence retrieval-trigger rate. A working active run should
normally have `avg_active_retrieval_calls > 0` and `active_step_trigger_rate > 0`.
Also check that `implementation_versions` contains only `direct_flare_v4` and
that `final_answer_rate` is close to 1 before scaling up. The report records
`forced_finalization_rate`, `finalization_active_retrieval_rate`,
`reasoning_stop_reasons`, and `termination_reasons`.

For tokenizer/output health, require all three to equal `1.0`:

- `clean_output_rate`
- `all_token_alignments_exact_rate`
- `answer_extraction_pattern_match_rate`

## 8. Run the full answerable set

First confirm the smoke results and API cost. Then omit `--max-examples`:

```bash
/content/.venv/bin/python \
  /content/flare_multihoprag/benchmarks/flare_multihoprag/run_flare.py \
  --model "$FLARE_MODEL" \
  --theta 0.8 \
  --beta 0.4 \
  --top-k 2 \
  --look-ahead-tokens 64 \
  --max-generation-tokens 256 \
  --min-reasoning-steps 2 \
  --concurrency 5 \
  --output /content/flare_multihoprag/benchmarks/flare_multihoprag/results/flare_v4_full.jsonl \
  --resume
```

These are the Direct FLARE settings from the paper's 2WikiMultiHopQA
configuration. For a MultiHop-RAG-specific retrieval-depth ablation, repeat the
experiment with `--top-k 4` and a different output filename.

By default the runner excludes the 301 `null_query` items. Add `--include-null`
only for a separate abstention experiment.

## Paper-style retrieval-policy controls

Keep the model, prompt, seed, corpus, and `top-k` fixed. Change only `theta`:

```bash
# Single-time RAG control: initial question retrieval only
/content/.venv/bin/python \
  /content/flare_multihoprag/benchmarks/flare_multihoprag/run_flare.py \
  --model "$FLARE_MODEL" --theta 0 --beta 0.4 --top-k 2 \
  --look-ahead-tokens 64 --max-generation-tokens 256 \
  --min-reasoning-steps 2 \
  --concurrency 5 \
  --output /content/flare_multihoprag/benchmarks/flare_multihoprag/results/rag_single_v4.jsonl

# Always-active forward-looking retrieval ablation
/content/.venv/bin/python \
  /content/flare_multihoprag/benchmarks/flare_multihoprag/run_flare.py \
  --model "$FLARE_MODEL" --theta 1 --beta 0.4 --top-k 2 \
  --look-ahead-tokens 64 --max-generation-tokens 256 \
  --min-reasoning-steps 2 \
  --concurrency 5 \
  --output /content/flare_multihoprag/benchmarks/flare_multihoprag/results/flare_always_v4.jsonl
```

`theta=0` disables active retrieval; `theta=1` activates it for virtually every
predicted sentence; `theta=0.8` is the paper's active FLARE setting. Evaluate
each JSONL with `evaluate.py` and report answer quality together with retrieval
calls and trigger rate.

## Stop the local model server

```bash
kill "$(cat /content/vllm-mistral.pid)"
```

## Colab persistence warning

Both `/content/.venv` and benchmark results disappear when the Colab runtime is
deleted. Download the JSONL and metrics files from the Files panel before ending
the session, or copy them to a mounted Google Drive directory.
