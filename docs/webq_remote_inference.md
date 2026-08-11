# Frozen WebQ remote inference

`data/webq_ragu` contains the frozen WebQuestions retrieval artifact used for
the Passage Utility-aligned experiment. The 400-item evaluation file is
`webq-test-400-seed10.jsonl`; its selected IDs and indices are recorded in the
adjacent `webq-test-400-seed10-ids.json` manifest.

The evaluation runner never performs retrieval. It reads the first five stored
Contriever-MSMARCO passages for each question, generates one greedy answer and
ten stochastic samples, and writes per-question uncertainty scores and AUROC.

## On the Linux GPU server

Install the project, then start an OpenAI-compatible vLLM server with output
log-probabilities enabled by the server's normal chat-completions API:

```bash
pip install -e ".[dev]"
pip install vllm
export VLLM_API_KEY='choose-a-secret'
vllm serve mistralai/Mistral-7B-Instruct-v0.3 --api-key "$VLLM_API_KEY"
```

Download the frozen artifact there if it has not been copied with the project:

```bash
hf download danny2507/ragu-webq-contriever-msmarco \
  README.md metadata.json webq-train.jsonl webq-dev.jsonl webq-test.jsonl \
  webq-test-400-seed10.jsonl webq-test-400-seed10-ids.json \
  --repo-type dataset --local-dir data/webq_ragu
```

In a second terminal on the server, run the UQ evaluation:

```bash
python eval/run_webq_remote.py \
  --base-url http://127.0.0.1:8000/v1 \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --api-key "$VLLM_API_KEY" \
  --output results/webq_remote/mistral7b_seed10.jsonl
```

The process appends each completed question. After an interruption, run the
same command with `--resume` to skip completed question IDs.

If the server does not expose `logprobs`, add `--no-token-logprobs`. This still
computes semantic entropy but omits token-uncertainty AUROC. For the full
method and a Passage Utility comparison, use a server that returns output
log-probabilities.

`correct_em` is normalized exact match. The Passage Utility paper used a
Qwen-based correctness judge, so use the same judge before making a numerical
claim of exact reproduction against its AUROC table.
