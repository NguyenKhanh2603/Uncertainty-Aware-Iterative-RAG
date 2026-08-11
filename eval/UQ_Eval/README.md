# Standalone RAGU baseline reproduction

This directory is deliberately self-contained. It does not import
`related_repos/ragu` or `src/uncertainty_rag`, so removing either directory
will not affect it. The copied/derived RAGU components retain their BSD
3-Clause attribution in the source headers.

It reproduces the PPL, regular-entropy, and probability-weighted semantic
entropy baselines on the frozen WebQ input. It uses the RAGU WebQ layout
`[passage-number] title` followed by passage text, `chat_directRagQA_REAR3`,
top-5 contexts, ten samples at temperature 1, top-p .9, top-k 50, and max 50
tokens.

Run it on the Linux server after starting Mistral vLLM:

```bash
python eval/UQ_Eval/run_webq_paper_baselines.py \
  --base-url http://127.0.0.1:8000/v1 \
  --api-key EMPTY \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --output results/webq_paper_baselines/mistral7b_seed10.jsonl
```

Use `--max-examples 5` first to validate that the server returns output log
probabilities. The first run also downloads RAGU's entailment model
`microsoft/deberta-v2-xlarge-mnli`; it needs GPU memory. Resume an interrupted
run with the same arguments plus `--resume`.

The output summary contains the three AUROCs. It uses RAGU's `Acc` label
(normalized gold answer contained in the generated answer), not `AccLM`.
AccLM and Passage Utility still need Laura's supplied annotations/checkpoint.

After the baseline run finishes, score our method without regenerating any QA
answers:

```bash
python eval/UQ_Eval/score_ours_from_ragu_samples.py \
  --input results/webq_paper_baselines/mistral7b_seed10.jsonl \
  --output results/webq_paper_baselines/mistral7b_seed10_with_ours.jsonl \
  --base-url http://127.0.0.1:8000/v1 \
  --api-key EMPTY \
  --claim-model mistralai/Mistral-7B-Instruct-v0.3
```

The unified output reports AUROC for our claim-level semantic uncertainty and
token uncertainty alongside RAGU PPL, regular entropy, and semantic entropy.
The default `--claim-mode extract` is our method. `--claim-mode answer` is a
faster whole-answer ablation and should not be presented as the full method.

To add p(True), run:

```bash
python eval/UQ_Eval/run_p_true_webq.py \
  --input results/webq_paper_baselines/mistral7b_seed10_with_ours.jsonl \
  --output results/webq_paper_baselines/mistral7b_seed10_with_ptrue.jsonl \
  --base-url http://127.0.0.1:8000/v1 \
  --api-key EMPTY \
  --model mistralai/Mistral-7B-Instruct-v0.3
```

On its first invocation this creates and saves 20 WebQ training demonstrations
in `ptrue_train_shots_seed10.jsonl`; that costs 20 greedy plus 200 sampled QA
generations. It then makes 400 one-token completion calls to obtain raw
`log p(" A")`, exactly the score used by RAGU (`p_false_fixed = 1 - exp(log p(A))`).
Later interrupted runs can use `--resume`; the cached demonstrations are reused.
The OpenAI-compatible server must support the `/v1/completions` endpoint and
return top log-probabilities for the next token. The default requests 20, the
vLLM maximum on the configured server.

To test whether our semantic U is significantly better than RAGU semantic
entropy on the *same* 400 examples, use the paired two-sided DeLong test:

```bash
python eval/UQ_Eval/delong_auroc.py \
  --input results/webq_paper_baselines/mistral7b_seed10_with_ptrue.jsonl \
  --first ours_semantic_uncertainty \
  --second semantic_entropy
```

The output gives the AUROC difference, z statistic, and two-sided p-value.
Use the same `correct_acc` label for both scores. A positive difference with
`p < 0.05` supports the limited claim that our score is better on this fixed
WebQ seed-10/raw-Acc evaluation.

## Upload the artifacts

On the remote machine, authenticate with a Hugging Face write token and run:

```bash
hf auth login
uv run --with huggingface_hub python eval/UQ_Eval/upload_results_to_hf.py
```

The default destination is the dataset repository
`danny2507/ragu-webq-mistral7b-uq-results`. It uploads `results/` JSON/JSONL
artifacts and this standalone `UQ_Eval` code, but not model files, source data,
or credentials. Add `--private` if the repository should not be public yet.

The local `p_true.py` contains the paper's p(True) prompt and probability
normalization helper. It needs a separately generated 20-shot training prompt,
so it is not mixed into the WebQ PPL/semantic runner.
