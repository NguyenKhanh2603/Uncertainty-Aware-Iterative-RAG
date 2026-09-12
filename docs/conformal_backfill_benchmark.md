# Conformal backfill benchmark

The implementation now separates three evidence-selection policies:

1. `fixed_top_k`: the original first K retrieved chunks.
2. `conformal_no_backfill`: only BY-accepted chunks inside the original Top-K.
3. `conformal_backfill`: keep those accepted Top-K chunks, then fill unused slots with
   BY-accepted candidates from ranks K+1 through L. An unaccepted chunk is never used merely
   to fill the context.

The benchmark reports evidence precision, conditional reserve support recall, support F1,
false-discovery diagnostics, chunks added by backfill, and queries where backfill restores at
least one support chunk. These are evidence-selection metrics, not downstream answer EM/F1.

## Run in Colab

Open `conformal_backfill_2_3_gpu_colab.ipynb`, set:

```python
MAX_QUESTIONS_PER_DATASET = 1000
```

and run all cells. The downloaded ZIP includes:

- `selection/backfill_benchmark/backfill_benchmark_summary.csv`
- `selection/backfill_benchmark/backfill_benchmark_summary.json`
- `selection/backfill_benchmark/backfill_benchmark_queries.jsonl.gz`

The 1000-question limit is per dataset, so four enabled datasets can produce up to 4000 total
queries. Retrieval/embedding uses the GPU; replaying p-values, alpha values, and the backfill
benchmark is CPU-only.

## Run from existing selection results

```bash
uv run python scripts/benchmark_conformal_backfill.py \
  --input selection/selection_decisions.jsonl.gz \
  --output-dir selection/backfill_benchmark \
  --alphas 0.01,0.025,0.05,0.10,0.20,0.30,0.50 \
  --max-context 10
```

The current BY-plus-at-most-K construction is experimental. The formal capped-risk proof and
the downstream generation comparison against attention pruning remain separate work items.

## Candidate search scope

Official bundles include a ragged candidate pool for each query. The retrieval runner keeps
that behavior by default. To retrieve a true fixed-size reserve from the full deduplicated
corpus stored in the bundle, pass:

```bash
uv run python scripts/generate_conformal_retrieval_log.py \
  --bundle-dir /path/to/bundle \
  --dataset hotpotqa \
  --output hotpotqa_top30_global_retrieval.jsonl.gz \
  --candidate-scope global_corpus \
  --retrieval-mode modality_aware \
  --top-l 30 \
  --min-per-modality 10
```

Candidate scope is part of the retrieval fingerprint. Rebuild both calibration banks and
evaluation retrieval logs after changing it; banks made from official per-query pools cannot
be reused for global-corpus retrieval.

To create separate, equally sized calibration and labelled validation query sets in one
bundle, prepare both source roles together and assign every training-source query to
calibration:

```bash
uv run python scripts/prepare_official_conformal_bundle.py \
  --output-dir /path/to/bundle \
  --datasets mmqa,webqa,hotpotqa,tatqa \
  --max-questions-per-role 1000

uv run python scripts/generate_conformal_retrieval_log.py \
  --bundle-dir /path/to/bundle \
  --dataset mmqa \
  --output mmqa_top30_global_retrieval.jsonl.gz \
  --candidate-scope global_corpus \
  --retrieval-mode modality_aware \
  --top-l 30 \
  --min-per-modality 10 \
  --split-policy official_holdout \
  --development-fraction 0
```

This produces 1000 `calibration` queries from the official training source and up to 1000
`test` queries from the labelled official dev/validation source. Both roles share the same
corpus fingerprint, so the calibration bank can be applied to the test rows without pipeline
drift or test leakage.

## Using a signal other than cosine similarity

[Probing-RAG](https://aclanthology.org/2025.findings-naacl.181/) does not score the relevance
of individual retrieved chunks. It first asks the generator to produce a rationale and answer,
mean-pools normalized hidden states for those generated tokens at several intermediate layers,
and uses binary feed-forward probes to decide whether another retrieval call is necessary. Its
released implementation keeps BM25 or Contriever as the retriever and uses the probes as an
iteration-level retrieval gate; see the
[official code](https://github.com/baekingeol/Probing-RAG).

Consequently, the released Probing-RAG logit must not replace `cosine_score` directly in the
current false-match bank. The two values represent different hypotheses:

- cosine or a reranker score asks whether a particular `(query, chunk)` pair is relevant;
- the Probing-RAG logit asks whether the generator's current answer needs another retrieval
  iteration.

There are two compatible integrations:

1. Keep the already-produced Top-L reserve, then use a Probing-RAG-style gate to
   stop or request another retrieval batch. This can reduce unnecessary retrieval calls but does
   not repair support/false separation inside the existing Top-L.
2. Train a new candidate-level hidden-state probe. For every Top-L pair, run a frozen open model
   on a fixed `(question, candidate)` prompt, extract a frozen layer/pooling representation, and
   train the probe target on `support_label`. Its support logit can then be the scalar score used
   to construct the conformal false-match bank. A text cross-encoder reranker is a cheaper first
   baseline for TAT-QA and HotpotQA; MMQA and WebQA need a multimodal model or an explicitly
   caption-only ablation.

The candidate probe must be trained before conformal calibration. Use three disjoint roles:
probe training from official train data, false-score calibration from a separate official train
subset, and final evaluation from the labelled official dev/validation split. Freeze the model
revision, prompt, layer set, pooling, probe checkpoint, precision, Top-L construction, and corpus
before calibration, and include them in the pipeline fingerprint. Training the probe and building
its conformal bank from the same 1000 calibration queries would make the reported p-values
optimistic.

The first diagnostic should compare candidate-level AUROC/AUPRC and the support-score/false-score
distributions for cosine-only, probe-only, and a frozen fusion. Only carry a signal into BY after
it shows stronger separation on development data. A better score can improve p-values, but it
does not remove BY's multiplicity penalty: increasing L still makes the first BY cutoff stricter.

### Frozen Top-30 TAT-QA diagnostic

A targeted experiment rescored the same 60,000 candidate pairs with the pinned
`BAAI/bge-reranker-v2-m3` cross-encoder. No new retrieval or larger candidate pool was used for
this comparison. The calibration role contains 1,000 official-train questions; the evaluation
role named `test` in the artifact contains 1,000 labelled official-dev questions.

| Score | Evaluation AUROC | Evaluation AUPRC | BY alpha | Precision | Conditional recall | Empty queries |
|---|---:|---:|---:|---:|---:|---:|
| Jina cosine | 0.762 | 0.186 | 0.10 | 0.522 | 0.069 | 0.933 |
| BGE reranker, modality bank | 0.906 | 0.378 | 0.10 | 0.491 | 0.097 | 0.912 |
| BGE reranker, pooled bank | 0.906 | 0.378 | 0.10 | 0.508 | 0.116 | 0.894 |

The candidate-level signal improves separation and roughly doubles support F1 for the pooled
bank (`0.113` to `0.189`), while most queries still remain empty at `alpha=0.10`. At
`alpha=0.20`, the pooled reranker reaches precision `0.437`, conditional recall `0.163`, and an
empty-query rate of `0.850`. These are model-selection diagnostics on labelled validation data,
not blind official-test results.

### Further targeted ablations

Increasing the reserve beyond the reranked Top-10 has limited headroom on this TAT-QA slice.
Top-10 already contains 793 of the 864 support rows found in Top-30 and reaches 738 of the 768
queries that have any support in Top-30. The large recall loss therefore occurs in statistical
selection after retrieval, rather than because `L=30` is too small.

A lightweight supervised fusion is the strongest tested improvement. It combines the frozen BGE
logit, cosine score, BGE rank, original cosine rank, modality, and modality interactions. For each
run, 500 official-train queries train logistic regression and a disjoint 500 official-train queries
construct the false-match bank. Across five deterministic 50/50 splits, evaluation on the same
1,000 labelled official-dev queries gives:

| Method at BY `alpha=0.10` | Precision | Conditional recall | Empty queries | AUROC | AUPRC |
|---|---:|---:|---:|---:|---:|
| BGE, modality bank using 500 bank queries | 0.473 | 0.100 | 0.910 | 0.906 | 0.378 |
| Fusion, mean over five splits | 0.702 | 0.145 | 0.866 | 0.929 | 0.455 |
| Fusion, observed split range | 0.676–0.752 | 0.135–0.150 | 0.862–0.874 | 0.928–0.930 | 0.442–0.469 |

This is still a development result. The split prevents training the fusion and constructing its
bank from the same queries, but a paper-ready claim also needs the clustered candidate dependence
within each query handled explicitly.

Several simple changes did not work:

- Conditioning the bank on coarse rank bins reduced precision to `0.203`, recall to `0.043`, and
  increased the empty-query rate to `0.959`.
- Testing only the reranker's Top-M candidates with a bank built from all Top-30 candidates caused
  post-selection bias. It looked more powerful but lost calibration, so it must not be reported as
  a valid conformal result.
- Replacing BY with ordinary BH on all 30 candidates raised recall to `0.237` and reduced empty
  queries to `0.769`, but micro precision fell to `0.316`. The
  [cfBH result](https://www.jmlr.org/papers/v24/22-1176.html) proves BH FDR control for particular
  exchangeable conformal constructions; the Top-L candidates here are clustered and dependent
  within query, so that theorem cannot be claimed without a new argument or cluster-level test.
- Conformal e-values with e-BH retained arbitrary-dependence safety in the diagnostic, but reached
  only `0.058` recall with `0.950` empty queries. It did not improve power here.

If support recall is the primary requirement, the statistical target should change from rejecting
false-match hypotheses to covering at least one support per retrievable query. A query-level
coverage calibration using the maximum support score reached `0.802` empirical query coverage at
an `0.80` target, selected `3.12` chunks per query, and reduced empty queries to `0.173`; its micro
precision was `0.204`. This is the expected precision/coverage tradeoff, and follows the retrieval
set objective used by [CONFLARE](https://arxiv.org/abs/2404.04287) and conformal retrieval work.
It should be evaluated by downstream answer EM/F1, since evidence precision alone does not tell
whether the generator can ignore distractors.

The next score-only comparison should use `Qwen/Qwen3-Reranker-0.6B` on the same frozen Top-30.
It is the same nominal size as BGE v2-m3 and the
[official Qwen evaluation](https://github.com/QwenLM/Qwen3-Embedding) reports stronger English and
multilingual reranking results. A task-specific instruction can also be frozen into the score
fingerprint. This comparison requires no new corpus retrieval.

For a usable non-empty RAG path, keep certified and fallback evidence as separate outputs. In one
fusion split, the certified set had precision `0.676`, conditional recall `0.147`, and an empty
rate of `0.863`. Adding the fusion-ranked Top-1 only when that set was empty produced precision
`0.537`, conditional recall `0.653`, zero empty queries, and `1.05` chunks per query. The conformal
claim applies only to the certified subset; fallback chunks must remain marked `unverified` and
must be evaluated through downstream answer EM/F1 and citation correctness. A Top-3 fallback
raised conditional recall to `0.802` but reduced evidence precision to `0.250`.
