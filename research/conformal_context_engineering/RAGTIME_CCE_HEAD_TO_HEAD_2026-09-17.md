# Head-to-head: Conformal Context Engineering and query-level conformal support

**Date:** 2026-09-17

## Decision

The Conformal Context Engineering (CCE) paper is closely related to this project, but calibrates a different error event. It is a necessary pointwise-evidence baseline, not a replacement for query-level all-support conformal prediction.

| Method | Calibration unit | Threshold fit on | Guarantee target |
|---|---|---|---|
| CCE positive-only conformal | relevant `(query, snippet)` pair | every relevant calibration score | retain a randomly drawn relevant snippet |
| Query-level all-support | one query | its lowest-scoring retrieved support | retain every retrieved support for a query |
| Legacy cosine + BY | candidate chunk | calibration false-score bank | candidate false-discovery control |

CCE uses `A(q,s)=1-cos(q,s)`, retains when `A <= tau`, and fits `tau` from the `(1-alpha)` quantile of positive calibration snippets. Its target is `P(s in K_q | r(q,s)=1) >= 1-alpha`. It does not state that all evidence for a multi-fact/multi-hop query is jointly retained. Query-level support instead calibrates `min_{s in S_q} score(q,s)`, changing the target to `S_q subseteq K_q` conditional on retrieval.

Sources: supplied [paper Markdown](../../baselines/2511.17908v2%20%281%29.md) and the authors' [supplementary repository](https://github.com/hltcoe/conformal-context-engineering).

## Release audit and comparability

The CCE repository releases prompts and illustrative JSON only. It does not release executable experiment code, retrieval runs, topic assignments, Llama-3.3-70B relevance labels, Qwen3-Embedding-8B vectors, GPT-4o scores, or generated answers. The paper uses 500-character overlapping snippets, Llama-generated binary labels, Qwen3-Embedding-8B for Conformal-Embedding, GPT-4o for Conformal-LLM, and NeuCLIR ARGUE F1. Its reported `F1 / ConRed%` values therefore cannot be numerically compared to this repository's chunk-support recall without the missing artifacts.

| Published CCE method | alpha=.05 | alpha=.10 | alpha=.20 |
|---|---:|---:|---:|
| Conformal-Embedding: ARGUE F1 / context reduction | .720 / 22.2% | .700 / 35.0% | .680 / 52.8% |
| Conformal-LLM: ARGUE F1 / context reduction | .710 / 46.5% | .700 / 58.0% | .680 / 57.8% |

## Public RAGTIME benchmark prepared here

Downloaded locally, excluded from Git by the existing `data` rule:

- `trec-ragtime/ragtime1` English corpus: `eng-docs.jsonl`, 3,399,011,290 bytes.
- Official English topics: 122 request variants.
- Official 2025 qrels: 1,044 positive English `(topic, document)` judgments across 30 topic titles.

Requests sharing one title have different report limits. To avoid topic leakage, their variants stay in one deterministic title-group role: 17 calibration and 13 test topic titles.

### Necessary proxy qualification

This is a **document-unit target comparison**, not an exact CCE reproduction:

1. Official document qrels are evaluation labels; CCE uses Llama-3.3 500-character snippet labels.
2. Every method receives the same TF-IDF cosine score, not Qwen3-Embedding-8B or GPT-4o.
3. The candidate pool is Top-30 from all 1,040 qrel-positive English documents plus a deterministic reservoir of 25,000 English corpus negatives. This isolates filtering, not production retrieval.
4. It does not run generator/ARGUE evaluation. The common outputs are retained context, precision, marginal relevant-document recall, and all-support coverage.

## Head-to-head results

`R` is micro relevant-document recall conditional on a positive candidate in Top-30. `All` is query all-support coverage on those retrievable queries. `ConRed` is one minus the retained fraction of Top-30.

| alpha | Method | Mean kept | ConRed | Precision | R | All | Empty |
|---:|---|---:|---:|---:|---:|---:|---:|
| .05 | CCE positive-only | 20.54 | 31.54% | 73.78% | 92.92% | 58.33% | 0.0% |
|  | Query all-support | 26.23 | 12.56% | 61.88% | 99.53% | 91.67% | 0.0% |
|  | Candidate BY | 0.00 | 100.00% | -- | 0.00% | 0.00% | 100.0% |
| .10 | CCE positive-only | 17.62 | 41.28% | 80.35% | 86.79% | 41.67% | 7.69% |
|  | Query all-support | 26.23 | 12.56% | 61.88% | 99.53% | 91.67% | 0.0% |
|  | Candidate BY | 0.00 | 100.00% | -- | 0.00% | 0.00% | 100.0% |
| .20 | CCE positive-only | 13.15 | 56.15% | 87.72% | 70.75% | 16.67% | 7.69% |
|  | Query all-support | 23.31 | 22.31% | 67.66% | 96.70% | 58.33% | 0.0% |
|  | Candidate BY | 0.38 | 98.72% | 100.00% | 2.36% | 0.00% | 92.31% |

Top-30 retrieval ceiling is 12/13 test topic titles (92.31%). At `.10`, CCE removes 41.28% of context and retains 86.79% of relevant candidate documents, but only 41.67% of queries retain every support. Query-level calibration retains 8.62 more documents, raises all-support coverage by 50 points, and raises marginal recall by 12.74 points. BY has the same empty-set failure already observed on TAT-QA, HotpotQA, MMQA, and WebQA.

This 17/13 split is descriptive, not a final statistical claim. Its direction matches the formal distinction: independently calibrated positives cannot protect the minimum-scoring support required by a multi-evidence report request.

## Contribution boundary

CCE already establishes model-agnostic conformal pointwise context filtering. The distinct contribution here is: (1) calibrating a query-level all-support event, (2) evaluating its context/coverage frontier against CCE's pointwise target and candidate BY on identical candidates, and (3) testing whether internal model signals improve the score before query-level calibration.

CCE remains the appropriate simpler method when high marginal evidence retention and aggressive context reduction are the product objective. Query-level support is appropriate when every retrieved support required by a multi-hop answer or report must survive pruning.

## Exact-reproduction path

An exact CCE comparison needs: their topic-disjoint retrieval runs; 500-character/100-character-overlap snippets; Llama-3.3-70B labels using their prompt; Qwen3-Embedding-8B and GPT-4o scores using their prompts; frozen calibration-only thresholds; and the NeuCLIR generator/nugget/AutoARGUE evaluation. The public release currently provides only the prompts, so this result must remain labelled public-data proxy until those artifacts are released or independently recreated.

## Artifacts

- Runner: [`run_ragtime_head_to_head.py`](run_ragtime_head_to_head.py)
- Summary: [`ragtime_head_to_head_summary.json`](results/ragtime_english_tfidf_top30_2026-09-17/ragtime_head_to_head_summary.json)
- Per-query score and labels: [`ragtime_head_to_head_predictions.npz`](results/ragtime_english_tfidf_top30_2026-09-17/ragtime_head_to_head_predictions.npz)
