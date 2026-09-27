# Source-code fidelity re-audit — CCE, CONFLARE, and TRAQ

**Verdict:** the three rows in [REPORT.md](REPORT.md) are useful *matched Jina score-threshold adaptations*, but they are **not executions of the three original repositories**.  They must not be described as exact literature reproductions or as head-to-head results against the complete published systems.

This audit rechecked the source repositories on 2026-09-27 at the exact commits below.  The local checkouts are intentionally excluded from this repository because they are third-party source trees; the links pin each inspected upstream file to its commit.

| Method shown in the report | Upstream source checked | Original code run in this artifact? | Fidelity verdict |
|---|---|---:|---|
| CCE Conformal-Embedding, Jina adaptation | [hltcoe/conformal-context-engineering @ `91732d6`](https://github.com/hltcoe/conformal-context-engineering/tree/91732d6058267f180ba9f47873d743288b2625af) | No executable retrieval implementation is released | Formula-level reimplementation only |
| CONFLARE source-question, Jina adaptation | [Mayo-Radiology-Informatics-Lab/conflare @ `ce081a4`](https://github.com/Mayo-Radiology-Informatics-Lab/conflare/tree/ce081a45fb452704daa87f3b37f601b4accc7a82) | No | Retrieval-threshold adaptation only |
| TRAQ retrieval Bonferroni, Jina adaptation | [shuoli90/TRAQ @ `e9b66b5`](https://github.com/shuoli90/TRAQ/tree/e9b66b5b7fd8fcd0b3c3dbfebc5a3391ca79aa55) | No | Retrieval-component adaptation only |

The practical source-code execution rate is therefore **0/3 original systems**.  For TRAQ, the scalar retrieval threshold convention was independently transcribed from source, but no original TRAQ code path or answer-set component was run.

## Why the three rows are close

All three rows consume the **same frozen Jina-v4 Top-L≤30 candidate scores** and make one monotonic decision: keep a candidate when its Jina cosine exceeds one dataset-wide cutoff.  They differ only in how that cutoff is calibrated:

| Row | Calibration score in the current runner | Test-time selector |
|---|---|---|
| CCE adaptation | Every labelled support cosine pooled across calibration questions; lower `alpha` quantile | `cosine >= cutoff` |
| CONFLARE adaptation | Maximum labelled-support cosine for each calibration question; lower `alpha` quantile | `cosine > cutoff` |
| TRAQ retrieval adaptation | Maximum labelled-support cosine for each calibration question; lower `alpha/2` quantile | `cosine >= cutoff` |

The implementation is [run_literature_protocol_1000cal.py](../../run_literature_protocol_1000cal.py), in [`cce_embedding_threshold`](../../run_literature_protocol_1000cal.py), [`conflare_threshold`](../../run_literature_protocol_1000cal.py), and [`traq_retrieval_threshold`](../../run_literature_protocol_1000cal.py).  Since these are all global cutoffs on the same score ordering, similar retained-chunk, precision, recall, and downstream values are expected when the three quantiles fall near each other.  Similarity of the rows is therefore not proof that the code accidentally reused one mask; each method constructs its own threshold and mask.  It *is* proof that this experiment cannot distinguish the complete literature systems.

The following independent audit rebuilt the three masks from the frozen logs and manifests.  “Same masks” is the number of the 100 held-out queries for which two methods selected exactly the same ordered candidate boolean mask; it does not compare only their aggregate metrics.  TRAQ's current `method="lower"` quantile has the same lower-order-statistic semantics as the upstream NumPy `interpolation="lower"` call.

| Dataset | CCE cutoff | CONFLARE cutoff | TRAQ cutoff | CCE = CONFLARE masks | CCE = TRAQ masks | CONFLARE = TRAQ masks |
|---|---:|---:|---:|---:|---:|---:|
| HotpotQA | 0.830209 | 0.872105 | 0.845252 | 36 / 100 | 69 / 100 | 42 / 100 |
| MMQA | 0.757050 | 0.767098 | 0.720308 | 73 / 100 | 53 / 100 | 42 / 100 |
| TAT-QA | 0.885476 | 0.898299 | 0.883185 | 71 / 100 | 93 / 100 | 66 / 100 |
| WebQA | 0.789195 | 0.813680 | 0.761551 | 25 / 100 | 36 / 100 | 15 / 100 |

## What each original repository actually does

### CCE / Conformal Context Engineering

The inspected upstream tree has only `README.md`, prompts, and example JSON: no Python package, retrieval executable, or released calibration pipeline.  The [README’s Conformal-Embedding definition](https://github.com/hltcoe/conformal-context-engineering/blob/91732d6058267f180ba9f47873d743288b2625af/README.md) defines nonconformity as `1 - cosine(query, chunk)` for chunks assigned relevance label `r=1`, then takes a `(1-alpha)` nonconformity quantile.  This is algebraically a lower-`alpha` cosine threshold, which is the formula used by the CCE adaptation.

That is the limit of the match.  The released materials describe a different embedding model, document segmentation, and relevance-labeling setup from the frozen Jina candidates and benchmark support labels used here.  There is no upstream executable that could have been run unchanged, so “CCE, Jina adaptation” is the accurate name.

### CONFLARE

The upstream [pipeline initialization](https://github.com/Mayo-Radiology-Informatics-Lab/conflare/blob/ce081a45fb452704daa87f3b37f601b4accc7a82/conflare/main.py#L18-L89) chunks input documents (default 1,500 characters with 200 overlap), embeds them with normalized `sentence-transformers/all-MiniLM-L6-v2`, and builds a Chroma cosine collection.  It then [generates a question from a source chunk](https://github.com/Mayo-Radiology-Informatics-Lab/conflare/blob/ce081a45fb452704daa87f3b37f601b4accc7a82/conflare/conformal/calibration.py#L31-L72), retrieves the **entire collection**, and accepts the first source or LLM-judged relevant chunk among up to 100 candidates ([evaluation](https://github.com/Mayo-Radiology-Informatics-Lab/conflare/blob/ce081a45fb452704daa87f3b37f601b4accc7a82/conflare/conformal/calibration.py#L176-L221)).

At inference it retrieves `vector_db.count()` documents and retains every document whose cosine **distance** is strictly below the `(1-error_rate)` percentile of those calibration distances ([filter](https://github.com/Mayo-Radiology-Informatics-Lab/conflare/blob/ce081a45fb452704daa87f3b37f601b4accc7a82/conflare/augmented_retrieval/rag.py#L54-L99)).  The current row preserves only the last scalar-threshold idea: it replaces generated source questions and LLM judgments with benchmark questions and labelled supports, substitutes Jina cosine for MiniLM/Chroma distance, and only sees the frozen Top-L≤30 pool.  It is not the CONFLARE pipeline.

### TRAQ

The upstream [threshold function](https://github.com/shuoli90/TRAQ/blob/e9b66b5b7fd8fcd0b3c3dbfebc5a3391ca79aa55/misc/utils.py#L619-L636) sets a retrieval cutoff to the lower `alpha` quantile of a true-retrieval score.  Its [Bonferroni procedure](https://github.com/shuoli90/TRAQ/blob/e9b66b5b7fd8fcd0b3c3dbfebc5a3391ca79aa55/run/traq/traq_chatgpt_semantic.py#L639-L657) assigns `alpha/2` to retrieval and `alpha/2` to an answer-prediction score.  The current adapter correctly uses a lower `alpha/2` quantile for its **retrieval-only** threshold.

But the full source reads pre-collected retrieval and generated-answer data ([inputs](https://github.com/shuoli90/TRAQ/blob/e9b66b5b7fd8fcd0b3c3dbfebc5a3391ca79aa55/run/traq/traq_chatgpt_semantic.py#L36-L61)), splits examples into 30% calibration, 30% validation/tuning, and 40% test ([split](https://github.com/shuoli90/TRAQ/blob/e9b66b5b7fd8fcd0b3c3dbfebc5a3391ca79aa55/run/traq/traq_chatgpt_semantic.py#L478-L518)), and filters generated semantic answer sets using a second calibrated threshold ([evaluation](https://github.com/shuoli90/TRAQ/blob/e9b66b5b7fd8fcd0b3c3dbfebc5a3391ca79aa55/run/traq/traq_chatgpt_semantic.py#L77-L153)).  None of those components exists in the Jina chunk-pruning run.  Its shared Qwen EM/F1 diagnostic is therefore not TRAQ’s claimed end-to-end coverage measurement.

## Consequence for the reported numbers

The values in [REPORT.md](REPORT.md) remain valid measurements of three explicitly defined selectors on the supplied `splits_khanh_27_09` split: 1,000 calibration questions, 100 held-out test questions, Jina-v4 candidate scores, Top-L≤30, and the shared Qwen2-VL-7B direct-answer diagnostic.  They do **not** establish that CCE, CONFLARE, or TRAQ themselves perform at those values.

The correct comparison label is:

> CCE-inspired pooled-support threshold; CONFLARE-inspired source-question threshold; and TRAQ retrieval-component Bonferroni threshold, all adapted to the same Jina candidate pool.

## What would count as a stronger comparison

1. **CCE:** reproduce the paper specification independently, since the checked repository contains no runnable source.  Match its embedding model, chunking, relevance labeling, and corpus protocol, then state that it is a reproduction from the paper rather than source-code execution.
2. **CONFLARE:** run its own code with a corpus-only calibration phase, generated questions, LLM relevance decisions, MiniLM embeddings, and retrieval over all chunks.  This gives high method fidelity but changes the retrieval environment, so it should be reported in a separate “original-pipeline” table from the common-Jina comparison.
3. **TRAQ:** run its retrieval **and** answer-set components on a compatible QA benchmark with its data-collection/semantic-clustering inputs.  It should be evaluated using retrieval and answer-set coverage, not micro chunk precision/recall alone.

For a controlled common-candidate benchmark, retain the current rows but call them adaptations and compare their calibration rules, rather than treating close values as three independent end-to-end baselines.
