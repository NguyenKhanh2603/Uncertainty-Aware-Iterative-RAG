# NeuCLIR: CCE versus query-level all-nugget conformal support

**Date:** 2026-09-17

## Protocol

This is a score-level head-to-head on the public CoverageBench NeuCLIR-2024
report-generation artifact. It supplies 19 topics, nugget-to-document qrels,
and frozen Qwen3-Embedding-8B initial-retrieval scores. A deterministic query
split has 10 calibration and 9 test topics. All methods use the identical
Top-30 ranked documents and scores.

NeuCLIR reports have many interchangeable documents per information nugget.
The relevant query-level event is therefore not retaining every relevant
document; it is retaining **at least one document for every nugget represented
in Top-30**. The query score for calibration is the lowest, across nuggets, of
the highest score among that nugget's candidate documents.

| Method | Conformal target |
|---|---|
| CCE positive-only | marginal retention of a positive candidate document |
| Query all-nugget | every retrievable information nugget has at least one retained document |
| Candidate BY | candidate false-discovery control |

This is a direct selection comparison using Qwen3 scores, but not downstream
F1: CoverageBench does not include document text or reports generated from
these filtered contexts. It must not be compared numerically with CCE's
published NeuCLIR ARGUE F1.

## Results

`Nugget micro` is the fraction of all retrievable nuggets with at least one
retained document. `All nugget` is the fraction of test queries retaining every
retrievable nugget.

| alpha | Method | Mean docs | Context reduction | Doc precision | Doc recall | Nugget micro | All nugget | Empty |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| .05 | CCE positive-only | 24.00 | 20.00% | 77.31% | 86.53% | 89.52% | 77.78% | 0.0% |
|  | Query all-nugget | 27.00 | 10.00% | 75.72% | 95.34% | 98.10% | 88.89% | 0.0% |
|  | Candidate BY | 0.00 | 100.00% | -- | 0.00% | 0.00% | 0.00% | 100.0% |
| .10 | CCE positive-only | 23.78 | 20.74% | 77.57% | 86.01% | 88.57% | 77.78% | 0.0% |
|  | Query all-nugget | 27.00 | 10.00% | 75.72% | 95.34% | 98.10% | 88.89% | 0.0% |
|  | Candidate BY | 3.22 | 89.26% | 86.21% | 12.95% | 10.48% | 11.11% | 88.89% |
| .20 | CCE positive-only | 22.44 | 25.19% | 79.70% | 83.42% | 83.81% | 77.78% | 11.11% |
|  | Query all-nugget | 23.44 | 21.85% | 77.73% | 84.97% | 83.81% | 77.78% | 11.11% |
|  | Candidate BY | 3.33 | 88.89% | 86.67% | 13.47% | 10.48% | 11.11% | 88.89% |

## Interpretation

At alpha `.10`, the proposed query-level target is better for report evidence:
it gains `+9.53` points nugget micro coverage and `+11.11` points all-nugget
coverage over CCE. It costs 3.22 extra documents on a Top-30 context and loses
1.85 points document precision. This is the intended precision/context versus
complete-evidence trade-off, not a claim that it is universally better.

At alpha `.20`, the two conformal thresholds land closer together on this small
calibration split. BY again maximizes pruning at the expense of almost all
evidence coverage.

## F1 status

This run does **not** establish a downstream F1 winner. CCE's published F1 is
ARGUE F1 on a separate NeuCLIR setup using 500-character snippets,
Llama-3.3-generated relevance labels, and a Llama-3.3 generator. To compare
F1 fairly, both selectors must be applied to the same retrieved documents, then
the same generator must create reports and the same nugget/AutoARGUE evaluator
must score them. The full NeuCLIR multilingual document collection contains
roughly 10 million documents; CoverageBench deliberately releases scores and
qrels rather than document text.

## Artifacts

- Runner: [`run_neuclir_nugget_head_to_head.py`](run_neuclir_nugget_head_to_head.py)
- Raw result: [`neuclir_qwen3_top30_2026-09-17.json`](results/neuclir_qwen3_top30_2026-09-17.json)
- Data source: [CoverageBench](https://huggingface.co/datasets/hltcoe/coveragebench)
