# Literature retrieval constructions on the inverse 1,000/100 split

This directory compares chunk-selection constructions on the same frozen
Jina-v4 Top-30 candidate logs, 1,000 calibration qids, and 100 held-out test
qids per dataset. The selection table is in [REPORT.md](REPORT.md).

The executable protocol is
[`run_literature_protocol_1000cal.py`](../../run_literature_protocol_1000cal.py).
It implements CCE Conformal-Embedding, the CONFLARE source-question
calibration rule, the TRAQ retrieval Bonferroni component, Fixed Top-10, and
query-level all-support cosine. The method scope and non-comparable full TRAQ
answer-set stage are documented in that file's module docstring and report.

The exact inputs are versioned under [splits](splits): each dataset has a
[calibration manifest](splits/hotpotqa/calibration_manifest.json), a
[test manifest](splits/hotpotqa/test_manifest.json), and the four-way audit in
[SPLIT_INTEGRITY.json](splits/SPLIT_INTEGRITY.json). The repeated HotpotQA
links are an example of the identical file layout for MMQA, TAT-QA, and WebQA.

The checked-out baseline sources used to derive the rules are:

- `baselines/conformal-context-engineering` at `91732d6058267f180ba9f47873d743288b2625af`;
- `baselines/conflare` at `ce081a45fb452704daa87f3b37f601b4accc7a82`;
- `baselines/TRAQ` at `e9b66b5b7fd8fcd0b3c3dbfebc5a3391ca79aa55`.

Downstream Qwen direct-answer outputs are appended to this directory after the
selection run. They are a shared QA diagnostic; they are not TRAQ's semantic
answer-prediction-set coverage metric.
