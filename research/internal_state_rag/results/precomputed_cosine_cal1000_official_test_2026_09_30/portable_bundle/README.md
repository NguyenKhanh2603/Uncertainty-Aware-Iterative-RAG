# Precomputed Jina-v4 cosine scores — calibration 1,000 + official full test

This portable bundle contains no model weights, corpus text, or images.  It stores the
exact Top-L candidate rows used by the experiment, including each candidate's cosine
score, modality, rank, support label, and pipeline fingerprint.

- `rowwise/`: original retrieval-log schema accepted directly by the conformal scripts;
- `querywise/`: one gzip JSONL record per query with a nested `candidates` list;
- `combined_reference_banks.json.gz`: false-score banks built only from calibration rows;
- `by_selection/`: candidate-wise BY decisions and summary on the official test rows;
- `splits/`: frozen calibration and test qid manifests;
- `MANIFEST.json`: file hashes, sizes, and query counts.

The retriever is `jinaai/jina-embeddings-v4` at its pinned repository revision.  All
methods consume the same cosine rows; running BY from this folder requires no GPU encode.
