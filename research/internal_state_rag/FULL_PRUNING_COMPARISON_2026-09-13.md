# Full pruning comparison

BY controls candidate-level false discovery and can return an empty set. Query-level conformal targets retention of every support chunk conditional on support existing in Top-30 and uses deterministic Top-1 fallback.

| Method | Rule | α | Scope (n) | Chunks | Precision | Micro recall | Mean query recall | All-support coverage | Empty |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Attention-only trained probe | query conformal | 0.20 | 150 | 2.19 | 40.2% | 76.7% | 79.1% | 75.3% | 0.0% |
| Attention only (position-controlled) | query conformal | 0.20 | 150 | 1.91 | 43.7% | 72.7% | 74.6% | 71.3% | 0.0% |
| Attention only (raw) | query conformal | 0.20 | 150 | 1.81 | 44.3% | 69.8% | 71.6% | 68.0% | 0.0% |
| BGE (matched) | query conformal | 0.20 | 150 | 1.91 | 41.6% | 69.2% | 72.1% | 68.0% | 0.0% |
| Cosine + BY (pooled) | BY-FDR | 0.20 | 1000 | 0.33 | 25.3% | 9.6% | — | — | 89.5% |
| Cosine + BY (modality) | BY-FDR | 0.20 | 1000 | 0.28 | 29.0% | 9.4% | — | — | 90.5% |
| BGE + LM-head + hidden probe (matched) | query conformal | 0.20 | 150 | 1.39 | 56.7% | 68.6% | 71.8% | 67.3% | 0.0% |
| Attention-only trained probe | query conformal | 0.10 | 150 | 17.17 | 6.4% | 95.9% | 97.0% | 96.0% | 0.0% |
| Attention only (position-controlled) | query conformal | 0.10 | 150 | 7.06 | 14.1% | 86.6% | 87.4% | 84.7% | 0.0% |
| Attention only (raw) | query conformal | 0.10 | 150 | 8.86 | 11.4% | 87.8% | 89.4% | 87.3% | 0.0% |
| BGE (matched) | query conformal | 0.10 | 150 | 5.19 | 19.4% | 87.8% | 89.7% | 86.7% | 0.0% |
| Cosine + BY (pooled) | BY-FDR | 0.10 | 1000 | 0.15 | 38.5% | 6.6% | — | — | 93.3% |
| Cosine + BY (modality) | BY-FDR | 0.10 | 1000 | 0.12 | 52.2% | 6.9% | — | — | 93.3% |
| BGE + LM-head + hidden probe (matched) | query conformal | 0.10 | 150 | 2.70 | 36.0% | 84.9% | 87.4% | 83.3% | 0.0% |
| Attention-only trained probe | query conformal | 0.05 | 150 | 28.39 | 4.0% | 99.4% | 99.7% | 99.3% | 0.0% |
| Attention only (position-controlled) | query conformal | 0.05 | 150 | 22.23 | 5.0% | 97.1% | 97.7% | 96.7% | 0.0% |
| Attention only (raw) | query conformal | 0.05 | 150 | 20.81 | 5.3% | 96.5% | 97.0% | 96.7% | 0.0% |
| BGE (matched) | query conformal | 0.05 | 150 | 17.40 | 6.2% | 94.2% | 95.0% | 94.0% | 0.0% |
| Cosine + BY (pooled) | BY-FDR | 0.05 | 1000 | 0.09 | 54.7% | 5.4% | — | — | 94.5% |
| Cosine + BY (modality) | BY-FDR | 0.05 | 1000 | 0.06 | 61.0% | 4.2% | — | — | 96.0% |
| BGE + LM-head + hidden probe (matched) | query conformal | 0.05 | 150 | 6.12 | 17.6% | 94.2% | 95.0% | 93.3% | 0.0% |
