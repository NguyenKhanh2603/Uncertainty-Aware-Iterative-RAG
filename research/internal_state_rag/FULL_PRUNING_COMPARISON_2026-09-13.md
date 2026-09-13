# Full pruning comparison

BY controls candidate-level false discovery and can return an empty set. Query-level conformal targets retention of every support chunk conditional on support existing in Top-30 and uses deterministic Top-1 fallback.

| Method | Rule | α | Scope (n) | Chunks | Precision | Micro recall | Mean query recall | All-support coverage | Empty |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Attention-only trained probe | query conformal | 0.20 | 150 | 2.08 | 41.7% | 75.6% | 78.1% | 74.0% | 0.0% |
| Attention only (position-controlled) | query conformal | 0.20 | 150 | 1.76 | 45.1% | 69.2% | 71.2% | 67.3% | 0.0% |
| Attention only (raw) | query conformal | 0.20 | 150 | 1.65 | 46.8% | 67.4% | 69.2% | 65.3% | 0.0% |
| BGE | query conformal | 0.20 | 768 | 3.19 | 28.5% | 80.7% | 83.8% | 80.1% | 0.0% |
| Cosine + BY (pooled) | BY-FDR | 0.20 | 1000 | 0.33 | 25.3% | 9.6% | — | — | 89.5% |
| Cosine + BY (modality) | BY-FDR | 0.20 | 1000 | 0.28 | 29.0% | 9.4% | — | — | 90.5% |
| BGE + LM-head + hidden probe | query conformal | 0.20 | 768 | 1.86 | 47.7% | 78.9% | 82.4% | 78.3% | 0.0% |
| Attention-only trained probe | query conformal | 0.10 | 150 | 13.89 | 7.8% | 94.8% | 96.3% | 94.7% | 0.0% |
| Attention only (position-controlled) | query conformal | 0.10 | 150 | 7.06 | 14.1% | 86.6% | 87.4% | 84.7% | 0.0% |
| Attention only (raw) | query conformal | 0.10 | 150 | 8.86 | 11.4% | 87.8% | 89.4% | 87.3% | 0.0% |
| BGE | query conformal | 0.10 | 768 | 5.73 | 17.4% | 88.4% | 91.0% | 87.4% | 0.0% |
| Cosine + BY (pooled) | BY-FDR | 0.10 | 1000 | 0.15 | 38.5% | 6.6% | — | — | 93.3% |
| Cosine + BY (modality) | BY-FDR | 0.10 | 1000 | 0.12 | 52.2% | 6.9% | — | — | 93.3% |
| BGE + LM-head + hidden probe | query conformal | 0.10 | 768 | 3.39 | 29.3% | 88.3% | 91.4% | 87.9% | 0.0% |
| Attention-only trained probe | query conformal | 0.05 | 150 | 24.01 | 4.7% | 97.7% | 98.7% | 97.3% | 0.0% |
| Attention only (position-controlled) | query conformal | 0.05 | 150 | 20.05 | 5.5% | 95.9% | 96.7% | 95.3% | 0.0% |
| Attention only (raw) | query conformal | 0.05 | 150 | 18.23 | 6.1% | 96.5% | 97.0% | 96.7% | 0.0% |
| BGE | query conformal | 0.05 | 768 | 15.16 | 7.0% | 94.3% | 95.7% | 93.8% | 0.0% |
| Cosine + BY (pooled) | BY-FDR | 0.05 | 1000 | 0.09 | 54.7% | 5.4% | — | — | 94.5% |
| Cosine + BY (modality) | BY-FDR | 0.05 | 1000 | 0.06 | 61.0% | 4.2% | — | — | 96.0% |
| BGE + LM-head + hidden probe | query conformal | 0.05 | 768 | 6.30 | 16.8% | 94.1% | 95.6% | 93.4% | 0.0% |
