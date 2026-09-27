# Báo cáo So sánh Tổng thể (1000 Calibration, 100 Test)

Báo cáo này tổng hợp kết quả của các baseline (ECIR CCE, CONFLARE, TRAQ) cùng với Fixed Top-10, Top-20 và phương pháp Conformal BH (cả dataset_pooled lẫn modality_aware).

### Dataset: HOTPOTQA
| Method | Kept Chunks | Precision | Recall | Empty Rate |
| :--- | :--- | :--- | :--- | :--- |
| Fixed Top-10 | 9.85 | 20.3% | 100.0% | 0.0% |
| Fixed Top-20 | 9.85 | 20.3% | 100.0% | 0.0% |
| ECIR CCE | N/A | 29.1% | 90.0% | N/A |
| CONFLARE | N/A | 29.2% | 90.0% | N/A |
| TRAQ retrieval | N/A | 25.0% | 93.0% | N/A |
| BH cosine (α=0.1, ctx=10) | 0.89 | 55.1% | 24.5% | 60.0% |
| BH cosine (α=0.9, ctx=10) | 8.11 | 22.3% | 90.5% | 3.0% |
| BH cosine (α=0.99, ctx=10) | 9.59 | 20.5% | 98.5% | 0.0% |
| BH cosine (α=0.99, ctx=20) | 9.59 | 20.5% | 98.5% | 0.0% |
| Conformal BH (modality_aware, α=0.1, ctx=10) | 0.89 | 55.1% | 24.5% | 60.0% |
| Conformal BH (modality_aware, α=0.9, ctx=10) | 8.11 | 22.3% | 90.5% | 3.0% |
| Conformal BH (modality_aware, α=0.99, ctx=10) | 9.59 | 20.5% | 98.5% | 0.0% |
| Conformal BH (modality_aware, α=0.99, ctx=20) | 9.59 | 20.5% | 98.5% | 0.0% |

### Dataset: MMQA
| Method | Kept Chunks | Precision | Recall | Empty Rate |
| :--- | :--- | :--- | :--- | :--- |
| Fixed Top-10 | 10.00 | 14.1% | 91.0% | 0.0% |
| Fixed Top-20 | 19.01 | 7.9% | 97.4% | 0.0% |
| ECIR CCE | N/A | 12.4% | 92.9% | N/A |
| CONFLARE | N/A | 12.3% | 92.3% | N/A |
| TRAQ retrieval | N/A | 10.9% | 97.4% | N/A |
| BH cosine (α=0.1, ctx=10) | 0.92 | 34.8% | 20.6% | 68.0% |
| BH cosine (α=0.9, ctx=10) | 8.88 | 14.6% | 83.9% | 2.0% |
| BH cosine (α=0.99, ctx=10) | 9.85 | 14.1% | 89.7% | 0.0% |
| BH cosine (α=0.99, ctx=20) | 18.41 | 8.1% | 96.1% | 0.0% |
| Conformal BH (modality_aware, α=0.1, ctx=10) | 0.69 | 37.7% | 16.8% | 72.0% |
| Conformal BH (modality_aware, α=0.9, ctx=10) | 6.62 | 11.0% | 47.1% | 18.0% |
| Conformal BH (modality_aware, α=0.99, ctx=10) | 8.36 | 9.6% | 51.6% | 7.0% |
| Conformal BH (modality_aware, α=0.99, ctx=20) | 14.60 | 6.0% | 56.8% | 7.0% |

### Dataset: TATQA
| Method | Kept Chunks | Precision | Recall | Empty Rate |
| :--- | :--- | :--- | :--- | :--- |
| Fixed Top-10 | 5.63 | 23.8% | 100.0% | 0.0% |
| Fixed Top-20 | 6.10 | 22.0% | 100.0% | 0.0% |
| ECIR CCE | N/A | 29.0% | 85.1% | N/A |
| CONFLARE | N/A | 28.8% | 84.3% | N/A |
| TRAQ retrieval | N/A | 25.9% | 89.6% | N/A |
| BH cosine (α=0.1, ctx=10) | 0.28 | 60.7% | 12.7% | 80.0% |
| BH cosine (α=0.9, ctx=10) | 4.48 | 25.7% | 85.8% | 10.0% |
| BH cosine (α=0.99, ctx=10) | 5.44 | 24.4% | 99.3% | 1.0% |
| BH cosine (α=0.99, ctx=20) | 5.91 | 22.5% | 99.3% | 1.0% |
| Conformal BH (modality_aware, α=0.1, ctx=10) | 0.18 | 61.1% | 8.2% | 84.0% |
| Conformal BH (modality_aware, α=0.9, ctx=10) | 2.74 | 13.1% | 26.9% | 31.0% |
| Conformal BH (modality_aware, α=0.99, ctx=10) | 3.22 | 11.5% | 27.6% | 26.0% |
| Conformal BH (modality_aware, α=0.99, ctx=20) | 3.46 | 10.7% | 27.6% | 26.0% |

### Dataset: WEBQA
| Method | Kept Chunks | Precision | Recall | Empty Rate |
| :--- | :--- | :--- | :--- | :--- |
| Fixed Top-10 | 9.98 | 14.1% | 80.6% | 0.0% |
| Fixed Top-20 | 19.61 | 8.4% | 93.7% | 0.0% |
| ECIR CCE | N/A | 12.0% | 81.7% | N/A |
| CONFLARE | N/A | 12.0% | 81.7% | N/A |
| TRAQ retrieval | N/A | 10.7% | 89.7% | N/A |
| BH cosine (α=0.1, ctx=10) | 0.68 | 36.8% | 14.3% | 77.0% |
| BH cosine (α=0.9, ctx=10) | 7.04 | 14.9% | 60.0% | 23.0% |
| BH cosine (α=0.99, ctx=10) | 8.49 | 14.3% | 69.1% | 14.0% |
| BH cosine (α=0.99, ctx=20) | 16.48 | 8.7% | 81.7% | 14.0% |
| Conformal BH (modality_aware, α=0.1, ctx=10) | 0.58 | 34.5% | 11.4% | 81.0% |
| Conformal BH (modality_aware, α=0.9, ctx=10) | 6.43 | 16.3% | 60.0% | 25.0% |
| Conformal BH (modality_aware, α=0.99, ctx=10) | 8.33 | 14.5% | 69.1% | 13.0% |
| Conformal BH (modality_aware, α=0.99, ctx=20) | 16.29 | 8.8% | 81.7% | 13.0% |

