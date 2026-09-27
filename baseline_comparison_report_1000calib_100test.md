# Báo cáo Đánh giá Phương pháp Conformal BH (1000 Calibration, 100 Test)

Dưới đây là bảng số liệu khi chạy pipeline với **1000 câu Calibration** và **100 câu Test** (được lấy từ file manifest trong thư mục `E:\splits`).

### 1. Dataset: HotpotQA
| Method | Kept Chunks | Precision | Recall | Empty Rate |
| :--- | :--- | :--- | :--- | :--- |
| Fixed Top-10 | 9.85 | 20.3% | 100.0% | 0.0% |
| BH cosine (α=0.1, ctx=10) | 0.89 | 55.1% | 24.5% | 60.0% |
| BH cosine (α=0.9, ctx=10) | 8.11 | 22.3% | 90.5% | 3.0% |
| BH cosine (α=0.99, ctx=10) | 9.59 | 20.5% | 98.5% | 0.0% |
| Conformal BH (modality_aware, α=0.1) | 0.89 | 55.1% | 24.5% | 60.0% |
| Conformal BH (modality_aware, α=0.9) | 8.11 | 22.3% | 90.5% | 3.0% |
| Conformal BH (modality_aware, α=0.99) | 9.59 | 20.5% | 98.5% | 0.0% |

### 2. Dataset: MMQA
| Method | Kept Chunks | Precision | Recall | Empty Rate |
| :--- | :--- | :--- | :--- | :--- |
| Fixed Top-10 | 10.00 | 14.1% | 91.0% | 0.0% |
| BH cosine (α=0.1, ctx=10) | 0.92 | 34.8% | 20.6% | 68.0% |
| BH cosine (α=0.9, ctx=10) | 8.88 | 14.6% | 83.9% | 2.0% |
| BH cosine (α=0.99, ctx=10) | 9.85 | 14.1% | 89.7% | 0.0% |
| Conformal BH (modality_aware, α=0.1) | 0.69 | 37.7% | 16.8% | 72.0% |
| Conformal BH (modality_aware, α=0.9) | 6.62 | 11.0% | 47.1% | 18.0% |
| Conformal BH (modality_aware, α=0.99) | 8.36 | 9.6% | 51.6% | 7.0% |

### 3. Dataset: TAT-QA
| Method | Kept Chunks | Precision | Recall | Empty Rate |
| :--- | :--- | :--- | :--- | :--- |
| Fixed Top-10 | 5.63 | 23.8% | 100.0% | 0.0% |
| BH cosine (α=0.1, ctx=10) | 0.28 | 60.7% | 12.7% | 80.0% |
| BH cosine (α=0.9, ctx=10) | 4.48 | 25.7% | 85.8% | 10.0% |
| BH cosine (α=0.99, ctx=10) | 5.44 | 24.4% | 99.3% | 1.0% |
| Conformal BH (modality_aware, α=0.1) | 0.18 | 61.1% | 8.2% | 84.0% |
| Conformal BH (modality_aware, α=0.9) | 2.74 | 13.1% | 26.9% | 31.0% |
| Conformal BH (modality_aware, α=0.99) | 3.22 | 11.5% | 27.6% | 26.0% |

### 4. Dataset: WebQA
| Method | Kept Chunks | Precision | Recall | Empty Rate |
| :--- | :--- | :--- | :--- | :--- |
| Fixed Top-10 | 9.98 | 14.1% | 80.6% | 0.0% |
| BH cosine (α=0.1, ctx=10) | 0.68 | 36.8% | 14.3% | 77.0% |
| BH cosine (α=0.9, ctx=10) | 7.04 | 14.9% | 60.0% | 23.0% |
| BH cosine (α=0.99, ctx=10) | 8.49 | 14.3% | 69.1% | 14.0% |
| Conformal BH (modality_aware, α=0.1) | 0.58 | 34.5% | 11.4% | 81.0% |
| Conformal BH (modality_aware, α=0.9) | 6.43 | 16.3% | 60.0% | 25.0% |
| Conformal BH (modality_aware, α=0.99) | 8.33 | 14.5% | 69.1% | 13.0% |
