# Quy trình Conformal Prediction for RAG (Đã Tối Ưu)

Hệ thống đã được dọn dẹp và tối giản hóa tối đa. Để chạy và ra được bảng kết quả chuẩn xác 100% so với phương pháp gốc, chỉ cần sử dụng 2 file Python duy nhất trong thư mục `scripts/`. 

Không cần sử dụng file .bat hay xây dựng Reference Bank phức tạp lưu ra ổ cứng, toàn bộ quá trình đã được tối ưu chạy trực tiếp trên RAM bằng thuật toán tìm kiếm nhị phân (`bisect`).

---

## 1. Xuất dữ liệu đã lọc (Filtering)
**File thực thi:** `scripts/export_downstream.py`

**Mô tả chức năng:**
*   **Bước 1:** Đọc trực tiếp 4 file dữ liệu thô (Top-30 Retreival của HotpotQA, MMQA, TATQA, WebQA) định dạng `.jsonl.gz`.
*   **Bước 2:** Tự động xây dựng `False-Null Bank` (phân loại theo dataset & modality) và `Support-Null Bank` (phân loại theo dataset) từ tập Calibration (1000 câu hỏi).
*   **Bước 3:** Áp dụng phương pháp Benjamini-Hochberg (BH) để cắt tỉa các chunk rác cho tập Test (100 câu hỏi). 
    *   Thực hiện đồng thời nhiều chiến lược: *Stage 1 Only*, *Two Stage*, và *Hybrid Fixed Top-10 + Stage 2* ở các ngưỡng alpha khác nhau.
    *   Hỗ trợ bù đắp (Backfill) chunk 
*   **Bước 4:** Xuất kết quả toàn bộ các chiến lược ra một file duy nhất: `downstream_export.json`.

**Cách chạy:**
`python scripts/export_downstream.py`


---

## 2. Đánh giá và in Bảng kết quả (Evaluation)
**File thực thi:** `scripts/evaluate_results.py`

**Mô tả chức năng:**
*   Đọc file `downstream_export.json` vừa được tạo ra.
*   Đối chiếu với Ground Truth (các chunk được gán nhãn `support`).
*   Tính toán các chỉ số cốt lõi: 
    *   **Kept:** Trung bình số lượng chunk được giữ lại đưa vào LLM.
    *   **Precision:** Tỷ lệ chunk chứa thông tin đúng trên tổng số chunk được đưa vào LLM.
    *   **Recall:** Tỷ lệ phủ của các chunk đúng so với tổng số chunk đúng có trong Top-30.
*   In ra bảng thống kê so sánh toàn bộ các method một cách trực quan trên Terminal.

**Cách chạy:**
`python scripts/evaluate_results.py`

---

## 3. Bảng Kết Quả Đánh Giá Toàn Bộ Hệ Thống

Dưới đây là thống kê chi tiết đầu ra (**Kept | Precision | Recall**) của 12 phương pháp/cấu hình đã chạy, mô phỏng chính xác logic lọc của hệ thống gốc:

| Method | HOTPOTQA | MMQA | TATQA | WEBQA |
| :--- | :--- | :--- | :--- | :--- |
| **fixed_top5** | 4.95 \| 33.7% \| 83.5% | 5.00 \| 24.0% \| 81.2% | 4.37 \| 29.7% \| 97.0% | 5.00 \| 23.0% \| 66.5% |
| **stage1_a0.1** | 0.89 \| 55.1% \| 24.5% | 0.70 \| 38.6% \| 19.0% | 0.33 \| 69.7% \| 17.5% | 0.58 \| 34.5% \| 11.2% |
| **stage1_a0.9** | 8.11 \| 22.3% \| 90.5% | 7.41 \| 15.5% \| 77.2% | 4.53 \| 25.6% \| 88.5% | 6.43 \| 16.3% \| 59.7% |
| **stage1_a0.99** | 9.59 \| 20.5% \| 98.5% | 9.43 \| 14.1% \| 88.8% | 5.43 \| 24.1% \| 98.0% | 8.33 \| 14.5% \| 69.3% |
| **twostage_a1_0.99_a2_0.15** | 6.92 \| 26.4% \| 91.5% | 8.93 \| 14.6% \| 87.2% | 4.23 \| 26.2% \| 85.5% | 6.92 \| 16.2% \| 64.5% |
| **twostage_a1_0.99_a2_0.2** | 6.08 \| 29.6% \| 90.0% | 8.31 \| 15.0% \| 84.8% | 3.88 \| 28.4% \| 84.5% | 6.44 \| 17.1% \| 63.7% |
| **twostage_a1_0.99_a2_0.3** | 5.15 \| 32.2% \| 83.0% | 6.73 \| 17.2% \| 80.2% | 3.21 \| 31.2% \| 76.0% | 5.43 \| 18.6% \| 58.2% |
| **top10_stage2_a2_0.15** | 6.94 \| 26.4% \| 91.5% | 9.79 \| 14.4% \| 93.2% | 4.23 \| 26.2% \| 85.5% | 8.08 \| 16.0% \| 74.0% |
| **top10_stage2_a2_0.2** | 6.09 \| 29.6% \| 90.0% | 9.72 \| 14.5% \| 93.2% | 3.90 \| 28.2% \| 84.5% | 7.73 \| 16.0% \| 71.8% |
| **top10_stage2_a2_0.3** | 5.15 \| 32.2% \| 83.0% | 8.52 \| 15.6% \| 89.6% | 3.28 \| 30.8% \| 77.0% | 6.88 \| 17.4% \| 68.8% |
| **top10_stage2_a2_0.4** | 3.85 \| 36.9% \| 71.0% | 6.20 \| 18.5% \| 77.8% | 2.78 \| 34.2% \| 73.5% | 5.89 \| 18.3% \| 60.3% |
