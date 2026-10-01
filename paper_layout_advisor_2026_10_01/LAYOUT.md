# Dàn bài bài báo: Two-Stage Conformal Context Selection for RAG

**Tác giả:** Hung Le

**Ngày cập nhật:** 01/10/2026

**Nội dung:** ý chính để phát triển bài báo, cách trình bày thí nghiệm, số hiện tại và hướng cải thiện.

## 1. Abstract

- Nêu vấn đề chọn context sau retrieval: giữ quá ít chunks có thể thiếu bằng chứng; giữ quá nhiều làm tăng nhiễu và chi phí sinh câu trả lời.
- Giới thiệu Two-Stage Conformal Selection: kiểm tra candidate bằng false-reference bank, sau đó prune bằng support-reference bank.
- Tóm tắt đánh giá trên HotpotQA, MMQA, TAT-QA và WebQA bằng ba nhóm chỉ số: chất lượng chọn evidence, chất lượng câu trả lời và chi phí tính toán.
- Nhấn mạnh khả năng điều chỉnh precision–recall và chất lượng–chi phí theo từng mức pruning.

## 2. Introduction

- Phân biệt retrieval tạo candidate pool và context selection quyết định chunks nào được đưa vào LLM. Bài báo tập trung vào bước thứ hai.
- Trình bày hạn chế của Fixed Top-k: một ngân sách cố định khó đáp ứng cả câu hỏi đơn giản và câu hỏi cần nhiều evidence.
- Đặt câu hỏi nghiên cứu: có thể giảm false chunks mà vẫn giữ đủ evidence để trả lời, và việc giảm context có cải thiện downstream hoặc chi phí không?
- Giới thiệu đóng góp dự kiến: hai reference banks, quyết định multiple testing theo query và đánh giá đồng thời selection/downstream/cost.

## 3. Related Work

### 3.1. CCE

- Trình bày xử lý sau retrieval trong code hiện tại: mỗi support query–chunk pair đóng góp nonconformity `1 − cosine`; lấy quantile từ calibration và giữ chunks vượt similarity cutoff.
- Support là chunk có nhãn chứa bằng chứng cho câu trả lời. Cách hiệu chuẩn này hướng tới bảo toàn support coverage.
- Liên hệ kết quả: MMQA/WebQA còn trung bình **18.40/23.36 chunks**, recall **91.0/86.3%**, nhưng precision chỉ **7.7/6.5%**.

### 3.2. CONFLARE

- Trình bày per-query calibration: lấy distance của support có cosine cao nhất trong mỗi calibration query; test giữ chunks dưới calibrated distance cutoff.
- Việc lấy best support tạo một record cho mỗi query nhưng không đại diện toàn bộ evidence của câu hỏi nhiều bước.
- Liên hệ kết quả HotpotQA: giảm context từ CCE **8.61** xuống **7.05 chunks**, precision vẫn **21.0%**, recall giảm **90.5% → 74.0%** và generation F1 giảm **73.02% → 68.78%**.

### 3.3. TRAQ

- Trình bày best-support retrieval score của từng calibration query; Bonferroni allocation dùng `α_R = α/2`, lấy lower quantile và giữ chunks đạt cutoff.
- Với `α=.10`, retrieval allocation `.05` tạo điểm hoạt động thiên về coverage.
- Liên hệ kết quả: MMQA/WebQA giữ **19.50/24.68 chunks**, recall **94.2/88.6%**, precision **7.5/6.3%**.

### 3.4. Research Gap

- Một cutoff đủ thấp để giữ support cũng cho nhiều false chunks đi qua khi hai nhóm cosine scores chồng lấn. Trên MMQA/WebQA, ba selectors còn khoảng **18–25 chunks/query** nhưng precision chỉ **6–8%**: coverage cao đi kèm context chứa nhiều nhiễu.
- Siết cutoff không tự động làm evidence tốt hơn. CONFLARE trên HotpotQA giảm số chunks nhưng không tăng precision, đồng thời mất recall và downstream F1.
- Khoảng trống cần giải quyết là **tách admission và pruning**, để điều chỉnh độ rộng của context và mức loại nhiễu riêng, thay vì chỉ dịch một support-derived cutoff.
- False bank cung cấp điểm so sánh với evidence không hỗ trợ; support bank kiểm tra score của candidate có phù hợp với evidence hỗ trợ hay không. Lợi ích được đánh giá bằng evidence giữ lại, câu trả lời và chi phí, theo từng dataset.

## 4. Methodology

### 4.1. Tổng quan

- Pipeline: candidate pool và cosine scores cố định → lookup reference banks → Stage 1 admission → Stage 2 pruning → giữ retrieval order → generator.
- Hai tham số `α₁` và `α₂` điều chỉnh riêng mức admission và mức pruning.
- Hình minh họa dự kiến thể hiện candidates/support/false trước và sau từng stage.

### 4.2. Xây reference banks

- Chỉ đọc calibration queries và nhãn evidence; calibration và test tách theo query ID.
- Gom cosine scores của chunks có nhãn false vào **false-reference bank**; gom scores có nhãn support vào **support-reference bank**; bỏ nhãn unknown khi xây banks.
- Sort scores và lưu bank size, dataset, cấu hình scoring và candidate depth. Báo riêng số calibration queries và số candidate scores trong mỗi bank.
- Đánh giá pooled banks và banks điều kiện theo modality/rank; quy định cách xử lý bank thiếu hoặc quá nhỏ trước khi đánh giá test.

### 4.3. Stage 1: candidate admission

- Tính upper-tail p-value đối với false bank: cosine cao so với false scores cho p-value nhỏ.
- Chạy BH hoặc BY trên candidates của cùng query; candidate bị reject false-null được admit vào context trung gian.
- Dùng mức `α₁` permissive để khảo sát high-recall operating point; đo recall và empty rate thực tế.

### 4.4. Stage 2: support-based pruning

- Tính lower-tail p-value đối với support bank: score thấp bất thường so với support scores cho p-value nhỏ.
- Chạy multiple testing trên tập qua Stage 1; candidate bị reject support-null được prune.
- Khảo sát ảnh hưởng của `α₂` đến precision, recall, all-support coverage và downstream. Phân tích assumptions của BH/BY và việc Stage 2 nhận tập đã được Stage 1 chọn.

### 4.5. Variants và ablations

- **Stage 1 only:** đo admission khi không prune.
- **Two-Stage:** Stage 1 rồi Stage 2.
- **Fixed Top-10 + Stage 2:** đo tác dụng pruning khi đầu vào có ngân sách cố định.
- **BH/BY, pooled/modality-aware, candidate depth và context cap:** xác định yếu tố tạo ra thay đổi và mức evidence mất đi.

## 5. Experimental Setup

- Datasets: HotpotQA — text/multi-hop; MMQA — text/table/image; TAT-QA — table-text reasoning; WebQA — text/image.
- Protocol dự kiến cho bảng chính: **1,000 calibration và 100 test queries mỗi dataset**, cùng query IDs, candidate pool, Jina-v4 cosine và generator giữa các methods.
- Baselines: Fixed Top-5/10/20, CCE, CONFLARE và TRAQ. Literature selectors dùng `α=.10`; TRAQ retrieval dùng `α_R=.05`.
- Configurations: Stage 1 `α₁∈{.10,.90,.99}`; Two-Stage cố định `α₁=.99`, sweep `α₂∈{.15,.20,.30,.40}`; hybrid dùng cùng sweep Stage 2.
- Generator: Qwen2-VL-7B, greedy decoding, tối đa 24 new tokens; text/table đưa vào context, image đưa vào vision processor.
- Metrics: kept chunks, support precision/recall, Selection F1, empty, any/all-support; downstream EM/token F1; input tokens, latency và approximate TFLOPs.
- Ghi rõ runtime, cách timing và công thức compute proxy; chọn operating point bằng calibration/validation và báo paired confidence intervals cho downstream differences.

## 6. Các bảng và hình thí nghiệm dự kiến

| Bảng | Dự kiến trình bày | Câu hỏi cần trả lời |
|---|---|---|
| 1. Dữ liệu và calibration | Số queries, candidate pool, support/modality distribution, bank sizes | Mỗi dataset có bao nhiêu evidence và calibration records? |
| 2. Context selection | Kept, precision, recall, Selection F1, empty, any/all-support cho các methods | Pruning loại nhiễu hay làm mất evidence? |
| 3. Downstream | EM/F1 trên bốn datasets và Overall | Selection tốt hơn có giúp trả lời tốt hơn? |
| 4. Chi phí | Input tokens, latency, approximate TFLOPs cùng downstream F1 | Đổi chất lượng lấy chi phí ở mức nào? |
| 5. Ablations | Stage 1, Two-Stage, hybrid; BH/BY; pooled/conditional banks | Thành phần nào tạo lợi ích, thành phần nào gây mất recall? |
| 6. Độ ổn định | Calibration-size/alpha sweep và confidence intervals | Kết quả nhạy với calibration và operating point ra sao? |

- Hình pipeline: cách xây hai banks và quyết định admit/prune.
- Hình precision–recall: thể hiện toàn sweep, kèm empty/all-support để đọc mức mất evidence.
- Hình quality–cost: generation F1 theo TFLOPs hoặc input tokens, thể hiện Stage 1, Two-Stage, hybrid và baselines.
- Case study: một query prune đúng false chunks, một query mất complementary support, một query multimodal khó.

## 7. Discussion và Conclusion

- Giải thích khác biệt giữa datasets: multi-hop cần đủ complementary evidence; table reasoning cần giữ operands; image context có chi phí khác text.
- Phân biệt Selection F1 và generation F1: evidence ít nhiễu hơn chưa chắc giúp answer quality nếu evidence cần thiết bị mất.
- Kết luận quanh khả năng tạo các operating points khác nhau và lựa chọn mức pruning phù hợp với mục tiêu quality/cost.

## 8. Số hiện tại

Các số dưới đây lấy từ bảng thí nghiệm cập nhật ngày 01/10/2026, trình bày ngắn để định hướng phần Results.

**Cách đọc bảng:** mỗi dataset có toàn bộ 15 method/configurations trong bảng số hiện tại. **In đậm: tốt nhất**; <ins>gạch chân: tốt nhì</ins>, xếp theo hai giá trị khác nhau của từng cột trong cùng dataset; các hàng bằng nhau được đánh dấu cùng hạng. ↑ là cao hơn tốt hơn; ↓ là thấp hơn về context/cost. Kept/cost cần đọc cùng recall và downstream. P/R/Selection F1 đo support chunks; EM/downstream F1 đo câu trả lời. **—** là số chưa có trong bảng nguồn.

### 8.1. HotpotQA: toàn bộ kết quả hiện tại

| Method | Kept ↓ | Support P (%) ↑ | Support R (%) ↑ | Selection F1 (%) ↑ | EM (%) ↑ | Downstream F1 (%) ↑ | Latency (s) ↓ | Approx. TFLOPs ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed Top-5 | 4.95 | 33.7 | 83.5 | <ins>48.1</ins> | 61.00 | 71.40 | <ins>0.22</ins> | 11.82 |
| CCE (α=.10) | 8.61 | 21.0 | 90.5 | 34.1 | 62.00 | 73.02 | 0.24 | 21.03 |
| CONFLARE (α=.10) | 7.05 | 21.0 | 74.0 | 32.7 | 57.00 | 68.78 | **0.21** | 17.35 |
| TRAQ (α=.10) | 8.17 | 21.4 | 87.5 | 34.4 | 62.00 | 73.52 | 0.23 | 19.91 |
| Stage 1 (α₁=.10) | **0.89** | **55.1** | 24.5 | 33.9 | 31.00 | 41.57 | 0.26 | **2.80** |
| Stage 1 (α₁=.90) | 8.11 | 22.3 | 90.5 | 35.8 | — | — | — | — |
| Stage 1 (α₁=.99) | 9.59 | 20.5 | **98.5** | 33.9 | **71.00** | **82.52** | 0.27 | 23.34 |
| Two-Stage (α₁=.99, α₂=.15) | 6.92 | 26.4 | <ins>91.5</ins> | 41.0 | <ins>64.00</ins> | <ins>76.96</ins> | 0.24 | 16.43 |
| Two-Stage (α₁=.99, α₂=.20) | 6.08 | 29.6 | 90.0 | 44.5 | 63.00 | 75.11 | 0.23 | 14.39 |
| Two-Stage (α₁=.99, α₂=.30) | 5.15 | 32.2 | 83.0 | 46.4 | 60.00 | 71.84 | 0.23 | 12.36 |
| Two-Stage (α₁=.99, α₂=.40) | — | — | — | — | 58.00 | 68.67 | 0.23 | <ins>9.34</ins> |
| Top-10 + S2 (α₂=.15) | 6.94 | 26.4 | <ins>91.5</ins> | 40.9 | 63.00 | 75.96 | 0.25 | 16.47 |
| Top-10 + S2 (α₂=.20) | 6.09 | 29.6 | 90.0 | 44.5 | 63.00 | 75.11 | 0.23 | 14.40 |
| Top-10 + S2 (α₂=.30) | 5.15 | 32.2 | 83.0 | 46.4 | 60.00 | 71.84 | 0.23 | 12.36 |
| Top-10 + S2 (α₂=.40) | <ins>3.85</ins> | <ins>36.9</ins> | 71.0 | **48.5** | 58.00 | 68.67 | 0.23 | <ins>9.34</ins> |

### 8.2. MMQA: toàn bộ kết quả hiện tại

| Method | Kept ↓ | Support P (%) ↑ | Support R (%) ↑ | Selection F1 (%) ↑ | EM (%) ↑ | Downstream F1 (%) ↑ | Latency (s) ↓ | Approx. TFLOPs ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed Top-5 | <ins>5.00</ins> | <ins>24.0</ins> | 77.4 | **36.6** | <ins>48.00</ins> | <ins>52.40</ins> | **0.24** | <ins>14.38</ins> |
| CCE (α=.10) | 18.40 | 7.7 | <ins>91.0</ins> | 14.2 | **49.00** | **54.08** | 0.44 | 47.45 |
| CONFLARE (α=.10) | 18.05 | 7.8 | 90.3 | 14.4 | 47.00 | 52.08 | 0.42 | 46.45 |
| TRAQ (α=.10) | 19.50 | 7.5 | **94.2** | 13.9 | <ins>48.00</ins> | 52.12 | 0.45 | 50.68 |
| Stage 1 (α₁=.10) | **0.59** | **38.6** | 14.8 | 21.4 | 23.00 | 29.24 | 0.27 | **2.84** |
| Stage 1 (α₁=.90) | 6.85 | 16.7 | 74.2 | 27.3 | — | — | — | — |
| Stage 1 (α₁=.99) | 9.43 | 14.1 | 85.8 | 24.2 | 47.00 | 52.10 | 1.96 | 53.16 |
| Two-Stage (α₁=.99, α₂=.15) | 9.26 | 14.4 | 85.8 | 24.7 | 47.00 | 52.03 | 0.95 | 34.06 |
| Two-Stage (α₁=.99, α₂=.20) | 9.24 | 14.4 | 85.8 | 24.7 | 47.00 | 52.03 | 0.85 | 31.12 |
| Two-Stage (α₁=.99, α₂=.30) | 8.30 | 15.3 | 81.9 | 25.8 | 47.00 | 51.93 | 0.64 | 24.45 |
| Two-Stage (α₁=.99, α₂=.40) | — | — | — | — | 42.00 | 47.29 | 0.34 | 15.82 |
| Top-10 + S2 (α₂=.15) | 9.79 | 14.4 | <ins>91.0</ins> | 24.9 | 47.00 | 51.53 | 0.32 | 24.12 |
| Top-10 + S2 (α₂=.20) | 9.72 | 14.5 | <ins>91.0</ins> | 25.0 | 47.00 | 51.53 | 0.32 | 23.99 |
| Top-10 + S2 (α₂=.30) | 8.52 | 15.6 | 85.8 | 26.4 | 45.00 | 49.03 | 0.30 | 21.54 |
| Top-10 + S2 (α₂=.40) | 6.20 | 18.5 | 74.2 | <ins>29.7</ins> | 47.00 | 51.75 | <ins>0.26</ins> | 16.42 |

### 8.3. TAT-QA: toàn bộ kết quả hiện tại

| Method | Kept ↓ | Support P (%) ↑ | Support R (%) ↑ | Selection F1 (%) ↑ | EM (%) ↑ | Downstream F1 (%) ↑ | Latency (s) ↓ | Approx. TFLOPs ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed Top-5 | 4.37 | 29.7 | <ins>97.0</ins> | <ins>45.5</ins> | <ins>37.00</ins> | <ins>57.30</ins> | 0.42 | 8.08 |
| CCE (α=.10) | 5.01 | 24.6 | 91.8 | 38.8 | 28.00 | 49.37 | <ins>0.36</ins> | 9.10 |
| CONFLARE (α=.10) | 4.64 | 25.0 | 86.6 | 38.8 | 25.00 | 44.52 | **0.35** | 8.58 |
| TRAQ (α=.10) | 5.09 | 24.6 | 93.3 | 38.9 | 28.00 | 47.95 | <ins>0.36</ins> | 9.26 |
| Stage 1 (α₁=.10) | **0.33** | **69.7** | 17.2 | 27.6 | 9.00 | 18.19 | 0.46 | **1.63** |
| Stage 1 (α₁=.90) | 4.53 | 25.6 | 86.6 | 39.5 | — | — | — | — |
| Stage 1 (α₁=.99) | 5.43 | 24.1 | **97.8** | 38.7 | **39.00** | **58.84** | 0.42 | 9.86 |
| Two-Stage (α₁=.99, α₂=.15) | 4.23 | 26.2 | 82.8 | 39.8 | <ins>37.00</ins> | 53.89 | 0.40 | 8.31 |
| Two-Stage (α₁=.99, α₂=.20) | 3.90 | 28.2 | 82.1 | 42.0 | 34.00 | 51.56 | 0.40 | 7.88 |
| Two-Stage (α₁=.99, α₂=.30) | 3.28 | 30.8 | 75.4 | 43.7 | 29.00 | 46.35 | 0.42 | 6.76 |
| Two-Stage (α₁=.99, α₂=.40) | — | — | — | — | 26.00 | 43.88 | 0.42 | 5.92 |
| Top-10 + S2 (α₂=.15) | 4.23 | 26.2 | 82.8 | 39.9 | <ins>37.00</ins> | 53.89 | 0.40 | 7.90 |
| Top-10 + S2 (α₂=.20) | 3.90 | 28.2 | 82.1 | 42.0 | 34.00 | 51.56 | 0.40 | 7.58 |
| Top-10 + S2 (α₂=.30) | 3.28 | 30.8 | 75.4 | 43.7 | 29.00 | 46.35 | 0.41 | 6.73 |
| Top-10 + S2 (α₂=.40) | <ins>2.78</ins> | <ins>34.2</ins> | 70.9 | **46.1** | 26.00 | 43.88 | 0.42 | <ins>5.91</ins> |

### 8.4. WebQA: toàn bộ kết quả hiện tại

| Method | Kept ↓ | Support P (%) ↑ | Support R (%) ↑ | Selection F1 (%) ↑ | EM (%) ↑ | Downstream F1 (%) ↑ | Latency (s) ↓ | Approx. TFLOPs ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed Top-5 | <ins>5.00</ins> | <ins>23.0</ins> | 65.7 | **34.1** | 9.00 | 27.90 | **0.27** | <ins>8.86</ins> |
| CCE (α=.10) | 23.36 | 6.5 | <ins>86.3</ins> | 12.1 | <ins>12.00</ins> | 28.89 | 0.47 | 50.99 |
| CONFLARE (α=.10) | 21.60 | 6.8 | 83.4 | 12.6 | <ins>12.00</ins> | 28.35 | 0.43 | 46.36 |
| TRAQ (α=.10) | 24.68 | 6.3 | **88.6** | 11.8 | **13.00** | 29.29 | 0.49 | 54.66 |
| Stage 1 (α₁=.10) | **0.58** | **34.5** | 11.4 | 17.1 | 7.00 | 23.96 | **0.27** | **2.74** |
| Stage 1 (α₁=.90) | 6.43 | 16.3 | 60.0 | 25.6 | — | — | — | — |
| Stage 1 (α₁=.99) | 8.33 | 14.5 | 69.1 | 24.0 | 10.00 | 27.67 | 3.77 | 55.07 |
| Two-Stage (α₁=.99, α₂=.15) | 7.38 | 15.6 | 65.7 | 25.2 | 10.00 | 26.84 | 1.46 | 27.16 |
| Two-Stage (α₁=.99, α₂=.20) | 7.08 | 15.5 | 62.9 | 24.9 | 9.00 | 26.08 | 1.25 | 23.37 |
| Two-Stage (α₁=.99, α₂=.30) | 6.47 | 16.7 | 61.7 | 26.3 | 9.00 | 25.71 | 0.79 | 16.88 |
| Two-Stage (α₁=.99, α₂=.40) | — | — | — | — | 9.00 | 27.93 | 0.68 | 13.69 |
| Top-10 + S2 (α₂=.15) | 8.08 | 16.0 | 73.7 | 26.2 | <ins>12.00</ins> | **30.02** | 0.38 | 13.83 |
| Top-10 + S2 (α₂=.20) | 7.73 | 16.0 | 70.9 | 26.2 | 9.00 | 28.09 | 0.38 | 13.23 |
| Top-10 + S2 (α₂=.30) | 6.88 | 17.4 | 68.6 | 27.8 | 11.00 | <ins>29.88</ins> | 0.33 | 11.70 |
| Top-10 + S2 (α₂=.40) | 5.89 | 18.3 | 61.7 | <ins>28.3</ins> | 9.00 | 27.88 | <ins>0.30</ins> | 9.70 |

### 8.5. Overall downstream

| Method | Overall EM (%) ↑ | Overall downstream F1 (%) ↑ |
|---|---:|---:|
| Fixed Top-5 | 38.75 | 52.25 |
| CCE (α=.10) | 37.75 | 51.34 |
| CONFLARE (α=.10) | 35.25 | 48.43 |
| TRAQ (α=.10) | 37.75 | 50.72 |
| Stage 1 (α₁=.10) | 17.50 | 28.24 |
| Stage 1 (α₁=.90) | — | — |
| Stage 1 (α₁=.99) | **41.75** | **55.28** |
| Two-Stage (α₁=.99, α₂=.15) | 39.50 | 52.43 |
| Two-Stage (α₁=.99, α₂=.20) | 38.25 | 51.20 |
| Two-Stage (α₁=.99, α₂=.30) | 36.25 | 48.96 |
| Two-Stage (α₁=.99, α₂=.40) | 33.75 | 46.95 |
| Top-10 + S2 (α₂=.15) | <ins>39.75</ins> | <ins>52.85</ins> |
| Top-10 + S2 (α₂=.20) | 38.25 | 51.57 |
| Top-10 + S2 (α₂=.30) | 36.25 | 49.28 |
| Top-10 + S2 (α₂=.40) | 35.00 | 48.05 |

- Overall giữ số được báo trong bảng cung cấp. F1 của Two-Stage `α₂=.40` được báo là **46.95%**; trung bình bốn ô dataset đã làm tròn là 46.9425%. Số tổng hợp sẽ được đối chiếu với log chưa làm tròn.
- Bảng còn thiếu downstream cho Stage 1 `.90` và selection cho Two-Stage `α₂=.40`; hai cấu hình vẫn được giữ đủ hàng, với các ô tương ứng là **—**.
- Dữ liệu bảng: [các số từ bản LaTeX cập nhật](../paper_assets/layout_results_supplied_2026_10_01.json).

### 8.6. Diễn giải số hiện tại

- **HotpotQA:** Stage 1 `.99` đạt **71.00 EM / 82.52 F1**. Two-Stage `.99/.15` đạt **64.00/76.96**, cao hơn CCE **3.94 điểm F1**, đồng thời giảm compute proxy **21.03 → 16.43 TFLOPs**. So với Stage 1, pruning giảm **29.6%** proxy nhưng mất **5.56 điểm F1**.
- **MMQA:** CCE đang có generation F1 cao nhất **54.08%**. Hybrid `α₂=.40` tăng Selection F1 từ CCE **14.2%** lên **29.7%**, nhưng generation F1 là **51.75%**. Fixed Top-5 đạt Selection F1 **36.6%** và generation F1 **52.40%**.
- **TAT-QA:** Stage 1 `.99` đạt **39.00 EM / 58.84 F1**; Fixed Top-5 đạt **37.00/57.30**. Two-Stage `.99/.15` đạt **37.00/53.89**: hơn ba literature rows về EM, nhưng thấp hơn Top-5 về F1.
- **WebQA:** Hybrid `α₂=.15` đạt F1 **30.02%**, so với TRAQ **29.29%**; context giảm **24.68 → 8.08 chunks**, proxy **54.66 → 13.83 TFLOPs**, recall giảm **88.6% → 73.7%**.
- **Overall:** Stage 1 `.99` đạt **41.75 EM / 55.28 F1**. Trong nhóm Two-Stage/hybrid, hybrid `.15` đạt **39.75/52.85**, so với Fixed Top-5 **38.75/52.25**.
- Kết quả hiện tại cho thấy pruning có điểm hữu ích về chi phí, nhưng việc tăng Selection F1 chưa chuyển thành downstream gain đồng đều giữa datasets.

## 9. Đang hoàn thiện để cải thiện kết quả

**Đang làm việc cùng anh Hưng để cải thiện phương pháp**, tập trung bảo toàn evidence khi prune và cải thiện đồng thời chất lượng chọn chunks, downstream và chi phí.

- **Bảo toàn multi-support evidence:** điều chỉnh Stage 2 với recall/all-support constraint trên calibration, tập trung vào các query mất complementary support hoặc table operands.
- **Conditioning theo modality/rank:** kiểm tra banks riêng cho text/table/image và rank bins để hạn chế trộn các nhóm có score distribution khác nhau; theo dõi bank size và mức ổn định.
- **Chọn operating point trên calibration:** khảo sát alpha và calibration-size để cân bằng answer quality với context cost; giữ Stage 1 permissive và Fixed Top-10 làm mốc so sánh.
- **Tách tác động budget và pruning:** bổ sung Fixed Top-10 và comparison cùng context budget để xác định phần lợi ích đến từ giảm số chunks, phần đến từ chọn evidence.
- **Hoàn thiện log và đo chi phí đồng nhất:** đối chiếu split/context IDs của bảng mới, input/image tokens và timing; kiểm tra các điểm latency cao ở MMQA/WebQA.
- **Hoàn thiện bảng và độ ổn định:** bổ sung downstream cho Stage 1 `.90`, selection cho Two-Stage `α₂=.40`, empty/any/all-support và paired bootstrap.
- **Hoàn thiện phân tích thống kê:** làm rõ điều kiện của p-values, BH/BY và selective procedure hai stages để nối diễn giải phương pháp với các error rates đo được.

Chi tiết bảng hiện tại và kế hoạch kỹ thuật nằm trong [layout đầy đủ](../PAPER_LAYOUT_UNCERTAINTY_AWARE_RAG.md#16-cập-nhật-layout-từ-bản-latex-ngày-01-10-2026).
