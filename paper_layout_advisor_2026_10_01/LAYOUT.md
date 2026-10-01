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

- **HotpotQA:** Stage 1 `.99` đạt **71.00 EM / 82.52 F1**. Two-Stage `.99/.15` đạt **64.00/76.96**, cao hơn CCE **3.94 điểm F1**, đồng thời giảm compute proxy **21.03 → 16.43 TFLOPs**. So với Stage 1, pruning giảm **29.6%** proxy nhưng mất **5.56 điểm F1**.
- **MMQA:** CCE đang có generation F1 cao nhất **54.08%**. Hybrid `α₂=.40` tăng Selection F1 từ CCE **14.2%** lên **29.7%**, nhưng generation F1 là **51.75%**. Fixed Top-5 đạt Selection F1 **36.6%** và generation F1 **52.40%**.
- **TAT-QA:** Stage 1 `.99` đạt **39.00 EM / 58.84 F1**; Fixed Top-5 đạt **37.00/57.30**. Two-Stage `.99/.15` đạt **37.00/53.89**: hơn ba literature rows về EM, nhưng thấp hơn Top-5 về F1.
- **WebQA:** Hybrid `α₂=.15` đạt F1 **30.02%**, so với TRAQ **29.29%**; context giảm **24.68 → 8.08 chunks**, proxy **54.66 → 13.83 TFLOPs**, recall giảm **88.6% → 73.7%**.
- **Overall:** Stage 1 `.99` đạt **41.75 EM / 55.28 F1**. Trong nhóm Two-Stage/hybrid, hybrid `.15` đạt **39.75/52.85**, so với Fixed Top-5 **38.75/52.25**.
- Kết quả hiện tại cho thấy pruning có điểm hữu ích về chi phí, nhưng việc tăng Selection F1 chưa chuyển thành downstream gain đồng đều giữa datasets.

## 9. Đang hoàn thiện để cải thiện kết quả

- **Bảo toàn multi-support evidence:** điều chỉnh Stage 2 với recall/all-support constraint trên calibration, tập trung vào các query mất complementary support hoặc table operands.
- **Conditioning theo modality/rank:** kiểm tra banks riêng cho text/table/image và rank bins để hạn chế trộn các nhóm có score distribution khác nhau; theo dõi bank size và mức ổn định.
- **Chọn operating point trên calibration:** khảo sát alpha và calibration-size để cân bằng answer quality với context cost; giữ Stage 1 permissive và Fixed Top-10 làm mốc so sánh.
- **Tách tác động budget và pruning:** bổ sung Fixed Top-10 và comparison cùng context budget để xác định phần lợi ích đến từ giảm số chunks, phần đến từ chọn evidence.
- **Hoàn thiện log và đo chi phí đồng nhất:** đối chiếu split/context IDs của bảng mới, input/image tokens và timing; kiểm tra các điểm latency cao ở MMQA/WebQA.
- **Hoàn thiện bảng và độ ổn định:** bổ sung downstream cho Stage 1 `.90`, selection cho Two-Stage `α₂=.40`, empty/any/all-support và paired bootstrap.
- **Hoàn thiện phân tích thống kê:** làm rõ điều kiện của p-values, BH/BY và selective procedure hai stages để nối diễn giải phương pháp với các error rates đo được.

Chi tiết bảng hiện tại và kế hoạch kỹ thuật nằm trong [layout đầy đủ](../PAPER_LAYOUT_UNCERTAINTY_AWARE_RAG.md#16-cập-nhật-layout-từ-bản-latex-ngày-01-10-2026).
