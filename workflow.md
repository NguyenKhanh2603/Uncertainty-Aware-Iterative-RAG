# Quy Trình Hoạt Động (Workflow) Của Hệ Thống Uncertainty-Aware RAG

Workflow bạn mô tả là **HOÀN TOÀN CHÍNH XÁC**. Để hệ thống hóa lại toàn bộ, dưới đây là tài liệu mô tả chi tiết từng bước hoạt động của hệ thống, bao gồm các công thức toán học và cơ chế nội bộ (như Hybrid Retrieval, Attention Saliency, và Attention Masking).

---

## 1. Khởi tạo (Initial Retrieval)
Khi người dùng đặt câu hỏi (Query), hệ thống sử dụng **Hybrid Retriever** để lấy ra **Top 10 Chunks** (nhằm giới hạn VRAM trên T4 GPU thay vì lấy tất cả).

---

## 2. Vòng Lặp Chính (Iterative RAG Loop)

### Bước 2.1: Sinh mẫu (Sampling) & Attention Saliency (Bộ lọc thô)
- **Hành động:** Hệ thống đưa câu hỏi và 10 chunks vào Mô hình LLM (Qwen2-VL) để sinh ra $N$ mẫu trả lời (samples) độc lập.
- **Attention Saliency:** Ngay trong quá trình sinh mẫu (forward pass), hệ thống âm thầm thu thập ma trận Attention của mô hình. Bằng cách dựa vào các thẻ định vị (Position tokens `<chunk_i>`), ta đo lường sự chú ý của LLM dành cho từng chunk.
- **Công thức Saliency:**
  $$ Saliency(C_k) = \sum_{t \in Output} \sum_{i \in C_k} A(t, i) $$
  *(Trong đó $A(t,i)$ là trọng số attention mô hình gán cho token $i$ thuộc chunk $k$ khi sinh ra token đầu ra $t$)*
- **Xử lý:** Nếu $Saliency(C_k) < 0.2$ (ngưỡng), chunk đó bị xem là "rác" (LLM không thèm đọc tới) và lập tức bị **loại bỏ ngay (Pruned)** mà không cần tốn thêm tài nguyên.

### Bước 2.2: Phân tích Claim & Gom Cụm (Clustering)
- Các mẫu sinh ra được trích xuất thành các sự kiện nguyên tử (Atomic Claims).
- Gom các mẫu có nội dung giống nhau thành các Cụm (Concepts). Có áp dụng **Factual Anchor Check** (tách riêng các cụm khác số liệu) và **ABSTAIN Normalization** (gộp chung các câu "Tôi không biết").

### Bước 2.3: Tính toán 3 Tín Hiệu (Uncertainty & Evidence)
Hệ thống tính toán 3 chỉ số độc lập để đánh giá độ tin cậy của bộ Context hiện tại:

1. **Độ bất định Ngữ nghĩa - $SE_{semantic}$ (Tín hiệu A):** Đo lường sự bất đồng giữa các câu trả lời.
   $$ SE_{semantic} = - \sum_{c \in Concepts} P(c) \log_2 P(c) $$
   *(Nếu $SE$ cao $\rightarrow$ LLM đang đưa ra nhiều câu trả lời khác nhau $\rightarrow$ Cần tìm thêm thông tin).*

2. **Độ nhiễu cục bộ - $U_{token}$ (Tín hiệu B):** Đo lường sự tự tin của LLM đối với từng từ ngữ nó nói ra.
   $$ U_{token} = \frac{1}{N} \sum_{i=1}^N \left( - \frac{1}{|K_i|} \sum_{j \in K_i} \log_2 P(y_{i,j}) \right) $$
   *(Nếu $U$ cao dù $SE$ thấp $\rightarrow$ LLM đồng thuận nhưng ngập ngừng $\rightarrow$ Có thể trong ngữ cảnh đang có thông tin mâu thuẫn gây nhiễu).*

3. **Tỉ lệ Bằng chứng - $Evidence Ratio$ (Tín hiệu C):** Dùng mô hình NLI (DeBERTa) quét xem Context có thực sự chứa nội dung chứng minh cho Câu trả lời hay không.
   $$ Evidence\_Ratio = \frac{|Supported\_Claims|}{|Total\_Claims|} $$
   *(Nếu Entailment Score < 0.75 $\rightarrow$ Trả lời bị coi là Ảo giác (Hallucination) $\rightarrow$ Cần lấy thêm thông tin).*

### Bước 2.4: Routing (Định tuyến 3-Signal)
Dựa vào 3 chỉ số trên cùng các ngưỡng (Threshold $\tau$), Router sẽ quyết định:

- **STOP (Dừng):** Khi $(SE \le \tau_{se}) \land (U \le \tau_{u}) \land (Evidence \ge \tau_{ev})$. Mọi thứ hoàn hảo, sinh kết quả cuối.
- **RETRIEVE (Tìm thêm):** Khi $(SE > \tau_{se})$ HOẶC $(Evidence < \tau_{ev})$. Hệ thống thiếu kiến thức hoặc kiến thức bị thiếu bằng chứng thực tế. (Chuyển sang Bước 3).
- **PRUNE (Lọc tinh):** Khi $(SE \le \tau_{se}) \land (Evidence \ge \tau_{ev})$ NHƯNG $(U > \tau_{u})$. Nghĩa là đáp án có vẻ đúng, nhưng trong đống Chunk đang có những đoạn văn gây mâu thuẫn nội bộ. (Chuyển sang Bước 4).

---

## 3. Pha Mở Rộng Tìm Kiếm (RETRIEVE Phase)
Khi thiếu kiến thức, thay vì tìm kiếm lại câu hỏi cũ, hệ thống dùng **Adaptive Query Expansion**:
1. LLM phân tích Giả thuyết (Hypothesis) hiện tại.
2. Sinh ra **3 từ khoá/câu hỏi biến thể (Query Variants)**.
3. Chạy **Hybrid Retrieval** cho cả 3 biến thể:
   - **Vector Search (Dense):** Dùng `SentenceTransformers` để tìm theo ý nghĩa (Semantic).
   - **BM25 (Sparse):** Dùng thuật toán `BM25Okapi` đếm tần suất từ vựng, bắt keyword chính xác (Exact match).
4. **Trộn kết quả (RRF - Reciprocal Rank Fusion):**
   $$ RRF\_Score(d) = \frac{\alpha}{60 + Rank_{dense}(d)} + \frac{1 - \alpha}{60 + Rank_{sparse}(d)} $$
   *(Với $\alpha=0.5$. Tài liệu nào nằm Top ở cả 2 phương pháp sẽ được đẩy lên cao nhất).*
5. Thêm các Chunk mới vào Context, loại bỏ trùng lặp $\rightarrow$ Quay lại Bước 2.1.

---

## 4. Pha Lọc Nhiễu (PRUNE Phase - Attention Masking)
Khi hệ thống rơi vào trạng thái nhiễu ($U_{token}$ cao), hệ thống kích hoạt **Attention Masking (Bộ lọc tinh)**.

Thay vì LOO (Leave-One-Out) truyền thống tốn $K$ lần gọi LLM, Attention Masking hoạt động song song trong 1 lần tính toán:
1. Đưa toàn bộ Query và $K$ chunks vào LLM.
2. Bằng cách thao tác trên Custom Attention Mask bên trong Pytorch, hệ thống **ép điểm attention của Chunk $i$ về 0** (Mô phỏng: Điều gì xảy ra nếu không có Chunk $i$?).
3. Mô hình trả ra $K$ đáp án khác nhau tương ứng với việc thiếu đi từng Chunk.
4. Tính lại độ bất định $SE_{semantic}$ cho mỗi kịch bản. 
   - Nếu "bỏ Chunk $i$ ra" mà làm $SE$ giảm đi $\rightarrow$ Chunk $i$ chính là tác nhân gây mâu thuẫn $\rightarrow$ **XÓA Chunk $i$**.
5. Cập nhật lại Context $\rightarrow$ Quay lại Bước 2.1.

---

## Tổng kết
Chu trình này liên tục lặp lại cho đến khi Context hội tụ toàn bộ thông tin bổ ích (qua Retrieval), sạch nhiễu (qua Saliency + Masking) và LLM tự tin tuyệt đối vào câu trả lời (thể hiện qua 3-Signal Routing thỏa mãn, lưu ý : max_iterations=5 để tránh vòng lặp vô tận)
