# Bản Cập Nhật Tối Ưu Hóa Hệ Thống RAG cho Google Colab (T4 GPU)

Tài liệu này ghi chú lại toàn bộ những thay đổi đã được áp dụng trực tiếp vào mã nguồn để hệ thống RAG có thể chạy mượt mà trên môi trường hạn chế tài nguyên như T4 GPU (16GB VRAM) của Google Colab, giải quyết dứt điểm tình trạng tràn RAM (OOM) khi làm việc với Qwen2-VL.

---

## 1. Dữ liệu & Đa Phương Thức (Modality)

- **`[MỚI] eval/datasets/multimodalqa_loader.py`**:
  - Tạo mới hoàn toàn class `MultiModalQALoader` để tự động tải, giải nén và tiền xử lý tập dữ liệu MultiModalQA từ định dạng `.jsonl.gz`.
  - Hỗ trợ chuyển đổi cấu trúc `Tables` sang định dạng Markdown để LLM dễ đọc hơn.
  - Xử lý các đường dẫn hình ảnh (images) thành các tham chiếu tuyệt đối, phục vụ cho việc nhúng vào Prompt.

- **`[MỚI] src/uncertainty_rag/modality/multimodal_handler.py`**:
  - Tạo mới class `MultimodalHandler`.
  - **Thay đổi quan trọng**: Thay vì mã hóa toàn bộ hình ảnh thành chuỗi Base64 (làm tăng kích thước dữ liệu và ngốn VRAM), hệ thống giờ đây sử dụng đường dẫn file dạng `file://...`. Mô hình Qwen2-VL sẽ tự động đọc từ đường dẫn này, tiết kiệm dung lượng bộ nhớ khổng lồ.

---

## 2. Vá Lỗi Pipeline & Retriever

- **`[SỬA] eval/run_eval.py`**:
  - **Chống lỗi Name Shadowing**: Thay đổi cách cấu hình `sys.path` ở đầu file để thư mục `eval/datasets` cục bộ không đè lên gói thư viện `datasets` của HuggingFace, giúp tránh lỗi khi Import.
  - **Khởi tạo LLM thông minh (Chống OOM)**: Nếu `llm` (mô hình chính) và `claim_llm` (mô hình trích xuất Claim) dùng chung 1 trọng số (ví dụ: Qwen2-VL), hệ thống sẽ dùng chung 1 Object Client duy nhất thay vì tải lại mô hình 2 lần vào GPU.
  - **Retriever Top 10**: Trong vòng lặp đánh giá `run_evaluation`, nếu ngữ cảnh đầu vào quá lớn, hệ thống sẽ gọi `dense_retriever.retrieve(top_k=10)` để chỉ nạp tối đa 10 Chunk ban đầu cho LLM (Context Window nhỏ lại -> Ít tốn RAM hơn). LLM sẽ phải dựa vào tính năng Active Retrieval để tự động lấy thêm thông tin nếu 10 Chunk này là chưa đủ.
  - Đăng ký `MultiModalQALoader` vào `DATASET_REGISTRY`.

---

## 3. Vá Lỗi Client & Chống OOM cho VLM

- **`[SỬA] src/uncertainty_rag/models/llm_client.py`**:
  - **Hỗ trợ 4-bit Quantization**: Trong hàm `__init__`, bổ sung cờ `load_in_4bit=True` và `BitsAndBytesConfig` để load mô hình siêu nhẹ.
  - **Giới hạn độ phân giải ảnh (`max_pixels`)**: Trong hàm `_prepare_inputs`, khi nhận diện được `image_url`, bổ sung trường `"max_pixels": 256 * 256` để ép Qwen2-VL nén ảnh xuống, tránh trường hợp ảnh quá to làm sập bộ nhớ. Đồng thời cắt tiền tố `file://` để tương thích với thư viện `qwen_vl_utils`.
  - **Cơ chế Sinh (Generate) Tuần Tự**: Trong hàm `generate`, sửa đổi tham số `num_return_sequences` về `1`. Để lấy $N$ mẫu (dùng cho thuật toán tính độ bất định), hệ thống sẽ chạy vòng lặp tuần tự $N$ lần (`for i in range(n)`). Trước đây, việc sinh song song 3 mẫu ($N=3$) cùng lúc đòi hỏi phân bổ ma trận Attention cực kỳ lớn, là nguyên nhân chính gây sập RAM.
  - **Đa dạng hóa mẫu (Diversity)**: Khi `temperature > 0`, bổ sung các tham số như `top_k=50`, `top_p=0.85`, và `repetition_penalty=1.1` vào cấu hình sinh để các mẫu trả về có sự đa dạng cần thiết.
  - Thêm các lệnh `print` log ra Terminal để theo dõi tiến độ sinh mẫu theo thời gian thực.

---

## 4. Vá Lỗi Prompt Ảo Giác (Hallucination)

- **`[SỬA] src/uncertainty_rag/core/sampler.py`**:
  - Viết lại `SYSTEM_PROMPT` để yêu cầu LLM đưa ra các giả thuyết (hypotheses) một cách hợp lý và đa dạng khi gặp trường hợp ngữ cảnh mơ hồ hoặc mâu thuẫn, thay vì copy y chang nội dung. Khuyến khích LLM tự chỉ ra sự không chắc chắn của nó nếu có.

- **`[SỬA] src/uncertainty_rag/core/claim_extractor.py`**:
  - Giảm `max_tokens` xuống còn `800`. Cung cấp Query Context trong prompt trích xuất claim để bảo đảm Claim trích xuất gắn liền với câu hỏi.
  - Sửa đổi Prompt cụ thể cho Ảnh (Image), Bảng (Table), và Văn bản (Text).

---

## 5. Tích hợp Mega Patch V2 (Kiến trúc RAG nâng cao)

- **`[MỚI] src/uncertainty_rag/core/evidence_checker.py`**:
  - Dùng NLI (DeBERTa) để quét các Claim do LLM sinh ra và đối chiếu với Document Context. Tính toán chỉ số `evidence_ratio` (Signal B - Evidence Sufficiency).

- **`[SỬA] src/uncertainty_rag/core/uncertainty.py`**:
  - Cập nhật `UncertaintyProfile` để chứa `evidence_ratio`.

- **`[SỬA] src/uncertainty_rag/core/router.py`**:
  - Triển khai kiến trúc **3-Signal Routing** (SE_semantic, U_token, và evidence_ratio). Quá trình STOP chỉ xảy ra khi cả 3 tín hiệu đạt yêu cầu.

- **`[SỬA] src/uncertainty_rag/core/retriever.py`**:
  - Thay thế DenseRetriever thành **Hybrid Retriever** kết hợp giữa Sparse (BM25Okapi) và Dense (SentenceTransformers). Dùng thuật toán RRF để trộn Rank.

- **`[SỬA] src/uncertainty_rag/core/semantic_cluster.py`**:
  - Thêm logic **Factual Anchor Check**: Kiểm tra con số/ngày tháng để chia tách ngay những answer khác biệt thực tế mà không cần hỏi NLI (Bypass NLI cho số liệu).
  - Thêm **ABSTAIN Normalization**: Gom cụm các phản hồi từ chối trả lời ("I don't know") thành 1 cụm ABSTAIN duy nhất, tránh bùng nổ SE entropy.

- **`[SỬA] src/uncertainty_rag/pipeline.py`**:
  - Cập nhật hàm `__init__` nhận `EvidenceChecker`.
  - Bổ sung **Adaptive Query Expansion (LLM Query Expansion)** ở pha RETRIEVE. LLM sẽ tự phát sinh thêm 3 từ khoá biến thể bổ sung cho giả thuyết, sau đó gọi Hybrid Search bằng các biến thể đó.

- **`[SỬA] src/uncertainty_rag/config.py`**:
  - Thêm `tau_evidence` vào ThresholdConfig (mặc định = 0.7).
  - Tăng cường `entailment_threshold` trong `nli_model.py` lên 0.75 để siết chặt tiêu chuẩn Evidence.

---

## 6. Tối ưu Loại Bỏ Ngữ Cảnh Dư Thừa (Chunk Pruning)

- **`[SỬA] src/uncertainty_rag/core/pruner.py`**:
  - **Xác định vị trí Chunk (Chunk Positional Tracking)**: Nhúng các tag định vị vị trí (vd: `<chunk_1>...</chunk_1>`) vào đầu mỗi chunk trong chuỗi prompt để dễ dàng ánh xạ (map) ngược lại từ ma trận Attention bên trong GPU về đúng chunk tương ứng.
  
  - **Attention Saliency Pruner (Bộ lọc thô - First-pass filter)**: Áp dụng cơ chế loại bỏ chunk dựa trên mức độ chú ý (Attention Saliency) của LLM. Tính toán trực tiếp điểm Attention (mức độ quan trọng) của từng chunk khi mô hình sinh ra giả thuyết.
    - *Khi nào dùng:* Áp dụng trước tiên khi số lượng chunk còn nhiều. Chỉ tốn 1 lần chạy (1 forward pass) là lấy được điểm của mọi chunk, giúp loại bỏ cực nhanh những chunk bị LLM ngó lơ (điểm chú ý < 0.02) với chi phí thấp nhất.

  - **Attention Masking (Bộ lọc tinh - Second-pass filter)**: Khi đánh giá loại bỏ chunk, áp dụng attention masking (Custom Mask) để "che" lần lượt từng chunk đi, ép điểm attention của chunk đó về 0. Bằng cách này, ta mô phỏng được thuật toán Leave-One-Out (bỏ 1 chunk ra xem kết quả thay đổi ra sao) một cách song song (parallel) cho toàn bộ các chunk trong cùng 1 lần tính toán.
    - *Khi nào dùng:* Áp dụng sau Saliency, trên những chunk còn lại, để đánh giá độ nhiễu một cách chính xác nhất. Tìm ra các chunk thực sự chứa thông tin mâu thuẫn (contradiction) làm nhiễu mô hình. Dù tối ưu hơn so với gọi rời rạc từng prompt, phương pháp này vẫn tốn tài nguyên sinh (generation) hơn Saliency.

Chiến lược kết hợp này đặc biệt hiệu quả trên T4 GPU vì nó giảm thiểu lượng token ngữ cảnh (context size) truyền vào cho các vòng lặp RAG tiếp theo, giải quyết trực tiếp nguy cơ cạn kiệt VRAM.
