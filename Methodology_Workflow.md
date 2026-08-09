# Quy Trình: RAG Lặp Lại Nhận Thức Bất Định Qua Mức Độ Tăng Thông Tin Ngữ Nghĩa (Uncertainty-Aware Iterative RAG)

Phương pháp này giới thiệu một hệ thống RAG (Retrieval-Augmented Generation) thông minh, có khả năng tự lượng hóa sự bất định (uncertainty) trong suy luận của mô hình ngôn ngữ lớn (LLM/VLM). Dựa vào đó, hệ thống sẽ tự động quyết định xem nên **cắt tỉa thông tin nhiễu** hay **tìm kiếm thêm thông tin mới**. 

Đặc biệt, hệ thống được thiết kế để **không phụ thuộc vào loại dữ liệu (modality-agnostic)**, giúp cốt lõi thuật toán hoạt động đồng nhất trên cả Văn bản (Text), Bảng biểu (Tables) và Hình ảnh (Images).

Dưới đây là workflow chi tiết từ đầu đến cuối, bao gồm các công thức toán học cốt lõi và các cơ chế thích ứng (adaptive):

---

## Bước 1: Trích xuất mệnh đề & Đồng nhất hóa dữ liệu (Sampling & Modality Handling)
Để xử lý câu trả lời dài mở (Open-ended Generation & Long Texts) cũng như dữ liệu đa phương tiện, phương pháp sử dụng trích xuất mệnh đề (Claim Extraction) thay vì tính toán trực tiếp trên toàn bộ chuỗi token.

**Tại sao cần xử lý đặc thù cho văn bản dài?**
Các phương pháp đo lường bất định bằng token (token-level uncertainty) thường thất bại với văn bản dài do "sự đa dạng từ vựng" (Lexical Diversity). LLM có thể diễn đạt cùng một ý nghĩa bằng nhiều từ ngữ khác nhau, khiến entropy của token rất cao dù mô hình hoàn toàn chắc chắn về mặt ngữ nghĩa.
*Ví dụ:*
- $s_1$: "Hà Nội là thủ đô của Việt Nam, nằm ở khu vực Đông Nam Á."
- $s_2$: "Thủ đô của nước Việt Nam là thành phố Hà Nội, thuộc Đông Nam Á."
Dù các chuỗi token hoàn toàn khác nhau, hai câu này chứa chung các mệnh đề sự thật (Hà Nội là thủ đô Việt Nam, Hà Nội thuộc Đông Nam Á). Nhờ công cụ NLI, chúng ta quy chúng về cùng một Cụm khái niệm (Concept) thay vì xem là sự bất định.

1. **Khởi tạo**: Nhận câu hỏi $Q$ và ngữ cảnh $C_t$.
2. **Lấy mẫu (Adaptive M)**: 
   - Thay vì luôn sinh $M=10$ mẫu rất tốn kém chi phí, ta áp dụng **Adaptive M** (Cost-Effective Sampling): Khởi đầu lấy một lượng mẫu nhỏ $M=3$. Nếu độ bất định $SE_{total}$ ban đầu lớn hơn một ngưỡng nhất định, hệ thống mới lấy thêm mẫu lên mức tối đa (ví dụ $M=10$).
3. **Trích xuất mệnh đề (Claim Extraction)**: 
   - Dùng JSON-mode chia nhỏ câu trả lời thành các sự thật cốt lõi (claims).
   - Với ảnh hoặc bảng, LLM/VLM cũng sẽ trích xuất ra các mệnh đề dưới dạng văn bản (text claims).

---

## Bước 2: Phân tích độ bất định ngữ nghĩa và nhiễu (Independent Signals)
Hệ thống mổ xẻ sự "phân vân" thành 2 tín hiệu độc lập: Bất định Ngữ nghĩa (Semantic Entropy) và Bất định Token (Token Uncertainty).

1. **Gom cụm ngữ nghĩa (Semantic Clustering)**: Dùng mô hình NLI kiểm tra sự kéo theo (entailment). Các mệnh đề đồng nghĩa được gom thành Cụm khái niệm (Concept $c$). Xác suất $P(c)$ xấp xỉ qua tần suất: $P(c | Q, C_t) = \frac{|s \in c|}{M}$.

2. **Tính toán 2 tín hiệu bất định**:
   - **Tín hiệu chính - Bất định Ngữ nghĩa ($SE_{semantic}$)**: Entropy của phân phối các Cụm khái niệm. Thể hiện sự mơ hồ về mặt ý nghĩa, mô hình đang phân vân giữa nhiều cách hiểu. Nếu chỉ số này cao, tức là mô hình thiếu kiến thức.
     $$SE_{semantic}(Y | Q, C_t) = - \sum_{c} P(c | Q, C_t) \log_2 P(c | Q, C_t)$$
   
   - **Tín hiệu phụ - Bất định Token ($U_{token}$)**: Thể hiện độ nhiễu (noise) trong quá trình sinh từ, hay còn gọi là đa dạng từ vựng. Được xấp xỉ bằng trung bình log-xác suất (negative logprob) của các "từ khóa chính" ($K_i$) trong mỗi mẫu $s_i$. 
     - **Cách xác định $K_i$**: Thay vì tính trên tất cả token (bao gồm cả các từ ngữ pháp như "the, is, in" có logprob gần 0 gây loãng kết quả), hệ thống chỉ giữ lại Danh từ (Nouns), Động từ (Verbs), Số liệu (Numbers). Bằng cách này, ta đo được chính xác sự "ngập ngừng" của LLM.
     $$U_{token}(Y | Q, C_t) \approx \frac{1}{M} \sum_{i=1}^M \left( -\frac{1}{|K_i|} \sum_{w \in K_i} \log P(w | Q, C_t, \theta) \right)$$
     *(Lưu ý: Hai tín hiệu $SE_{semantic}$ và $U_{token}$ nằm trên 2 thang đo khác nhau và được hệ thống đánh giá hoàn toàn độc lập).*

---

## Bước 3: Định tuyến hành động bằng Ngưỡng thích ứng (Adaptive Routing Decision)
Thay vì dùng ngưỡng cố định tĩnh dễ gây ra sai sót trên các tập dữ liệu khác nhau, hệ thống tính toán **Ngưỡng thích ứng (Adaptive Thresholds)** dựa trên mức độ bất định đo được ở vòng lặp đầu tiên ($t=0$):
- $\tau_{token} = \alpha \times U_{token}^{(0)}$ (ví dụ: $\alpha = 0.5$)
- $\tau_{semantic} = \beta \times SE_{semantic}^{(0)}$ (ví dụ: $\beta = 0.5$)

**Tại sao gọi là Adaptive (Thích ứng)?**
$\alpha$ và $\beta$ là các hệ số điều chỉnh (hyperparameters) được chọn cố định một lần. Khi vận hành thực tế:
- Ngay khi nhận câu hỏi, hệ thống đo mức độ bất định "gốc" ở vòng lặp đầu ($t=0$), ví dụ $U_{token}^{(0)} = 3.5$ và $SE_{semantic}^{(0)} = 1.2$.
- Dựa vào $\alpha, \beta$, hệ thống tự chốt vạch đích (ngưỡng dừng) riêng cho câu hỏi đó: $\tau_{token} = 1.75$ và $\tau_{semantic} = 0.6$. Từ vòng lặp thứ $1, 2, 3...$ trở đi, nó sẽ dùng đúng 2 con số này làm mốc quyết định.

Hệ thống áp dụng **Định tuyến Hai tín hiệu Độc lập (Two-Signal Independent Routing)**:

- **TRUY XUẤT (RETRIEVE)**: Nếu $SE_{semantic} > \tau_{semantic}$. Mô hình đang mơ hồ về mặt ý nghĩa, chứng tỏ nó bị thiếu kiến thức để trả lời. (Chuyển sang Bước 5).
  
- **CẮT TỈA (PRUNE)**: Nếu $SE_{semantic} \le \tau_{semantic}$ NHƯNG $U_{token} > \tau_{token}$. Mô hình đã chốt được một ý nghĩa chung, nhưng quá trình sinh từ lại rất ngập ngừng và nhiễu. Chứng tỏ ngữ cảnh chứa thông tin thừa/gây nhiễu cần loại bỏ. (Chuyển sang Bước 4).
  
- **DỪNG (STOP)**: Nếu $SE_{semantic} \le \tau_{semantic}$ VÀ $U_{token} \le \tau_{token}$. Cả 2 tín hiệu đều an toàn.
  - *An toàn (Safety)*: Giới hạn vòng lặp `MAX_ITERATIONS = 5` để tránh chạy vô hạn.
  - *Hội tụ (Convergence)*: Nếu $SE_{semantic}$ không sụt giảm >5% trong `patience` vòng lặp liên tiếp, hệ thống dừng sớm.

---

## Bước 4: Cắt tỉa ngữ cảnh (Context Pruning - Xử lý nhiễu)
Kỹ thuật Leave-One-Out (LOO) truyền thống đòi hỏi chi phí tính toán khổng lồ $O(N \times M)$ lần gọi LLM. Để giải quyết điểm nghẽn này, hệ thống đề xuất 3 hướng tiếp cận, có thể lựa chọn tùy thuộc vào hạ tầng triển khai:

### Hướng 1: Cắt tỉa 2 giai đoạn (Two-phase Pruning) - Cách tiếp cận truyền thống
Sử dụng các công cụ NLP cơ bản để lọc rác trước khi gọi LLM.
1. **Lọc sơ bộ bằng NLI (Pre-filter)**: Nhanh chóng loại các đoạn tài liệu (chunk) gây mâu thuẫn (contradiction) bằng mô hình NLI với chi phí cực rẻ.
2. **Phân tích Leave-One-Out song song (Batch-parallel)**: Với các chunk còn lại, chạy song song để tính độ biến thiên của $SE_{semantic}$ khi loại bỏ từng chunk $c_i$:
   $$\Delta SE(c_i) = SE_{semantic}(Y | Q, C_t \setminus \{c_i\}) - SE_{semantic}(Y | Q, C_t)$$
   - Nếu $\Delta SE(c_i) \le 0$: Rác/Nhiễu $\rightarrow$ Xóa bỏ $c_i$.
   - Nếu $\Delta SE(c_i) > 0$: Thông tin hữu ích $\rightarrow$ Giữ lại.
- **Ưu điểm**: Dễ cài đặt, không cần can thiệp sâu vào cấu trúc phần cứng của LLM (chạy được qua các API đóng).
- **Nhược điểm**: Vẫn tốn chi phí gọi LLM rất lớn ở giai đoạn 2 nếu bộ lọc NLI không lọc được nhiều.

### Hướng 2: Attention-Guided Gray-Zone Pruning
Tối ưu hóa bằng cách phân vùng ngữ cảnh thành các mức độ rủi ro trước khi gọi LLM.
1. **Phân vùng bằng mô hình Reranker/NLI nhẹ**: Chấm điểm tất cả các chunk để chia làm 3 vùng:
   - **Chắc chắn rác (Score thấp)**: Xóa ngay lập tức.
   - **Chắc chắn tốt (Score cao)**: Giữ lại tuyệt đối.
   - **Vùng nghi ngờ (Gray-Zone)**: Chỉ giữ lại 2-3 chunk nằm ở lằn ranh.
2. **Đo lường LOO trên vùng Gray-Zone**: Chỉ thực hiện đo lường $\Delta SE$ cho 2-3 chunk nghi ngờ này (có thể soi chiếu thêm Attention Weights của LLM).
- **Ưu điểm**: Cắt giảm 80-90% số lượng truy vấn LLM. Giảm $N$ xuống con số rất nhỏ ($N' \le 3$), tốc độ cực nhanh.
- **Nhược điểm**: Phụ thuộc hoàn toàn vào độ chính xác của Reranker. Tiềm ẩn rủi ro rất lớn làm mất các chunk có giá trị (False Negatives) do Reranker chấm điểm sai, làm giảm độ chính xác tổng thể của RAG.

### Hướng 3: Attention Masking (Zero-Cost LOO ở mức Hệ thống)
Giải quyết triệt để nút thắt cổ chai mà KHÔNG cần dùng bộ lọc ngoài, bảo toàn 100% độ chính xác của LOO thông qua can thiệp hệ thống (System-Level).
1. **Gửi 1 Prompt duy nhất**: Tạo 1 prompt chứa toàn bộ $N$ chunk ngữ cảnh thay vì tạo $N$ prompt rời rạc.
2. **Khai thác Attention Masking**: Truyền vào hệ thống một batch gồm $N$ `attention_mask` khác nhau.
   - Mask 1: Che khuất (mask) chunk $c_1$ (trọng số attention $= -\infty$).
   - Mask 2: Che khuất chunk $c_2$
   - ... (đến Mask $N$).
3. **Tính toán $\Delta SE$ song song**: GPU (thông qua vLLM/HuggingFace) chỉ tính KV-Cache đúng 1 lần cho toàn bộ phần văn bản dùng chung, sau đó sinh mẫu song song trên từng mask và tính $\Delta SE(c_i)$:
   $$\Delta SE(c_i) = SE_{semantic}(Y | Q, C_t \text{ masked } c_i) - SE_{semantic}(Y | Q, C_t)$$
   - Nếu $\Delta SE(c_i) \le 0$: Bỏ chunk này đi giúp giảm hoặc giữ nguyên bất định $\rightarrow$ Nó là Rác/Nhiễu $\rightarrow$ Chốt xóa $c_i$.
   - Nếu $\Delta SE(c_i) > 0$: Chunk mang thông tin hữu ích $\rightarrow$ Giữ lại.
- **Ưu điểm**: Chính xác tuyệt đối 100% (so với LOO gốc) vì không phải bỏ sót bất kỳ chunk nào cho mô hình lọc ngoài. Thông lượng cực cao do tính KV-Cache 1 lần duy nhất.
- **Nhược điểm**: Khó lập trình. Đòi hỏi quyền truy cập Local GPU và tùy chỉnh sâu vào tensor/code của framework (như vLLM hay HuggingFace). Không chạy được với API.

### Hướng 4: Automatic Prefix Caching (APC) - Tối ưu hóa cho API
Nếu buộc phải sử dụng các API đóng (như OpenAI, Anthropic) và không thể can thiệp Attention Masking, hệ thống có thể tận dụng tính năng Prefix Caching để giảm thiểu chi phí của LOO. 

**Lưu ý chí mạng**: KV-Cache hoạt động nghiêm ngặt từ trái sang phải (Left-to-Right). Bất kỳ sự thay đổi nào ở vị trí $X$ sẽ làm mất toàn bộ cache từ $X$ trở về sau. Do đó, cần tuân thủ quy tắc sắp xếp Prompt:
1. **Cố định "Đầu não" (Top-of-Prompt)**: Đặt toàn bộ `System Prompt` $\rightarrow$ `Instruction` $\rightarrow$ `Câu hỏi Q` $\rightarrow$ `Few-shot examples` lên trên cùng. Đảm bảo phần này tuyệt đối không thay đổi giữa các request LOO. (Nên cấu trúc dài $>1024$ tokens để vượt ngưỡng kích hoạt cache tự động của API).
2. **Đặt Ngữ cảnh ở Đuôi (Bottom-of-Prompt)**: Toàn bộ danh sách tài liệu RAG ($C_t$) phải nằm ở cuối cùng của Prompt ($[\text{Chunk}_1 + \text{Chunk}_2 + \text{Chunk}_3...]$).
- **Cách hoạt động**: Khi gửi $N$ request LOO, dù việc xóa $\text{Chunk}_1$ làm mất toàn bộ cache của các chunk đứng sau nó, API vẫn luôn giữ được 100% Cache Hit cho phần "Đầu não" khổng lồ phía trên.
- **Ưu điểm**: Dễ triển khai. Tiết kiệm được $30\% - 60\%$ chi phí token và giảm đáng kể độ trễ cho mỗi vòng LOO qua API.
- **Nhược điểm**: Vẫn tốn chi phí tính toán lại (Cache Miss) cho các chunk nằm sau chunk bị xóa. Không nhanh và rẻ triệt để như Attention Masking.
### Hướng 5: Dựa vào Trọng số Chú ý (Attention-based Saliency)
- **Cách hoạt động**: Trích xuất trực tiếp **Attention Weights** khi LLM sinh ra các token gây bất định. Chunk nào nhận được sự chú ý (attention score) cao nhưng sinh ra mâu thuẫn $\rightarrow$ Tác nhân nhiễu. Chunk nào có attention $\approx 0$ $\rightarrow$ Rác/Distractor.
- **Tính chất**: Hoạt động với 1 lần forward-pass duy nhất. Tiết kiệm tuyệt đối thời gian.

> **💡 Lựa chọn trong Triển khai Thực tế (Thử nghiệm Cốt lõi):** 
> Để kiểm thử và báo cáo chính xác nhất sức mạnh của thuật toán RAG lặp lại, hệ thống sẽ **chỉ sử dụng Hướng 3 (Attention Masking)**. Việc bỏ qua Reranker/Gray-Zone (Hướng 2) giúp đảm bảo không có chunk giá trị nào bị rơi rớt vô lý, giữ nguyên tính nguyên bản của LOO nhưng vẫn đạt được tốc độ thời gian thực.

Kết quả cuối cùng của Bước 4 luôn là tập ngữ cảnh sạch hoàn toàn ($C_{clean}$). Vòng lặp quay lại Bước 2.

---

## Bước 5: Truy xuất chủ động (Active Retrieval - Bổ sung kiến thức)
Giải quyết triệt để lỗi "tính toán vòng vo" (Circular EIG - dùng LLM tự thẩm định giả định của chính nó) của hệ thống cũ thông qua cơ chế **EIG Ngầm định (Implicit EIG via Iterative Loop Evaluation)**:

1. **Chọn giả thuyết (Hypothesis)**: Lấy luôn Cụm khái niệm (Concept) có xác suất cao nhất hiện tại làm giả thuyết $d'$. Không cần tính EIG giả định cho $d'$ để tiết kiệm chi phí.
2. **Truy xuất**: Dùng $d'$ làm truy vấn (query) để lấy top-K tài liệu thực tế $d_{new}$ từ Vector Database. Hợp nhất ngữ cảnh: $C_{t+1} = C_{clean} \cup \{d_{new}\}$.
3. **Đánh giá EIG thực tế (Actual EIG)**: 
   - Đi tới vòng lặp tiếp theo ($t+1$) và tính lại độ bất định.
   - Lượng thông tin thực tế nhận được (Actual Information Gain) chính là sự sụt giảm bất định giữa 2 vòng lặp:
     $$\Delta SE = SE_{t} - SE_{t+1}$$
   - Nếu $\Delta SE \le 0$ (tức là tài liệu mang về không có tác dụng), nó sẽ ngay lập tức kích hoạt điều kiện kiểm tra hội tụ và dừng sớm ở các vòng lặp sau. 

---
**Tóm tắt vòng lặp (Iterative Loop)**:
Tính $SE_{semantic}$ và $U_{token}$ độc lập $\rightarrow$ Định tuyến qua ma trận 2 tín hiệu (RETRIEVE/PRUNE/STOP) $\rightarrow$ Cắt tỉa (đo $\Delta SE$ khi rút chunk) $\rightarrow$ Truy xuất (đo Implicit EIG qua $\Delta SE$ của vòng tiếp theo) $\rightarrow$ Lặp lại tới khi hội tụ hoặc hết số vòng an toàn.

---
## Ghi chú Triển khai Thực tế (Local GPU Hosting)
Phương pháp này đặc biệt tối ưu khi chạy trên các LLM mã nguồn mở được host trên GPU nội bộ (Local GPU) thông qua các framework như **vLLM** hay **HuggingFace Transformers** thay vì dùng Web API (như OpenAI). Việc host model local mang lại 2 lợi thế tuyệt đối cho thuật toán này:
1. **Truy cập Logprobs đầy đủ**: Khác với API thương mại có thể bị giới hạn hoặc tính phí cao khi trả về logprobs, việc chạy local cho phép truy xuất trực tiếp, toàn quyền và miễn phí vào ma trận `logits` / `logprobs` từ trọng số $\theta$ của mô hình, giúp tính toán $U_{token}$ chính xác tuyệt đối.
2. **Cắt tỉa Zero-Cost qua Attention Masking**: Chạy LOO thông thường (Bước 4) đòi hỏi tính toán cực lớn, nhưng với Local GPU, ta có thể can thiệp thẳng vào hệ thống để đưa vào $N$ `attention_mask` khác nhau. Các framework (như vLLM hay HuggingFace) sẽ tính toán KV-Cache 1 lần duy nhất rồi xử lý song song các mẫu khuyết. Điều này giúp tăng thông lượng (throughput) lên hàng chục lần, biến thuật toán cắt tỉa LOO từ rào cản lý thuyết thành giải pháp thời gian thực hoàn hảo mà không làm rớt mất ngữ cảnh quan trọng.
