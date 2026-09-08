# Modality-Aware Attention Pruning: phân tích và đề xuất

Ngày soạn: 04/09/2026

## 1. Bối cảnh

Benchmark hiện tại gồm bốn tập dữ liệu, mỗi tập 200 câu:

- **MMQA**: dữ liệu hỗn hợp text, table và image.
- **WebQA**: trong bundle hiện tại, bằng chứng đúng chủ yếu là image.
- **HotpotQA**: bằng chứng là text.
- **TAT-QA**: dữ liệu gồm table và text, trong đó table đóng vai trò rất quan trọng.

Hai lần chạy cần so sánh là:

| Kết quả | Source GitHub | Fast runner trên Hugging Face | Cách tính attention |
|---|---|---|---|
| `(3)` | `554d4d0` | `2bde966a` | Attention density, tức mass chia cho số token |
| `(4)` | `1690106` | `3dd72ff2` | Total attention mass, không chia cho số token |

Kết quả `(3)` đã có phần code của nhánh `feat/improve-uncertainty-pipeline` được cherry-pick và đã có hardening cho image/table alignment trong core library. Điểm chưa được sửa trong `(3)` là công thức scoring riêng của `fast_runner.py` vẫn dùng attention density.

Kết quả `(4)` kế thừa các sửa đổi trên, đồng thời đổi fast runner sang total attention mass và lưu trace của từng chunk.

## 2. Đề xuất ban đầu của nhóm

Nhận định ban đầu của nhóm là:

> Ở bản `(3)`, MMQA và HotpotQA có kết quả tương đối tốt hơn các baseline chọn chunk đơn giản, trong khi WebQA và TAT-QA yếu hơn. WebQA phụ thuộc vào image và TAT-QA phụ thuộc nhiều vào table. Image và table thường được biểu diễn bằng rất nhiều token, nên khi attention mass bị chia cho số token, score của chúng trở nên thấp và gold chunk dễ bị prune.
>
> Ở bản `(4)`, support recall và support precision tăng mạnh nhưng answer EM/F1 không tăng tương ứng, thậm chí giảm trên MMQA và HotpotQA. Nguyên nhân có thể là total attention mass ưu tiên các chunk có nhiều token, giúp giữ image/table nhưng lại dễ loại các text chunk ngắn.
>
> Vì vậy, cần xử lý sự chênh lệch số token giữa text, image và table. Một ý tưởng là chia các modality vào các “room” riêng để text cạnh tranh với text, image cạnh tranh với image và table cạnh tranh với table.

So sánh trực tiếp dưới đây dùng **773 câu mà attention chạy thành công ở cả `(3)` và `(4)`**, nhờ đó tránh làm chênh lệch metric chỉ vì hai lần chạy có số câu OOM khác nhau:

| Dataset | Answer EM `(3) → (4)` | Answer F1 `(3) → (4)` | Support recall `(3) → (4)` | Support precision `(3) → (4)` |
|---|---:|---:|---:|---:|
| MMQA | 0.168 → 0.098 | 0.198 → 0.121 | 0.192 → 0.502 | 0.070 → 0.173 |
| WebQA | 0.000 → 0.010 | 0.125 → 0.245 | 0.200 → 0.995 | 0.057 → 0.483 |
| HotpotQA | 0.280 → 0.225 | 0.395 → 0.318 | 0.565 → 0.410 | 0.322 → 0.219 |
| TAT-QA | 0.150 → 0.170 | 0.258 → 0.301 | 0.618 → 0.943 | 0.256 → 0.415 |
| **Tổng** | **0.149 → 0.127** | **0.246 → 0.251** | **0.401 → 0.720** | **0.180 → 0.328** |

Bảng này làm rõ hai điểm. Thứ nhất, nhận định “support recall và precision tăng mạnh” đúng ở mức tổng thể và đặc biệt rõ trên hai tập phụ thuộc nhiều vào modality dài là WebQA và TAT-QA. Thứ hai, cải thiện này **không xảy ra trên mọi dataset**: HotpotQA bị giảm cả support recall, support precision và answer quality; MMQA giữ được nhiều gold support hơn nhưng EM/F1 vẫn giảm. Đây chính là dấu hiệu cho thấy total mass đã sửa được bias chống chunk dài của `(3)`, nhưng đồng thời tạo ra bias mới ưu tiên chunk dài.

Đây là một chẩn đoán đúng về hướng tổng thể: hai công thức hiện tại nằm ở hai cực và tạo ra hai loại length bias đối nghịch nhau.

## 3. Đề xuất đầy đủ từ Gemini

Gemini đề xuất **Modality-Aware Pruning/Normalization**, với các luận điểm chính sau.

### 3.1. Lợi ích kỳ vọng

1. **Giảm modality bias**

   Khi text, image và table có số token rất khác nhau, xếp tất cả chunk trong cùng một bảng điểm có thể khiến một modality được ưu tiên hoặc bị trừng phạt chỉ vì cách tokenizer biểu diễn nó. Chia nhóm theo modality giúp các image cạnh tranh nội bộ với image, text với text và table với table.

2. **Có thể dùng ngưỡng riêng cho từng modality**

   Table thường dài và chứa nhiều thông tin; image chứa thông tin nén nhưng được bung thành nhiều visual token; text thường ngắn hơn. Do đó có thể cân nhắc threshold hoặc cơ chế giữ chunk khác nhau cho từng modality.

3. **Ổn định hơn so với dùng một công thức chung cho mọi modality**

   Một phép chuẩn hóa duy nhất có thể không phù hợp đồng thời với text, image và table vì độ dài token và phân phối attention của chúng khác nhau.

### 3.2. Rủi ro được Gemini nêu ra

1. **Mất khả năng đối chiếu chéo nếu chạy từng modality riêng biệt**

   Nếu text, image và table được đưa vào ba lần forward độc lập, model sẽ không nhìn thấy quan hệ giữa các modality. Ví dụ, model không thể liên kết cụm từ trong text với một chi tiết trong image.

2. **Quota cứng không phù hợp với mọi câu hỏi**

   HotpotQA có thể chỉ cần text, trong khi WebQA có thể cần image. Nếu luôn giữ một số lượng cố định từ từng room, hệ thống sẽ giữ cả modality không liên quan và lãng phí context budget.

### 3.3. Thiết kế Gemini đề xuất

Gemini không khuyến nghị chạy model ba lần. Thay vào đó:

1. Đưa tất cả text, image và table vào cùng một prompt.
2. Chạy VLM một lần để giữ nguyên cross-modal interaction.
3. Sau khi lấy attention score, chia chunk thành các nhóm theo modality.
4. Chuẩn hóa score riêng trong nhóm image, text và table.
5. Lấy các chunk đứng đầu mỗi nhóm rồi gộp lại, hoặc đặt trọng số cân bằng giữa các nhóm.

Có thể biểu diễn phiên bản đơn giản như sau:

```python
for modality in ("text", "image", "table"):
    group_scores = raw_scores[chunks_of(modality)]
    normalized_scores[chunks_of(modality)] = group_scores / group_scores.sum()

selected = top_chunks_from_each_modality(normalized_scores)
```

Gemini cho rằng hướng này có thể trở thành một ablation study hoặc một phiên bản mới của thuật toán.

## 4. Kết quả thực nghiệm từ `(3)` và `(4)`

### 4.1. Hai công thức hiện tại

Gọi:

- `mass_i` là tổng attention từ final prompt token tới toàn bộ token thuộc chunk `i`;
- `n_i` là số token của chunk `i`.

Bản `(3)` dùng attention density:

```python
raw_i = mass_i / max(1, n_i)
score_i = raw_i / sum(all_raw)
```

Bản `(4)` dùng total attention mass:

```python
raw_i = mass_i
score_i = raw_i / sum(all_raw)
```

Cả hai giữ chunk khi normalized score lớn hơn hoặc bằng `0.1`, đồng thời có fallback giữ chunk có score lớn nhất nếu không chunk nào qua ngưỡng.

### 4.2. Paired comparison

Trên 773 câu mà attention hoàn thành ở cả hai lần chạy:

| Dataset | EM `(3) → (4)` | F1 `(3) → (4)` | Support recall `(3) → (4)` | Chunk giữ trung bình `(3) → (4)` |
|---|---:|---:|---:|---:|
| MMQA | 0.168 → 0.098 | 0.198 → 0.121 | 0.192 → 0.502 | 4.13 → 4.24 |
| WebQA | 0.000 → 0.010 | 0.125 → 0.245 | 0.200 → 0.995 | 3.58 → 2.08 |
| HotpotQA | 0.280 → 0.225 | 0.395 → 0.318 | 0.565 → 0.410 | 3.66 → 3.82 |
| TAT-QA | 0.150 → 0.170 | 0.258 → 0.301 | 0.618 → 0.943 | 3.64 → 3.37 |
| **Tổng** | **0.149 → 0.127** | **0.246 → 0.251** | **0.401 → 0.720** | **3.74 → 3.35** |

Các kết luận trực tiếp:

- `(4)` cải thiện support recall rất mạnh, đặc biệt trên WebQA và TAT-QA.
- F1 tổng gần như không đổi; EM tổng giảm nhẹ.
- HotpotQA text-only giảm rõ rệt khi đổi sang total mass.
- MMQA giữ được nhiều support hơn nhưng answer quality giảm và số attention OOM tăng từ 15 lên 24.
- Top-10 answer, EM, F1 và PPL giống nhau trên cả 800 câu, nên khác biệt chủ yếu đến từ attention pruning.

### 4.3. Bằng chứng về length bias trong `(4)`

Tương quan giữa `token_count` và `attention_score` trong trace `(4)`:

| Dataset | Correlation giữa độ dài và score |
|---|---:|
| WebQA | 0.84 |
| TAT-QA | 0.59 |
| HotpotQA | 0.43 |
| MMQA Top-10 text | 0.43 |

WebQA cho thấy vấn đề rõ nhất:

- Support image có score trung bình khoảng `0.415`.
- Distractor image có score trung bình khoảng `0.455`.
- Support image chỉ có score cao hơn distractor image ở 88/200 câu.
- Tuy vậy, `(4)` giữ cả hai ảnh trong 199/200 câu.

Do đó support recall `0.995` không có nghĩa attention đã phân biệt đúng ảnh chứa bằng chứng. Nó phần lớn có nghĩa thuật toán giữ gần như toàn bộ image và loại phần lớn text.

MMQA cũng cho tín hiệu tương tự:

- Khoảng 23.5% support image được giữ.
- Khoảng 27.2% distractor image được giữ.

Total mass giúp image không còn nhận score bằng 0, nhưng chưa chứng minh được rằng score đó đo đúng relevance.

## 5. Phân tích đề xuất “chia room”

### 5.1. Trường hợp nên tránh: forward riêng từng modality

Thiết kế sau không nên dùng làm phương pháp chính:

```text
Forward 1: chỉ text
Forward 2: chỉ image
Forward 3: chỉ table
```

Nó có các vấn đề:

- Mất cross-modal interaction.
- Score từ ba forward khác nhau không nằm trên cùng một điều kiện và khó so sánh.
- Tăng runtime và VRAM traffic.
- Không xử lý tốt câu hỏi cần kết hợp nhiều modality.

Nó chỉ có thể hữu ích như một ablation để đo riêng đóng góp của từng modality.

### 5.2. Trường hợp hợp lý: một forward chung, hậu xử lý theo modality

Thiết kế sau giữ được ưu điểm quan trọng:

```text
Một multimodal prompt
→ một VLM forward
→ lấy raw attention mass của từng chunk
→ hiệu chỉnh score theo modality
→ chọn chunk
```

Đây là hướng đáng thử vì không làm mất cross-modal attention.

### 5.3. Tại sao normalize mỗi room về tổng 1 chưa đủ tốt?

Giả sử mỗi modality được chuẩn hóa độc lập:

```python
room_score_i = raw_i / sum(raw_in_same_modality)
```

Thiết kế này tạo ra các bias mới:

1. Nếu chỉ có một table, table đó tự động nhận score `1.0`, kể cả khi không liên quan.
2. Hai image chia tổng `1.0`, trong khi mười text cũng chia tổng `1.0`; số lượng chunk trong room ảnh hưởng trực tiếp tới score.
3. Score `0.3` của image room không chắc tương đương score `0.3` của text room.
4. Nếu bắt buộc giữ top-1 mỗi room, hệ thống luôn giữ cả modality không liên quan.
5. Một threshold chung như `0.1` không còn ý nghĩa ổn định khi số room hoặc số chunk thay đổi.
6. Room normalization không sửa length bias bên trong cùng một modality: hai text chunk hoặc hai image có độ dài rất khác nhau vẫn bị lệch.

Replay chẩn đoán từ trace `(4)` cũng không cho thấy equal-room normalization thống trị phương pháp global:

| Dataset | Phương pháp | Support recall | Support precision | Số chunk giữ |
|---|---|---:|---:|---:|
| WebQA | Global total mass | 0.995 | 0.483 | 2.08 |
| WebQA | Equal rooms | 0.935 | 0.284 | 3.42 |
| TAT-QA | Global total mass | khoảng 0.960 | 0.425 | 3.29 |
| TAT-QA | Equal rooms | khoảng 0.928 | 0.460 | 2.78 |

Equal rooms có thể cải thiện precision trong một số trường hợp như TAT-QA, nhưng có thể giảm recall, tăng số chunk phải giữ hoặc giữ modality nhiễu. Vì thế nó phù hợp làm ablation hơn là lựa chọn mặc định.

## 6. Proposal khuyến nghị

### 6.1. Mục tiêu thiết kế

Phương pháp mới cần đồng thời:

1. Không phạt image/table chỉ vì chúng có nhiều token.
2. Không ưu tiên chunk chỉ vì nó dài.
3. Giữ một forward multimodal chung.
4. Không dùng quota cứng cho mọi modality.
5. Có score so sánh được giữa text, image và table.
6. Đánh giá selection dưới cùng một context/token budget.

### 6.2. Giai đoạn A: power length normalization

Thay hai cực `(3)` và `(4)` bằng một họ công thức liên tục:

\[
r_i = \frac{m_i}{(n_i + c)^{\alpha}}
\]

Trong đó:

- `alpha = 0` tương đương `(4)`;
- `alpha = 1` tương đương `(3)`;
- `0 < alpha < 1` là mức hiệu chỉnh trung gian;
- `c` là hằng số nhỏ để tránh vấn đề với span rất ngắn.

Ablation tối thiểu nên thử:

```text
alpha ∈ {0.00, 0.25, 0.50, 0.75, 1.00}
```

Replay từ trace hiện tại cho thấy `alpha = 0.25` là một điểm khởi đầu đáng thử:

| Dataset | α | Support recall | Support precision |
|---|---:|---:|---:|
| WebQA | 0.00 | 0.995 | 0.483 |
| WebQA | 0.25 | 0.985 | 0.393 |
| WebQA | 0.50 | 0.945 | 0.314 |
| WebQA | 1.00 | 0.200 | 0.057 |
| HotpotQA | 0.00 | 0.410 | 0.219 |
| HotpotQA | 0.25 | 0.455 | 0.244 |
| HotpotQA | 0.50 | 0.500 | 0.277 |
| HotpotQA | 1.00 | 0.565 | 0.322 |
| TAT-QA | 0.00 | khoảng 0.960 | 0.425 |
| TAT-QA | 0.25 | khoảng 0.968 | 0.386 |
| TAT-QA | 0.50 | khoảng 0.933 | 0.350 |
| TAT-QA | 1.00 | khoảng 0.617 | 0.258 |

Đây là replay selection từ attention trace chứ chưa phải một lần chạy answer generation hoàn chỉnh. Với MMQA nhiều vòng, thay đổi selection ở vòng đầu sẽ thay đổi attention của vòng sau nên không thể replay chính xác toàn bộ từ trace `(4)`.

### 6.3. Giai đoạn B: length correction riêng theo modality

Một `alpha` chung vẫn khó phù hợp với mọi modality. Có thể mở rộng thành:

\[
r_i = \log(m_i + \epsilon) - \alpha_{z_i}\log(n_i + c)
\]

Trong đó `z_i` là modality của chunk. Hệ thống học hoặc tune riêng:

```text
alpha_text
alpha_image
alpha_table
```

Giá trị khởi tạo để ablation, không phải giá trị kết luận:

```text
alpha_text  ∈ {0.50, 0.75, 1.00}
alpha_image ∈ {0.00, 0.25, 0.50}
alpha_table ∈ {0.00, 0.25, 0.50}
```

Các hệ số phải được chọn trên validation set, không tune trực tiếp trên 800 câu test hiện tại.

### 6.4. Giai đoạn C: calibrate bằng phân phối validation

Không nên normalize mỗi room về tổng 1 trong từng câu. Thay vào đó, ước lượng phân phối score của từng modality trên validation set rồi chuẩn hóa robust:

\[
z_i = \frac{r_i - \operatorname{median}(r \mid modality_i)}
{\operatorname{MAD}(r \mid modality_i) + \epsilon}
\]

Trong đó MAD là median absolute deviation.

Ưu điểm:

- Một singleton table không tự động được score 1.
- Score giữa các modality có ý nghĩa so sánh tốt hơn.
- Hiệu chỉnh được cả scale và variance riêng của mỗi modality.
- Ít nhạy với outlier hơn mean/std.

Nếu đủ dữ liệu validation, có thể thay median/MAD bằng một calibrator học được như logistic regression hoặc isotonic regression để ánh xạ feature sang xác suất chunk là support.

Feature đầu vào có thể gồm:

```text
log(attention_mass)
log(token_count)
modality
chunk position
retrieval score
attention entropy
```

### 6.5. Giai đoạn D: modality gate mềm theo câu hỏi

Thay vì quota cứng, ước lượng mức độ câu hỏi cần từng modality:

\[
g_m = P(modality=m \mid question)
\]

Sau đó kết hợp với calibrated score:

\[
f_i = z_i + \lambda\log(g_{z_i} + \epsilon)
\]

Ví dụ:

- Câu hỏi về màu sắc, hình dạng hoặc “shown/pictured” tăng prior cho image.
- Câu hỏi về tổng, chênh lệch, revenue hoặc số liệu tăng prior cho table.
- Câu hỏi dạng entity/relation có thể tăng prior cho text.

Gate phải là soft prior. Không nên đặt score của modality khác về 0 vì câu hỏi đa phương thức có thể cần kết hợp nhiều nguồn.

### 6.6. Giai đoạn E: soft reserve và token budget

Thay vì luôn giữ top-1 của mỗi room:

1. Chỉ reserve top-1 của modality nếu `g_m` vượt một ngưỡng validation.
2. Gộp phần còn lại và chọn global theo calibrated score.
3. Dừng theo tổng context-token budget hoặc cumulative calibrated probability, không chỉ theo số chunk.

Ví dụ:

```python
selected = []

for modality in active_modalities(question):
    if modality_gate[modality] >= gate_threshold[modality]:
        selected.append(best_chunk_of(modality))

remaining = sort_by_final_score(chunks_not_selected)
selected += fill_until_token_budget(remaining, token_budget)
```

Soft reserve giữ được tinh thần “chia room” nhưng không bắt buộc đưa một ảnh/table vô nghĩa vào mọi câu hỏi.

## 7. Thuật toán đề xuất tổng hợp

```python
# 1. Một forward chung để giữ cross-modal interaction.
mass = extract_final_token_attention_mass(all_chunks)
length = chunk_token_counts(all_chunks)

# 2. Length correction riêng theo modality.
adjusted = []
for chunk, m, n in zip(all_chunks, mass, length):
    alpha = alpha_by_modality[chunk.modality]
    adjusted.append(log(m + eps) - alpha * log(n + c))

# 3. Calibration theo thống kê validation của từng modality.
calibrated = []
for chunk, value in zip(all_chunks, adjusted):
    stats = validation_stats[chunk.modality]
    calibrated.append((value - stats.median) / (stats.mad + eps))

# 4. Query-conditioned soft modality prior.
gates = modality_gate(question)
final_score = [
    score + lambda_gate * log(gates[chunk.modality] + eps)
    for chunk, score in zip(all_chunks, calibrated)
]

# 5. Optional soft reserve, sau đó fill theo token budget.
selected = reserve_best_from_active_modalities(
    all_chunks, final_score, gates, gate_thresholds
)
selected = fill_by_global_score_until_budget(
    selected, all_chunks, final_score, context_token_budget
)
```

## 8. Kế hoạch ablation

Nên tách từng thay đổi để biết chính xác phần nào tạo ra cải thiện:

| ID | Phương pháp | Mục đích |
|---|---|---|
| A0 | Density, `alpha=1` | Tái lập `(3)` |
| A1 | Total mass, `alpha=0` | Tái lập `(4)` |
| A2 | Global `alpha=0.25` | Kiểm tra điểm trung gian đơn giản |
| A3 | Global `alpha=0.50` | Kiểm tra square-root normalization |
| A4 | Equal-room normalization | Kiểm tra đúng đề xuất Gemini |
| A5 | Per-modality alpha | Tách length bias theo modality |
| A6 | Per-modality robust calibration | Làm score giữa modality so sánh được |
| A7 | A6 + soft modality gate | Điều chỉnh theo loại câu hỏi |
| A8 | A7 + token-budget selection | Kiểm soát chi phí và độ dài context |

Mỗi phương pháp cần báo cáo:

- Answer EM và token F1.
- Support recall, precision và support F1.
- Recall/precision tại cùng retained-token budget.
- Số text/image/table được giữ.
- Tổng text token, visual token và table token được giữ.
- OOM rate, latency và peak VRAM.
- UQ AUROC bằng entropy và PPL.
- Paired bootstrap confidence interval giữa các phương pháp.

Không nên chọn phương pháp chỉ bằng support recall. Một thuật toán giữ gần như mọi chunk có thể có recall rất cao nhưng không thực hiện pruning hữu ích. WebQA `(4)` là ví dụ trực tiếp: recall cao vì hầu như cả support image lẫn distractor image đều được giữ.

## 9. Khuyến nghị cuối cùng

Đề xuất của nhóm và Gemini đúng ở điểm quan trọng: **phải làm scoring nhận biết modality và không được để số token quyết định relevance**. Việc giữ tất cả modality trong cùng một forward rồi hậu xử lý score là lựa chọn đúng.

Tuy nhiên, không nên dùng equal-room normalization và quota cứng làm thuật toán cuối cùng. Nó dễ thay length bias bằng group-size bias và modality-quota bias.

Thứ tự triển khai khuyến nghị:

1. Dùng trace `(4)` để hoàn tất offline ablation cho `alpha ∈ {0, 0.25, 0.5, 0.75, 1}`.
2. Chạy benchmark thật với `alpha=0.25` và `alpha=0.5` để đo answer EM/F1, OOM và latency.
3. Thêm `alpha` riêng cho text/image/table.
4. Học robust calibration trên validation split.
5. Sau cùng mới thêm soft modality gate và soft reserve.

Proposal chính nên được gọi là:

> **Query-Conditioned, Modality-Calibrated Attention Pruning under a Token Budget**

Nó giữ lại ý tưởng tốt nhất của “chia room”, nhưng tránh chạy model riêng theo modality và tránh bắt buộc giữ chunk nhiễu từ từng room.
