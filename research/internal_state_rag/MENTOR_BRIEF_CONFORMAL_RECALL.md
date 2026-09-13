# Trao đổi với mentor: conformal cosine precision cao nhưng recall thấp

## Kết luận một câu

Nút thắt chính nằm ở bước **candidate-wise BY selection**, không nằm ở việc
Top-L thiếu candidates. Retrieve full corpus và tăng L có thể cải thiện retrieval
coverage một ít, nhưng không giải quyết việc BY loại gần hết support đã có trong
Top-L. Nếu mục tiêu sản phẩm là giữ đủ evidence cho RAG, cần chuyển guarantee từ
candidate-wise FDR sang query-level support coverage, hoặc tách certified set và
fallback set.

## Số liệu chính

Thí nghiệm TAT-QA dùng 1,000 calibration queries từ official train và 1,000
evaluation queries từ official dev. Với Top-30:

| Score | BY alpha | Precision | Recall trong Top-L | Query empty |
|---|---:|---:|---:|---:|
| Jina cosine | 0.10 | 52.2% | **6.9%** | **93.3%** |
| BGE, modality bank | 0.10 | 49.1% | 9.7% | 91.2% |
| BGE, pooled bank | 0.10 | 50.8% | 11.6% | 89.4% |
| BGE + metadata/rank fusion | 0.10 | **70.2%** | 14.5% | 86.6% |
| BGE, pooled bank | 0.20 | 43.7% | 16.3% | 85.0% |

“Recall trong Top-L” là số support được chọn chia cho số support đã có sẵn
trong Top-L. Recall 6.9% vì vậy không thể được giải thích bằng việc corpus
retrieval chưa tìm thấy support: support đã ở trong reserve nhưng conformal loại
nó.

Precision 52.2% phải được đọc cùng empty rate 93.3%. Selector có thể trông khá
precision vì chỉ trả evidence cho một nhóm rất nhỏ các query dễ. Metric precision
không phạt việc 933/1,000 query không nhận được certified context.

`alpha=0.10` là mức FDR mục tiêu, không phải cosine threshold bằng 0.10. BY dùng:

```text
p_(i) <= i * alpha / (L * H_L)
```

Ở alpha 0.10, cutoff đầu tiên xấp xỉ:

- L=10: 0.00341;
- L=30: 0.000834.

Tăng L từ 10 lên 30 làm cutoff đầu tiên nghiêm hơn khoảng 4.1 lần.

## Tăng L và retrieve full corpus có tác dụng gì?

Top-10 đã chứa 793/864 support rows được tìm thấy trong Top-30 và cover 738/768
queries mà Top-30 có support. Tăng từ Top-10 lên Top-30 chỉ thêm 71 support rows
và 30 covered queries, nhưng thêm 20 hypotheses vào BY cho mọi query.

Retrieve full corpus vẫn cần thiết vì nó:

- bảo đảm mỗi query thật sự có đủ L candidates;
- sửa candidate pool ngắn/ragged;
- có thể tìm support nằm ngoài candidate list cũ;
- giữ corpus fingerprint giống nhau giữa calibration và test.

Nhưng sau khi support đã vào Top-L, full-corpus retrieval không làm p-value của
support đủ nhỏ để vượt BY. Nó sửa retrieval input, không sửa selection rule.

Backfill cũng chỉ thêm candidate ở rank `K+1..L` nếu candidate đó đã pass BY.
Nó không được phép lấy một chunk bị reject chỉ để lấp chỗ trống. Vì vậy Top-L
lớn hơn không giúp khi nguyên nhân là support đã vào reserve nhưng không pass
BY.

## Hướng giải quyết đề xuất

### 1. Query-level conformal coverage

Đổi câu hỏi thống kê từ “chunk này có chắc chắn không phải false match?” thành
“prediction set có giữ ít nhất một, hoặc toàn bộ, support cần thiết không?”.

Với score:

```text
z(BGE) + 0.20 z(Yes-vs-No LM-head)
       + 0.75 z(layer-30 relevance probe)
```

kết quả trên 256 fresh queries tại alpha 0.05 là:

| Target | Score | Chunks giữ / 30 | Support recall | Query coverage |
|---|---|---:|---:|---:|
| Any support | BGE | 5.00 | 92.3% | 94.5% |
| Any support | Internal fusion | **3.49** | **92.4%** | **95.3%** |
| All support | BGE | 26.76 | 99.4% | 99.2% |
| All support | Internal fusion | **9.92** | 97.9% | 97.3% |

Đây là hướng khuyến nghị khi recall/evidence coverage là yêu cầu chính. Claim
thống kê lúc này là query-level coverage, không còn là candidate-wise BY-FDR.

### 2. BY certified set cộng fallback tách biệt

Nếu bắt buộc giữ BY-FDR, output hai nhóm:

- certified chunks do BY chọn;
- fallback Top-1/Top-3, đánh dấu rõ là unverified.

Một split đã cho kết quả:

| Policy | Precision | Recall | Empty rate | Mean chunks |
|---|---:|---:|---:|---:|
| Certified fusion | 67.6% | 14.7% | 86.3% | chưa cố định |
| Certified + Top-1 fallback | 53.7% | 65.3% | 0% | 1.05 |
| Certified + Top-3 fallback | 25.0% | 80.2% | 0% | chưa báo cáo |

Guarantee chỉ áp dụng cho certified subset. Không được gọi fallback là
conformal-certified evidence.

### 3. Score tốt hơn trong BY

BGE và internal fusion giúp separation, nhưng không loại bỏ multiplicity penalty
của BY. Thí nghiệm tiếp theo phải train pairwise probe trên train split, khóa
prompt/layer/weights, tạo false-score bank từ 1,000 development calibration
queries riêng, rồi đánh giá một lần trên test role. Không dùng cùng examples để
train probe và calibrate p-values.

## Đề xuất chốt với mentor

- Giữ L=30 làm reserve; không tăng L để chữa recall.
- Giữ full-corpus retrieval để candidate construction đúng và reproducible.
- Nếu mục tiêu là RAG answer quality và evidence coverage: dùng internal fusion
  với all-support query-level conformal, alpha 0.05.
- Nếu mục tiêu bắt buộc là candidate-wise FDR: giữ BY, thêm fallback tách nhãn,
  và chấp nhận precision--recall trade-off.
- Chạy frozen method trên bank 1,000 dev queries của từng dataset trước khi đưa
  ra claim paper-level.

## Những câu không nên claim

- Không nói “full-corpus retrieval thất bại”. Nó đã sửa candidate construction;
  bottleneck hiện tại nằm ở selection.
- Không nói “conformal bị hỏng”. BY đang bảo vệ candidate-wise false discovery,
  nhưng guarantee đó không khớp trực tiếp với nhu cầu luôn có evidence cho RAG.
- Không so precision của candidate-wise BY với coverage method mà bỏ qua việc
  hai phương pháp kiểm soát hai event khác nhau.
- Chưa claim internal fusion đã giải quyết BY. Kết quả mạnh hiện tại thuộc
  query-level coverage; candidate-wise internal-BY vẫn cần bank 1,000 queries
  độc lập.
