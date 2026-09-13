# Causal internal-state probe và calibration-size study

## Câu hỏi nghiên cứu

Ba thí nghiệm được chạy theo thứ tự đã khóa trước: (2) tạo causal-value label
bằng cách bỏ từng chunk, (3) học causal-value từ hidden states rồi dùng nó để
prune, và (1) kiểm tra thêm dữ liệu, hard-negative mining, cùng số query cần cho
conformal calibration. Mọi kết quả bên dưới dùng Top-30 full-corpus candidates.
Test chỉ dùng 1.000 official-dev queries; model selection và calibration không
dùng test labels.

## 1. Causal teacher: chunk nào thật sự làm model tăng xác suất gold answer?

Với mỗi query và từng chunk `i`, teacher đo

```text
delta_i = log p(gold answer | query, all Top-30 chunks)
        - log p(gold answer | query, Top-30 except chunk i)
```

`delta_i > 0` nghĩa là bỏ chunk làm xác suất gold answer giảm. Qwen2.5-3B-Instruct
được giữ frozen. Batched leave-one-out giảm thời gian từ khoảng 26.3 xuống 14.8
giây/query trong benchmark batch-size 4; hai cách cho score tương quan 0.99969
và giữ nguyên Top-1/Top-5. Run chính gồm 128 queries, tách 96 train và 32 causal
holdout, hoàn tất trong 1,650.78 giây (khoảng 27.5 phút), trung bình 12.72
giây/query.

| Statistic trên 128 queries | Support | Non-support |
|---|---:|---:|
| Số chunks | 141 | 3.699 |
| Mean causal-value | 2.3665 | 0.0015 |
| Fraction có causal-value dương | 77.30% | 52.61% |

Causal-value chứa tín hiệu: support chunks có mean drop lớn hơn rõ rệt. Nhưng nó
không đồng nhất với relevance label. Nếu dùng raw causal-value để xếp support,
mean query AP là 68.66%, MRR 71.48%, Top-1 support 67.97%; BGE trên đúng 128
queries đạt lần lượt 79.24%, 81.57%, và 72.66%. Leave-one-out đo **mức model
đang sử dụng chunk trong một context cụ thể**, không đo đầy đủ chunk có liên quan
hay không. Một support có thể dư thừa nên drop nhỏ; một distractor cũng có thể
làm đổi gold likelihood do tương tác hoặc nhiễu.

## 2. Causal RankNet từ hidden states

Probe dùng query-centered hidden state của từng chunk và pairwise RankNet loss
theo thứ tự causal-value. Bốn cấu hình được chọn chỉ trên 32 causal holdout
queries, mỗi cấu hình ensemble ba seeds.

| Probe | Holdout Spearman | Top-causal MRR | Top-causal Top-1 |
|---|---:|---:|---:|
| Layer 30 linear | 0.0751 | **49.40%** | **37.50%** |
| Layer 30 MLP | 0.1083 | 31.03% | 18.75% |
| Layers 18/24/30/35 MLP | **0.1242** | 37.16% | 25.00% |
| Late-4 MLP + BGE/LM-head/rank | 0.1214 | 38.69% | 25.00% |

Late-4 MLP thắng theo tiêu chí Spearman đã định trước, chứng tỏ hidden trajectory
có một lượng thông tin yếu về causal usage. Tuy nhiên, khi refit trên toàn bộ 128
queries và đánh giá 768 retrievable test queries, causal probe chỉ đạt support AP
52.04%, so với 79.96% của three-signal baseline. Tuning trên causal holdout chọn
trọng số causal bằng **0.0**; mọi trọng số dương từ 0.05 đến 2.0 đều không tăng AP.

| Test ranking, n=768 retrievable | Mean query AP | MRR | Top-1 support |
|---|---:|---:|---:|
| BGE | 74.45% | 76.77% | 66.41% |
| BGE + LM-head + hidden relevance probe | **79.96%** | **82.22%** | **72.66%** |
| Causal RankNet alone | 52.04% | 54.35% | 42.58% |
| Baseline + causal RankNet | **79.96%** | **82.22%** | **72.66%** |

Kết quả này bác bỏ phiên bản đơn giản của giả thuyết “internal state có thể
distill single leave-one-out drop thành relevance pruner”. Nó không bác bỏ toàn
bộ hướng internal-state. Góc mới hợp lý hơn là tách hai đại lượng: **semantic
relevance** và **conditional indispensability**. Causal-value nên làm label cho
redundancy/diversity hoặc cho quyết định thay thế một chunk, thay vì cộng tuyến
tính trực tiếp vào relevance score. Một teacher tốt hơn cần đo expected marginal
value qua nhiều context subsets (Shapley-style hoặc random-subset masking), hoặc
đo answer-margin giữa gold và một answer đối chứng; single leave-one-out phụ
thuộc mạnh vào 29 chunks còn lại.

## 3. Hard-negative mining và learning curve

Layer-30 relevance probe được train trên 452 queries. Mỗi hard-negative cấu hình
giữ mọi support và chỉ giữ top 3/5/10/15 non-support theo BGE. Cấu hình được chọn
bằng 5-fold group OOF trên train, không nhìn test.

| Negative set | Probe OOF AP | Fusion OOF AP |
|---|---:|---:|
| **Tất cả negatives** | **78.37%** | **84.59%** |
| Top-3 BGE negatives | 72.27% | 83.69% |
| Top-5 BGE negatives | 74.66% | 84.06% |
| Top-10 BGE negatives | 75.77% | 84.00% |
| Top-15 BGE negatives | 77.95% | 84.49% |

OOF chọn lại toàn bộ negatives. Chỉ train trên hard negatives làm mất thông tin
về biên score toàn query và không cải thiện pruning. Learning curve của cấu hình
được chọn cho thấy phần lớn gain xuất hiện sau 128 queries:

| Probe-training queries | Retrievable train | Test fusion AP | MRR | Top-1 support |
|---:|---:|---:|---:|---:|
| 32 | 30 | 73.15% | 75.17% | 62.11% |
| 64 | 52 | 76.17% | 78.31% | 66.80% |
| 128 | 96 | 76.00% | 77.82% | 66.54% |
| 256 | 201 | 79.39% | 81.35% | 71.35% |
| 452 | 351 | **79.83%** | **81.71%** | **72.14%** |

Đường cong chưa bão hòa hoàn toàn, nhưng gain 256→452 nhỏ. Muốn có bước nhảy
lớn cần label/objective tốt hơn; tăng thêm cùng loại dữ liệu chỉ hứa hẹn gain
nhỏ.

## 4. Cần bao nhiêu calibration queries?

Bank cố định có 452 queries, trong đó 363 queries có support trong Top-30. Với
mỗi cỡ bank nhỏ hơn, 200 random permutations được tạo từ seed 1291; các subset
lồng nhau trong cùng repeat. Raw log ghi threshold, finite-sample order, mọi
metric và SHA-256 của subset query IDs. Các khoảng dưới đây là percentile
2.5–97.5% qua 200 calibration subsets, trên cùng 768 retrievable test queries.

### Alpha 0.10, three-signal fusion

| Retrievable calibration queries | Mean chunks (95% range) | Mean precision | Mean micro recall | All-support coverage (95% range) |
|---:|---:|---:|---:|---:|
| 20 | 4.48 (1.70–12.92) | 28.83% | 88.41% | 88.12% (75.38–97.66%) |
| 32 | 4.06 (1.92–8.37) | 28.64% | 88.73% | 88.45% (77.34–96.24%) |
| 64 | 3.55 (2.26–6.00) | 29.80% | 88.45% | 88.18% (81.33–93.93%) |
| 128 | 3.43 (2.53–4.48) | 29.65% | 88.78% | 88.50% (83.72–91.30%) |
| 256 | 3.29 (2.75–3.82) | 30.49% | 88.57% | 88.31% (85.68–90.10%) |
| 363 full bank | 3.23 | 30.85% | 88.66% | 88.41% |

20–64 calibration queries không đủ để dự báo context cost: cùng một method có
thể giữ khoảng 2 chunks hoặc hơn 10 chunks chỉ vì chọn bank khác. 128 dùng được
cho pilot nhưng còn rộng. 256 retrievable queries là mức tối thiểu thực dụng ở
alpha 0.10 trong thí nghiệm này; 363 vẫn nên được dùng khi có sẵn. Dòng full-bank
chỉ có một subset nên độ lệch chuẩn bằng 0 không phải bằng chứng rằng một bank
363-query mới sẽ ổn định tuyệt đối.

### Tóm tắt theo alpha

| Alpha | Cỡ bank | Mean chunks (95% range) | Mean micro recall | All-support coverage (95% range) |
|---:|---:|---:|---:|---:|
| 0.20 | 128 | 1.99 (1.66–2.38) | 78.79% | 78.44% (74.87–82.55%) |
| 0.20 | 256 | 1.93 (1.79–2.05) | 78.22% | 77.72% (76.04–79.30%) |
| 0.10 | 128 | 3.43 (2.53–4.48) | 88.78% | 88.50% (83.72–91.30%) |
| 0.10 | 256 | 3.29 (2.75–3.82) | 88.57% | 88.31% (85.68–90.10%) |
| 0.05 | 128 | 5.83 (3.89–8.37) | 93.54% | 93.20% (90.10–96.22%) |
| 0.05 | 256 | 5.33 (4.42–6.44) | 93.04% | 92.77% (91.15–94.53%) |

Quantile tail chỉ có xấp xỉ `alpha × n` observations quyết định threshold. Với
363 retrievable queries, effective tail chỉ khoảng 36 ở alpha 0.10 và 18 ở
alpha 0.05. Một planning heuristic là nhắm ít nhất 50 tail observations:

| Alpha | Retrievable calibration cần cho ~50 tail points | Raw queries nếu retrievability ~80% |
|---:|---:|---:|
| 0.20 | 250 | ~313 |
| 0.10 | 500 | ~625 |
| 0.05 | 1.000 | ~1.250 |

Đây là heuristic về ổn định threshold, không phải cỡ mẫu tối thiểu trong định lý
split conformal. Nếu đã có 1.000 development queries mỗi dataset và khoảng 80%
retrievable, nên giữ chúng làm bank riêng: khoảng 800 usable queries cho alpha
0.10 và khoảng 40 tail observations cho alpha 0.05. Không nên lấy bớt một nửa
bank đó để train probe nếu mục tiêu paper là claim calibration chắc.

Observed all-support coverage trên official-dev thấp hơn nominal ở full bank:
77.73% so với target 80% tại alpha 0.20, 88.41% so với 90% tại alpha 0.10, và
92.06% so với 95% tại alpha 0.05. Tăng calibration size giảm variance nhưng
không tự sửa distribution shift giữa official-train calibration và official-dev
test. Cần calibration cùng distribution, calibration theo dataset/modality, hoặc
weighted conformal nếu muốn phát biểu guarantee dưới shift.

## Kết luận để trao đổi với mentor

- High precision/low recall ban đầu đến từ candidate-wise BY; query-level
  all-support conformal là sửa đúng về mục tiêu thống kê.
- Internal relevance fusion hiện là điểm vận hành tốt nhất: tại alpha 0.10 giữ
  3.23/30 chunks, precision 30.85%, micro recall 88.66% trên retrievable test.
- Causal leave-one-out teacher tìm được tín hiệu mới trong model, nhưng tín hiệu
  đó là conditional usage và không thể cộng trực tiếp như relevance. Causal
  RankNet bị validation loại với weight 0.
- Hard-negative mining không cải thiện; thêm relevance training từ 256 lên 452
  chỉ tăng AP 0.43 điểm phần trăm.
- Với alpha 0.10, dùng ít nhất 256 retrievable calibration queries cho pilot và
  ưu tiên 500+ cho kết quả ổn định. Với alpha 0.05, 363 vẫn còn quá ít ở phần
  tail; bank khoảng 1.000 retrievable queries hợp lý hơn.
- Hướng nghiên cứu mới đáng thử tiếp là causal marginal value qua random context
  subsets hoặc answer-margin attribution, sau đó dùng nó để phát hiện redundancy
  giữa chunks. Đây là objective khác relevance và có cơ hội cải thiện frontier
  thay vì chỉ nới threshold.

## Artifacts

- `results/causal_value_batched_n128_seed809.json`: 3.840 leave-one-out labels.
- `results/causal_ranknet_n96_holdout32_cal776_test1000.json`: architecture,
  holdout, fusion, ranking và conformal results.
- `results/causal_ranknet_n96_holdout32_cal776_test1000_predictions.npz`: raw
  causal/baseline/fusion predictions.
- `results/hard_negative_calibration_curve_train452_cal363_test1000.json`: full
  learning curve và 200-repeat calibration log.
- `results/hard_negative_calibration_curve_train452_cal363_test1000_predictions.npz`:
  scores và labels để tái phân tích.
