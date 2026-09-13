# Full pruning comparison

BY controls candidate-level false discovery and can return an empty set. Query-level conformal targets retention of every support chunk conditional on support existing in Top-30 and uses deterministic Top-1 fallback.

| Method | Rule | α | Scope (n) | Chunks | Precision | Micro recall | Mean query recall | All-support coverage | Empty |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Attention-only trained probe | query conformal | 0.20 | 150 | 2.19 | 40.2% | 76.7% | 79.1% | 75.3% | 0.0% |
| Attention only (position-controlled) | query conformal | 0.20 | 150 | 1.91 | 43.7% | 72.7% | 74.6% | 71.3% | 0.0% |
| Attention only (raw) | query conformal | 0.20 | 150 | 1.81 | 44.3% | 69.8% | 71.6% | 68.0% | 0.0% |
| BGE (matched) | query conformal | 0.20 | 150 | 1.91 | 41.6% | 69.2% | 72.1% | 68.0% | 0.0% |
| Cosine + BY (pooled) | BY-FDR | 0.20 | 1000 | 0.33 | 25.3% | 9.6% | — | — | 89.5% |
| Cosine + BY (modality) | BY-FDR | 0.20 | 1000 | 0.28 | 29.0% | 9.4% | — | — | 90.5% |
| BGE + LM-head + hidden probe (matched) | query conformal | 0.20 | 150 | 1.39 | 56.7% | 68.6% | 71.8% | 67.3% | 0.0% |
| Attention-only trained probe | query conformal | 0.10 | 150 | 17.17 | 6.4% | 95.9% | 97.0% | 96.0% | 0.0% |
| Attention only (position-controlled) | query conformal | 0.10 | 150 | 7.06 | 14.1% | 86.6% | 87.4% | 84.7% | 0.0% |
| Attention only (raw) | query conformal | 0.10 | 150 | 8.86 | 11.4% | 87.8% | 89.4% | 87.3% | 0.0% |
| BGE (matched) | query conformal | 0.10 | 150 | 5.19 | 19.4% | 87.8% | 89.7% | 86.7% | 0.0% |
| Cosine + BY (pooled) | BY-FDR | 0.10 | 1000 | 0.15 | 38.5% | 6.6% | — | — | 93.3% |
| Cosine + BY (modality) | BY-FDR | 0.10 | 1000 | 0.12 | 52.2% | 6.9% | — | — | 93.3% |
| BGE + LM-head + hidden probe (matched) | query conformal | 0.10 | 150 | 2.70 | 36.0% | 84.9% | 87.4% | 83.3% | 0.0% |
| Attention-only trained probe | query conformal | 0.05 | 150 | 28.39 | 4.0% | 99.4% | 99.7% | 99.3% | 0.0% |
| Attention only (position-controlled) | query conformal | 0.05 | 150 | 22.23 | 5.0% | 97.1% | 97.7% | 96.7% | 0.0% |
| Attention only (raw) | query conformal | 0.05 | 150 | 20.81 | 5.3% | 96.5% | 97.0% | 96.7% | 0.0% |
| BGE (matched) | query conformal | 0.05 | 150 | 17.40 | 6.2% | 94.2% | 95.0% | 94.0% | 0.0% |
| Cosine + BY (pooled) | BY-FDR | 0.05 | 1000 | 0.09 | 54.7% | 5.4% | — | — | 94.5% |
| Cosine + BY (modality) | BY-FDR | 0.05 | 1000 | 0.06 | 61.0% | 4.2% | — | — | 96.0% |
| BGE + LM-head + hidden probe (matched) | query conformal | 0.05 | 150 | 6.12 | 17.6% | 94.2% | 95.0% | 93.3% | 0.0% |

## Cách đọc bảng

- Hai dòng cosine dùng candidate-wise Benjamini–Yekutieli (BY). Nó kiểm soát false discovery và được phép trả context rỗng. Kết quả lấy trên toàn bộ 1.000 test queries. Alpha sweep gốc tự đánh dấu là diagnostic vì tái sử dụng p-value từ development smoke, nên chưa phải kết quả paper-ready.
- Các dòng còn lại dùng split conformal ở query level, với mục tiêu giữ **tất cả** support chunks đang có trong Top-30. Nếu threshold không giữ chunk nào thì dùng deterministic Top-1 fallback.
- So sánh matched dùng đúng 136 calibration queries chung và đúng cùng 150 locked test queries. Cả 150 test queries này đều có ít nhất một support trong Top-30, nên đây là kết quả conditional on retrievability.
- Internal fusion gồm `z(BGE) + 0.20 z(Yes-vs-No LM-head) + 0.75 z(layer-30 hidden-state probe)`. Hidden probe được train trên 96 queries tách rời. Attention-only trained probe được train trên 120 queries khác và chỉ nhận attention mass/fraction, không nhận BGE.

## Ranking trên cùng 150 test queries

| Signal | Mean query AP | MRR | Top-1 support |
|---|---:|---:|---:|
| BGE | 71.8% | 73.6% | 61.3% |
| Attention raw | 72.2% | 74.2% | 66.0% |
| Attention position-controlled | 73.2% | 75.4% | 67.3% |
| Attention-only trained probe | 76.1% | 78.6% | 70.7% |
| BGE + LM-head + hidden probe | **77.8%** | **79.8%** | 70.0% |

## Kết luận ở α = 0.10

Candidate-wise BY chính là nguyên nhân của hiện tượng “high precision, almost everything is dropped”: modality-aware BY đạt 52.2% precision nhưng recall chỉ 6.9%, giữ 0.12 chunk/query và để rỗng 93.3% queries. Chỉ tăng alpha từ 0.05 lên 0.20 vẫn không sửa được: recall mới 9.4% và 90.5% queries còn rỗng.

Attention có tín hiệu relevance thật. Attention raw và position-controlled đều nâng recall lên khoảng 87% với query-level conformal. Tuy nhiên, để đạt mức đó, chúng phải giữ lần lượt 8.86 và 7.06 chunks/query, khiến precision chỉ còn 11.4% và 14.1%. Probe được train từ attention giúp ranking, nhưng tail calibration xấu: ở α=0.10 nó phải giữ 17.17 chunks để đạt 95.9% micro recall.

Internal fusion cho trade-off tốt nhất trong matched comparison: giữ 2.70 chunks/query, precision 36.0%, micro recall 84.9%, và all-support coverage 83.3%. So với position-controlled attention, fusion giảm 4.36 chunks/query và tăng precision 21.9 điểm phần trăm, đổi lại micro recall giảm 1.7 điểm phần trăm. Trên validation lớn gồm 768 retrievable test queries, internal fusion ở α=0.10 giữ 3.39 chunks, đạt 29.3% precision, 88.3% micro recall và 87.9% all-support coverage.

Vì thế attention-only không phải lời giải cuối. Tín hiệu internal hữu ích nhất khi bổ sung cho reranker: LM-head và hidden probe sửa thứ tự candidate, còn query-level conformal quyết định mức pruning theo recall target. Nếu mentor muốn ưu tiên recall khoảng 90%, điểm vận hành α=0.10 của internal fusion hợp lý hơn BY; α=0.05 tăng recall lên 94.1% nhưng giữ 6.30 chunks trên validation lớn.

## Provenance

- Fresh A100 smoke rerun: 3/3 queries, Top-30, 12 answer tokens tối đa, cả original và reversed order.
- Attention numbers: raw traces đã checkpoint từ 120 train + 150 calibration + 150 locked test queries; phép fit/calibration/evaluation trong bảng được chạy lại với code mới.
- Internal large validation: 96 probe-train + 904 calibration queries (714 retrievable) + 1.000 test queries (768 retrievable), không overlap query ID.

## Thí nghiệm cải thiện frontier

Ba hướng tiếp theo đã được kiểm tra mà không chạy lại feature extraction:

| Thí nghiệm, α=0.10 | Test n | Chunks | Precision | Micro recall | All-support coverage | Kết luận |
|---|---:|---:|---:|---:|---:|---|
| Baseline internal fusion, probe 96 | 768 | 3.388 | 29.32% | 88.31% | 87.89% | Mốc so sánh |
| Query-adaptive threshold | 768 | 4.549 | 21.89% | 88.54% | 87.24% | Không cải thiện frontier |
| Thêm attention vào fusion | 150 | 2.700 | 36.05% | 84.88% | 83.33% | Weight được chọn trên train là 0.0; attention bị loại |
| **Probe train 452 + retuned fusion** | **768** | **3.233** | **30.85%** | **88.66%** | **88.41%** | Pareto improvement nhỏ |

So với baseline 96-query probe, probe 452-query ở α=0.10 giảm 0.155 chunk/query, paired-bootstrap 95% CI `[-0.241, -0.072]`, và tăng precision 1.53 điểm phần trăm, CI `[+0.65, +2.39]`. Recall tăng 0.35 điểm, CI `[-1.50, +2.18]`, và all-support coverage tăng 0.52 điểm, CI `[-1.43, +2.47]`; hai thay đổi coverage này chưa có ý nghĩa thống kê. Vì vậy kết luận chắc nhất là thêm probe-training data cải thiện efficiency/precision ở mức recall không phân biệt được với baseline.

Không có bằng chứng rằng threshold prediction hay cộng attention đơn giản sẽ tạo bước nhảy lớn. Với 768 retrievable test queries chỉ có trung bình 1.125 support chunks/query, trong khi hệ thống đang giữ 3.23 chunks; do đó vẫn còn headroom nếu score phân biệt support tốt hơn.

## Follow-up: causal-value, hard negatives, và calibration size

Các hướng còn lại ở trên đã được chạy. Leave-one-out causal teacher trên 128
queries cho support mean log-probability drop 2.3665, so với 0.0015 cho
non-support. Tuy nhiên raw causal-value xếp support kém BGE (AP 68.66% so với
79.24%). Late-4-layer RankNet dự đoán causal order với holdout Spearman 0.1242,
nhưng support AP trên test chỉ 52.04%; tuning chọn causal fusion weight bằng 0.0.
Causal usage vì vậy chưa phải một relevance score tốt.

Hard-negative mining cũng không cải thiện: 5-fold OOF chọn lại toàn bộ negatives.
Learning curve tăng test fusion AP từ 79.39% ở 256 probe-training queries lên
79.83% ở 452, cho thấy thêm cùng loại labels chỉ còn gain nhỏ.

Calibration-size sweep 200 repeats cho thấy ở α=0.10, 20 retrievable queries làm
số chunks giữ lại dao động 95% từ 1.70 đến 12.92; 128 queries còn 2.53–4.48;
256 còn 2.75–3.82. Full bank 363 giữ 3.23 chunks. Vì vậy 256 retrievable queries
là mức pilot thực dụng cho α=0.10, còn khoảng 500+ phù hợp hơn nếu cần threshold
ổn định. Chi tiết và raw-log provenance nằm trong
`CAUSAL_PROBE_AND_CALIBRATION_STUDY_2026-09-13.md`.
