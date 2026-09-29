# Chẩn đoán CCE, CONFLARE và TRAQ trên split `splits_khanh_27_09`

Phân tích này đọc trực tiếp selector trong
[`run_literature_protocol_1000cal.py`](../../run_literature_protocol_1000cal.py), các
cutoff trong [`summary.json`](summary.json), retrieval metrics trong
[`selection_summary.json`](selection_summary.json), và downstream metrics trong
[`REPORT.md`](REPORT.md). Candidate pool, Jina cosine scores, calibration/test split và
generator được giữ cố định. Phần dưới phân tích trực tiếp ba toán tử post-retrieval trong run
này: calibration score, threshold, selected chunks và downstream output.

## 1. Code thực sự quyết định giữ chunk như thế nào

- **CCE** gộp mọi cosine score của support chunks trong calibration set. Với
  nonconformity $a=1-s$, code lấy quantile $1-\alpha$ của $a$, tương đương lấy lower
  $\alpha$ quantile của support cosine. Candidate test được giữ khi score lớn hơn hoặc bằng
  cutoff. Một calibration query có nhiều support chunks đóng góp nhiều điểm hơn query chỉ có
  một support chunk.
- **CONFLARE** lấy support score lớn nhất của mỗi calibration query, chuyển thành
  distance $1-s$, rồi lấy percentile $1-\alpha$. Candidate test được giữ nếu score lớn hơn
  cutoff. Mỗi calibration query đóng góp đúng một score và các support còn lại bị collapse.
- **TRAQ** cũng lấy support score lớn nhất của mỗi calibration query,
  nhưng dùng retrieval budget $\alpha_R=\alpha/2=0.05$. Cutoff vì thế thường thấp hơn và
  selector permissive hơn.

Cả ba selector cuối cùng đều là một **dataset-wide scalar cosine cutoff**. Chúng không dùng
false-score distribution, không điều chỉnh score theo query, không condition theo modality
hoặc rank, và không xử lý family of candidates trong một query bằng multiple testing.

## 2. Kết quả cho thấy selector giữ quá nhiều chunk

`Pool/query` là candidate pool trung bình trước pruning. `Pool support` là tỷ lệ support thật
trong toàn pool. `Kept %` là tỷ lệ tất cả candidates vượt cutoff. AUC đo khả năng một cosine
score ngẫu nhiên xếp support cao hơn false; 0.5 là gần như random.

| Dataset | Pool/query | Pool support | Cosine AUC | CCE kept % / P / R | CONFLARE kept % / P / R | TRAQ kept % / P / R |
|---|---:|---:|---:|---:|---:|---:|
| HotpotQA | 9.85 | 20.3% | 0.543 | 87.4% / 21.0% / 90.5% | 71.6% / 21.0% / 74.0% | 82.9% / 21.4% / 87.5% |
| MMQA | 21.87 | 7.1% | 0.609 | 84.1% / 7.7% / 91.0% | 82.5% / 7.8% / 90.3% | 89.2% / 7.5% / 94.2% |
| TAT-QA | 6.13 | 21.9% | 0.620 | 81.7% / 24.6% / 91.8% | 75.7% / 25.0% / 86.6% | 83.0% / 24.6% / 93.3% |
| WebQA | 28.11 | 6.2% | 0.621 | 83.1% / 6.5% / 86.3% | 76.8% / 6.8% / 83.4% | 87.8% / 6.3% / 88.6% |

Ba dấu hiệu quan trọng:

1. **Pruning yếu:** selector giữ 71.6%–89.2% pool. MMQA còn 18.05–19.50 chunks/query;
   WebQA còn 21.60–24.68 chunks/query.
2. **Hầu như không enrich support:** HotpotQA có 20.3% support trong pool nhưng output chỉ
   đạt 21.0%–21.4% precision. WebQA có 6.2% support trong pool và output chỉ đạt 6.3%–6.8%.
   TAT-QA tốt hơn một chút, từ 21.9% lên 24.6%–25.0%, nhưng vẫn giữ hơn ba phần tư pool.
3. **Số false chunks vẫn rất lớn:** mỗi query, CCE giữ trung bình 6.80 false chunks trên
   HotpotQA, 16.99 trên MMQA, 3.78 trên TAT-QA và 21.85 trên WebQA. TRAQ còn permissive hơn:
   6.42, 18.04, 3.84 và 23.13 false chunks/query.

Đây không đơn thuần là lỗi chọn `alpha`. Với $\alpha=.10$, cutoff được đặt gần lower 10% của
support distribution để ưu tiên coverage. Khi support và false score overlap mạnh, cutoff đủ
thấp để giữ 90% support cũng tất yếu giữ phần lớn false candidates.

## 3. Cosine score chưa đủ tách support và false

| Dataset | Support cosine q10 / median / q90 | False cosine q10 / median / q90 | CCE cutoff | CONFLARE cutoff | TRAQ cutoff |
|---|---:|---:|---:|---:|---:|
| HotpotQA | .836 / .910 / .951 | .815 / .900 / .947 | .830 | .872 | .845 |
| MMQA | .770 / .887 / .944 | .705 / .864 / .927 | .757 | .767 | .720 |
| TAT-QA | .894 / .938 / .960 | .849 / .924 / .958 | .885 | .898 | .883 |
| WebQA | .756 / .900 / .947 | .748 / .871 / .929 | .789 | .814 | .762 |

- Trên HotpotQA, support median .910 và false median .900 gần như chồng lên nhau; AUC .543
  giải thích vì sao cả ba methods cho precision gần base rate dù giữ số chunk khác nhau.
- MMQA, TAT-QA và WebQA có AUC khoảng .61–.62: score có signal, nhưng chưa đủ mạnh để một
  cutoff toàn dataset vừa giữ recall khoảng 90% vừa loại nhiều false chunks.
- CONFLARE thường có cutoff cao nhất nên prune mạnh hơn. Tuy nhiên, signal không đủ sạch;
  phần recall mất đi không luôn đổi thành downstream gain.
- TRAQ dùng $\alpha_R=.05$, nên cutoff thấp hơn và thường giữ nhiều chunks nhất. Đây là hành
  vi đúng với một coverage-oriented cutoff, không phải một precision-oriented pruner.

## 4. Cơ chế tạo ra failure pattern trong bảng kết quả

### 4.1. False-score distribution

Vì threshold được suy ra từ support distribution, nó phải đủ thấp để giữ các support khó.
Khi false scores overlap với support scores, chính cutoff thấp này cho nhiều false chunks đi
qua. Hậu quả được thấy trực tiếp ở MMQA và WebQA: false prevalence là 92.9% và 93.8%, còn
output precision chỉ 7.5%–7.8% và 6.3%–6.8%. False-bank p-value ở Stage 1 được đưa vào để
đánh giá candidate theo mức độ nó khác false distribution, nhắm trực tiếp vào failure này.

### 4.2. Query-relative score distribution

Vì từng candidate được so độc lập với một cutoff, số chunks đã được giữ và kích thước pool
không ảnh hưởng tới quyết định tiếp theo. Hậu quả là context size tăng gần theo pool size:
MMQA bắt đầu với 21.87 candidates/query và còn 18.05–19.50; WebQA bắt đầu với 28.11 và còn
21.60–24.68. Query-level BH/BY đưa toàn bộ candidate family vào cùng một quyết định để kiểm
soát hiện tượng này.

### 4.3. Modality

| Dataset / modality | Candidates | Support prevalence | Cosine AUC |
|---|---:|---:|---:|
| MMQA image | 1,087 | 3.2% | 0.598 |
| MMQA table | 100 | 49.0% | 0.734 |
| MMQA text | 1,000 | 7.1% | 0.627 |
| TAT-QA table | 100 | 78.0% | 0.832 |
| TAT-QA text | 513 | 10.9% | 0.620 |
| WebQA image | 1,275 | 6.0% | 0.555 |
| WebQA text | 1,536 | 6.4% | 0.683 |

Vì pooled cutoff trộn các score regimes này, cùng một ngưỡng không thể đồng thời chặt cho
WebQA image gần random và mềm cho TAT-QA table có evidence rõ. Hậu quả phù hợp với WebQA:
selector giữ 76.8%–87.8% pool nhưng precision chỉ 6.3%–6.8%. Modality-conditioned banks kiểm
tra candidate trong đúng score regime của nó; ablation pooled versus modality-aware sẽ xác
định mức cải thiện thực tế.

### 4.4. Retrieval rank

Rank vẫn chứa signal bổ sung. Top-1 precision là 42% trên HotpotQA và 45% trên TAT-QA, cao hơn
precision sau global threshold. Trên MMQA và WebQA, Top-1 precision chỉ 20%, nên rank không đủ
dùng một mình. Cách hợp lý là condition bank theo rank bin hoặc dùng rank như covariate, thay
vì quay lại một hard Top-$k$ cutoff.

### 4.5. Multi-support structure

CONFLARE và TRAQ collapse mỗi query thành maximum support score, nên cutoff phản ánh support
dễ nhất thay vì support yếu nhất cần giữ. Trên HotpotQA, CONFLARE vẫn có any-support coverage
88% nhưng all-support coverage chỉ 60%; TRAQ tăng all-support lên 79% bằng cutoff permissive
hơn và giữ 8.17/9.85 chunks/query. Vì vậy all-support coverage phải trở thành constraint khi
chọn operating point, thay vì chỉ tối đa hóa precision.

## 5. Downstream không nói rằng cứ giữ nhiều hơn hoặc ít hơn là tốt hơn

| Dataset | CCE chunks / F1 | CONFLARE chunks / F1 | TRAQ chunks / F1 | Quan sát |
|---|---:|---:|---:|---|
| HotpotQA | 8.61 / .653 | 7.05 / .623 | 8.17 / .642 | CONFLARE prune mạnh hơn nhưng mất 16.5 điểm recall và giảm 3.0 F1 points so với CCE. |
| MMQA | 18.40 / .436 | 18.05 / .462 | 19.50 / .435 | CONFLARE có downstream F1 cao nhất dù retrieval metrics gần nhau; bớt noise có thể giúp. |
| TAT-QA | 5.01 / .490 | 4.64 / .432 | 5.09 / .500 | TRAQ giữ recall cao nhất và có F1 tốt nhất; prune evidence làm task này giảm rõ. |
| WebQA | 23.36 / .260 | 21.60 / .255 | 24.68 / .246 | Thay đổi 3 chunks/query gần như không thay đổi EM (.11); selection signal hiện tại chưa tác động mạnh tới generator. |

Kết quả cho thấy không có quy luật đơn điệu “ít chunk hơn luôn tốt” hoặc “recall cao hơn luôn
tốt”. MMQA có dấu hiệu hưởng lợi từ noise reduction, TAT-QA phụ thuộc coverage, còn WebQA bị
giới hạn bởi score/image signal hoặc generator hơn là bởi vài chunks chênh lệch. Paper nên
đánh giá Pareto theo từng dataset và dùng downstream quality cùng selection metrics.

## 6. Hàm ý trực tiếp cho phương pháp Two-Stage

- **Stage 1 có lý do tồn tại:** dùng false bank để tạo signal mà ba support-only cutoffs bỏ
  qua. Tuy nhiên, nếu false/support cosine overlap như HotpotQA, Stage 1 sẽ phải chấp nhận một
  precision–recall trade-off rõ ràng.
- **Stage 2 cần được trình bày như precision-oriented pruning:** support bank có thể loại
  candidates không giống evidence, nhưng bản thân cosine không bảo đảm tách sạch hai lớp.
- **Modality-aware và rank-conditioned banks là ablation ưu tiên:** phân tích trên cho thấy
  heterogeneity đủ lớn để kỳ vọng tốt hơn pooled bank.
- **Operating point phải chọn trên calibration set:** báo Pareto sweep và chọn alpha bằng
  objective có recall constraint, thay vì chọn test alpha có F1 cao nhất.
- **Head-to-head phải chạy cùng split:** các Two-Stage numbers dùng split khác không được đặt
  vào bảng này. Cần chạy proposed selectors trên chính `splits_khanh_27_09`, candidate pool
  và downstream protocol này.

Finding hiện tại không phải “literature baselines kém”. Finding là: trong frozen Jina pool,
ba support-calibrated cutoffs đều đạt coverage cao bằng cách giữ phần lớn pool, vì
cosine support/false separability thấp và cutoff không dùng false, query, modality hoặc rank
information. Đây là failure mode cụ thể mà Two-Stage phải chứng minh nó khắc phục được.
