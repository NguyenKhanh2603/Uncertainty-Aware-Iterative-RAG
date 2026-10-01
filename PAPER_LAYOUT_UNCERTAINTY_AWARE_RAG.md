# Dàn bài bài báo: Uncertainty-Aware Conformal Context Selection for RAG

> **Trạng thái:** layout và writing plan, chưa phải bản thảo hoàn chỉnh.  
> **Ngôn ngữ dự kiến của paper:** tiếng Anh.  
> **Mục tiêu độ dài:** 8–10 trang nội dung chính, chưa tính tài liệu tham khảo và phụ lục.

## 0. Working Title và thông điệp trung tâm

### 0.1. Working title đề xuất

- **Uncertainty-Aware Context Selection for Retrieval-Augmented Generation**
- **Two-Stage Conformal Context Selection for Efficient Retrieval-Augmented Generation**
- **From Candidate Recall to Context Precision: Two-Stage Conformal Selection for RAG**

### 0.2. Câu hỏi nghiên cứu trung tâm

- Sau khi retriever trả về một candidate pool, làm thế nào để chọn số lượng context theo từng query thay vì luôn dùng Fixed Top-$k$?
- Có thể dùng hai calibration banks — false evidence và support evidence — để tách hai quyết định:
  - candidate nào có đủ tín hiệu để được đưa vào context;
  - candidate nào có score thấp bất thường so với support evidence và nên bị prune?
- Việc prune có cải thiện được downstream EM/F1 và giảm generation cost hay chỉ đổi precision lấy recall?

### 0.3. Thông điệp nên xuyên suốt paper

- Fixed Top-$k$ tạo một operating point duy nhất và không phản ánh độ khó khác nhau giữa các query.
- Two-Stage Conformal Selection tạo một **đường trade-off có thể điều chỉnh** giữa support recall, support precision, context size và generation cost.
- Giá trị chính của phương pháp không nên được mô tả là thắng mọi baseline trên mọi dataset.
- Claim phù hợp với số hiện có là:
  - Stage 1 permissive có thể tạo candidate set recall cao;
  - Stage 2 dịch operating point về phía context ngắn và precision cao hơn;
  - một số operating point cải thiện downstream QA hoặc giữ chất lượng gần tương đương với chi phí thấp hơn;
  - operating point tốt nhất phụ thuộc dataset và mục tiêu sử dụng.

### 0.4. Đóng góp dự kiến

- Đề xuất một post-retrieval framework hai giai đoạn với false-reference bank và support-reference bank.
- Thực hiện multiple testing theo query thay vì dùng một global cosine threshold cho mọi query.
- Đánh giá đồng thời ba lớp kết quả:
  - context-selection quality;
  - downstream answer quality;
  - context/generation cost.
- So sánh các toán tử post-retrieval trên bốn text/multimodal datasets với cùng split,
  candidate pool, cosine scores và generator.
- Phân tích calibration size, modality conditioning, $\alpha$ sweep, BH/BY và các ablation của từng stage.

---

## 1. Abstract

- Mở đầu bằng vấn đề:
  - RAG thường truyền một số lượng cố định các chunk vào generator;
  - context quá ngắn có thể bỏ evidence;
  - context quá dài tăng noise và generation cost.
- Nêu khoảng trống:
  - similarity threshold hoặc Fixed Top-$k$ không biểu diễn uncertainty theo từng query;
  - các conformal retrieval methods hiện có chủ yếu dùng một calibration distribution hoặc một global cutoff.
- Giới thiệu phương pháp trong hai câu:
  - Stage 1 kiểm tra candidate đối với false-score reference bank;
  - Stage 2 kiểm tra các candidate sống sót đối với support-score reference bank và prune các candidate support-incompatible.
- Nêu protocol:
  - frozen Jina-v4 candidate pools;
  - 1,000 calibration và 100 held-out test queries cho mỗi dataset;
  - HotpotQA, MMQA, TAT-QA và WebQA;
  - Qwen2-VL-7B làm downstream generator.
- Tóm tắt kết quả bằng 2–3 con số đã được xác nhận trên cùng protocol:
  - một con số về Selection F1 hoặc support recall;
  - một con số về downstream F1;
  - một con số về latency hoặc approximate TFLOPs.
- Kết thúc bằng kết luận có giới hạn:
  - two-stage selection cung cấp các operating point tốt hơn trong quality–cost space trên một số dataset;
  - lợi ích phụ thuộc dataset và độ chặt của pruning.
- Chỉ điền latency/TFLOPs sau khi Colab run đủ 1,200 generations hoàn tất.

---

## 2. Introduction

### 2.1. Bối cảnh và vấn đề

- Giới thiệu RAG và vai trò của retrieved context trong việc giảm hallucination.
- Mô tả hai lỗi đối lập của Fixed Top-$k$:
  - $k$ nhỏ: thiếu multi-hop hoặc multimodal evidence;
  - $k$ lớn: thêm false evidence, tăng prompt length và compute.
- Dùng một ví dụ trực quan:
  - query đơn giản chỉ cần 1–2 chunk;
  - query multi-hop cần nhiều evidence;
  - cùng một $k$ không phù hợp cho cả hai.
- Phân biệt hai bài toán:
  - retrieval tạo candidate pool;
  - context selection/pruning quyết định candidate nào thật sự vào generator.

### 2.2. Hạn chế của các hướng hiện tại

- Fixed Top-$k$ không có calibration và không thích ứng theo uncertainty.
- Global similarity threshold thích ứng số chunk nhưng vẫn dùng một cutoff chung.
- Các post-retrieval conformal selectors hiện tại dùng support-side calibration để bảo toàn
  relevant-snippet coverage, nhưng chưa tách false-null screening và support-null pruning
  thành hai quyết định theo query.
- Cosine score có thể có distribution khác nhau giữa dataset, modality, rank bin và query type.

### 2.3. Ý tưởng chính

- Xây hai reference banks từ calibration queries tách biệt:
  - $\mathcal{B}_{false}$ chứa score của false evidence;
  - $\mathcal{B}_{supp}$ chứa score của support evidence.
- Stage 1 giữ candidate có score không điển hình đối với false bank.
- Stage 2 prune candidate có score thấp bất thường đối với support bank.
- $\alpha_1$ và $\alpha_2$ tạo hai nút điều chỉnh khác nhau:
  - độ rộng của candidate context;
  - cường độ pruning.

### 2.4. Research questions

- **RQ1:** Two-stage selection thay đổi precision–recall frontier như thế nào so với Fixed Top-$k$ và single-stage conformal baselines?
- **RQ2:** Selection quality có chuyển thành downstream EM/F1 cao hơn không?
- **RQ3:** Context ngắn hơn giảm latency và compute proxy bao nhiêu?
- **RQ4:** Kết quả nhạy thế nào với $\alpha_1$, $\alpha_2$, calibration size, modality conditioning và candidate depth $L$?
- **RQ5:** BH và BY tạo khác biệt gì khi candidate p-values phụ thuộc lẫn nhau trong cùng query?

### 2.5. Contributions

- Viết 3–4 contribution bullets, mỗi bullet gắn với một kết quả hoặc artifact kiểm chứng được.
- Không dùng các cụm “strictly controls FDR under arbitrary dependence” nếu chỉ chạy BH.
- Không claim Stage 1 “bounds FNR” trừ khi có theorem riêng với định nghĩa và giả định đầy đủ.
- Không claim “consistently outperforms all baselines” khi bảng có dataset mà Fixed Top-5 hoặc literature baseline cao hơn.

### 2.6. Figure mở đầu

- **Figure 1:** cùng một candidate pool qua ba pipeline:
  - Fixed Top-$k$;
  - single-threshold conformal selection;
  - proposed false-bank screening + support-bank pruning.
- Minh họa số chunk và support/false labels ở mỗi bước.
- Caption nêu đây là post-retrieval selection; retriever không bị thay đổi.

---

## 3. Related Work

### 3.1. Retrieval-Augmented Generation và context selection

- Tóm tắt retrieve-then-generate pipeline.
- Phân biệt retriever training, reranking và post-retrieval pruning.
- Tổng hợp các hướng dynamic context sizing, adaptive retrieval và evidence filtering.
- Chỉ ra paper này tập trung vào **post-retrieval candidate selection**.

### 3.2. Conformal prediction cho retrieval và RAG

- Giới thiệu exchangeability, calibration set và finite-sample marginal validity ở mức cần thiết.
- Trình bày ba post-retrieval rules đúng theo code đang chạy:
  - **CCE:** hiệu chuẩn nonconformity $1-s$ từ mọi labelled support pair, rồi giữ candidate
    nếu $s$ vượt support-derived cutoff;
  - **CONFLARE:** lấy best support score của từng calibration query, hiệu chuẩn một
    source-question cutoff, rồi giữ candidate vượt cutoff;
  - **TRAQ:** lấy best support score của từng calibration query, dành mức
    $\alpha_R=\alpha/2$ cho retrieval threshold, rồi giữ candidate vượt cutoff.
- Cả ba bắt đầu từ cùng candidate pool sau retrieval. Vì vậy phần so sánh tập trung hoàn toàn
  vào calibration object, threshold và tập chunk được chuyển cho generator.

### 3.3. Multiple hypothesis testing trong context selection

- Giới thiệu quyết định theo từng candidate như một family of hypotheses trong một query.
- Tóm tắt BH step-up procedure.
- Ghi đúng phạm vi:
  - BH có FDR guarantee dưới independence hoặc các dạng positive dependence phù hợp;
  - BY dùng harmonic correction và hợp lệ dưới arbitrary dependence nhưng thường conservative hơn.
- Giải thích tại sao p-values trong một query có thể phụ thuộc:
  - các chunk đến từ cùng retriever ranking;
  - chunk cùng document hoặc cùng modality;
  - score chịu chung query representation.

### 3.4. Research Gap

#### 3.4.1. CCE

- **Input:** toàn bộ candidate chunks và Jina cosine score $s(q,c)$ sau retrieval.
- **Calibration:** gom score của **mọi labelled support chunk** trên calibration set và đặt
  nonconformity $a(q,c)=1-s(q,c)$. Code lấy quantile $(1-\alpha)$ của $a$.
- **Selection:** giữ từng test chunk nếu $1-s(q,c)\le \hat q_{1-\alpha}$, tương đương
  $s(q,c)\ge 1-\hat q_{1-\alpha}$.
- **Làm được:** bảo toàn support recall cao: 90.5% HotpotQA, 91.0% MMQA, 91.8% TAT-QA và
  86.3% WebQA.
- **Chưa ổn:** gộp mọi support pair làm query có nhiều labelled supports đóng góp nhiều điểm;
  threshold không quan sát false distribution và không thay đổi theo query. Kết quả là giữ
  81.7%–87.4% pool, với precision chỉ 6.5% trên WebQA và 7.7% trên MMQA.

#### 3.4.2. CONFLARE

- **Input:** cùng candidate pool và cùng cosine score sau retrieval.
- **Calibration:** với mỗi calibration query, code lấy **maximum cosine của các support
  chunks**, chuyển thành distance $d=1-s$, rồi lấy percentile $(1-\alpha)$ của các distances.
- **Selection:** giữ test chunk nếu distance của nó nhỏ hơn calibrated cutoff.
- **Làm được:** cutoff thường chặt hơn CCE/TRAQ, nên giữ ít chunk nhất trong ba methods:
  7.05 trên HotpotQA, 18.05 trên MMQA, 4.64 trên TAT-QA và 21.60 trên WebQA. Trên MMQA,
  downstream F1 .462 là tốt nhất trong ba baseline.
- **Chưa ổn:** maximum operation chỉ đại diện support dễ nhất của mỗi query; nó không hiệu
  chuẩn khả năng giữ đủ multi-support evidence. HotpotQA recall giảm xuống 74.0% và F1 giảm
  từ .653 của CCE xuống .623; TAT-QA recall giảm xuống 86.6% và F1 còn .432.

#### 3.4.3. TRAQ

- **Input:** cùng candidate pool và cùng cosine score sau retrieval.
- **Calibration:** lấy maximum support score của mỗi calibration query. Với $\alpha=.10$,
  code dùng $\alpha_R=\alpha/2=.05$ và lấy lower $.05$ quantile làm similarity cutoff.
- **Selection:** giữ test chunk nếu score lớn hơn hoặc bằng cutoff.
- **Làm được:** threshold permissive giúp đạt recall cao nhất trên MMQA (94.2%), TAT-QA
  (93.3%) và WebQA (88.6%). Trên TAT-QA, nó đạt downstream F1 .500, cao nhất trong ba methods.
- **Chưa ổn:** recall cao đạt được bằng cách giữ 82.9%–89.2% pool. TRAQ còn trung bình 19.50
  chunks/query trên MMQA và 24.68 trên WebQA; precision tương ứng chỉ 7.5% và 6.3%.

#### 3.4.4. Gap chung và cách phương pháp đề xuất xử lý

- **Support-derived cutoff dẫn đến keep-too-many khi hai lớp overlap.** CCE đặt cutoff theo
  lower tail của support scores; CONFLARE và TRAQ đặt cutoff theo best-support score của mỗi
  calibration query. Để bảo toàn support coverage, cutoff phải đủ thấp cho các support khó.
  Tuy nhiên, false scores nằm gần support scores: cosine AUC chỉ .543 trên HotpotQA và khoảng
  .61–.62 trên ba datasets còn lại. Vì vậy, một cutoff đủ thấp để giữ support cũng cho phần lớn
  false chunks đi qua. Điều này thể hiện ở việc ba methods giữ 71.6%–89.2% pool, trong khi
  precision gần bằng tỷ lệ support trước pruning: HotpotQA tăng từ 20.3% lên 21.0%–21.4%,
  WebQA chỉ tăng từ 6.2% lên 6.3%–6.8%.

- **Quyết định độc lập trên từng candidate làm context size tăng theo candidate pool.** Một
  chunk chỉ cần vượt global cutoff; việc query đã có bao nhiêu chunks được giữ không ảnh hưởng
  quyết định của chunk tiếp theo. Do đó các query có pool lớn dễ chuyển một context dài sang
  generator. MMQA có 21.87 candidates/query và còn 18.05–19.50; WebQA có 28.11 và còn
  21.60–24.68. Đây là nguyên nhân trực tiếp khiến recall cao nhưng chi phí context và số false
  chunks vẫn lớn.

- **Best-support calibration không đại diện cho việc giữ đủ evidence.** CONFLARE và TRAQ rút
  mỗi calibration query về support score lớn nhất. Ngưỡng vì thế được quyết định bởi support
  dễ nhất, trong khi multi-hop query có thể còn các support yếu hơn. Trên HotpotQA, CONFLARE
  đạt any-support coverage 88% nhưng all-support coverage chỉ 60%; support recall giảm còn
  74.0% và downstream F1 giảm từ .653 của CCE xuống .623. TRAQ dùng cutoff permissive hơn nên
  all-support coverage tăng lên 79%, đổi lại giữ 8.17/9.85 chunks mỗi query.

- **Một pooled cutoff tạo cùng quyết định cho các modality có độ tách lớp rất khác nhau.**
  Trên TAT-QA, table cosine có AUC .832 nhưng text chỉ .620. Trên WebQA, text đạt .683 còn
  image chỉ .555, gần random. Khi các score regimes này dùng chung một cutoff, selector không
  thể đồng thời chặt với nhóm có nhiều false scores và mềm với nhóm chứa evidence khó. Pattern
  này phù hợp với WebQA: 21.60–24.68 chunks/query nhưng precision chỉ 6.3%–6.8%.

- Các failure mechanisms trên dẫn trực tiếp tới thiết kế đề xuất:
  - **Stage 1 false-bank screening** so candidate với false distribution, nhắm trực tiếp vào
    hiện tượng cutoff support-side giữ quá nhiều false chunks;
  - **BH/BY theo query** đặt quyết định trong toàn candidate family, để số lượng và p-values
    của các chunks cùng query ảnh hưởng đến tập được giữ;
  - **modality- và rank-conditioned banks** so candidate với calibration regime gần nó hơn,
    thay vì trộn các nhóm có score separability khác nhau;
  - **Stage 2 support-bank pruning** loại support-incompatible candidates sau khi Stage 1 đã
    bảo toàn candidate set rộng;
  - operating point được chọn với recall/all-support constraint để pruning không lặp lại lỗi
    mất secondary evidence của best-support calibration.

### 3.5. Kết quả post-retrieval hiện tại cho thấy điều gì

Phân tích đầy đủ và số liệu truy vết nằm tại
[`BASELINE_SIGNAL_DIAGNOSIS.md`](research/internal_state_rag/results/full_context_selection_splits_khanh_27_09_2026_09_27/BASELINE_SIGNAL_DIAGNOSIS.md).

`Support rate before pruning` là số support chunks chia cho tổng số retrieved candidates
trước khi bất kỳ selector nào chạy.

| Dataset | Support rate before pruning | Cosine AUC | CCE kept / P / R | CONFLARE kept / P / R | TRAQ kept / P / R |
|---|---:|---:|---:|---:|---:|
| HotpotQA | 20.3% | .543 | 87.4% / 21.0% / 90.5% | 71.6% / 21.0% / 74.0% | 82.9% / 21.4% / 87.5% |
| MMQA | 7.1% | .609 | 84.1% / 7.7% / 91.0% | 82.5% / 7.8% / 90.3% | 89.2% / 7.5% / 94.2% |
| TAT-QA | 21.9% | .620 | 81.7% / 24.6% / 91.8% | 75.7% / 25.0% / 86.6% | 83.0% / 24.6% / 93.3% |
| WebQA | 6.2% | .621 | 83.1% / 6.5% / 86.3% | 76.8% / 6.8% / 83.4% | 87.8% / 6.3% / 88.6% |

- Ba methods giữ **71.6%–89.2% toàn candidate pool**. CCE còn 18.40 chunks/query trên MMQA
  và 23.36 trên WebQA; TRAQ còn 19.50 và 24.68.
- Precision gần bằng tỷ lệ support trước pruning. HotpotQA tăng từ 20.3% lên khoảng 21%; WebQA
  tăng từ 6.2% lên 6.3%–6.8%. Selector đạt recall cao chủ yếu bằng cách cho gần hết pool đi
  qua, chưa thực sự enrich evidence.
- Cosine AUC chỉ .543 trên HotpotQA và khoảng .61–.62 trên ba datasets còn lại. Support và
  false score overlap mạnh, nên một cutoff thấp đủ giữ khoảng 90% support cũng giữ phần lớn
  false chunks.
- TRAQ thường giữ nhiều nhất vì $\alpha_R=.05$ tạo cutoff thấp hơn. CONFLARE thường prune
  nhiều hơn, nhưng phần recall mất đi không luôn đổi thành downstream gain.

### 3.6. Các giả thuyết thực nghiệm rút ra từ Research Gap

- **H1 — False-bank enrichment:** tại cùng support recall, Stage 1 phải giảm false
  chunks/query và tăng precision so với support-derived global cutoffs. Hiệu quả kỳ vọng rõ
  nhất trên MMQA và WebQA, nơi false-chunk rate trước pruning lần lượt là 92.9% và 93.8%.
- **H2 — Query-family selection:** BH/BY theo query phải làm retained context ít phụ thuộc vào
  raw pool size hơn independent thresholding, đồng thời giảm empty rate so với một cutoff quá
  chặt.
- **H3 — Conditional calibration:** modality-aware banks phải cải thiện Selection F1 so với
  pooled banks trên MMQA, TAT-QA và WebQA. Ablation cần báo riêng text/table/image để xác định
  improvement đến từ nhóm nào.
- **H4 — Multi-support preservation:** Two-Stage phải tăng precision mà không làm all-support
  coverage sụp như CONFLARE trên HotpotQA. Vì vậy operating point được chọn bằng calibration
  objective có recall hoặc all-support constraint, không chỉ tối đa hóa precision.

### 3.7. Downstream diagnosis và research gap

- HotpotQA: CCE giữ 8.61 chunks, F1 .653; CONFLARE giảm còn 7.05 chunks nhưng F1 giảm xuống
  .623 vì recall mất 16.5 điểm.
- MMQA: CONFLARE có F1 .462, cao hơn CCE .436 và TRAQ .435 dù retrieval metrics gần nhau;
  đây là dấu hiệu noise reduction có thể giúp.
- TAT-QA: TRAQ giữ recall cao nhất và đạt F1 .500; CONFLARE prune mạnh hơn và giảm còn .432,
  cho thấy task cần giữ evidence coverage.
- WebQA: 21.60–24.68 chunks vẫn cho cùng EM .11 và F1 chỉ .246–.260; thay đổi cutoff hiện tại
  chưa tác động đáng kể tới generator.
- Không có quy luật đơn điệu “ít chunk hơn tốt hơn”. Khoảng trống cần kiểm chứng là liệu dùng
  false bank, query-level multiple testing, modality và rank conditioning có đẩy được Pareto
  frontier ra ngoài các global support cutoffs trên cùng split hay không.
- Two-Stage phải được đánh giá trên chính candidate pool và split này. Các con số từ split
  khác không được gộp vào cùng bảng so sánh.

### 3.8. Bảng so sánh related work dự kiến

| Method | Calibration object | Decision granularity | Output | Multiple testing | Comparison role |
|---|---|---|---|---|---|
| Fixed Top-$k$ | None | global budget | first $k$ chunks | No | non-calibrated baseline |
| CCE | mọi support-pair nonconformity $1-\cos(e_q,e_c)$ | independent global threshold | chunks below nonconformity cutoff | No | high-recall support cutoff |
| CONFLARE | best-support distance của mỗi calibration query | independent global threshold | chunks below distance cutoff | No | stricter support cutoff |
| TRAQ | best-support score của mỗi calibration query, $\alpha_R=\alpha/2$ | independent global threshold | chunks above similarity cutoff | No | permissive coverage cutoff |
| Stage 1 only | false-score bank | per query | admitted set | BH or BY | ablation |
| Two-Stage | false + support banks | per query | screened then pruned set | BH/BY per stage | proposed method |
| Top-$k$ + Stage 2 | support bank | per query after fixed budget | pruned Top-$k$ | BH/BY | stage-isolation ablation |

---

## 4. Problem Formulation

### 4.1. Post-retrieval setting

- Với query $q$, retriever trả candidate pool:

$$
\mathcal{C}_q=\{(c_i,s_i,r_i,m_i)\}_{i=1}^{L_q},
$$

  trong đó $s_i$ là score, $r_i$ là rank và $m_i$ là modality.
- Candidate pool có thể ragged; $L_q$ không nhất thiết bằng Top-$L$ tối đa.
- Retriever, corpus revision và preprocessing được đóng băng giữa calibration/test.

### 4.2. Evidence labels

- $y_i\in\{support,false,unknown\}$.
- `support` là annotated evidence cho answer.
- `false` là candidate đã xác định không hỗ trợ.
- `unknown` không được dùng để xây false/support banks.
- Nêu rõ chunk-level support annotation không đánh giá mọi dạng semantic usefulness cho generator.

### 4.3. Mục tiêu tối ưu

- Chọn $\mathcal{S}_q\subseteq\mathcal{C}_q$ sao cho:
  - support recall đủ cao;
  - support precision tăng;
  - empty-context rate thấp;
  - downstream answer quality không giảm;
  - context/generation cost giảm.
- Không gộp các mục tiêu này thành một claim duy nhất.
- Dùng Selection F1 và Pareto analysis để mô tả trade-off.

### 4.4. Evaluation levels

- **Chunk level:** micro precision, recall, Selection F1.
- **Query level:** any-support, all-support, empty rate, chunks/query.
- **Downstream level:** EM, token F1, numerical accuracy.
- **Efficiency level:** latency, input tokens, context chunks, approximate TFLOPs.

---

## 5. Methodology

### 5.1. Overview of Two-Stage Conformal Context Selection

- Trình bày full pipeline:
  - frozen candidate retrieval;
  - condition-specific reference-bank lookup;
  - Stage 1 false-null screening;
  - Stage 2 support-null pruning;
  - optional rank-preserving context cap;
  - generator.
- **Figure 2:** sơ đồ pipeline và dữ liệu đi qua từng stage.
- **Algorithm 1:** pseudocode train/calibration phase và test phase.

### 5.2. Calibration split và pipeline fingerprints

- Calibration queries và test queries tách theo `qid`.
- Lưu fingerprints:
  - retriever ID;
  - corpus revision;
  - preprocessing hash;
  - maximum Top-$L$;
  - query-type rule;
  - score type.
- Nếu fingerprint không khớp, không được dùng bank cũ để score test candidate.

### 5.3. Reference-bank construction

- Định nghĩa stratum:

$$
g=(dataset, modality[, query\ type, rank\ bin]).
$$

- False bank:

$$
\mathcal{B}_{false}^{g}=
\operatorname{sort}\{s(q,c):y(q,c)=false,\ g(q,c)=g\}.
$$

- Support bank:

$$
\mathcal{B}_{supp}^{g}=
\operatorname{sort}\{s(q,c):y(q,c)=support,\ g(q,c)=g\}.
$$

- Nêu cách xử lý bank nhỏ hoặc missing:
  - block selection;
  - fallback hierarchy được định trước;
  - pooled-bank ablation;
  - không chọn fallback sau khi nhìn test result.
- Báo bank sizes theo dataset/modality trong Appendix.

### 5.4. Stage 1: false-null candidate screening

- Giả thuyết làm việc:

$$
H_{0,i}^{(1)}:\ c_i\text{ có score tương thích với false-evidence bank}.
$$

- Upper-tail p-value:

$$
p_i^{(1)}=
\frac{1+\sum_{s\in\mathcal{B}_{false}^{g_i}}\mathbb{I}[s\ge s_i]}
{1+|\mathcal{B}_{false}^{g_i}|}.
$$

- Chạy BH/BY trong family các candidate của cùng query.
- Rejection của $H_0^{(1)}$ được đưa vào $\mathcal{S}_1$.
- Diễn giải $\alpha_1$ theo hành vi thực nghiệm:
  - $\alpha_1$ lớn thường admit nhiều candidate hơn;
  - recall có thể tăng nhưng không gọi đây là FNR guarantee nếu chưa có theorem.

### 5.5. Stage 2: support-null pruning

- Với $c_i\in\mathcal{S}_1$, đặt giả thuyết:

$$
H_{0,i}^{(2)}:\ c_i\text{ có score tương thích với support-evidence bank}.
$$

- Lower-tail p-value:

$$
p_i^{(2)}=
\frac{1+\sum_{s\in\mathcal{B}_{supp}^{g_i}}\mathbb{I}[s\le s_i]}
{1+|\mathcal{B}_{supp}^{g_i}|}.
$$

- Candidate bị reject ở Stage 2 được prune.
- $\alpha_2$ lớn tạo pruning mạnh hơn trong implementation hiện tại.
- Formal interpretation cần viết chính xác:
  - nếu p-values hợp lệ và dependence assumptions thỏa mãn, multiple-testing guarantee gắn với false discoveries trong **tập bị reject/prune**;
  - điều này không tự động tương đương với chunk false-positive rate trong tập được giữ.

### 5.6. Final context, rank order và context cap

- Giữ retrieval order khi đưa selected chunks vào generator.
- Nếu có cap $K$, mô tả exact rule:
  - lấy tối đa $K$ selected chunks theo retrieval rank;
  - hoặc verified backfill từ ranks $K+1\ldots L$.
- Tách rõ:
  - uncapped multiple-testing procedure;
  - capped procedure dùng trong downstream experiments.
- Không chuyển guarantee của uncapped procedure sang capped procedure nếu chưa có proof.

### 5.7. Method variants

- **Stage 1 only:** false-bank screening, không support pruning.
- **Two-Stage:** Stage 1 rồi Stage 2.
- **Top-10 + Stage 2:** dùng Fixed Top-10 làm input để đo riêng tác dụng Stage 2.
- **Pooled banks:** không conditioning theo modality.
- **Modality-aware banks:** conditioning theo text/table/image.
- **BH vs BY:** cùng p-values, khác multiple-testing correction.

### 5.8. Statistical scope và assumptions

- Nêu exchangeability assumptions giữa calibration và test candidates trong cùng stratum.
- Nêu dependence assumption của BH.
- Dùng BY nếu muốn claim robustness dưới arbitrary dependence.
- Hai-stage selective reuse cần được phân tích riêng vì Stage 2 chỉ nhận candidates đã qua Stage 1.
- Context cap và hyperparameter selection có thể thay đổi guarantee.
- Nếu chưa có theorem cho full pipeline, mô tả Stage 2 là calibrated hypothesis-testing heuristic và báo empirical error rates.

### 5.9. Computational complexity

- Bank construction: sort một lần, $O(M_g\log M_g)$ cho mỗi stratum.
- P-value lookup: binary search, $O(\log M_g)$ mỗi candidate.
- BH: $O(L_q\log L_q)$ mỗi query.
- Stage 2: $O(|\mathcal{S}_1|\log|\mathcal{S}_1|)$.
- Selection overhead nhỏ so với multimodal generation; đo thực nghiệm riêng nếu có thể.

---

## 6. Experimental Setup

### 6.1. Datasets

- **HotpotQA:** text, multi-hop reasoning.
- **MMQA:** multimodal QA với text/table/image evidence.
- **TAT-QA:** table-text financial reasoning.
- **WebQA:** multimodal open-domain QA.
- Với mỗi dataset, báo:
  - số calibration/test queries;
  - candidate-pool mean/min/max;
  - support chunks/query;
  - modality distribution;
  - percentage queries có ít nhất một labelled support.

### 6.2. Data split

- Main split: `splits_khanh_27_09`.
- 1,000 calibration và 100 held-out test queries mỗi dataset.
- Zero qid overlap.
- Hyperparameters phải chọn từ calibration hoặc một validation subset, không chọn bằng test F1.
- Báo SHA-256 của split archive và link artifact.

### 6.3. Retrieval and candidate pool

- Frozen `jinaai/jina-embeddings-v4` scores.
- Candidate pool dataset-provided, ragged, capped ở Top-$L=30$ khi đủ candidates.
- Nêu rõ nhiều query không có đủ 30 candidates.
- Không reretrieve hoặc thay đổi candidate pool giữa methods.

### 6.4. Generator protocol

- `Qwen/Qwen2-VL-7B-Instruct`, pinned revision.
- Greedy decoding, `max_new_tokens=24`.
- Context đứng trước prompt.
- Image bounds: `min_pixels=3136`, `max_pixels=200704`.
- Prompt exact string được đưa vào Appendix.
- Latency đo chỉ quanh `model.generate` với CUDA synchronization.

### 6.5. Baselines

- Fixed Top-5, Fixed Top-10 và Fixed Top-20 khi candidate pool đủ lớn.
- CCE support-pair nonconformity cutoff.
- CONFLARE per-query best-support distance cutoff.
- TRAQ per-query best-support retrieval cutoff với $\alpha_R=\alpha/2$.
- Với mỗi method, báo exact calibration sample count, quantile convention, numerical cutoff,
  comparison operator (`<`, `<=`, `>=`) và candidate pool trước/sau selection.

### 6.6. Proposed configurations

- Stage 1:
  - $\alpha_1\in\{0.10,0.30,0.50,0.90,0.99\}$ hoặc tập đã chạy đầy đủ.
- Two-Stage:
  - cố định $\alpha_1=0.99$;
  - sweep $\alpha_2\in\{0.15,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90\}$.
- Top-10 + Stage 2:
  - cùng sweep $\alpha_2$ để isolate Stage 2.
- Chọn một operating point chính bằng calibration objective, chẳng hạn:

$$
\max_{\alpha_2}\ \mathrm{SelectionF1}_{cal}
\quad\text{s.t.}\quad
\mathrm{Recall}_{cal}\ge r_{min}.
$$

- Các sweep khác đặt trong Appendix; main table không nên cherry-pick test optimum cho từng dataset.

### 6.7. Metrics

- Context selection:
  - retained chunks/query;
  - micro precision/recall/Selection F1;
  - empty rate;
  - any-support/all-support coverage.
- Downstream generation:
  - EM;
  - token F1;
  - numerical accuracy cho TAT-QA.
- Efficiency:
  - input tokens;
  - generation-only latency;
  - median/P95 latency nếu đủ runs;
  - approximate TFLOPs theo exact formula đã dùng.
- Ghi rõ approximate TFLOPs không phải profiler-measured hardware FLOPs.

### 6.8. Statistical analysis

- Downstream model là deterministic với greedy decoding; đổi random seed không tạo ba independent model runs.
- Dùng paired bootstrap trên test queries để tạo 95% confidence intervals cho EM/F1 differences.
- Dùng bootstrap theo query cho selection metrics.
- Với latency:
  - chạy ba lần trên cùng loại GPU/runtime nếu cần mean ± standard deviation;
  - hoặc báo median/P95 và loại model-loading time.
- Báo effect size và confidence interval, không chỉ bold giá trị lớn nhất.

### 6.9. Reproducibility

- Link code, split manifests, frozen selection IDs và exact model revision.
- Lưu `RUN_CONFIG.json`, environment versions và GPU name.
- Progress log theo `(method,dataset,qid)` để resume.
- Công bố commands cho calibration, selection và downstream evaluation.

---

## 7. Main Results

### 7.1. Context-selection quality

- **Table 1:** Kept chunks, precision, recall và Selection F1 trên bốn datasets.
- Sắp xếp rows theo nhóm:
  - Fixed Top-$k$;
  - literature post-retrieval selectors;
  - Stage 1 only;
  - Two-Stage;
  - Top-10 + Stage 2.
- Bôi đậm best value trong cùng context-budget region hoặc báo Pareto-optimal rows.
- Không bôi đậm precision riêng nếu recall đã sụp mạnh.
- Phân tích dự kiến:
  - Stage 2 tăng precision khi $\alpha_2$ tăng;
  - recall giảm theo cường độ pruning;
  - effect mạnh khác nhau giữa dataset.

### 7.2. Downstream answer quality

- **Table 2:** EM/F1/Numeric cho cùng exact test qids và exact selected contexts.
- Trả lời RQ2:
  - context-selection F1 cao hơn có luôn cải thiện QA F1 không;
  - dataset nào hưởng lợi từ context ngắn;
  - dataset nào cần giữ recall cao.
- Phân tích ba post-retrieval baselines trên `splits_khanh_27_09`:
  - HotpotQA: CCE/CONFLARE/TRAQ đạt F1 .653/.623/.642;
  - MMQA: .436/.462/.435;
  - TAT-QA: .490/.432/.500;
  - WebQA: .260/.255/.246.
- Điền Two-Stage và Top-$k$ + Stage 2 vào cùng bảng sau khi chạy đúng candidate pool, split
  và generator protocol này.

### 7.3. Quality–cost trade-off

- **Table 3:** chunks/query, input tokens, latency và approximate TFLOPs.
- **Figure 3:** downstream F1 theo approximate TFLOPs.
- **Figure 4:** downstream F1 theo chunks/query.
- Đánh dấu Pareto frontier thay vì chỉ chọn một row “best”.
- Phân tích:
  - giảm chunks có thực sự giảm tokens/latency không;
  - image-heavy datasets có quan hệ chunks–compute khác text-only datasets;
  - phương pháp nào đạt gần-best F1 với chi phí thấp hơn.
- Chỉ điền ba baseline latency/TFLOPs từ Colab artifact có đủ 1,200 rows và `status=complete`.

### 7.4. Head-to-head comparison với CCE, CONFLARE và TRAQ

- So sánh ở ba chế độ công bằng:
  - cùng $\alpha$ nominal;
  - cùng mean context budget;
  - Pareto frontier không ép cùng threshold.
- Với mỗi baseline, trả lời ngắn:
  - calibration object khác gì;
  - output context size khác gì;
  - selection F1 khác gì;
  - downstream/cost khác gì.
- Phân tích chỉ dựa trên calibration rule, selected chunks và downstream output sau retrieval.

### 7.5. Head-to-head comparison với Fixed Top-$k$

- Fixed Top-5 là baseline quan trọng vì có context nhỏ và Selection F1 mạnh trên một số dataset.
- Các số preliminary không hỗ trợ claim “dynamic selection luôn tốt hơn Fixed Top-5”:
  - MMQA Fixed Top-5 Selection F1 36.6%, cao hơn các proposed rows được liệt kê;
  - WebQA Fixed Top-5 Selection F1 34.1%, cao hơn proposed maximum 28.8% trong sweep hiện tại;
  - HotpotQA proposed Top-10 + Stage 2 ở $\alpha_2=.50$ đạt 49.9% so với Fixed Top-5 48.1%;
  - TAT-QA proposed ở $\alpha_2=.50$ đạt 46.9% so với Fixed Top-5 45.5%.
- Định vị contribution ở khả năng tạo nhiều operating points và tối ưu downstream/cost, thay vì phủ nhận Fixed Top-$k$.

### 7.6. Error analysis

- Chọn 3–5 query mỗi dataset:
  - Stage 2 loại đúng false chunks;
  - Stage 2 loại nhầm support chunk;
  - empty-context failure;
  - multimodal support bị score thấp;
  - candidate pool không chứa gold evidence.
- Với mỗi case, hiển thị:
  - retrieval rank;
  - modality;
  - cosine score;
  - Stage 1/2 p-values;
  - decision;
  - generated answer.


### 7.7. Kết quả mới từ bản LaTeX ngày 01-10-2026

- Mục 16 giữ toàn bộ số downstream/resource mới, kèm bảng từng dataset và phân tích các operating points.
- Khi phát triển mục 7.1–7.5, dùng kết quả mới cùng manifest/log của nó; các kết quả đã truy vết ở mục 7.2 tiếp tục giữ riêng.
- Bổ sung so sánh Stage 1 permissive với Two-Stage: trên HotpotQA và TAT-QA, Stage 1 $.99$ có downstream cao nhất trong bảng mới, còn pruning chuyển điểm hoạt động về phía chi phí thấp hơn.


---

## 8. Ablation Studies

### 8.1. Stage contribution

- So sánh Stage 1 only, Two-Stage và Top-10 + Stage 2.
- Xác định gain đến từ adaptive admission hay support pruning.

### 8.2. Alpha sensitivity

- Vẽ precision, recall, Selection F1, empty rate và chunks/query theo $\alpha_1$ hoặc $\alpha_2$.
- Không chỉ báo ba điểm tốt nhất.
- Báo operating-point stability giữa calibration và test.

### 8.3. Calibration size

- Sweep 50, 100, 250, 500 và 1,000 calibration queries nếu dữ liệu cho phép.
- Báo bank size theo modality và variance của threshold/p-values.
- Xác định số calibration queries tối thiểu để kết quả ổn định.

### 8.4. Pooled versus modality-aware banks

- So sánh pooled bank với text/table/image-conditioned banks.
- Kiểm tra modality-aware có giúp MMQA/WebQA hay chỉ làm banks quá nhỏ.
- Báo missing/underpowered-bank rate.

### 8.5. BH versus BY

- Giữ nguyên p-values và $\alpha$, chỉ thay correction.
- So sánh context size, precision, recall và empty rate.
- Dùng kết quả để định lượng giá của arbitrary-dependence robustness.

### 8.6. Candidate depth $L$ và context cap $K$

- Sweep $L$ trong phạm vi candidate pool thật sự có sẵn.
- Báo fraction query có ít hơn $L$ candidates.
- So sánh uncapped selection, capped selection và verified backfill.
- Không kết luận tăng $L$ vô ích nếu candidate pool cũ chưa thật sự được retrieve từ full corpus.

---

## 9. Discussion

### 9.1. Khi nào Two-Stage hữu ích

- Candidate pool có nhiều false evidence nhưng vẫn chứa support ở score tương đối cao.
- Generator nhạy với distractor context.
- Context budget hoặc multimodal compute là ràng buộc quan trọng.

### 9.2. Khi nào phương pháp không giúp

- Candidate pool ban đầu thiếu support evidence.
- Cosine score không tách được support và false distributions.
- Support bank quá nhỏ hoặc bị distribution shift.
- Aggressive pruning gây empty contexts hoặc loại mất complementary evidence.

### 9.3. Dataset-specific behavior

- HotpotQA cần multi-hop evidence nên all-support coverage quan trọng.
- MMQA/WebQA có modality imbalance và image processing cost.
- TAT-QA cần table-text evidence; chunk count nhỏ không đồng nghĩa với đủ operands để reasoning.

### 9.4. Practical operating-point selection

- Nếu ưu tiên recall: dùng permissive Stage 1 hoặc low-pruning Stage 2.
- Nếu ưu tiên cost: chọn point trên Pareto frontier với context cap.
- Nếu ưu tiên downstream quality: chọn $\alpha$ bằng validation objective, không dùng test labels.
- Có thể dùng dataset-specific $\alpha$, nhưng phải chọn trước test.

### 9.5. Interpretation of uncertainty

- P-values đo mức tương thích của score với calibration banks.
- Chúng không phải xác suất chunk đúng hoặc sai.
- Cosine uncertainty chỉ phản ánh uncertainty của score pipeline, không bao phủ toàn bộ epistemic uncertainty của generator.

---

## 10. Limitations and Threats to Validity

### 10.1. Formal validity

- Candidate-level exchangeability có thể bị vi phạm bởi retrieval ranking.
- P-values trong một query phụ thuộc lẫn nhau.
- Stage 2 chạy sau selection của Stage 1 tạo selective-inference issue.
- Context cap/backfill có thể làm guarantee của uncapped procedure không còn áp dụng.
- Tách formal theorem cho procedure nào đã chứng minh và empirical method cho phần còn lại.

### 10.2. Kiểm soát post-retrieval comparison

- Mọi method nhận cùng qids, candidate chunk IDs, Jina scores và support labels.
- Chỉ calibration rule và chunk-selection decision được thay đổi giữa các methods.
- Lưu exact threshold, selected chunk IDs và context đưa vào generator để kết quả có thể
  kiểm tra ở từng query.

### 10.3. Dataset and label limitations

- Chỉ bốn datasets và 100 held-out test queries mỗi dataset.
- Support annotations có thể không đầy đủ.
- Unknown chunks không được dùng trong precision/recall denominator cần được mô tả rõ.

### 10.4. Efficiency measurement

- Generation-only latency không bao gồm retrieval, selection, image loading hoặc preprocessing.
- Approximate TFLOPs hiện tại là proxy từ parameter count và sequence length.
- Không diễn giải proxy như hardware-counter FLOPs.

### 10.5. Generator dependence

- Main downstream evaluation dùng một generator Qwen2-VL-7B.
- Kết quả có thể đổi với model size, context window và prompting.
- Nếu đủ tài nguyên, thêm một text-only hoặc larger-model validation subset.

---

## 11. Conclusion

- Nhắc lại vấn đề Fixed Top-$k$ và noisy context.
- Tóm tắt false-bank screening và support-bank pruning trong 2–3 câu.
- Kết luận theo bằng chứng:
  - two-stage procedure tạo trade-off có thể điều chỉnh;
  - một số operating points cải thiện downstream quality hoặc efficiency;
  - kết quả phụ thuộc dataset và calibration quality.
- Nêu hướng tiếp theo:
  - theorem cho selective two-stage procedure và capped context;
  - larger post-retrieval evaluations;
  - adaptive selection of operating point without test leakage.

---

## 12. Planned Figures and Tables

### 12.1. Figures

- **Figure 1:** Fixed Top-$k$ versus single-stage versus Two-Stage overview.
- **Figure 2:** reference-bank construction và test-time algorithm.
- **Figure 3:** Selection precision–recall frontier trên bốn datasets.
- **Figure 4:** downstream F1 versus approximate TFLOPs.
- **Figure 5:** retained chunks/precision/recall theo $\alpha_2$.
- **Figure 6:** pooled versus modality-aware banks hoặc calibration-size stability.

### 12.2. Main tables

- **Table 1:** dataset, split và candidate-pool statistics.
- **Table 2:** calibration object và exact post-retrieval decision rule của từng method.
- **Table 3:** context-selection metrics.
- **Table 4:** downstream EM/F1/Numeric.
- **Table 5:** context size, tokens, latency và approximate TFLOPs.
- **Table 6:** core ablations.

### 12.3. Appendix tables

- Full $\alpha_1$/$\alpha_2$ sweep.
- Calibration-bank sizes theo dataset/modality.
- Any-support, all-support và empty rates.
- BH versus BY.
- Per-dataset bootstrap confidence intervals.
- Exact environment, model revision và commands.

---

## 13. Recommended Page Budget

| Section | Suggested length |
|---|---:|
| Abstract | 180–220 words |
| Introduction | 1.0 page |
| Related Work | 1.0 page |
| Problem + Method | 2.0–2.5 pages |
| Experimental Setup | 1.0–1.5 pages |
| Main Results | 2.0 pages |
| Ablations + Discussion | 1.0–1.5 pages |
| Limitations + Conclusion | 0.5–0.75 page |

---

## 14. Claim Audit trước khi viết bản camera-ready

### 14.1. Những câu có thể dùng sau khi số liệu được xác nhận

- “The proposed support-null pruning stage exposes a controllable precision–recall trade-off.”
- “Several operating points lie on a better downstream-quality/compute frontier than the post-retrieval baselines.”
- “The benefit is dataset dependent, with the clearest downstream gains on HotpotQA and TAT-QA in the current evaluation.”
- “The evaluation controls the candidate pool, retriever scores, split and generator across post-retrieval methods.”

### 14.2. Những câu chưa nên dùng

- “Stage 1 mathematically bounds the false negative rate.”
- “BH is valid under arbitrary p-value dependence.”
- “Stage 2 strictly controls the false positive rate of retained chunks.”
- “Our method consistently outperforms all baselines on all datasets.”
- “TFLOPs are hardware-measured,” nếu vẫn dùng công thức proxy hiện tại.

### 14.3. Các việc phải khóa trước khi viết Results cuối

- Hoàn thành Colab run 1,200 generations cho ba literature baselines.
- Chạy proposed rows trên đúng `splits_khanh_27_09` nếu các số hiện tại đến từ split khác.
- Chọn $\alpha$ bằng calibration/validation protocol được định trước.
- Tính paired bootstrap confidence intervals.
- Kiểm tra mọi table lấy cùng exact qids, candidate pools và generator revision.
- Gắn link code/result artifact cho từng table.

---

## 15. Writing Order đề xuất

1. Khóa Problem Formulation, notation và exact procedure.
2. Hoàn thành Experimental Setup và bảng exact post-retrieval decision rules.
3. Chạy đủ experiments trên cùng candidate pool và khóa main tables.
4. Viết Results từ evidence đã khóa, không viết claim trước số liệu.
5. Viết Introduction và Abstract sau khi biết contribution nào thực sự đứng vững.
6. Viết Limitations và formal scope song song với Methodology.
7. Chuyển outline sang LaTeX/Overleaf sau khi headings và main tables ổn định.


---

## 16. Cập nhật layout từ bản LaTeX ngày 01-10-2026

### 16.1. Thông tin mới và vị trí đưa vào paper

- Bản LaTeX cung cấp thêm **downstream EM/F1 và latency/TFLOPs** cho Stage 1, Two-Stage và Top-10 + Stage 2 trên cả bốn datasets. Đây là dữ liệu đầu vào mới cho phần Results và quality–cost analysis.
- Metadata bản thảo: title `Uncertainty-aware RAG`, author `Hung Le`, date `September 2026`; working titles tại mục 0.1 vẫn giữ để chọn khi hoàn thiện bài.
- Dữ liệu hai bảng được lưu tại [layout_results_supplied_2026_10_01.json](paper_assets/layout_results_supplied_2026_10_01.json). Các số tại mục 16 được chép từ bản LaTeX vừa cung cấp; không thay các số đã truy vết theo run tại mục 3.7 và 7.2.
- Các baseline downstream trong bản mới khác bản đang có: ví dụ HotpotQA CCE F1 **73.02%** so với **65.3%** tại mục 7.2. Khi viết Results, gắn bảng mới với split, selected chunk IDs, generator prompt, runtime và raw outputs tương ứng; không ghép hai bộ downstream vào cùng một bảng.
- Thêm kết quả mới vào mục 7.1–7.5 và các Figure quality–cost, giữ tất cả configurations/ablations đã lên kế hoạch.
- Giữ riêng **Selection F1** (chunk evidence) và **Generation F1** (answer tokens); hai metric phản ánh hai mục tiêu khác nhau.

### 16.2. Experimental configurations từ bản mới

- **Baselines:** Fixed Top-5; CCE, CONFLARE và TRAQ tại $\alpha=.10$. Retrieval allocation của TRAQ là $\alpha_R=.05$ theo code hiện tại.
- **Stage 1 only:** $\alpha_1\in\{.10,.90,.99\}$. Bản mới có selection cho cả ba mức nhưng chỉ có downstream cho $.10$ và $.99$.
- **Two-Stage:** $\alpha_1=.99$, $\alpha_2\in\{.15,.20,.30,.40\}$. Bản mới có downstream cho cả bốn mức, selection chỉ tới $.30$.
- **Top-10 + Stage 2:** $\alpha_2\in\{.15,.20,.30,.40\}$, có cả selection và downstream.
- Giữ các sweep rộng hơn trong mục 6.6 và 8.2. Phần thiếu dùng ký hiệu **—**, không suy ra từ variant khác.
- Chốt rõ cap/rank-preserving backfill trong config: Stage 1 $.99$ còn 8.33 chunks/query trên WebQA, trong khi CCE/TRAQ còn 23.36/24.68; budget và pruning đều có thể đóng góp vào giảm cost. Thêm Fixed Top-10 và comparison cùng budget để tách hai tác động này.

### 16.3. Bảng downstream mới theo từng dataset

- EM/F1 theo phần trăm, latency theo giây/query, TFLOPs là compute proxy theo công thức của runner. Bảng giữ toàn bộ số từ LaTeX, không giữ các đánh dấu bold chọn best bị sai trong bản đó.

#### 16.3.1. HotpotQA

| Method | EM (%) | F1 (%) | Latency (s) | Approx. TFLOPs |
|---|---:|---:|---:|---:|
| Fixed Top-5 | 61.00 | 71.40 | 0.22 | 11.82 |
| CCE (α=.10) | 62.00 | 73.02 | 0.24 | 21.03 |
| CONFLARE (α=.10) | 57.00 | 68.78 | 0.21 | 17.35 |
| TRAQ (α=.10) | 62.00 | 73.52 | 0.23 | 19.91 |
| Stage 1 (α₁=.10) | 31.00 | 41.57 | 0.26 | 2.80 |
| Stage 1 (α₁=.90) | — | — | — | — |
| Stage 1 (α₁=.99) | 71.00 | 82.52 | 0.27 | 23.34 |
| Two-Stage (α₁=.99, α₂=.15) | 64.00 | 76.96 | 0.24 | 16.43 |
| Two-Stage (α₁=.99, α₂=.20) | 63.00 | 75.11 | 0.23 | 14.39 |
| Two-Stage (α₁=.99, α₂=.30) | 60.00 | 71.84 | 0.23 | 12.36 |
| Two-Stage (α₁=.99, α₂=.40) | 58.00 | 68.67 | 0.23 | 9.34 |
| Top-10 + S2 (α₂=.15) | 63.00 | 75.96 | 0.25 | 16.47 |
| Top-10 + S2 (α₂=.20) | 63.00 | 75.11 | 0.23 | 14.40 |
| Top-10 + S2 (α₂=.30) | 60.00 | 71.84 | 0.23 | 12.36 |
| Top-10 + S2 (α₂=.40) | 58.00 | 68.67 | 0.23 | 9.34 |

#### 16.3.2. MMQA

| Method | EM (%) | F1 (%) | Latency (s) | Approx. TFLOPs |
|---|---:|---:|---:|---:|
| Fixed Top-5 | 48.00 | 52.40 | 0.24 | 14.38 |
| CCE (α=.10) | 49.00 | 54.08 | 0.44 | 47.45 |
| CONFLARE (α=.10) | 47.00 | 52.08 | 0.42 | 46.45 |
| TRAQ (α=.10) | 48.00 | 52.12 | 0.45 | 50.68 |
| Stage 1 (α₁=.10) | 23.00 | 29.24 | 0.27 | 2.84 |
| Stage 1 (α₁=.90) | — | — | — | — |
| Stage 1 (α₁=.99) | 47.00 | 52.10 | 1.96 | 53.16 |
| Two-Stage (α₁=.99, α₂=.15) | 47.00 | 52.03 | 0.95 | 34.06 |
| Two-Stage (α₁=.99, α₂=.20) | 47.00 | 52.03 | 0.85 | 31.12 |
| Two-Stage (α₁=.99, α₂=.30) | 47.00 | 51.93 | 0.64 | 24.45 |
| Two-Stage (α₁=.99, α₂=.40) | 42.00 | 47.29 | 0.34 | 15.82 |
| Top-10 + S2 (α₂=.15) | 47.00 | 51.53 | 0.32 | 24.12 |
| Top-10 + S2 (α₂=.20) | 47.00 | 51.53 | 0.32 | 23.99 |
| Top-10 + S2 (α₂=.30) | 45.00 | 49.03 | 0.30 | 21.54 |
| Top-10 + S2 (α₂=.40) | 47.00 | 51.75 | 0.26 | 16.42 |

#### 16.3.3. TAT-QA

| Method | EM (%) | F1 (%) | Latency (s) | Approx. TFLOPs |
|---|---:|---:|---:|---:|
| Fixed Top-5 | 37.00 | 57.30 | 0.42 | 8.08 |
| CCE (α=.10) | 28.00 | 49.37 | 0.36 | 9.10 |
| CONFLARE (α=.10) | 25.00 | 44.52 | 0.35 | 8.58 |
| TRAQ (α=.10) | 28.00 | 47.95 | 0.36 | 9.26 |
| Stage 1 (α₁=.10) | 9.00 | 18.19 | 0.46 | 1.63 |
| Stage 1 (α₁=.90) | — | — | — | — |
| Stage 1 (α₁=.99) | 39.00 | 58.84 | 0.42 | 9.86 |
| Two-Stage (α₁=.99, α₂=.15) | 37.00 | 53.89 | 0.40 | 8.31 |
| Two-Stage (α₁=.99, α₂=.20) | 34.00 | 51.56 | 0.40 | 7.88 |
| Two-Stage (α₁=.99, α₂=.30) | 29.00 | 46.35 | 0.42 | 6.76 |
| Two-Stage (α₁=.99, α₂=.40) | 26.00 | 43.88 | 0.42 | 5.92 |
| Top-10 + S2 (α₂=.15) | 37.00 | 53.89 | 0.40 | 7.90 |
| Top-10 + S2 (α₂=.20) | 34.00 | 51.56 | 0.40 | 7.58 |
| Top-10 + S2 (α₂=.30) | 29.00 | 46.35 | 0.41 | 6.73 |
| Top-10 + S2 (α₂=.40) | 26.00 | 43.88 | 0.42 | 5.91 |

#### 16.3.4. WebQA

| Method | EM (%) | F1 (%) | Latency (s) | Approx. TFLOPs |
|---|---:|---:|---:|---:|
| Fixed Top-5 | 9.00 | 27.90 | 0.27 | 8.86 |
| CCE (α=.10) | 12.00 | 28.89 | 0.47 | 50.99 |
| CONFLARE (α=.10) | 12.00 | 28.35 | 0.43 | 46.36 |
| TRAQ (α=.10) | 13.00 | 29.29 | 0.49 | 54.66 |
| Stage 1 (α₁=.10) | 7.00 | 23.96 | 0.27 | 2.74 |
| Stage 1 (α₁=.90) | — | — | — | — |
| Stage 1 (α₁=.99) | 10.00 | 27.67 | 3.77 | 55.07 |
| Two-Stage (α₁=.99, α₂=.15) | 10.00 | 26.84 | 1.46 | 27.16 |
| Two-Stage (α₁=.99, α₂=.20) | 9.00 | 26.08 | 1.25 | 23.37 |
| Two-Stage (α₁=.99, α₂=.30) | 9.00 | 25.71 | 0.79 | 16.88 |
| Two-Stage (α₁=.99, α₂=.40) | 9.00 | 27.93 | 0.68 | 13.69 |
| Top-10 + S2 (α₂=.15) | 12.00 | 30.02 | 0.38 | 13.83 |
| Top-10 + S2 (α₂=.20) | 9.00 | 28.09 | 0.38 | 13.23 |
| Top-10 + S2 (α₂=.30) | 11.00 | 29.88 | 0.33 | 11.70 |
| Top-10 + S2 (α₂=.40) | 9.00 | 27.88 | 0.30 | 9.70 |

#### 16.3.5. Overall downstream

- Overall giữ số được báo trong bản cung cấp. Trung bình F1 từ bốn ô đã làm tròn của Two-Stage $.40$ là 46.9425%; bản cung cấp ghi 46.95%. Khi có log, tính lại Overall từ số chưa làm tròn.

| Method | Overall EM (%) | Overall F1 (%) |
|---|---:|---:|
| Fixed Top-5 | 38.75 | 52.25 |
| CCE (α=.10) | 37.75 | 51.34 |
| CONFLARE (α=.10) | 35.25 | 48.43 |
| TRAQ (α=.10) | 37.75 | 50.72 |
| Stage 1 (α₁=.10) | 17.50 | 28.24 |
| Stage 1 (α₁=.90) | — | — |
| Stage 1 (α₁=.99) | 41.75 | 55.28 |
| Two-Stage (α₁=.99, α₂=.15) | 39.50 | 52.43 |
| Two-Stage (α₁=.99, α₂=.20) | 38.25 | 51.20 |
| Two-Stage (α₁=.99, α₂=.30) | 36.25 | 48.96 |
| Two-Stage (α₁=.99, α₂=.40) | 33.75 | 46.95 |
| Top-10 + S2 (α₂=.15) | 39.75 | 52.85 |
| Top-10 + S2 (α₂=.20) | 38.25 | 51.57 |
| Top-10 + S2 (α₂=.30) | 36.25 | 49.28 |
| Top-10 + S2 (α₂=.40) | 35.00 | 48.05 |

### 16.4. Bảng context-selection mới theo từng dataset

- Kept là số chunks trung bình/query; P/R/Selection F1 theo phần trăm. Các ô trống phản ánh số chưa có trong bản LaTeX.

#### 16.4.1. HotpotQA

| Method | Kept | Precision (%) | Recall (%) | Selection F1 (%) |
|---|---:|---:|---:|---:|
| Fixed Top-5 | 4.95 | 33.7 | 83.5 | 48.1 |
| CCE (α=.10) | 8.61 | 21.0 | 90.5 | 34.1 |
| CONFLARE (α=.10) | 7.05 | 21.0 | 74.0 | 32.7 |
| TRAQ (α=.10) | 8.17 | 21.4 | 87.5 | 34.4 |
| Stage 1 (α₁=.10) | 0.89 | 55.1 | 24.5 | 33.9 |
| Stage 1 (α₁=.90) | 8.11 | 22.3 | 90.5 | 35.8 |
| Stage 1 (α₁=.99) | 9.59 | 20.5 | 98.5 | 33.9 |
| Two-Stage (α₁=.99, α₂=.15) | 6.92 | 26.4 | 91.5 | 41.0 |
| Two-Stage (α₁=.99, α₂=.20) | 6.08 | 29.6 | 90.0 | 44.5 |
| Two-Stage (α₁=.99, α₂=.30) | 5.15 | 32.2 | 83.0 | 46.4 |
| Two-Stage (α₁=.99, α₂=.40) | — | — | — | — |
| Top-10 + S2 (α₂=.15) | 6.94 | 26.4 | 91.5 | 40.9 |
| Top-10 + S2 (α₂=.20) | 6.09 | 29.6 | 90.0 | 44.5 |
| Top-10 + S2 (α₂=.30) | 5.15 | 32.2 | 83.0 | 46.4 |
| Top-10 + S2 (α₂=.40) | 3.85 | 36.9 | 71.0 | 48.5 |

#### 16.4.2. MMQA

| Method | Kept | Precision (%) | Recall (%) | Selection F1 (%) |
|---|---:|---:|---:|---:|
| Fixed Top-5 | 5.00 | 24.0 | 77.4 | 36.6 |
| CCE (α=.10) | 18.40 | 7.7 | 91.0 | 14.2 |
| CONFLARE (α=.10) | 18.05 | 7.8 | 90.3 | 14.4 |
| TRAQ (α=.10) | 19.50 | 7.5 | 94.2 | 13.9 |
| Stage 1 (α₁=.10) | 0.59 | 38.6 | 14.8 | 21.4 |
| Stage 1 (α₁=.90) | 6.85 | 16.7 | 74.2 | 27.3 |
| Stage 1 (α₁=.99) | 9.43 | 14.1 | 85.8 | 24.2 |
| Two-Stage (α₁=.99, α₂=.15) | 9.26 | 14.4 | 85.8 | 24.7 |
| Two-Stage (α₁=.99, α₂=.20) | 9.24 | 14.4 | 85.8 | 24.7 |
| Two-Stage (α₁=.99, α₂=.30) | 8.30 | 15.3 | 81.9 | 25.8 |
| Two-Stage (α₁=.99, α₂=.40) | — | — | — | — |
| Top-10 + S2 (α₂=.15) | 9.79 | 14.4 | 91.0 | 24.9 |
| Top-10 + S2 (α₂=.20) | 9.72 | 14.5 | 91.0 | 25.0 |
| Top-10 + S2 (α₂=.30) | 8.52 | 15.6 | 85.8 | 26.4 |
| Top-10 + S2 (α₂=.40) | 6.20 | 18.5 | 74.2 | 29.7 |

#### 16.4.3. TAT-QA

| Method | Kept | Precision (%) | Recall (%) | Selection F1 (%) |
|---|---:|---:|---:|---:|
| Fixed Top-5 | 4.37 | 29.7 | 97.0 | 45.5 |
| CCE (α=.10) | 5.01 | 24.6 | 91.8 | 38.8 |
| CONFLARE (α=.10) | 4.64 | 25.0 | 86.6 | 38.8 |
| TRAQ (α=.10) | 5.09 | 24.6 | 93.3 | 38.9 |
| Stage 1 (α₁=.10) | 0.33 | 69.7 | 17.2 | 27.6 |
| Stage 1 (α₁=.90) | 4.53 | 25.6 | 86.6 | 39.5 |
| Stage 1 (α₁=.99) | 5.43 | 24.1 | 97.8 | 38.7 |
| Two-Stage (α₁=.99, α₂=.15) | 4.23 | 26.2 | 82.8 | 39.8 |
| Two-Stage (α₁=.99, α₂=.20) | 3.90 | 28.2 | 82.1 | 42.0 |
| Two-Stage (α₁=.99, α₂=.30) | 3.28 | 30.8 | 75.4 | 43.7 |
| Two-Stage (α₁=.99, α₂=.40) | — | — | — | — |
| Top-10 + S2 (α₂=.15) | 4.23 | 26.2 | 82.8 | 39.9 |
| Top-10 + S2 (α₂=.20) | 3.90 | 28.2 | 82.1 | 42.0 |
| Top-10 + S2 (α₂=.30) | 3.28 | 30.8 | 75.4 | 43.7 |
| Top-10 + S2 (α₂=.40) | 2.78 | 34.2 | 70.9 | 46.1 |

#### 16.4.4. WebQA

| Method | Kept | Precision (%) | Recall (%) | Selection F1 (%) |
|---|---:|---:|---:|---:|
| Fixed Top-5 | 5.00 | 23.0 | 65.7 | 34.1 |
| CCE (α=.10) | 23.36 | 6.5 | 86.3 | 12.1 |
| CONFLARE (α=.10) | 21.60 | 6.8 | 83.4 | 12.6 |
| TRAQ (α=.10) | 24.68 | 6.3 | 88.6 | 11.8 |
| Stage 1 (α₁=.10) | 0.58 | 34.5 | 11.4 | 17.1 |
| Stage 1 (α₁=.90) | 6.43 | 16.3 | 60.0 | 25.6 |
| Stage 1 (α₁=.99) | 8.33 | 14.5 | 69.1 | 24.0 |
| Two-Stage (α₁=.99, α₂=.15) | 7.38 | 15.6 | 65.7 | 25.2 |
| Two-Stage (α₁=.99, α₂=.20) | 7.08 | 15.5 | 62.9 | 24.9 |
| Two-Stage (α₁=.99, α₂=.30) | 6.47 | 16.7 | 61.7 | 26.3 |
| Two-Stage (α₁=.99, α₂=.40) | — | — | — | — |
| Top-10 + S2 (α₂=.15) | 8.08 | 16.0 | 73.7 | 26.2 |
| Top-10 + S2 (α₂=.20) | 7.73 | 16.0 | 70.9 | 26.2 |
| Top-10 + S2 (α₂=.30) | 6.88 | 17.4 | 68.6 | 27.8 |
| Top-10 + S2 (α₂=.40) | 5.89 | 18.3 | 61.7 | 28.3 |

### 16.5. Phân tích mới để phát triển phần Results

- **HotpotQA — pruning đổi quality lấy cost:** Stage 1 $.99$ có EM/F1 cao nhất trong bảng mới (**71.00/82.52**). Two-Stage $.15$ đạt **64.00/76.96**, giảm TFLOPs từ 23.34 xuống 16.43 (**29.6%**) nhưng mất **5.56 điểm F1**. So với CCE, cùng row Two-Stage tăng **3.94 điểm F1**, giảm **21.9%** compute proxy và giảm chunks từ 8.61 xuống 6.92. Figure quality–cost cần hiển thị cả Stage 1 để thấy hai operating points.
- **MMQA — selection gain chưa chuyển thành downstream gain:** CCE vẫn có generation EM/F1 cao nhất (**49.00/54.08**). Top-10 + S2 $.40$ tăng Selection F1 từ CCE **14.2%** lên **29.7%**, nhưng generation F1 là **51.75%**, thấp hơn CCE **2.33 điểm**. Fixed Top-5 đạt Selection F1 **36.6%** và generation F1 **52.40%**; cần giữ baseline này trong phân tích.
- **TAT-QA — permissive Stage 1 giữ evidence tốt nhất trong bảng:** Stage 1 $.99$ đạt **39.00 EM / 58.84 F1**; Fixed Top-5 đạt **37.00/57.30**. Two-Stage $.15$ và Top-10 + S2 $.15$ đạt **37.00/53.89**: cao hơn ba literature rows về EM, nhưng F1 thấp hơn Fixed Top-5 **3.41 điểm**. Selection F1 của Top-10 + S2 $.40$ là **46.1%**, chỉ hơn Fixed Top-5 **0.6 điểm**, trong khi generation F1 giảm còn **43.88%**. Phân tích errors nên tập trung vào mất table operands/complementary evidence.
- **WebQA — hybrid có điểm quality–cost đáng chú ý:** Top-10 + S2 $.15$ đạt F1 **30.02%** so với TRAQ **29.29%**, tăng **0.73 điểm**; proxy giảm **54.66 → 13.83 TFLOPs (74.7%)**, latency **.49 → .38 s (22.4%)**, chunks **24.68 → 8.08**. Recall đồng thời giảm **88.6% → 73.7%**. Hai tỷ lệ giảm compute/latency khác nhau; không gọi proxy reduction là mức tăng tốc tương đương.
- **Overall — best row tùy nhóm:** trong các Two-Stage/hybrid rows, Top-10 + S2 $.15$ có Overall EM/F1 cao nhất (**39.75/52.85**), hơn Fixed Top-5 **1.00 điểm EM / .60 điểm F1**. Xét cả Stage 1 only, Stage 1 $.99$ mới là best Overall (**41.75/55.28**). Main table cần thể hiện cả hai kết luận.
- **Recall và Selection F1 là hai đánh đổi riêng:** trên WebQA, hybrid $.15$ đạt Selection F1 **26.2%**, cao hơn CCE/TRAQ **12.1/11.8%**, nhưng vẫn thấp hơn Fixed Top-5 **34.1%**. Trên MMQA cũng có pattern tương tự. Không dùng tăng Selection F1 so với literature rows để suy ra thắng tất cả baselines.
- **Latency cần giải thích bằng log:** Stage 1 $.99$ trên MMQA/WebQA có latency **1.96/3.77 s**, cao hơn CCE **.44/.47 s**, dù kept chunks ít hơn. Đây là điểm cần kiểm tra runtime, image/token counts và cách timing giữa các runs; số chunks không đủ giải thích latency.

### 16.6. Cập nhật Research Gap và narrative theo bảng mới

- **CCE:** support-pair cutoff giữ recall cao nhưng trên MMQA/WebQA còn **18.40/23.36 chunks**, precision chỉ **7.7/6.5%**. Điều này đặt ra bài toán tăng evidence density và giảm context cost, đồng thời bảo toàn generation quality; MMQA CCE vẫn đạt F1 tốt nhất **54.08%** trong bảng mới.
- **CONFLARE:** best-support cutoff giảm HotpotQA context **8.61 → 7.05** so với CCE, nhưng precision vẫn **21.0%** và recall giảm **90.5% → 74.0%**; generation F1 giảm **73.02% → 68.78%**. Đây là ví dụ giảm context mà chưa đổi thành evidence enrichment, và mất coverage đi kèm mất answer quality.
- **TRAQ:** retrieval allocation $.05$ giữ nhiều context trên MMQA/WebQA (**19.50/24.68 chunks**) và recall **94.2/88.6%**, nhưng precision **7.5/6.3%**. Trên WebQA, hybrid $.15$ cho điểm cost–quality thấp chi phí hơn; trên MMQA, TRAQ recall cao hơn hybrid nhưng generation F1 **52.12%** cũng cao hơn hybrid $.15$ **51.53%**. Cần giải thích riêng từng dataset.
- **Thiết kế đề xuất:** false-reference bank để kiểm tra admission; support-reference bank để prune score không tương thích với support. Hai stages tách việc mở rộng context và giảm noise thành hai tham số; kết quả có thể tạo các điểm quality–cost hữu ích nhưng vẫn phụ thuộc vào cosine separability và evidence cần cho reasoning.
- **Ablation mới cần ưu tiên:** Stage 1 $.99$ versus Two-Stage $.15$; Fixed Top-10 versus Top-10 + S2; Two-Stage versus hybrid cùng $\alpha_2$. Báo any/all-support, empty rate và per-modality breakdown cùng downstream để xác định evidence nào mất.

### 16.7. Viết Methodology từ bản LaTeX mới

- Giữ hai công thức p-values trong mục 5.4–5.5; bổ sung sơ đồ cho mỗi phép reject: **Stage 1 reject false-null → admit**, **Stage 2 reject support-null → prune**.
- Với $N=L_q$ ở Stage 1, BH tìm $k=\max\{j:p_{(j)}\le j\alpha_1/N\}$; nếu không có $j$ thỏa thì không admit chunk. Stage 2 dùng $M=|\mathcal S_1|$ và $\alpha_2$, prune các rejected candidates; nếu $M=0$ thì context cuối rỗng.
- Thêm pseudocode build banks: đọc calibration qids → lấy frozen Top-L candidates và labels → bỏ unknown → gom false/support scores theo stratum → sort → lưu sizes/fingerprint → test lookup, tuyệt đối không thêm test scores vào bank.
- Phân biệt **high-recall operating point** với guarantee: false-null BH kiểm soát false discoveries trong admitted set dưới assumptions phù hợp; không tự tạo FNR bound. Support-null BH gắn với false discoveries trong pruned set; không tự tạo retained-context FPR/FDR bound. Formal scope tiếp tục theo mục 5.8 và 10.1.
- Hiệu chuẩn support bank cho candidates qua Stage 1 cần được thiết kế/kiểm chứng cho selective procedure. Ghi rõ bank xây từ tất cả support hay support sau screening, và strategy calibration độc lập nếu áp dụng.
- Reference-bank sample là candidate score, không phải query count. Báo cả 1,000 calibration queries và số false/support records thực tế, theo pooled/conditional configuration.

### 16.8. Tables, figures và việc hoàn thiện tiếp

- **Table generation:** dùng bảng riêng từng dataset hoặc hai panels để đọc được EM/F1/latency/TFLOPs; giữ Overall EM/F1. Đánh dấu best của toàn bảng và best của nhóm proposed riêng biệt nếu cần.
- **Table selection:** giữ Kept/P/R/Selection F1, thêm any/all-support và empty khi có logs. Ô thiếu của Stage 1 $.90$ downstream và Two-Stage $.40$ selection tiếp tục để **—** đến khi có output.
- **Figure quality–cost:** plot generation F1 theo approximate TFLOPs, ghi Stage 1 $.99$, Two-Stage $.15$, hybrid $.15$ và Fixed Top-5 để thấy trade-off hiện có.
- **Figure pruning:** plot precision/recall/all-support theo $\alpha_2$; đánh dấu downstream F1 trên panel riêng, không đồng nhất hai loại F1.
- **Timing metadata:** giữ công thức proxy `2 × parameter_count × total_sequence_length`; lưu input/output tokens, image pixels/tokens, CUDA synchronization, GPU/runtime và phạm vi timing để giải thích các điểm latency khác biệt.
- **Truy vết bảng mới:** attach raw per-query log và run config từ người chạy; làm paired bootstrap cho các chênh lệch nhỏ như Overall +.60 điểm F1 và WebQA +.73 điểm F1 trước khi diễn giải độ ổn định.
- **Code dùng để tạo Word:** [export_paper_layout_word.py](scripts/export_paper_layout_word.py). Bản Word giữ heading đánh số, bullets, bảng, hyperlinks và công thức để chỉnh sửa.
