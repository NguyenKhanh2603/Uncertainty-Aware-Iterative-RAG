# Những gì đã chạy đến 16-09-2026

File này là index của các thí nghiệm retrieval/chunk pruning. Mục tiêu là giúp
quay lại đúng artifact, code và protocol thay vì so sánh nhầm các con số dùng
khác candidate pool, model hoặc mục tiêu conformal.

## Cách đọc kết quả

- `P` là chunk precision; `R` là micro support recall, thường điều kiện trên
  truy vấn có support trong Top-30. `E2E all` tính cả retrieval failures.
- `BY` là candidate-wise Benjamini--Yekutieli với false-score p-values, `L=30`
  và context cap `K=10`. Nó có thể trả về empty set.
- `query conformal` giữ support theo mục tiêu `any-support` hoặc `all-support`.
  Nó không cùng statistical target với BY, nên không được diễn giải là cùng một
  guarantee.
- Các bảng Qwen2-VL-2B dùng split official train thành probe/calibration và
  official dev/validation làm test. Qwen2-VL-7B dùng pilot 100/100/100 mỗi
  dataset; không trộn hai protocol này trong bootstrap claim.

## Kết luận ngắn

1. Cosine + BY có precision cao khi nó chọn được chunk, nhưng recall rất thấp
   vì BY rỗng ở phần lớn query. Tăng `L` không sửa được lỗi này.
2. Thay cosine score trong BY bằng Jina m0 cải thiện precision và recall ở
   TAT-QA/HotpotQA/MMQA, nhưng BY vẫn quá conservative; WebQA gần như rỗng.
3. Attention-only có recall cao vì giữ nhiều chunk, đổi lại precision rất thấp.
4. Với query-level conformal, Jina m0 là selector thực dụng mạnh nhất. Internal
   signal cải thiện cosine rõ rệt, còn gain trên Jina m0 là nhỏ và phụ thuộc
   dataset.
5. BGE + internal attention có kết quả tốt trên một locked TAT-QA sample, nhưng
   chưa đủ rộng để thay thế Jina m0 trên bốn dataset.

## 1. Attention-only baseline: recall cao, precision rất thấp

Protocol: attention từ draft answer về Top-30 chunks, đảo thứ tự chunk để giảm
position effect, sau đó query-level all-support conformal ở `alpha=.1`.
Qwen2-VL-2B, official test role. Các giá trị dưới đây là `mean chunks / P / R`.

| Dataset | Attention-only | Diễn giải |
|---|---:|---|
| TAT-QA | 21.28 / 4.80% / 91.79% | Giữ gần hết context để đạt recall. |
| HotpotQA | 14.04 / 12.39% / 94.96% | Recall cao nhưng không phải ranker chính xác. |
| MMQA | 16.34 / 7.20% / 93.93% | Tương tự. |
| WebQA | 22.21 / 5.57% / 98.25% | Ví dụ rõ nhất của high recall / low precision. |

- Tổng hợp: [four-dataset snapshot](research/internal_state_rag/results/qwen2vl_jina4/FOUR_DATASET_RESULTS_SNAPSHOT_2026-09-14.md).
- JSON: [TAT-QA](research/internal_state_rag/results/qwen2vl_jina4/tatqa_attention_only_report.json), [HotpotQA](research/internal_state_rag/results/qwen2vl_jina4/hotpotqa_attention_only_report.json), [MMQA](research/internal_state_rag/results/qwen2vl_jina4/mmqa_attention_only_report.json), [WebQA](research/internal_state_rag/results/qwen2vl_jina4/webqa_attention_only_report.json).
- Code extract: [run_attention_only_clean_features.py](research/internal_state_rag/run_attention_only_clean_features.py); analyzer: [analyze_attention_only_clean_conformal.py](research/internal_state_rag/analyze_attention_only_clean_conformal.py).
- Caveat: một phần attention trace non-finite được tie về zero; invalid test-query
  counts là 547/1000 TAT-QA, 205/1000 HotpotQA, 190/1000 MMQA, 32/250 WebQA.

## 2. Original conformal BY với cosine: alpha sweep

Đây là proposal gốc: false-score bank từ calibration false chunks, candidate
p-values, BY trên 30 hypotheses/query, cap 10. Bảng là pooled bank ở `alpha=.1`.

| Dataset | Mean chunks | P | R | Empty | E2E all |
|---|---:|---:|---:|---:|---:|
| TAT-QA | 0.208 | 25.00% | 5.08% | 93.7% | 4.4% |
| HotpotQA | 0.238 | 69.33% | 9.04% | 85.1% | 2.6% |
| MMQA | 0.065 | 50.77% | 2.78% | 96.3% | 1.9% |
| WebQA | 0.000 | -- | 0.00% | 100.0% | 0.0% |

Alpha `.20/.10/.05`, pooled và modality-conditioned banks đều có trong
[BY cosine vs Jina report](research/internal_state_rag/results/by_score_ablation_2026-09-16/BY_COSINE_VS_JINA_RERANKER_2026-09-16.md).

### Sweep candidate pool Top-L (`alpha=.10`, pooled, `K=10`)

Mỗi prefix tái tạo false-score calibration bank và BY decision đúng với `L`
đó, không reuse p-value của Top-30. `Ceiling` là retrieval availability; `R`
là conditional micro support recall.

| Dataset | L | Ceiling | Mean chunks | P | R | Empty |
|---|---:|---:|---:|---:|---:|---:|
| TAT-QA | 5 / 10 / 20 / 30 | 73.3 / 80.6 / 88.2 / 91.9% | .196 / .263 / .226 / .208 | 30.61 / 22.05 / 23.45 / 25.00% | 7.69 / 6.61 / 5.44 / 5.08% | 91.8 / 92.7 / 93.6 / 93.7% |
| HotpotQA | 5 / 10 / 20 / 30 | 98.3 / 98.9 / 99.4 / 99.6% | .262 / .277 / .257 / .238 | 69.08 / 64.26 / 65.76 / 69.33% | 11.32 / 10.41 / 9.42 / 9.04% | 83.8 / 84.1 / 84.7 / 85.1% |
| MMQA | 5 / 10 / 20 / 30 | 91.3 / 93.1 / 94.3 / 94.8% | .091 / .093 / .072 / .065 | 56.04 / 52.69 / 50.00 / 50.77% | 4.87 / 4.44 / 3.10 / 2.78% | 94.5 / 94.7 / 96.0 / 96.3% |
| WebQA | 5 / 10 / 20 / 30 | 40.8 / 54.8 / 66.4 / 72.8% | .020 / .016 / .016 / .000 | 20.00 / 0 / 0 / --% | .86 / 0 / 0 / 0% | 99.6 / 99.6 / 99.6 / 100.0% |

Tăng L tăng retrieval ceiling nhưng giảm BY recall gần như đơn điệu: BY có nhiều
hypotheses hơn và false bank nhận thêm lower-score negatives. Fresh full-corpus
Top-50 xác nhận điều này, không chỉ là prefix artifact:

| Dataset | L=30: Ceiling / P / R / Empty | L=50: Ceiling / P / R / Empty |
|---|---|---|
| TAT-QA | 91.9% / 25.00% / 5.08% / 93.7% | 94.2% / 26.13% / 4.90% / 93.8% |
| HotpotQA | 99.6% / 69.33% / 9.04% / 85.1% | 99.7% / 69.27% / 7.63% / 86.9% |
| MMQA | 94.8% / 50.77% / 2.78% / 96.3% | 95.6% / 52.63% / 2.46% / 96.7% |

### Alpha sweep at fixed `L=30`, pooled, `K=10`

Ô có dạng `mean chunks / P / R / empty`. Nới alpha tăng recall, nhưng precision
giảm và vẫn không đưa BY thành high-recall selector.

| Dataset | alpha=.05 | alpha=.10 | alpha=.20 | alpha=.50 |
|---|---|---|---|---|
| TAT-QA | .132 / 24.2% / 3.1% / 95.9% | .208 / 25.0% / 5.1% / 93.7% | .354 / 20.3% / 7.0% / 90.6% | .774 / 15.6% / 11.8% / 84.4% |
| HotpotQA | .118 / 74.6% / 4.8% / 91.8% | .238 / 69.3% / 9.0% / 85.1% | .474 / 57.6% / 15.0% / 77.5% | 1.060 / 49.5% / 28.8% / 58.5% |
| MMQA | .046 / 60.9% / 2.4% / 97.1% | .065 / 50.8% / 2.8% / 96.3% | .150 / 46.0% / 5.8% / 92.7% | .420 / 37.1% / 13.1% / 83.9% |
| WebQA | .000 / -- / 0.0% / 100.0% | .000 / -- / 0.0% / 100.0% | .064 / 12.5% / .9% / 98.8% | .432 / 4.6% / 2.2% / 94.0% |

Removing `K=10` at alpha `.50` also is not a solution: TAT-QA recall only goes
from 11.83% to 13.78% while precision drops 15.63% to 7.34%; the other datasets
change recall by at most 0.88 points. BY makes too few discoveries *before* the
cap applies.

- Detailed Top-L/alpha/cap report: [BY cosine Top-L sweep](research/internal_state_rag/results/by_topl_sweep_2026-09-16/BY_COSINE_TOPL_SWEEP_2026-09-16.md).
- Raw sweep outputs: [directory](research/internal_state_rag/results/by_topl_sweep_2026-09-16/), including fresh [Top-50 retrieval logs](research/internal_state_rag/results/by_topl_sweep_2026-09-16/top50_logs/).

- JSON mới nhất: [TAT-QA](research/internal_state_rag/results/by_score_ablation_2026-09-16/tatqa_by_cosine.json), [HotpotQA](research/internal_state_rag/results/by_score_ablation_2026-09-16/hotpotqa_by_cosine.json), [MMQA](research/internal_state_rag/results/by_score_ablation_2026-09-16/mmqa_by_cosine.json), [WebQA](research/internal_state_rag/results/by_score_ablation_2026-09-16/webqa_by_cosine.json).
- Code: [analyze_current_cosine_by.py](research/internal_state_rag/analyze_current_cosine_by.py). Script hiện hỗ trợ `--score-field`, `--order-field`, và `--top-l` để tái tạo bank/decision cho từng candidate-pool size.

## 3. Conformal BY với Jina m0 reranker score

Giữ nguyên Jina embeddings-v4 Top-30 candidate pool, thay score dùng cho
false-bank/p-value/ranking bằng `jina_reranker_score`. Đây là score ablation,
không phải đổi candidate pool.

| Dataset | Mean chunks | P | R | Empty | E2E all |
|---|---:|---:|---:|---:|---:|
| TAT-QA | 0.113 | 75.22% | 8.31% | 91.3% | 7.4% |
| HotpotQA | 0.293 | 88.40% | 14.19% | 78.4% | 6.7% |
| MMQA | 0.309 | 89.64% | 23.34% | 72.5% | 24.4% |
| WebQA | 0.044 | 0.00% | 0.00% | 98.8% | 0.0% |

Reranker làm BY tốt hơn ở ba dataset đầu, nhưng không biến BY thành một
high-recall selector. Tại `alpha=.2`, pooled Jina-BY recall là 16.72% TAT-QA,
21.97% HotpotQA, 28.73% MMQA và 1.75% WebQA; precision giảm theo expected
trade-off. Full sweep, modality ablation và protocol nằm trong
[report](research/internal_state_rag/results/by_score_ablation_2026-09-16/BY_COSINE_VS_JINA_RERANKER_2026-09-16.md).

- JSON: [TAT-QA](research/internal_state_rag/results/by_score_ablation_2026-09-16/tatqa_by_jina_m0.json), [HotpotQA](research/internal_state_rag/results/by_score_ablation_2026-09-16/hotpotqa_by_jina_m0.json), [MMQA](research/internal_state_rag/results/by_score_ablation_2026-09-16/mmqa_by_jina_m0.json), [WebQA](research/internal_state_rag/results/by_score_ablation_2026-09-16/webqa_by_jina_m0.json).
- Code: [analyze_current_cosine_by.py](research/internal_state_rag/analyze_current_cosine_by.py); reranking: [rerank_jina_m0.py](research/internal_state_rag/rerank_jina_m0.py).

## 4. Query-level conformal: cosine, Jina reranker, and Qwen2-VL-2B fusion

Đây là official-test evaluation với Jina-v4 Top-30, Qwen2-VL-2B internal
features và query-level all-support conformal, `alpha=.1`. Mỗi ô là
`mean chunks / P / R`, condition trên Top-30 retrievable queries.

| Dataset | Cosine | Cosine + internal | Jina m0 | Jina m0 + internal |
|---|---:|---:|---:|---:|
| TAT-QA | 10.84 / 9.08% / 88.37% | 6.27 / 16.06% / 90.52% | 3.92 / 26.24% / 92.38% | 3.93 / 26.30% / 92.77% |
| HotpotQA | 9.66 / 17.90% / 94.41% | 3.89 / 44.24% / 93.92% | 2.91 / 59.24% / 94.19% | 2.76 / 62.59% / 94.25% |
| MMQA | 8.14 / 14.30% / 92.92% | 4.84 / 24.16% / 93.43% | 2.93 / 40.13% / 93.85% | 2.94 / 40.07% / 94.02% |
| WebQA | 19.07 / 5.85% / 88.65% | not in this 2B table | 7.29 / 15.38% / 89.08% | 5.35 / 20.66% / 87.77% |

Điểm chính: internal fusion giúp cosine mạnh trên TAT/Hotpot/MMQA. Với Jina
m0, gain recall là +0.39, +0.05, +0.17 điểm phần trăm ở ba dataset đó và
-1.31 trên WebQA; chưa có evidence cho universal reranker-fusion gain.

- Summary: [four-dataset snapshot](research/internal_state_rag/results/qwen2vl_jina4/FOUR_DATASET_RESULTS_SNAPSHOT_2026-09-14.md), [cross-dataset report](research/internal_state_rag/results/qwen2vl_jina4/CROSS_DATASET_REPORT_2026-09-14.md).
- Per-dataset JSON: [TAT-QA](research/internal_state_rag/results/qwen2vl_jina4/tatqa_ablation_report.json), [HotpotQA](research/internal_state_rag/results/qwen2vl_jina4/hotpotqa_ablation_report.json), [MMQA](research/internal_state_rag/results/qwen2vl_jina4/mmqa_ablation_report.json), [WebQA](research/internal_state_rag/results/qwen2vl_jina4/webqa_ablation_report.json).
- Code: [run_qwen2vl_pairwise_features.py](research/internal_state_rag/run_qwen2vl_pairwise_features.py), [analyze_qwen2vl_jina_ablation.py](research/internal_state_rag/analyze_qwen2vl_jina_ablation.py).

## 5. Qwen2-VL-7B candidate fusion: cosine and Jina m0

Protocol khác mục 4: 100 disjoint probe-train + 100 calibration + 100 test
queries mỗi dataset, Qwen2-VL-7B BF16, aligned Jina reranker artifacts,
query-level all-support conformal ở `alpha=.1`.

| Dataset | Cosine | Cosine + 7B internal | Jina m0 | Jina m0 + 7B internal |
|---|---:|---:|---:|---:|
| TAT-QA | 9.11 / 10.86% / 85.71% | 3.49 / 29.87% / 90.48% | 3.36 / 31.05% / 90.48% | 3.20 / 32.30% / 89.52% |
| HotpotQA | 9.98 / 17.94% / 97.28% | 4.08 / 44.36% / 98.37% | 3.13 / 55.59% / 94.57% | 2.79 / 61.29% / 92.93% |
| MMQA | 11.95 / 9.81% / 96.46% | 4.74 / 25.17% / 98.23% | 2.85 / 41.51% / 97.35% | 2.85 / 41.51% / 97.35% |
| WebQA | 19.17 / 5.89% / 92.94% | 7.97 / 13.44% / 88.24% | 8.07 / 13.63% / 90.59% | 7.61 / 14.63% / 91.76% |

Kết luận: cosine + 7B fusion cải thiện rõ trên ba dataset đầu, WebQA mất
4.71 điểm recall. Jina + 7B chỉ tốt hơn Jina rõ ở WebQA (+1.18 điểm recall,
bootstrap CI vẫn chứa 0); các dataset khác trade ít recall lấy ít chunk hơn.

- Primary report: [Qwen2-VL-7B candidate fusion](research/internal_state_rag/results/qwen2vl_7b_jina4/QWEN2VL_7B_CANDIDATE_FUSION_100_REPORT_2026-09-15.md).
- JSON aligned: [TAT-QA](research/internal_state_rag/results/qwen2vl_7b_jina4/tatqa_fusion_ablation_100_aligned.json), [HotpotQA](research/internal_state_rag/results/qwen2vl_7b_jina4/hotpotqa_fusion_ablation_100_aligned.json), [MMQA](research/internal_state_rag/results/qwen2vl_7b_jina4/mmqa_fusion_ablation_100_aligned.json), [WebQA](research/internal_state_rag/results/qwen2vl_7b_jina4/webqa_fusion_ablation_100_aligned.json).
- Code: [run_qwen2vl_pairwise_features.py](research/internal_state_rag/run_qwen2vl_pairwise_features.py), [analyze_qwen2vl_jina_ablation.py](research/internal_state_rag/analyze_qwen2vl_jina_ablation.py).
- Không dùng `research/internal_state_rag/results/qwen2vl_7b_jina4/fusion_features/` hay các JSON không có suffix `_aligned`: chúng dùng retrieval file thiếu Jina score, nên invalid cho Jina fusion.

## 6. BGE conformal fusion

### BGE + Qwen hidden/LM residual, multi-dataset

Qwen2.5-3B, frozen dense Top-30, BGE reranker, OOF-selected logistic hidden
probe và query-level all-support conformal. `alpha=.1`:

| Dataset | BGE | BGE + LM-head + hidden |
|---|---:|---:|
| TAT-QA | 5.391 / 18.33% / 87.85% | **3.233 / 30.85% / 88.66%** |
| HotpotQA | **2.257 / 72.38% / 92.56%** | same; OOF chọn internal weights = 0 |
| MMQA text/table pilot | 1.841 / 64.46% / 91.86% | 1.766 / 66.99% / 91.60% |

TAT-QA là gain rõ nhất. HotpotQA là negative control quan trọng: BGE đã gần
saturation nên OOF tắt internal path. MMQA chỉ là text/table pilot, không gồm
image-support.

- Main report: [multi-dataset BGE fusion](research/internal_state_rag/MULTIDATASET_INTERNAL_FUSION_RESULTS_2026-09-13.md).
- Result JSON: [TAT-QA](research/internal_state_rag/results/pairwise_conformal_clean_n96_cal904_test1000.json), [HotpotQA](research/internal_state_rag/results/hotpotqa_internal_fusion_train500_cal500_test1000.json), [MMQA text/table](research/internal_state_rag/results/mmqa_text_table_internal_fusion_train477_cal523_test300.json).
- Code: [run_pairwise_relevance_features.py](research/internal_state_rag/run_pairwise_relevance_features.py), [analyze_multidataset_internal_fusion.py](research/internal_state_rag/analyze_multidataset_internal_fusion.py), [analyze_pairwise_conformal_clean_split.py](research/internal_state_rag/analyze_pairwise_conformal_clean_split.py).

### BGE + position-controlled attention

Đây là locked 150-query TAT-QA evaluation riêng: `alpha=.05` fusion giữ 13.39
chunks, 95.93% support recall và 95.33% all-support coverage; BGE giữ 16.08,
94.19% và 94.00%. Downstream answer F1 difference chưa statistically reliable.

- Report: [full Top-30 BGE-attention validation](research/internal_state_rag/RESULTS_FULL_TOP30_FUSION_VALIDATION.md).
- Raw/analysis artifacts: [locked test folder](research/internal_state_rag/results/locked_test_top30_n150_seed211/).
- Code: [run_full_topl_validation.py](research/internal_state_rag/run_full_topl_validation.py), [analyze_full_topl_validation.py](research/internal_state_rag/analyze_full_topl_validation.py), [run_pruned_answer_validation.py](research/internal_state_rag/run_pruned_answer_validation.py).

## 7. Additional internal-signal experiments and negative results

- [Causal LOO, hidden deltas, head masking and Probing-RAG gate](research/internal_state_rag/results/qwen2vl_jina4/CAUSAL_AND_PROBING_GATE_REPORT_2026-09-14.md): gold-answer LOO is promising on TAT-QA/MMQA pilot but does not beat Jina consistently; raw attention-head shortlist fails causal controls. The early 2B gate table has an order-alignment correction; use the 7B report below for the aligned gate.
- [Qwen2-VL-7B Probing-RAG gate](research/internal_state_rag/results/qwen2vl_7b_jina4/QWEN2VL_7B_PROBING_GATE_100_REPORT_2026-09-15.md): query-level gate did not produce a robust new pruning gain.
- [DIRECTER plausibility adaptation](research/internal_state_rag/RESULTS_DIRECTER_PLAUSIBILITY.md): probability-ratio plausibility is not a candidate relevance signal; it may only be useful later as a set-level output-fidelity veto.
- Code: [run_qwen2vl_causal_evidence_pilot.py](research/internal_state_rag/run_qwen2vl_causal_evidence_pilot.py), [directer_plausibility.py](research/internal_state_rag/directer_plausibility.py), [analyze_probing_gate_conformal.py](research/internal_state_rag/analyze_probing_gate_conformal.py).

## Recommended interpretation for the next run

Use Jina m0 or BGE as external selector. Use query-level conformal when the
product requirement is retaining support, and report BY separately if
candidate-wise certification is mandatory. Internal states are worth pursuing
as a residual or causal chunk-value signal, but the present evidence does not
support a universal fixed fusion weight or raw attention as the standalone
selector.
