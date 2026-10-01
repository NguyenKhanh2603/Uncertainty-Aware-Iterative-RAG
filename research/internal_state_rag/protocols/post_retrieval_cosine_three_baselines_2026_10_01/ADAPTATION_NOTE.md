# Ghi chú thay đổi sau retrieval: CCE, CONFLARE và TRAQ

Folder này nhận **candidate chunks đã retrieve cùng cosine score**. Đầu vào dùng Jina-v4, candidate pool do từng dataset cung cấp, Top-L tối đa 30; có thể ít hơn 30. Từ đó chạy calibration → chọn context → đánh giá support → đưa context cho cùng Qwen2-VL-7B. Cả ba dùng 1.000 calibration queries mỗi dataset và official test disjoint. Không có reranker, normalization theo query, modality-specific cutoff, Top-1 fallback hoặc cap Top-10 ở bước chọn context.

## 1. CCE Conformal-Embedding

**Xử lý sau retrieval ở nguồn:** mỗi relevant query–snippet pair đóng góp một nonconformity `A = 1 − cosine`; lấy quantile `1 − α` trên các relevant pairs và giữ candidate khi `A ≤ τ`. Các cặp cùng query vẫn là nhiều calibration records. [Định nghĩa nguồn, revision 91732d6](https://github.com/hltcoe/conformal-context-engineering/blob/91732d6058267f180ba9f47873d743288b2625af/README.md).

**Thay đổi khi dùng cosine chung:** Qwen3-Embedding cosine được thay bằng Jina-v4 cosine của các chunks đã có. Relevant labels dùng benchmark support labels thay cho relevance judgments của setup nguồn. Mỗi support candidate xuất hiện trong calibration Top-L đóng góp một record. Code [cce.py](cce.py) dùng `np.quantile(1 − positive_cosines, 1 − α, method="higher")`; giữ `cosine ≥ 1 − τ`. README nguồn không chỉ rõ interpolation, nên `higher` là lựa chọn của implementation hiện tại. Đây là code local triển khai công thức từ README; checkout nguồn chỉ có supplementary materials, không có executable calibration runner. Bước chọn không dùng score từ LLM.

## 2. CONFLARE source-question

**Xử lý sau retrieval ở nguồn:** calibration evaluator đi theo thứ tự distance tăng dần, dừng ở chunk đầu tiên là source chunk hoặc được relevance evaluator chấp nhận; ghi một distance cho question thành công. Lúc lọc context, lấy `np.percentile(calibration_distances, 100 × (1 − α))` và giữ `distance < τ`. [Calibration evaluator](https://github.com/Mayo-Radiology-Informatics-Lab/conflare/blob/ce081a45fb452704daa87f3b37f601b4accc7a82/conflare/conformal/calibration.py#L176-L221), [filter](https://github.com/Mayo-Radiology-Informatics-Lab/conflare/blob/ce081a45fb452704daa87f3b37f601b4accc7a82/conflare/augmented_retrieval/rag.py#L54-L99).

**Thay đổi khi dùng cosine chung:** distance dùng `1 − Jina cosine`. Calibration records dùng benchmark human questions và labelled supports thay cho generated source questions và relevance-evaluator decisions. Trong candidate pool đóng băng, first relevant chunk tương ứng support có cosine cao nhất; query không có support trong pool không đóng góp record. Code [conflare.py](conflare.py) giữ đúng percentile mặc định của NumPy và dấu `<` trên distance, tương đương `cosine > 1 − τ`. Relevance labels chỉ được đọc ở calibration và tính metric; lúc test, quyết định giữ chunk chỉ dùng score/cutoff. Pool chỉ gồm candidates đã cung cấp, nên các relevant chunks ngoài Top-L không thể tham gia calibration hoặc được chọn.

## 3. TRAQ retrieval Bonferroni

**Xử lý sau retrieval ở nguồn:** retrieval calibration dùng best true-retrieval score của từng question; `compute_threshold` lấy lower quantile. Bonferroni retrieval allocation dùng `α_R = α/2`; giữ candidate đạt retrieval cutoff. [Lower quantile](https://github.com/shuoli90/TRAQ/blob/e9b66b5b7fd8fcd0b3c3dbfebc5a3391ca79aa55/misc/utils.py#L619-L636), [allocation](https://github.com/shuoli90/TRAQ/blob/e9b66b5b7fd8fcd0b3c3dbfebc5a3391ca79aa55/run/traq/traq_chatgpt_semantic.py#L639-L657).

**Thay đổi khi dùng cosine chung:** true-retrieval score dùng maximum Jina-v4 cosine của labelled-support candidates trong mỗi calibration query. Calibration split dùng manifest chung; query không có support trong pool không đóng góp record. Code [traq.py](traq.py) dùng `np.quantile(best_support_cosines, α/2, method="lower")`, tương ứng NumPy `interpolation="lower"` ở code nguồn; giữ `cosine ≥ τ`. Với `α=.10`, retrieval cutoff dùng quantile `.05`. Folder này chỉ dùng retrieval threshold của TRAQ để chọn chunks; downstream trả lời bằng shared generator dưới đây.

## 4. Downstream chung sau chọn context

Cả ba đưa selected chunks theo retrieval rank cho cùng generator [generator.py](generator.py). Text/table được chèn nguyên văn; image được đưa vào vision processor bằng file ảnh thật. Image bounds `3.136–200.704` pixels; greedy decoding, tối đa 24 new tokens, Qwen2-VL-7B revision trong [config.json](config.json). Đây là xử lý ảnh ở generation; calibration vẫn pooled theo dataset.

Chat content bắt đầu bằng `Answer using only the supplied context. Return only the short answer.\nQuestion: …\n\nContext:\n`, sau đó lần lượt các chunks. Prompt order được giữ đúng code của run đã hoàn thành. EM/token F1/Numeric dùng [answer_metrics.py](answer_metrics.py); Numeric là kiểm tra số theo implementation chung, không phải official TAT-QA/WebQA evaluator.

## 5. Frozen thresholds và calibrate lại

Run official-full đã công bố **reuse 1.000-query thresholds của `splits_khanh_27_09`**; nó không fit lại thresholds từ scores newly encoded. Vì vậy [run_selection.py](run_selection.py) mặc định dùng [frozen_thresholds.json](frozen_thresholds.json) để replay đúng kết quả đó. `--recalibrate` fit từ 1.000 calibration queries trong portable cosine bundle và ghi vào output mới. Hai chế độ được log rõ trong summary. Method IDs giữ tên của run hiện tại; nếu đổi α khi recalibrate, giá trị thật nằm ở trường `alpha` trong summary.

Các thuật toán trong folder được tách trực tiếp từ `run_literature_protocol_1000cal.py` ở commit `fa6dd68`; không nhập code legacy proxy. Các thay đổi ở trên nằm ở calibration input, score và shared generation protocol; rules riêng của ba selectors được giữ và kiểm tra bằng replay kết quả official-full.
