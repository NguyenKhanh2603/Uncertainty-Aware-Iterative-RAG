# Three post-retrieval cosine baselines — 2026-10-01

Code CCE Conformal-Embedding, CONFLARE source-question và TRAQ retrieval Bonferroni đã tách vào một folder độc lập. Chỉ cần copy folder này để chạy. Input là cosine logs đã có; runner bắt đầu từ calibration/chọn chunks, không làm retrieval hoặc encoding.

- [Ghi chú thay đổi từng method so với nguồn](ADAPTATION_NOTE.md)
- [CCE](cce.py), [CONFLARE](conflare.py), [TRAQ](traq.py)
- [Selection runner](run_selection.py), [downstream runner](run_downstream.py)
- [Shared generator](generator.py), [metrics](answer_metrics.py), [config](config.json)
- [Frozen thresholds của run đã công bố](frozen_thresholds.json)
- [Pinned upstream revisions và file nguồn](SOURCE_PROVENANCE.json)
- [Kiểm tra replay](VALIDATION.md)
- [Kết quả official-full trước khi tách folder](https://github.com/NguyenKhanh2603/Uncertainty-Aware-Iterative-RAG/blob/results/qwen2vl-jina4-four-datasets-2026-09-14/research/internal_state_rag/results/precomputed_cosine_cal1000_official_test_2026_09_30/three_baselines/REPORT.md)

## 1. Giữ, thay và bỏ gì so với bản gốc?

Folder này là **implementation local cho ba selectors sau retrieval**, nhận chung Jina-v4 cosine và candidate pool Top-L tối đa 30. Embedding, chunking và retrieval riêng của từng repo được thay bằng input đã tính sẵn. Calibration dùng benchmark questions/support labels và split chung. Sau chọn context, cả ba dùng cùng Qwen2-VL-7B để đo downstream; calibration vẫn pooled theo dataset.

### 1.1. CCE Conformal-Embedding

- **Giữ:** mỗi relevant query–chunk pair là một calibration record; nonconformity `A = 1 − cosine`; lấy quantile `1 − α` và giữ candidate khi `A ≤ τ`, tương đương `cosine ≥ 1 − τ`.
- **Thay:** cosine từ Qwen3-Embedding-8B bằng Jina-v4 cosine; snippets của nguồn bằng chunks hiện có; relevance judgments bằng benchmark support labels. [cce.py](cce.py) dùng `method="higher"` cho quantile. README nguồn không quy định interpolation này; đó là lựa chọn của code hiện tại.
- **Bỏ khỏi setup chạy:** bước LLM gán relevance labels. Không đưa nhánh **Conformal-LLM** dùng LLM rating vào folder này: đó là một lựa chọn scorer khác của CCE, không phải bước tiếp theo của Conformal-Embedding.

Nguồn: [CCE README tại revision đã kiểm tra](https://github.com/hltcoe/conformal-context-engineering/blob/91732d6058267f180ba9f47873d743288b2625af/README.md). Repo nguồn cung cấp supplementary materials; selector ở đây triển khai công thức trong README, không phải calibration runner Python lấy nguyên từ repo đó.

### 1.2. CONFLARE source-question

- **Giữ:** mỗi calibration question thành công đóng góp distance của relevant chunk đầu tiên theo thứ tự distance tăng dần; cutoff là `np.percentile(distances, 100 × (1 − α))`; test giữ `distance < τ` với dấu so sánh strict.
- **Thay:** generated source questions bằng benchmark questions; kiểm tra source chunk/LLM relevance bằng labelled supports; distance bằng `1 − Jina cosine`. Trong pool cố định, relevant chunk đầu tiên là support có cosine cao nhất. Query calibration không có support trong pool không đóng góp record. [conflare.py](conflare.py) giữ percentile mặc định của NumPy và chọn `cosine > 1 − τ`.
- **Bỏ khỏi setup chạy:** LLM sinh câu hỏi từ source chunks và LLM đánh giá relevance ở calibration. Không chạy Chroma/scan toàn collection vì candidates đã là input. QA chain/prompt riêng được thay bằng shared Qwen downstream.

Nguồn: [calibration](https://github.com/Mayo-Radiology-Informatics-Lab/conflare/blob/ce081a45fb452704daa87f3b37f601b4accc7a82/conflare/conformal/calibration.py), [context filter](https://github.com/Mayo-Radiology-Informatics-Lab/conflare/blob/ce081a45fb452704daa87f3b37f601b4accc7a82/conflare/augmented_retrieval/rag.py).

### 1.3. TRAQ retrieval Bonferroni

- **Giữ:** một best true-retrieval score cho mỗi calibration question; lower quantile tại `α_R = α/2`; test giữ `score ≥ τ`. Với `α=.10`, retrieval allocation vẫn là `.05`.
- **Thay:** true-retrieval score bằng maximum Jina cosine của labelled supports trong pool; dùng calibration manifest chung. Query calibration không có support trong pool không đóng góp record. [traq.py](traq.py) dùng `method="lower"`, tương ứng `interpolation="lower"` trong code nguồn.
- **Bỏ khỏi setup chạy sau retrieval:** lấy mẫu nhiều câu trả lời/answer-confidence, semantic clustering, calibration phía answer và tạo semantic answer prediction set. Folder chỉ dùng Bonferroni retrieval cutoff để chọn chunks, rồi sinh một short answer bằng shared Qwen downstream. Phần error budget phía answer không được chuyển sang retrieval; `α_R` vẫn bằng `α/2`. Các nhánh tuning allocation/PAC/Fisher không được đưa vào folder.

Nguồn: [lower quantile](https://github.com/shuoli90/TRAQ/blob/e9b66b5b7fd8fcd0b3c3dbfebc5a3391ca79aa55/misc/utils.py), [Bonferroni allocation và answer-set pipeline](https://github.com/shuoli90/TRAQ/blob/e9b66b5b7fd8fcd0b3c3dbfebc5a3391ca79aa55/run/traq/traq_chatgpt_semantic.py).

### 1.4. Hệ quả với kết quả trong folder

- Cả ba chỉ chọn được chunks trong pool đã cung cấp; support ngoài Top-L không được bổ sung. Labels của test dùng tính metrics, không dùng quyết định giữ chunks.
- Không thêm reranker, cutoff theo modality, Top-1 fallback hoặc cap Top-10 sau selection. Text/table/image được xử lý theo modality khi đưa vào generator chung.
- EM/F1 đo một câu trả lời từ shared generator; không đo semantic answer-set coverage của TRAQ. [ADAPTATION_NOTE.md](ADAPTATION_NOTE.md) ghi thêm protocol và nguồn code.

## 2. Input cosine

[Tải ZIP khoảng 41,5 MB](https://github.com/NguyenKhanh2603/Uncertainty-Aware-Iterative-RAG/blob/results/qwen2vl-jina4-four-datasets-2026-09-14/research/internal_state_rag/results/precomputed_cosine_cal1000_official_test_2026_09_30/precomputed_cosine_cal1000_official_test_2026_09_30.zip?raw=1). Giải nén, trỏ `--bundle` tới thư mục `portable_bundle` chứa `rowwise/` và `splits/`.

| Dataset | Calibration | Official labelled test |
|---|---:|---:|
| HotpotQA | 1.000 | 7.405 |
| MMQA | 1.000 | 2.441 |
| TAT-QA | 1.000 | 1.668 |
| WebQA | 1.000 | 4.966 |

## 3. Chạy support metrics bằng CPU

Trong folder này:

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r requirements-selection.txt
.venv/bin/python run_selection.py --bundle /path/to/portable_bundle --output outputs/frozen
```

Default dùng frozen thresholds để replay đúng run đã công bố. Để calibrate lại từ scores trong ZIP:

```bash
.venv/bin/python run_selection.py --bundle /path/to/portable_bundle --output outputs/recalibrated --recalibrate
```

Output gồm `selection_summary.json` với Chunks, Precision, Support recall, Empty, Any/All support và `{dataset}_selection.jsonl.gz` với selected chunk IDs cho mỗi query/method. Recall đo support có trong Top-L; Any/All chỉ tính query có support trong Top-L. CCE, CONFLARE và TRAQ đều pooled theo dataset. Alpha mặc định `.10`; TRAQ retrieval dùng `.05`.

## 4. Chạy downstream trên GPU

Cần normalized corpus, question answers và ảnh thật. ZIP cosine chỉ chứa score/IDs; downstream đọc local data bundle có cấu trúc:

```text
DATA_BUNDLE/
  hotpotqa/{questions.jsonl,corpus.jsonl}
  mmqa/{questions.jsonl,corpus.jsonl,images/...}
  tatqa/{questions.jsonl,corpus.jsonl}
  webqa/{questions.jsonl,corpus.jsonl,images/...}
```

Image content path trong corpus phải trỏ tới asset tồn tại trên máy chạy; nếu chuyển bundle sang Colab cần đổi absolute path máy gốc thành relative path hoặc đường dẫn Colab. Runner báo lỗi khi ảnh thiếu.

Giữ Torch/CUDA đang work. Máy gốc dùng `torch 2.7.1+cu118`; requirements downstream ghi các package version thực tế của run, không tự thay Torch.

```bash
uv pip install --python .venv/bin/python -r requirements-downstream.txt
.venv/bin/python run_downstream.py \
  --selection-dir outputs/frozen \
  --data-bundle /path/to/data_bundle \
  --output outputs/downstream
```

Có thể thêm `--model /path/to/local/Qwen2-VL-7B-Instruct`. Model revision, pixel limits và token budget nằm trong config. Predictions được append và resume theo dataset/qid; thay config hoặc selection cần output folder mới. Common generator được dùng cho cả ba method, với text/table/image theo modality.

## 5. Kiểm tra thuật toán

```bash
.venv/bin/python -m unittest test_rules.py
```

`VALIDATION.md` ghi kết quả replay trên tất cả 16.480 test queries và kiểm tra parity với code nguồn local đã chạy.
