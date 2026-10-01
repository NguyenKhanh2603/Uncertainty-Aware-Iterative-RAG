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

## 1. Input cosine

[Tải ZIP khoảng 41,5 MB](https://github.com/NguyenKhanh2603/Uncertainty-Aware-Iterative-RAG/blob/results/qwen2vl-jina4-four-datasets-2026-09-14/research/internal_state_rag/results/precomputed_cosine_cal1000_official_test_2026_09_30/precomputed_cosine_cal1000_official_test_2026_09_30.zip?raw=1). Giải nén, trỏ `--bundle` tới thư mục `portable_bundle` chứa `rowwise/` và `splits/`.

| Dataset | Calibration | Official labelled test |
|---|---:|---:|
| HotpotQA | 1.000 | 7.405 |
| MMQA | 1.000 | 2.441 |
| TAT-QA | 1.000 | 1.668 |
| WebQA | 1.000 | 4.966 |

## 2. Chạy support metrics bằng CPU

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

## 3. Chạy downstream trên GPU

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

## 4. Kiểm tra thuật toán

```bash
.venv/bin/python -m unittest test_rules.py
```

`VALIDATION.md` ghi kết quả replay trên tất cả 16.480 test queries và kiểm tra parity với code nguồn local đã chạy.
