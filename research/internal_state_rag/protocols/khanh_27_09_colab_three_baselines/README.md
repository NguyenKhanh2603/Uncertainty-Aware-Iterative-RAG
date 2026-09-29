# Colab standalone: 3 literature baselines on `splits_khanh_27_09`

Folder này chạy lại downstream QA và đo latency/FLOPs cho đúng ba context-selection baselines:

1. CCE Conformal-Embedding, Jina adaptation (`alpha=.10`)
2. CONFLARE source-question, Jina adaptation (`alpha=.10`)
3. TRAQ retrieval Bonferroni, Jina adaptation (`alpha=.10`, `alpha_R=.05`)

Input đã được đóng băng: 4 dataset × 100 held-out test queries, exact selected chunk IDs của từng baseline, và split 1,000 calibration / 100 test không overlap. Runner không retrieve, calibrate, rerank hoặc select lại chunk.

## Chạy trên Google Colab

Chọn **A100 GPU** để giữ cùng model dtype/protocol. Mở [`Khanh_27_09_3_baselines_A100.ipynb`](Khanh_27_09_3_baselines_A100.ipynb) trên Colab và chạy lần lượt tất cả cell. Notebook sẽ:

- clone đúng branch kết quả;
- cài phiên bản Transformers/Qwen processor đã khóa;
- tải test-only data bundle từ Hugging Face;
- chạy 1,200 generations với Qwen2-VL-7B;
- resume từ `progress_log.jsonl` nếu runtime bị ngắt;
- xuất `final_downstream_results.csv`, `RESOURCE_RESULTS.md`, `summary.json`, `RUN_CONFIG.json`, và per-query log.

Có thể chạy tương đương bằng shell:

```bash
pip install -r requirements.txt
hf download danny2507/khanh-27-09-three-baselines-colab \
  khanh_27_09_test_bundle_v1.tar.gz --repo-type dataset --local-dir .
tar -xzf khanh_27_09_test_bundle_v1.tar.gz
python run_colab.py --data-root data_bundle --output-dir colab_results
```

## Protocol đo

- Model: `Qwen/Qwen2-VL-7B-Instruct`
- Revision: `eed13092ef92e448dd6875b2a00151bd3f7db0ac`
- dtype: `bfloat16`; device placement: `auto`; attention: SDPA
- Prompt: `Answer using only the supplied context. Return only the short answer.\nQuestion: {question}`
- Context text/table: `Context: {content}`; ảnh dùng `min_pixels=3136`, `max_pixels=200704`
- Decoding: greedy, `max_new_tokens=24`
- Latency: wall time chỉ bao quanh `model.generate`, có `torch.cuda.synchronize()` trước và sau
- Approx. FLOPs: `2 × parameter_count × generated_ids.shape[1]`, đúng công thức được yêu cầu

`Approx. TFLOPs` là proxy để giữ cùng cách tính giữa các method. Nó không phải profiler FLOPs và không mô hình hóa đầy đủ attention/vision compute. So sánh latency có ý nghĩa khi cả ba method chạy trong cùng một Colab runtime, GPU type và software versions.

## Resume và lưu Drive

Đặt `--output-dir` trong Google Drive để không mất progress khi Colab disconnect:

```bash
python run_colab.py \
  --data-root /content/data_bundle \
  --output-dir /content/drive/MyDrive/khanh_27_09_colab_results
```

Mỗi query hoàn thành được append và flush ngay vào `progress_log.jsonl`. Chạy lại đúng command sẽ skip các khóa `(method, dataset, qid)` đã có. Không đổi model, method hoặc dataset list trong cùng output directory; runner kiểm tra `RUN_CONFIG.json` và sẽ dừng nếu cấu hình khác.

## Nội dung folder

- `run_colab.py`: runner độc lập, gồm cả EM/F1 implementation và report writer.
- `selections/`: exact saved context IDs của ba baseline cho 400 test queries.
- `splits/`: qid manifests và split-integrity record.
- `REPORT_TEMPLATE.md`: báo cáo selection/downstream hiện tại; hai cột resource sẽ được cập nhật sau khi Colab trả kết quả.
- `build_portable_bundle.py`: script provenance dùng để tạo test-only bundle từ dữ liệu đầy đủ.
- `DATA_BUNDLE.json`: URL, SHA-256, kích thước và content counts của bundle đã publish.

Không dùng các số local chạy dở trước khi đóng gói Colab. Bảng cuối phải lấy từ một Colab run hoàn chỉnh có `summary.json` với `status: complete` và đủ 1,200 dòng progress.
