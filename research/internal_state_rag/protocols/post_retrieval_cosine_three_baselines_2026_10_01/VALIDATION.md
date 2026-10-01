# Validation — 2026-10-01

Selection được replay bằng code trong folder này trên cùng portable cosine bundle. Threshold mode: `frozen`, `alpha=.10`.

| Dataset | Calibration | Test queries | Method/query contexts checked |
|---|---:|---:|---:|
| HotpotQA | 1.000 | 7.405 | 22.215 |
| MMQA | 1.000 | 2.441 | 7.323 |
| TAT-QA | 1.000 | 1.668 | 5.004 |
| WebQA | 1.000 | 4.966 | 14.898 |
| Total | 4.000 | 16.480 | 49.440 |

- Tất cả support metrics khớp chính xác `three_baselines/summary.json` của run completed.
- Selected chunk IDs và thứ tự context của mỗi method/query khớp predictions downstream completed.
- Calibration functions khi fit lại trên bundled scores trả cùng threshold/metadata như functions ở runner cũ `run_literature_protocol_1000cal.py`.
- Boundary regression checks: CCE dùng pooled support pairs + `higher`; CONFLARE dùng per-query maxima + percentile và strict cutoff; TRAQ dùng per-query maxima + lower quantile tại `.05`. Cả ba tests pass.
- Generator class và 5 answer-metric functions khớp AST với code đã chạy: prompt content order, ảnh, greedy generation và metric semantics được giữ nguyên. Lần kiểm tra này không chạy lại GPU generation.

[Machine-readable audit](VALIDATION.json), [replay checker](verify_replay.py), [unit tests](test_rules.py).

Chạy lại verification từ repository root:

```bash
.venv-cu118/bin/python research/internal_state_rag/protocols/post_retrieval_cosine_three_baselines_2026_10_01/run_selection.py \
  --bundle research/internal_state_rag/results/precomputed_cosine_cal1000_official_test_2026_09_30/portable_bundle \
  --output /tmp/post_retrieval_three_baselines_frozen_20261001

.venv-cu118/bin/python research/internal_state_rag/protocols/post_retrieval_cosine_three_baselines_2026_10_01/verify_replay.py \
  --repo-root . --selection-output /tmp/post_retrieval_three_baselines_frozen_20261001 \
  --output /tmp/post_retrieval_three_baselines_validation.json
```

Parity checker đọc các predictions completed local của run official-full. Chạy support selection chỉ cần portable cosine bundle.
