# Downstream QA for the 100-query baseline-comparison report

**Source selection report:** [baseline_comparison_report_100_queries_detailed.md](https://github.com/NguyenKhanh2603/Uncertainty-Aware-Iterative-RAG/blob/docs/add-detailed-report/baseline_comparison_report_100_queries_detailed.md).<br>
**Authoritative split:** [`report_100_calibration_100_test_question_ids_2026-09-19.csv`](../../protocols/baseline_report_100/report_100_calibration_100_test_question_ids_2026-09-19.csv).<br>
**Runner:** [`run_baseline_report_100_downstream.py`](../../run_baseline_report_100_downstream.py).<br>
**Splits:** [`splits/`](splits/) contains the exact 100 calibration qids and 100 held-out test qids used here for each dataset. Both plans come directly from `report_100_calibration_100_test_question_ids_2026-09-19.csv` in the supplied protocol archive.<br>
**Leakage audit:** [`CSV_SPLIT_INTEGRITY.json`](CSV_SPLIT_INTEGRITY.json) verifies 100 unique calibration qids, 100 unique test qids, and zero within-dataset overlap for every dataset.

This rerun reads the source report's explicit CSV protocol, verifies that calibration and held-out test qids are disjoint, then joins those qids to the frozen Jina-v4 Top-30 retrieval logs. Every row uses greedy Qwen2-VL-7B-Instruct output with at most 24 new tokens. CCE, CONFLARE, and TRAQ are retrieval adapters; BH is the source report's modality-conditioned selection component: false-score calibration banks are stratified by `{dataset, modality}`, then BH is applied across the candidate set of each query.

## hotpotqa (complete; n=100)

| Method | Chunks | Precision | Recall | Empty | Any support | All support | EM | F1 | Numeric |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_top10 | 10.00 | 17.7% | 96.2% | 0.0% | 99.0% | 93.0% | 0.410 | 0.559 | 0.430 |
| fixed_top20 | 20.00 | 9.2% | 99.5% | 0.0% | 99.0% | 99.0% | 0.340 | 0.513 | 0.360 |
| cce_cosine_alpha_0.10 | 11.19 | 14.9% | 90.8% | 1.0% | 99.0% | 84.0% | 0.430 | 0.580 | 0.450 |
| conflare_cosine_alpha_0.10 | 10.80 | 15.5% | 90.8% | 1.0% | 99.0% | 84.0% | 0.410 | 0.558 | 0.430 |
| traq_cosine_alpha_0.10 | 14.97 | 11.4% | 92.9% | 0.0% | 100.0% | 87.0% | 0.390 | 0.542 | 0.420 |
| bh_modality_cosine_alpha_0.10_ctx10 | 0.74 | 63.5% | 25.5% | 66.0% | 34.0% | 18.0% | 0.220 | 0.351 | 0.240 |
| bh_modality_cosine_alpha_0.90_ctx10 | 8.19 | 18.6% | 82.6% | 11.0% | 88.0% | 77.0% | 0.410 | 0.550 | 0.420 |
| bh_modality_cosine_alpha_0.99_ctx10 | 9.90 | 17.8% | 95.7% | 1.0% | 98.0% | 92.0% | 0.410 | 0.559 | 0.430 |
| bh_modality_cosine_alpha_0.99_ctx20 | 19.80 | 9.2% | 98.9% | 1.0% | 98.0% | 98.0% | 0.340 | 0.513 | 0.360 |

The recomputed selector masks match every rounded selection row published in the source report.

### Calibration and selection metadata

```json
{
  "support_score_bank_size": 185,
  "thresholds": {
    "cce_cosine_alpha_0.10": 0.5741243362426758,
    "conflare_cosine_alpha_0.10": 0.5756383657455444,
    "traq_cosine_alpha_0.10": 0.5566290616989136
  },
  "false_score_bank_sizes_by_modality": {
    "text": 2815
  },
  "bh": {
    "conditioning": [
      "dataset",
      "modality"
    ],
    "procedure": "BH p-value step-up, then smallest-p-value context cap; rank breaks ties",
    "note": "The post-BH cap is an experimental context policy; no capped-procedure FDR guarantee is claimed."
  },
  "published_selection_audit": {
    "matches_all_evaluated_rows": true,
    "evaluated_methods": [
      "fixed_top10",
      "fixed_top20",
      "cce_cosine_alpha_0.10",
      "conflare_cosine_alpha_0.10",
      "traq_cosine_alpha_0.10",
      "bh_modality_cosine_alpha_0.10_ctx10",
      "bh_modality_cosine_alpha_0.90_ctx10",
      "bh_modality_cosine_alpha_0.99_ctx10",
      "bh_modality_cosine_alpha_0.99_ctx20"
    ],
    "mismatched_methods": [],
    "rows": {
      "fixed_top10": {
        "published": {
          "mean_chunks": 10.0,
          "precision": 0.177,
          "support_recall": 0.962,
          "empty_rate": 0.0
        },
        "recomputed": {
          "mean_chunks": 10.0,
          "precision": 0.177,
          "support_recall": 0.9620000000000001,
          "empty_rate": 0.0
        },
        "matches_published_rounding": true
      },
      "fixed_top20": {
        "published": {
          "mean_chunks": 20.0,
          "precision": 0.092,
          "support_recall": 0.995,
          "empty_rate": 0.0
        },
        "recomputed": {
          "mean_chunks": 20.0,
          "precision": 0.092,
          "support_recall": 0.995,
          "empty_rate": 0.0
        },
        "matches_published_rounding": true
      },
      "cce_cosine_alpha_0.10": {
        "published": {
          "mean_chunks": 11.19,
          "precision": 0.149,
          "support_recall": 0.908,
          "empty_rate": 0.01
        },
        "recomputed": {
          "mean_chunks": 11.19,
          "precision": 0.149,
          "support_recall": 0.9079999999999999,
          "empty_rate": 0.01
        },
        "matches_published_rounding": true
      },
      "conflare_cosine_alpha_0.10": {
        "published": {
          "mean_chunks": 10.8,
          "precision": 0.155,
          "support_recall": 0.908,
          "empty_rate": 0.01
        },
        "recomputed": {
          "mean_chunks": 10.8,
          "precision": 0.155,
          "support_recall": 0.9079999999999999,
          "empty_rate": 0.01
        },
        "matches_published_rounding": true
      },
      "traq_cosine_alpha_0.10": {
        "published": {
          "mean_chunks": 14.97,
          "precision": 0.114,
          "support_recall": 0.929,
          "empty_rate": 0.0
        },
        "recomputed": {
          "mean_chunks": 14.97,
          "precision": 0.114,
          "support_recall": 0.929,
          "empty_rate": 0.0
        },
        "matches_published_rounding": true
      },
      "bh_modality_cosine_alpha_0.10_ctx10": {
        "published": {
          "mean_chunks": 0.74,
          "precision": 0.635,
          "support_recall": 0.255,
          "empty_rate": 0.66
        },
        "recomputed": {
          "mean_chunks": 0.74,
          "precision": 0.635,
          "support_recall": 0.255,
          "empty_rate": 0.66
        },
        "matches_published_rounding": true
      },
      "bh_modality_cosine_alpha_0.90_ctx10": {
        "published": {
          "mean_chunks": 8.19,
          "precision": 0.186,
          "support_recall": 0.826,
          "empty_rate": 0.11
        },
        "recomputed": {
          "mean_chunks": 8.19,
          "precision": 0.18600000000000003,
          "support_recall": 0.826,
          "empty_rate": 0.11
        },
        "matches_published_rounding": true
      },
      "bh_modality_cosine_alpha_0.99_ctx10": {
        "published": {
          "mean_chunks": 9.9,
          "precision": 0.178,
          "support_recall": 0.957,
          "empty_rate": 0.01
        },
        "recomputed": {
          "mean_chunks": 9.9,
          "precision": 0.17800000000000002,
          "support_recall": 0.9570000000000001,
          "empty_rate": 0.01
        },
        "matches_published_rounding": true
      },
      "bh_modality_cosine_alpha_0.99_ctx20": {
        "published": {
          "mean_chunks": 19.8,
          "precision": 0.092,
          "support_recall": 0.989,
          "empty_rate": 0.01
        },
        "recomputed": {
          "mean_chunks": 19.8,
          "precision": 0.092,
          "support_recall": 0.9890000000000001,
          "empty_rate": 0.01
        },
        "matches_published_rounding": true
      }
    }
  }
}
```

## mmqa (complete; n=100)

| Method | Chunks | Precision | Recall | Empty | Any support | All support | EM | F1 | Numeric |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_top10 | 10.00 | 10.9% | 96.5% | 0.0% | 100.0% | 96.0% | 0.470 | 0.512 | 0.490 |
| fixed_top20 | 20.00 | 5.5% | 97.3% | 0.0% | 100.0% | 97.0% | 0.460 | 0.497 | 0.480 |
| cce_cosine_alpha_0.10 | 14.19 | 7.4% | 92.9% | 3.0% | 97.0% | 92.0% | 0.510 | 0.554 | 0.530 |
| conflare_cosine_alpha_0.10 | 13.02 | 8.0% | 92.0% | 3.0% | 97.0% | 91.0% | 0.490 | 0.533 | 0.510 |
| traq_cosine_alpha_0.10 | 20.43 | 5.4% | 97.3% | 1.0% | 100.0% | 97.0% | 0.490 | 0.537 | 0.510 |
| bh_modality_cosine_alpha_0.10_ctx10 | 0.21 | 28.6% | 5.3% | 93.0% | 13.0% | 11.0% | 0.160 | 0.193 | 0.170 |
| bh_modality_cosine_alpha_0.90_ctx10 | 7.79 | 11.8% | 81.4% | 15.0% | 87.0% | 83.0% | 0.480 | 0.524 | 0.500 |
| bh_modality_cosine_alpha_0.99_ctx10 | 9.41 | 10.9% | 91.2% | 5.0% | 95.0% | 90.0% | 0.490 | 0.526 | 0.510 |
| bh_modality_cosine_alpha_0.99_ctx20 | 18.81 | 5.5% | 92.0% | 5.0% | 95.0% | 91.0% | 0.450 | 0.487 | 0.470 |

The recomputed selector masks match every rounded selection row published in the source report.

### Calibration and selection metadata

```json
{
  "support_score_bank_size": 124,
  "thresholds": {
    "cce_cosine_alpha_0.10": 0.5828832387924194,
    "conflare_cosine_alpha_0.10": 0.5880766034126282,
    "traq_cosine_alpha_0.10": 0.5536381602287292
  },
  "false_score_bank_sizes_by_modality": {
    "image": 492,
    "table": 327,
    "text": 2057
  },
  "bh": {
    "conditioning": [
      "dataset",
      "modality"
    ],
    "procedure": "BH p-value step-up, then smallest-p-value context cap; rank breaks ties",
    "note": "The post-BH cap is an experimental context policy; no capped-procedure FDR guarantee is claimed."
  },
  "published_selection_audit": {
    "matches_all_evaluated_rows": true,
    "evaluated_methods": [
      "fixed_top10",
      "fixed_top20",
      "cce_cosine_alpha_0.10",
      "conflare_cosine_alpha_0.10",
      "traq_cosine_alpha_0.10",
      "bh_modality_cosine_alpha_0.10_ctx10",
      "bh_modality_cosine_alpha_0.90_ctx10",
      "bh_modality_cosine_alpha_0.99_ctx10",
      "bh_modality_cosine_alpha_0.99_ctx20"
    ],
    "mismatched_methods": [],
    "rows": {
      "fixed_top10": {
        "published": {
          "mean_chunks": 10.0,
          "precision": 0.109,
          "support_recall": 0.965,
          "empty_rate": 0.0
        },
        "recomputed": {
          "mean_chunks": 10.0,
          "precision": 0.109,
          "support_recall": 0.965,
          "empty_rate": 0.0
        },
        "matches_published_rounding": true
      },
      "fixed_top20": {
        "published": {
          "mean_chunks": 20.0,
          "precision": 0.055,
          "support_recall": 0.973,
          "empty_rate": 0.0
        },
        "recomputed": {
          "mean_chunks": 20.0,
          "precision": 0.055,
          "support_recall": 0.973,
          "empty_rate": 0.0
        },
        "matches_published_rounding": true
      },
      "cce_cosine_alpha_0.10": {
        "published": {
          "mean_chunks": 14.19,
          "precision": 0.074,
          "support_recall": 0.929,
          "empty_rate": 0.03
        },
        "recomputed": {
          "mean_chunks": 14.19,
          "precision": 0.07400000000000001,
          "support_recall": 0.929,
          "empty_rate": 0.03
        },
        "matches_published_rounding": true
      },
      "conflare_cosine_alpha_0.10": {
        "published": {
          "mean_chunks": 13.02,
          "precision": 0.08,
          "support_recall": 0.92,
          "empty_rate": 0.03
        },
        "recomputed": {
          "mean_chunks": 13.02,
          "precision": 0.08,
          "support_recall": 0.92,
          "empty_rate": 0.03
        },
        "matches_published_rounding": true
      },
      "traq_cosine_alpha_0.10": {
        "published": {
          "mean_chunks": 20.43,
          "precision": 0.054,
          "support_recall": 0.973,
          "empty_rate": 0.01
        },
        "recomputed": {
          "mean_chunks": 20.43,
          "precision": 0.054000000000000006,
          "support_recall": 0.973,
          "empty_rate": 0.01
        },
        "matches_published_rounding": true
      },
      "bh_modality_cosine_alpha_0.10_ctx10": {
        "published": {
          "mean_chunks": 0.21,
          "precision": 0.286,
          "support_recall": 0.053,
          "empty_rate": 0.93
        },
        "recomputed": {
          "mean_chunks": 0.21,
          "precision": 0.28600000000000003,
          "support_recall": 0.053,
          "empty_rate": 0.93
        },
        "matches_published_rounding": true
      },
      "bh_modality_cosine_alpha_0.90_ctx10": {
        "published": {
          "mean_chunks": 7.79,
          "precision": 0.118,
          "support_recall": 0.814,
          "empty_rate": 0.15
        },
        "recomputed": {
          "mean_chunks": 7.79,
          "precision": 0.11800000000000001,
          "support_recall": 0.8140000000000001,
          "empty_rate": 0.15
        },
        "matches_published_rounding": true
      },
      "bh_modality_cosine_alpha_0.99_ctx10": {
        "published": {
          "mean_chunks": 9.41,
          "precision": 0.109,
          "support_recall": 0.912,
          "empty_rate": 0.05
        },
        "recomputed": {
          "mean_chunks": 9.41,
          "precision": 0.109,
          "support_recall": 0.912,
          "empty_rate": 0.05
        },
        "matches_published_rounding": true
      },
      "bh_modality_cosine_alpha_0.99_ctx20": {
        "published": {
          "mean_chunks": 18.81,
          "precision": 0.055,
          "support_recall": 0.92,
          "empty_rate": 0.05
        },
        "recomputed": {
          "mean_chunks": 18.81,
          "precision": 0.055,
          "support_recall": 0.92,
          "empty_rate": 0.05
        },
        "matches_published_rounding": true
      }
    }
  }
}
```

## tatqa (complete; n=100)

| Method | Chunks | Precision | Recall | Empty | Any support | All support | EM | F1 | Numeric |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_top10 | 10.00 | 8.9% | 84.8% | 0.0% | 89.0% | 84.0% | 0.190 | 0.292 | 0.270 |
| fixed_top20 | 20.00 | 5.1% | 96.2% | 0.0% | 97.0% | 96.0% | 0.150 | 0.253 | 0.230 |
| cce_cosine_alpha_0.10 | 20.10 | 4.7% | 89.5% | 5.0% | 92.0% | 90.0% | 0.150 | 0.256 | 0.230 |
| conflare_cosine_alpha_0.10 | 20.02 | 4.7% | 89.5% | 5.0% | 92.0% | 90.0% | 0.150 | 0.263 | 0.240 |
| traq_cosine_alpha_0.10 | 24.08 | 4.2% | 95.2% | 3.0% | 95.0% | 95.0% | 0.140 | 0.248 | 0.220 |
| bh_modality_cosine_alpha_0.10_ctx10 | 1.15 | 12.2% | 13.3% | 77.0% | 23.0% | 19.0% | 0.040 | 0.107 | 0.100 |
| bh_modality_cosine_alpha_0.90_ctx10 | 8.39 | 8.8% | 70.5% | 11.0% | 77.0% | 71.0% | 0.160 | 0.251 | 0.230 |
| bh_modality_cosine_alpha_0.99_ctx10 | 9.70 | 8.9% | 81.9% | 3.0% | 85.0% | 81.0% | 0.190 | 0.287 | 0.260 |
| bh_modality_cosine_alpha_0.99_ctx20 | 19.40 | 5.1% | 94.3% | 3.0% | 94.0% | 94.0% | 0.130 | 0.249 | 0.230 |

The recomputed selector masks match every rounded selection row published in the source report.

### Calibration and selection metadata

```json
{
  "support_score_bank_size": 103,
  "thresholds": {
    "cce_cosine_alpha_0.10": 0.5897385478019714,
    "conflare_cosine_alpha_0.10": 0.59063560962677,
    "traq_cosine_alpha_0.10": 0.5600771307945251
  },
  "false_score_bank_sizes_by_modality": {
    "table": 1283,
    "text": 1614
  },
  "bh": {
    "conditioning": [
      "dataset",
      "modality"
    ],
    "procedure": "BH p-value step-up, then smallest-p-value context cap; rank breaks ties",
    "note": "The post-BH cap is an experimental context policy; no capped-procedure FDR guarantee is claimed."
  },
  "published_selection_audit": {
    "matches_all_evaluated_rows": true,
    "evaluated_methods": [
      "fixed_top10",
      "fixed_top20",
      "cce_cosine_alpha_0.10",
      "conflare_cosine_alpha_0.10",
      "traq_cosine_alpha_0.10",
      "bh_modality_cosine_alpha_0.10_ctx10",
      "bh_modality_cosine_alpha_0.90_ctx10",
      "bh_modality_cosine_alpha_0.99_ctx10",
      "bh_modality_cosine_alpha_0.99_ctx20"
    ],
    "mismatched_methods": [],
    "rows": {
      "fixed_top10": {
        "published": {
          "mean_chunks": 10.0,
          "precision": 0.089,
          "support_recall": 0.848,
          "empty_rate": 0.0
        },
        "recomputed": {
          "mean_chunks": 10.0,
          "precision": 0.08900000000000001,
          "support_recall": 0.848,
          "empty_rate": 0.0
        },
        "matches_published_rounding": true
      },
      "fixed_top20": {
        "published": {
          "mean_chunks": 20.0,
          "precision": 0.051,
          "support_recall": 0.962,
          "empty_rate": 0.0
        },
        "recomputed": {
          "mean_chunks": 20.0,
          "precision": 0.051,
          "support_recall": 0.9620000000000001,
          "empty_rate": 0.0
        },
        "matches_published_rounding": true
      },
      "cce_cosine_alpha_0.10": {
        "published": {
          "mean_chunks": 20.1,
          "precision": 0.047,
          "support_recall": 0.895,
          "empty_rate": 0.05
        },
        "recomputed": {
          "mean_chunks": 20.1,
          "precision": 0.047,
          "support_recall": 0.895,
          "empty_rate": 0.05
        },
        "matches_published_rounding": true
      },
      "conflare_cosine_alpha_0.10": {
        "published": {
          "mean_chunks": 20.02,
          "precision": 0.047,
          "support_recall": 0.895,
          "empty_rate": 0.05
        },
        "recomputed": {
          "mean_chunks": 20.02,
          "precision": 0.047,
          "support_recall": 0.895,
          "empty_rate": 0.05
        },
        "matches_published_rounding": true
      },
      "traq_cosine_alpha_0.10": {
        "published": {
          "mean_chunks": 24.08,
          "precision": 0.042,
          "support_recall": 0.952,
          "empty_rate": 0.03
        },
        "recomputed": {
          "mean_chunks": 24.08,
          "precision": 0.042,
          "support_recall": 0.9520000000000001,
          "empty_rate": 0.03
        },
        "matches_published_rounding": true
      },
      "bh_modality_cosine_alpha_0.10_ctx10": {
        "published": {
          "mean_chunks": 1.15,
          "precision": 0.122,
          "support_recall": 0.133,
          "empty_rate": 0.77
        },
        "recomputed": {
          "mean_chunks": 1.15,
          "precision": 0.122,
          "support_recall": 0.133,
          "empty_rate": 0.77
        },
        "matches_published_rounding": true
      },
      "bh_modality_cosine_alpha_0.90_ctx10": {
        "published": {
          "mean_chunks": 8.39,
          "precision": 0.088,
          "support_recall": 0.705,
          "empty_rate": 0.11
        },
        "recomputed": {
          "mean_chunks": 8.39,
          "precision": 0.08800000000000001,
          "support_recall": 0.705,
          "empty_rate": 0.11
        },
        "matches_published_rounding": true
      },
      "bh_modality_cosine_alpha_0.99_ctx10": {
        "published": {
          "mean_chunks": 9.7,
          "precision": 0.089,
          "support_recall": 0.819,
          "empty_rate": 0.03
        },
        "recomputed": {
          "mean_chunks": 9.7,
          "precision": 0.08900000000000001,
          "support_recall": 0.8190000000000001,
          "empty_rate": 0.03
        },
        "matches_published_rounding": true
      },
      "bh_modality_cosine_alpha_0.99_ctx20": {
        "published": {
          "mean_chunks": 19.4,
          "precision": 0.051,
          "support_recall": 0.943,
          "empty_rate": 0.03
        },
        "recomputed": {
          "mean_chunks": 19.4,
          "precision": 0.051,
          "support_recall": 0.943,
          "empty_rate": 0.03
        },
        "matches_published_rounding": true
      }
    }
  }
}
```

## webqa (complete; n=100)

| Method | Chunks | Precision | Recall | Empty | Any support | All support | EM | F1 | Numeric |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_top10 | 10.00 | 5.8% | 68.2% | 0.0% | 81.0% | 74.0% | 0.000 | 0.125 | 0.050 |
| fixed_top20 | 20.00 | 4.0% | 92.9% | 0.0% | 97.0% | 94.0% | 0.000 | 0.128 | 0.050 |
| cce_cosine_alpha_0.10 | 21.98 | 3.5% | 91.8% | 8.0% | 95.0% | 93.0% | 0.000 | 0.174 | 0.050 |
| conflare_cosine_alpha_0.10 | 21.71 | 3.5% | 90.6% | 8.0% | 94.0% | 92.0% | 0.000 | 0.174 | 0.050 |
| traq_cosine_alpha_0.10 | 26.16 | 3.2% | 97.6% | 2.0% | 98.0% | 98.0% | 0.000 | 0.140 | 0.050 |
| bh_modality_cosine_alpha_0.10_ctx10 | 0.33 | 3.0% | 1.2% | 95.0% | 31.0% | 31.0% | 0.020 | 0.441 | 0.110 |
| bh_modality_cosine_alpha_0.90_ctx10 | 7.69 | 6.4% | 57.6% | 23.0% | 74.0% | 69.0% | 0.000 | 0.208 | 0.060 |
| bh_modality_cosine_alpha_0.99_ctx10 | 9.10 | 6.6% | 70.6% | 9.0% | 82.0% | 76.0% | 0.000 | 0.163 | 0.070 |
| bh_modality_cosine_alpha_0.99_ctx20 | 18.20 | 4.1% | 87.1% | 9.0% | 92.0% | 89.0% | 0.000 | 0.164 | 0.070 |

The recomputed selector masks match every rounded selection row published in the source report.

### Calibration and selection metadata

```json
{
  "support_score_bank_size": 88,
  "thresholds": {
    "cce_cosine_alpha_0.10": 0.5742910504341125,
    "conflare_cosine_alpha_0.10": 0.5756952047348023,
    "traq_cosine_alpha_0.10": 0.5444818735122681
  },
  "false_score_bank_sizes_by_modality": {
    "image": 2012,
    "text": 900
  },
  "bh": {
    "conditioning": [
      "dataset",
      "modality"
    ],
    "procedure": "BH p-value step-up, then smallest-p-value context cap; rank breaks ties",
    "note": "The post-BH cap is an experimental context policy; no capped-procedure FDR guarantee is claimed."
  },
  "published_selection_audit": {
    "matches_all_evaluated_rows": true,
    "evaluated_methods": [
      "fixed_top10",
      "fixed_top20",
      "cce_cosine_alpha_0.10",
      "conflare_cosine_alpha_0.10",
      "traq_cosine_alpha_0.10",
      "bh_modality_cosine_alpha_0.10_ctx10",
      "bh_modality_cosine_alpha_0.90_ctx10",
      "bh_modality_cosine_alpha_0.99_ctx10",
      "bh_modality_cosine_alpha_0.99_ctx20"
    ],
    "mismatched_methods": [],
    "rows": {
      "fixed_top10": {
        "published": {
          "mean_chunks": 10.0,
          "precision": 0.058,
          "support_recall": 0.682,
          "empty_rate": 0.0
        },
        "recomputed": {
          "mean_chunks": 10.0,
          "precision": 0.057999999999999996,
          "support_recall": 0.682,
          "empty_rate": 0.0
        },
        "matches_published_rounding": true
      },
      "fixed_top20": {
        "published": {
          "mean_chunks": 20.0,
          "precision": 0.04,
          "support_recall": 0.929,
          "empty_rate": 0.0
        },
        "recomputed": {
          "mean_chunks": 20.0,
          "precision": 0.04,
          "support_recall": 0.929,
          "empty_rate": 0.0
        },
        "matches_published_rounding": true
      },
      "cce_cosine_alpha_0.10": {
        "published": {
          "mean_chunks": 21.98,
          "precision": 0.035,
          "support_recall": 0.918,
          "empty_rate": 0.08
        },
        "recomputed": {
          "mean_chunks": 21.98,
          "precision": 0.035,
          "support_recall": 0.9179999999999999,
          "empty_rate": 0.08
        },
        "matches_published_rounding": true
      },
      "conflare_cosine_alpha_0.10": {
        "published": {
          "mean_chunks": 21.71,
          "precision": 0.035,
          "support_recall": 0.906,
          "empty_rate": 0.08
        },
        "recomputed": {
          "mean_chunks": 21.71,
          "precision": 0.035,
          "support_recall": 0.9059999999999999,
          "empty_rate": 0.08
        },
        "matches_published_rounding": true
      },
      "traq_cosine_alpha_0.10": {
        "published": {
          "mean_chunks": 26.16,
          "precision": 0.032,
          "support_recall": 0.976,
          "empty_rate": 0.02
        },
        "recomputed": {
          "mean_chunks": 26.16,
          "precision": 0.032,
          "support_recall": 0.976,
          "empty_rate": 0.02
        },
        "matches_published_rounding": true
      },
      "bh_modality_cosine_alpha_0.10_ctx10": {
        "published": {
          "mean_chunks": 0.33,
          "precision": 0.03,
          "support_recall": 0.012,
          "empty_rate": 0.95
        },
        "recomputed": {
          "mean_chunks": 0.33,
          "precision": 0.03,
          "support_recall": 0.012,
          "empty_rate": 0.95
        },
        "matches_published_rounding": true
      },
      "bh_modality_cosine_alpha_0.90_ctx10": {
        "published": {
          "mean_chunks": 7.69,
          "precision": 0.064,
          "support_recall": 0.576,
          "empty_rate": 0.23
        },
        "recomputed": {
          "mean_chunks": 7.69,
          "precision": 0.064,
          "support_recall": 0.5760000000000001,
          "empty_rate": 0.23
        },
        "matches_published_rounding": true
      },
      "bh_modality_cosine_alpha_0.99_ctx10": {
        "published": {
          "mean_chunks": 9.1,
          "precision": 0.066,
          "support_recall": 0.706,
          "empty_rate": 0.09
        },
        "recomputed": {
          "mean_chunks": 9.1,
          "precision": 0.066,
          "support_recall": 0.706,
          "empty_rate": 0.09
        },
        "matches_published_rounding": true
      },
      "bh_modality_cosine_alpha_0.99_ctx20": {
        "published": {
          "mean_chunks": 18.2,
          "precision": 0.041,
          "support_recall": 0.871,
          "empty_rate": 0.09
        },
        "recomputed": {
          "mean_chunks": 18.2,
          "precision": 0.040999999999999995,
          "support_recall": 0.871,
          "empty_rate": 0.09
        },
        "matches_published_rounding": true
      }
    }
  }
}
```
