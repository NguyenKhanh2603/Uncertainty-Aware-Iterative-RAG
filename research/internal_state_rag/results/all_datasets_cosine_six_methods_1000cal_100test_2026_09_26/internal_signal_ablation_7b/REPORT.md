# Qwen-7B internal-signal ablation: 1,000 calibration / 100 held-out test

Each dataset partitions the same 1,000 parent calibration qids into a disjoint probe-training role and conformal-calibration role. All rows use the same frozen Top-30 cosine candidates and the same 100 held-out test qids. LM-head is the direct Yes-vs-No logit signal; the hidden probe is an L2 logistic relevance probe selected with grouped OOF AP on probe-training qids. Every row uses an all-support query-level split-conformal threshold at α=0.10 and a deterministic Top-1 fallback.

## hotpotqa (complete; probe=500, calibration=500, test=100)

| Method | Chunks | Precision | Support recall | Empty | Any support | All support | EM | F1 | Numeric |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Query-level cosine (matched 500-ish calibration) | 9.89 | 16.2% | 93.0% | 0.0% | 99.0% | 88.0% | 0.400 | 0.508 | 0.420 |
| LM-head only | 8.20 | 19.9% | 94.8% | 0.0% | 96.0% | 91.0% | 0.430 | 0.551 | 0.440 |
| Hidden-state probe only | 2.67 | 60.7% | 94.2% | 0.0% | 96.0% | 91.0% | 0.440 | 0.576 | 0.460 |
| LM-head + hidden probe | 2.67 | 60.7% | 94.2% | 0.0% | 96.0% | 91.0% | 0.440 | 0.576 | 0.460 |
| Cosine + LM-head + hidden probe | 2.50 | 65.2% | 94.8% | 0.0% | 98.0% | 92.0% | 0.440 | 0.576 | 0.460 |

Selected hidden-probe layer/C: `19` / `0.03`.

```json
{
  "internal_lm_hidden": {
    "lm_head": 0.0,
    "hidden_probe": 0.01,
    "mean_query_ap": 0.9176958254988558,
    "mrr": 0.9544356261022927,
    "top1_support_rate": 0.9212121212121213
  },
  "cosine_internal": {
    "lm_head": 1.0,
    "hidden_probe": 2.0,
    "mean_query_ap": 0.9354257947439765,
    "mrr": 0.9723256373256373,
    "top1_support_rate": 0.9535353535353536
  },
  "reranker_internal": {
    "lm_head": 0.0,
    "hidden_probe": 0.0,
    "mean_query_ap": 0.8066395005052182,
    "mrr": 0.9187229297077782,
    "top1_support_rate": 0.8808080808080808
  }
}
```

## mmqa (complete; probe=504, calibration=496, test=100)

| Method | Chunks | Precision | Support recall | Empty | Any support | All support | EM | F1 | Numeric |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Query-level cosine (matched 500-ish calibration) | 8.07 | 13.5% | 96.5% | 0.0% | 100.0% | 96.0% | 0.500 | 0.547 | 0.520 |
| LM-head only | 8.55 | 12.7% | 96.5% | 0.0% | 98.0% | 96.0% | 0.500 | 0.537 | 0.520 |
| Hidden-state probe only | 4.17 | 25.9% | 95.6% | 0.0% | 98.0% | 95.0% | 0.500 | 0.545 | 0.520 |
| LM-head + hidden probe | 4.12 | 26.2% | 95.6% | 0.0% | 98.0% | 95.0% | 0.500 | 0.545 | 0.520 |
| Cosine + LM-head + hidden probe | 3.70 | 29.7% | 97.3% | 0.0% | 99.0% | 97.0% | 0.500 | 0.539 | 0.520 |

Selected hidden-probe layer/C: `19` / `0.003`.

```json
{
  "internal_lm_hidden": {
    "lm_head": 0.01,
    "hidden_probe": 0.35,
    "mean_query_ap": 0.8569368471485277,
    "mrr": 0.8813259167543991,
    "top1_support_rate": 0.8146551724137931
  },
  "cosine_internal": {
    "lm_head": 1.5,
    "hidden_probe": 2.0,
    "mean_query_ap": 0.8716179312679994,
    "mrr": 0.8958549431638133,
    "top1_support_rate": 0.8297413793103449
  },
  "reranker_internal": {
    "lm_head": 0.0,
    "hidden_probe": 0.0,
    "mean_query_ap": 0.7969087107195786,
    "mrr": 0.8437633714032234,
    "top1_support_rate": 0.7758620689655172
  }
}
```

## tatqa (complete; probe=516, calibration=484, test=100)

| Method | Chunks | Precision | Support recall | Empty | Any support | All support | EM | F1 | Numeric |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Query-level cosine (matched 500-ish calibration) | 10.95 | 8.4% | 87.6% | 0.0% | 91.0% | 87.0% | 0.190 | 0.293 | 0.270 |
| LM-head only | 10.11 | 9.9% | 95.2% | 0.0% | 99.0% | 95.0% | 0.200 | 0.315 | 0.270 |
| Hidden-state probe only | 4.26 | 22.1% | 89.5% | 0.0% | 94.0% | 89.0% | 0.150 | 0.249 | 0.240 |
| LM-head + hidden probe | 4.14 | 23.2% | 91.4% | 0.0% | 96.0% | 91.0% | 0.170 | 0.280 | 0.280 |
| Cosine + LM-head + hidden probe | 3.96 | 24.5% | 92.4% | 0.0% | 97.0% | 92.0% | 0.150 | 0.253 | 0.260 |

Selected hidden-probe layer/C: `19` / `0.01`.

```json
{
  "internal_lm_hidden": {
    "lm_head": 0.35,
    "hidden_probe": 1.5,
    "mean_query_ap": 0.7814686660777834,
    "mrr": 0.7989282871873965,
    "top1_support_rate": 0.6862348178137652
  },
  "cosine_internal": {
    "lm_head": 1.5,
    "hidden_probe": 2.0,
    "mean_query_ap": 0.8153774365932022,
    "mrr": 0.8361593085277296,
    "top1_support_rate": 0.7489878542510121
  },
  "reranker_internal": {
    "lm_head": 0.0,
    "hidden_probe": 0.0,
    "mean_query_ap": 0.6623128342910399,
    "mrr": 0.6832826368168833,
    "top1_support_rate": 0.5587044534412956
  }
}
```

## webqa (complete; probe=465, calibration=535, test=100)

| Method | Chunks | Precision | Support recall | Empty | Any support | All support | EM | F1 | Numeric |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Query-level cosine (matched 500-ish calibration) | 18.89 | 4.2% | 92.9% | 0.0% | 97.0% | 94.0% | 0.000 | 0.122 | 0.050 |
| LM-head only | 16.24 | 4.9% | 94.1% | 0.0% | 97.0% | 96.0% | 0.000 | 0.151 | 0.040 |
| Hidden-state probe only | 6.91 | 9.4% | 76.5% | 0.0% | 89.0% | 83.0% | 0.000 | 0.120 | 0.040 |
| LM-head + hidden probe | 6.90 | 9.9% | 80.0% | 0.0% | 92.0% | 85.0% | 0.000 | 0.123 | 0.040 |
| Cosine + LM-head + hidden probe | 5.77 | 11.6% | 78.8% | 0.0% | 90.0% | 84.0% | 0.000 | 0.122 | 0.040 |

Selected hidden-probe layer/C: `19` / `0.03`.

```json
{
  "internal_lm_hidden": {
    "lm_head": 0.1,
    "hidden_probe": 0.75,
    "mean_query_ap": 0.768689369735609,
    "mrr": 0.8201708139409338,
    "top1_support_rate": 0.7288557213930348
  },
  "cosine_internal": {
    "lm_head": 0.35,
    "hidden_probe": 2.0,
    "mean_query_ap": 0.7768298413696992,
    "mrr": 0.8207949031053902,
    "top1_support_rate": 0.7263681592039801
  },
  "reranker_internal": {
    "lm_head": 0.0,
    "hidden_probe": 0.0,
    "mean_query_ap": 0.56980651368291,
    "mrr": 0.6471713051554282,
    "top1_support_rate": 0.5323383084577115
  }
}
```

## Audit

The qid partition and zero-overlap checks are recorded in `splits/SPLIT_INTEGRITY.json`. `analysis/{dataset}/fusion_predictions.npz` contains the five exact masks and score arrays used here; `*_downstream_predictions.jsonl` contains the selected contexts and answers for each held-out qid.
