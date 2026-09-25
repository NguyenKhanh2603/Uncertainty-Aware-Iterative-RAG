# Cosine conformal selector comparison

Every row within a dataset uses the same frozen Top-30 cosine candidates and its disjoint 100-query calibration plan. Query-level cosine is the pure per-query z-scored cosine threshold with a deterministic Top-1 empty-context fallback. BH context caps are experimental rank-ordered policies and have no claimed capped-procedure FDR guarantee.

## hotpotqa (complete; n=1000)

| Method | Chunks | Precision | Recall | Empty | Any support | All support | EM | F1 | Numeric |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| query_level_cosine_internal_fusion_alpha_0.10 | 3.40 | 48.8% | 95.5% | 0.0% | 99.1% | 92.3% | 0.399 | 0.526 | 0.422 |

## Protocol

```json
{
  "selector_families": [
    "CCE",
    "CONFLARE",
    "TRAQ retrieval",
    "BY cosine",
    "query-level cosine",
    "BH cosine"
  ],
  "top_l": 30,
  "calibration_queries_per_dataset": 100,
  "BH_note": "BH normally needs independence or suitable positive dependence; its rank cap is experimental."
}
```
