# Literature-protocol comparison: 1,000 calibration / 100 test

Every row uses the same frozen Jina Top-30 candidates and the same disjoint qid manifests. CCE, CONFLARE, and TRAQ follow their distinct published **retrieval calibration units**. They are adapted to benchmark human questions and frozen labelled supports; the original paper models, source corpora, and end-to-end generators are not substituted or claimed. TRAQ's row is its retrieval component with Bonferroni allocation, not the full answer-prediction-set method.

## hotpotqa (selection_complete; calibration=1000, test=100)

| Method | Chunks | Precision | Support recall | Empty | Any support* | All support* | EM | F1 | Numeric |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed Top-10 | 10.00 | 16.0% | 93.0% | 0.0% | 99.0% | 87.8% | pending | pending | pending |
| CCE Conformal-Embedding (Jina adaptation) | 8.75 | 17.8% | 90.7% | 0.0% | 96.9% | 85.7% | pending | pending | pending |
| CONFLARE source-question (Jina adaptation) | 2.14 | 57.0% | 70.9% | 8.0% | 88.8% | 55.1% | pending | pending | pending |
| TRAQ retrieval, Bonferroni (Jina adaptation) | 3.39 | 41.3% | 81.4% | 3.0% | 92.9% | 71.4% | pending | pending | pending |
| Query-level all-support cosine | 10.60 | 15.1% | 93.0% | 0.0% | 99.0% | 87.8% | pending | pending | pending |

*Any/all-support coverage is conditional on at least one labelled support being present in the frozen Top-30. The retrieval ceiling is recorded in the JSON below; missing retrieval is never counted as coverage.*

```json
{
  "cce_conformal_embedding_jina_alpha_0.10": {
    "positive_pairs": 1733,
    "nonconformity_threshold": 0.5464885532855988,
    "similarity_threshold": 0.45351144671440125,
    "calibration_unit": "labelled_relevant_question_chunk_pair"
  },
  "conflare_source_question_jina_alpha_0.10": {
    "calibration_records": 991,
    "distance_threshold": 0.4665285348892212,
    "similarity_threshold": 0.5334714651107788,
    "calibration_unit": "one_first_known_relevant_chunk_per_question",
    "question_source": "benchmark_human_question_with_labelled_support"
  },
  "traq_retrieval_bonferroni_jina_alpha_0.10": {
    "retrievable_calibration_queries": 991,
    "similarity_threshold": 0.5000492334365845,
    "total_alpha": 0.1,
    "alpha_retrieval": 0.05,
    "alpha_answer": 0.05,
    "calibration_unit": "one_best_true_retrieval_score_per_question",
    "scope": "retrieval_component_only; no_answer_prediction_set"
  },
  "query_level_all_support_cosine_alpha_0.10": {
    "threshold": -0.08596044143713166,
    "finite_sample_order": 893,
    "retrievable_calibration_queries": 991,
    "top1_fallback_queries": 0
  },
  "calibration_queries": 1000,
  "calibration_retrievable_queries": 991
}
```

## mmqa (selection_complete; calibration=1000, test=100)

| Method | Chunks | Precision | Support recall | Empty | Any support* | All support* | EM | F1 | Numeric |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed Top-10 | 10.00 | 10.9% | 96.5% | 0.0% | 100.0% | 95.7% | pending | pending | pending |
| CCE Conformal-Embedding (Jina adaptation) | 10.00 | 10.0% | 88.5% | 5.0% | 95.7% | 87.1% | pending | pending | pending |
| CONFLARE source-question (Jina adaptation) | 5.70 | 16.0% | 80.5% | 8.0% | 89.2% | 78.5% | pending | pending | pending |
| TRAQ retrieval, Bonferroni (Jina adaptation) | 9.97 | 10.0% | 88.5% | 5.0% | 95.7% | 87.1% | pending | pending | pending |
| Query-level all-support cosine | 9.05 | 12.0% | 96.5% | 0.0% | 100.0% | 95.7% | pending | pending | pending |

*Any/all-support coverage is conditional on at least one labelled support being present in the frozen Top-30. The retrieval ceiling is recorded in the JSON below; missing retrieval is never counted as coverage.*

```json
{
  "cce_conformal_embedding_jina_alpha_0.10": {
    "positive_pairs": 1192,
    "nonconformity_threshold": 0.3954514265060425,
    "similarity_threshold": 0.6045485734939575,
    "calibration_unit": "labelled_relevant_question_chunk_pair"
  },
  "conflare_source_question_jina_alpha_0.10": {
    "calibration_records": 927,
    "distance_threshold": 0.3691904306411743,
    "similarity_threshold": 0.6308095693588257,
    "calibration_unit": "one_first_known_relevant_chunk_per_question",
    "question_source": "benchmark_human_question_with_labelled_support"
  },
  "traq_retrieval_bonferroni_jina_alpha_0.10": {
    "retrievable_calibration_queries": 927,
    "similarity_threshold": 0.6048312187194824,
    "total_alpha": 0.1,
    "alpha_retrieval": 0.05,
    "alpha_answer": 0.05,
    "calibration_unit": "one_best_true_retrieval_score_per_question",
    "scope": "retrieval_component_only; no_answer_prediction_set"
  },
  "query_level_all_support_cosine_alpha_0.10": {
    "threshold": 0.04547165129619883,
    "finite_sample_order": 836,
    "retrievable_calibration_queries": 927,
    "top1_fallback_queries": 0
  },
  "calibration_queries": 1000,
  "calibration_retrievable_queries": 927
}
```

## tatqa (selection_complete; calibration=1000, test=100)

| Method | Chunks | Precision | Support recall | Empty | Any support* | All support* | EM | F1 | Numeric |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed Top-10 | 10.00 | 8.9% | 84.8% | 0.0% | 87.9% | 82.4% | pending | pending | pending |
| CCE Conformal-Embedding (Jina adaptation) | 17.80 | 5.2% | 87.6% | 6.0% | 89.0% | 86.8% | pending | pending | pending |
| CONFLARE source-question (Jina adaptation) | 16.52 | 5.6% | 87.6% | 7.0% | 89.0% | 86.8% | pending | pending | pending |
| TRAQ retrieval, Bonferroni (Jina adaptation) | 21.57 | 4.5% | 92.4% | 4.0% | 93.4% | 91.2% | pending | pending | pending |
| Query-level all-support cosine | 11.30 | 8.1% | 87.6% | 0.0% | 90.1% | 85.7% | pending | pending | pending |

*Any/all-support coverage is conditional on at least one labelled support being present in the frozen Top-30. The retrieval ceiling is recorded in the JSON below; missing retrieval is never counted as coverage.*

```json
{
  "cce_conformal_embedding_jina_alpha_0.10": {
    "positive_pairs": 1061,
    "nonconformity_threshold": 0.3985612988471985,
    "similarity_threshold": 0.6014387011528015,
    "calibration_unit": "labelled_relevant_question_chunk_pair"
  },
  "conflare_source_question_jina_alpha_0.10": {
    "calibration_records": 941,
    "distance_threshold": 0.39303016662597656,
    "similarity_threshold": 0.6069698333740234,
    "calibration_unit": "one_first_known_relevant_chunk_per_question",
    "question_source": "benchmark_human_question_with_labelled_support"
  },
  "traq_retrieval_bonferroni_jina_alpha_0.10": {
    "retrievable_calibration_queries": 941,
    "similarity_threshold": 0.5787214040756226,
    "total_alpha": 0.1,
    "alpha_retrieval": 0.05,
    "alpha_answer": 0.05,
    "calibration_unit": "one_best_true_retrieval_score_per_question",
    "scope": "retrieval_component_only; no_answer_prediction_set"
  },
  "query_level_all_support_cosine_alpha_0.10": {
    "threshold": -0.03361823925974544,
    "finite_sample_order": 848,
    "retrievable_calibration_queries": 941,
    "top1_fallback_queries": 0
  },
  "calibration_queries": 1000,
  "calibration_retrievable_queries": 941
}
```

## webqa (selection_complete; calibration=1000, test=100)

| Method | Chunks | Precision | Support recall | Empty | Any support* | All support* | EM | F1 | Numeric |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed Top-10 | 10.00 | 5.8% | 68.2% | 0.0% | 72.9% | 62.9% | pending | pending | pending |
| CCE Conformal-Embedding (Jina adaptation) | 19.82 | 3.6% | 83.5% | 9.0% | 85.7% | 80.0% | pending | pending | pending |
| CONFLARE source-question (Jina adaptation) | 18.27 | 3.8% | 82.4% | 10.0% | 84.3% | 78.6% | pending | pending | pending |
| TRAQ retrieval, Bonferroni (Jina adaptation) | 22.17 | 3.5% | 91.8% | 8.0% | 92.9% | 90.0% | pending | pending | pending |
| Query-level all-support cosine | 17.63 | 4.4% | 90.6% | 0.0% | 94.3% | 88.6% | pending | pending | pending |

*Any/all-support coverage is conditional on at least one labelled support being present in the frozen Top-30. The retrieval ceiling is recorded in the JSON below; missing retrieval is never counted as coverage.*

```json
{
  "cce_conformal_embedding_jina_alpha_0.10": {
    "positive_pairs": 1435,
    "nonconformity_threshold": 0.41339462995529175,
    "similarity_threshold": 0.5866053700447083,
    "calibration_unit": "labelled_relevant_question_chunk_pair"
  },
  "conflare_source_question_jina_alpha_0.10": {
    "calibration_records": 876,
    "distance_threshold": 0.4059034585952759,
    "similarity_threshold": 0.5940965414047241,
    "calibration_unit": "one_first_known_relevant_chunk_per_question",
    "question_source": "benchmark_human_question_with_labelled_support"
  },
  "traq_retrieval_bonferroni_jina_alpha_0.10": {
    "retrievable_calibration_queries": 876,
    "similarity_threshold": 0.5730915665626526,
    "total_alpha": 0.1,
    "alpha_retrieval": 0.05,
    "alpha_answer": 0.05,
    "calibration_unit": "one_best_true_retrieval_score_per_question",
    "scope": "retrieval_component_only; no_answer_prediction_set"
  },
  "query_level_all_support_cosine_alpha_0.10": {
    "threshold": -0.4762793413219147,
    "finite_sample_order": 790,
    "retrievable_calibration_queries": 876,
    "top1_fallback_queries": 0
  },
  "calibration_queries": 1000,
  "calibration_retrievable_queries": 876
}
```

## Scope and provenance

- CCE source: `baselines/conformal-context-engineering` (commit `91732d6058267f180ba9f47873d743288b2625af`).
- CONFLARE source: `baselines/conflare` (commit `ce081a45fb452704daa87f3b37f601b4accc7a82`).
- TRAQ source: `baselines/TRAQ` (commit `e9b66b5b7fd8fcd0b3c3dbfebc5a3391ca79aa55`).
- Exact qid manifests are copied under `splits/`; `SPLIT_INTEGRITY.json` verifies calibration/test disjointness.
- Downstream EM/F1 is a shared deterministic Qwen direct-answer diagnostic. It is not TRAQ answer-set coverage and must not be presented as TRAQ's end-to-end guarantee.
