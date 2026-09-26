# Audited Qwen-7B internal-signal ablation

This artifact evaluates a frozen Qwen2-VL-7B relevance signal against query-level cosine pruning. Each dataset begins with 1,000 parent calibration qids, divided into a disjoint probe-training role and conformal-calibration role; the same 100 held-out test qids are used by all five methods.

The full selection and Qwen downstream table is in [REPORT.md](REPORT.md).

## Reproduction and audit

- [Pipeline runner](../../../run_inverse_1000cal_internal_signal_ablation.sh)
- [Split/materialization code](../../../prepare_inverse_1000cal_internal_signal_plans.py)
- [Internal-score analysis](../../../analyze_qwen2vl_jina_ablation.py)
- [Selection and downstream evaluator](../../../run_inverse_1000cal_internal_signal_ablation.py)
- [Feature extractor](../../../run_qwen2vl_pairwise_features.py)
- [Method-positioning note](../../../QUERY_LEVEL_CONFORMAL_POSITIONING_CCE_TRAQ_CONFLARE.md)
- [Programmatic completion audit](FINALIZATION_AUDIT.json)

| Dataset | Probe train | Conformal calibration | Held-out test | Downstream records |
|---|---:|---:|---:|---:|
| hotpotqa | 500 | 500 | 100 | 100 |
| mmqa | 504 | 496 | 100 | 100 |
| tatqa | 516 | 484 | 100 | 100 |
| webqa | 465 | 535 | 100 | 100 |

## Exact inputs and per-query artifacts

- **hotpotqa:** [probe qids](splits/hotpotqa/probe_train_manifest.json), [conformal-calibration qids](splits/hotpotqa/conformal_calibration_manifest.json), [test qids](splits/hotpotqa/test_manifest.json), [analysis](analysis/hotpotqa/fusion_analysis.json), and [downstream predictions](hotpotqa_downstream_predictions.jsonl).
- **mmqa:** [probe qids](splits/mmqa/probe_train_manifest.json), [conformal-calibration qids](splits/mmqa/conformal_calibration_manifest.json), [test qids](splits/mmqa/test_manifest.json), [analysis](analysis/mmqa/fusion_analysis.json), and [downstream predictions](mmqa_downstream_predictions.jsonl).
- **tatqa:** [probe qids](splits/tatqa/probe_train_manifest.json), [conformal-calibration qids](splits/tatqa/conformal_calibration_manifest.json), [test qids](splits/tatqa/test_manifest.json), [analysis](analysis/tatqa/fusion_analysis.json), and [downstream predictions](tatqa_downstream_predictions.jsonl).
- **webqa:** [probe qids](splits/webqa/probe_train_manifest.json), [conformal-calibration qids](splits/webqa/conformal_calibration_manifest.json), [test qids](splits/webqa/test_manifest.json), [analysis](analysis/webqa/fusion_analysis.json), and [downstream predictions](webqa_downstream_predictions.jsonl).

The large Qwen feature caches are intentionally not versioned here. Their exact qid manifests, frozen Top-30 candidate ordering, extractor code, model revision, and analysis artifacts above are sufficient to reproduce the masks.
