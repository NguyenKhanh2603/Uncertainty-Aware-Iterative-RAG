"""Evaluate Jina retrieval/reranking and Qwen2-VL internal-only conformal pruning."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from research.internal_state_rag.analyze_hidden_chunk_probe import query_metrics, query_z
from research.internal_state_rag.analyze_multidataset_internal_fusion import (
    C_VALUES,
    WEIGHTS,
    fit_predict_probe,
    select_probe,
    validate_inputs,
)
from research.internal_state_rag.analyze_pairwise_conformal_clean_split import (
    conditional_metrics,
    conformal_threshold,
    end_to_end_metrics,
    fixed_k_metrics,
    keep_mask,
    paired_bootstrap,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--probe-train", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions-output", type=Path, required=True)
    parser.add_argument("--alphas", default="0.2,0.1,0.05")
    parser.add_argument("--candidate-layers", default="")
    parser.add_argument("--c-values", default=",".join(str(value) for value in C_VALUES))
    parser.add_argument("--seed", type=int, default=733)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    return parser.parse_args()


def choose_weights(labels, cosine, reranker, lm_head, hidden):
    retrievable = labels.any(axis=1)
    z = {
        "cosine": query_z(cosine),
        "reranker": query_z(reranker),
        "lm": query_z(lm_head),
        "hidden": query_z(hidden),
    }

    def choose(base: str | None):
        candidates = []
        for lm_weight in WEIGHTS:
            for hidden_weight in WEIGHTS:
                if base is None and lm_weight == 0 and hidden_weight == 0:
                    continue
                score = lm_weight * z["lm"] + hidden_weight * z["hidden"]
                if base is not None:
                    score = z[base] + score
                candidates.append(
                    {
                        "lm_head": lm_weight,
                        "hidden_probe": hidden_weight,
                        **query_metrics(labels[retrievable], score[retrievable]),
                    }
                )
        return max(
            candidates,
            key=lambda row: (
                row["mean_query_ap"],
                row["mrr"],
                -(row["lm_head"] + row["hidden_probe"]),
            ),
        )

    return {
        "internal_lm_hidden": choose(None),
        "cosine_internal": choose("cosine"),
        "reranker_internal": choose("reranker"),
    }


def make_scores(part, hidden, weights):
    cosine = query_z(part["cosine_scores"].astype(np.float64))
    reranker = query_z(part["reranker_scores"].astype(np.float64))
    lm_head = query_z(part["lm_relevance_scores"].astype(np.float64))
    hidden = query_z(hidden)

    def internal(name, base=None):
        row = weights[name]
        score = row["lm_head"] * lm_head + row["hidden_probe"] * hidden
        return score if base is None else base + score

    methods = {
        "jina_v4_cosine": cosine,
        "lm_head_only": lm_head,
        "hidden_probe_only": hidden,
        "internal_lm_hidden": internal("internal_lm_hidden"),
        "jina_m0_reranker": reranker,
        "cosine_internal": internal("cosine_internal", cosine),
        "reranker_internal": internal("reranker_internal", reranker),
    }
    if "bge_reranker_scores" in part.files and np.isfinite(part["bge_reranker_scores"]).all():
        methods["bge_v2_m3_reranker"] = query_z(
            part["bge_reranker_scores"].astype(np.float64)
        )
    return methods


def main() -> None:
    args = parse_args()
    train = np.load(args.probe_train)
    calibration = np.load(args.calibration)
    test = np.load(args.test)
    validate_inputs(train, calibration, test)
    layers = train["layer_ids"].astype(int).tolist()
    candidate_layers = (
        {int(x) for x in args.candidate_layers.split(",") if x.strip()}
        if args.candidate_layers
        else set(layers)
    )
    c_values = tuple(float(x) for x in args.c_values.split(",") if x.strip())
    train_labels = train["labels"].astype(bool)
    layer_index, selected_c, train_hidden_oof, audit = select_probe(
        train["features"].astype(np.float32),
        train_labels,
        layers,
        candidate_layers=candidate_layers,
        c_values=c_values,
    )
    cal_hidden, test_hidden = fit_predict_probe(
        train, [calibration, test], layer_index=layer_index, c=selected_c
    )
    weights = choose_weights(
        train_labels,
        train["cosine_scores"].astype(np.float64),
        train["reranker_scores"].astype(np.float64),
        train["lm_relevance_scores"].astype(np.float64),
        train_hidden_oof,
    )
    train_scores = make_scores(train, train_hidden_oof, weights)
    cal_scores = make_scores(calibration, cal_hidden, weights)
    test_scores = make_scores(test, test_hidden, weights)
    cal_labels = calibration["labels"].astype(bool)
    test_labels = test["labels"].astype(bool)
    train_retrievable = train_labels.any(axis=1)
    cal_retrievable = cal_labels.any(axis=1)
    test_retrievable = test_labels.any(axis=1)

    conformal = {}
    masks_to_save = {}
    bootstrap = {}
    for alpha_index, alpha in enumerate(float(x) for x in args.alphas.split(",")):
        alpha_key = str(alpha)
        conformal[alpha_key] = {}
        masks = {}
        for name, cal_score in cal_scores.items():
            threshold, order = conformal_threshold(
                cal_score[cal_retrievable],
                cal_labels[cal_retrievable],
                alpha=alpha,
                coverage_target="all_support",
            )
            mask = keep_mask(test_scores[name], threshold)
            masks[name] = mask
            masks_to_save[f"mask_{name}_alpha_{alpha_key}"] = mask
            conformal[alpha_key][name] = {
                "threshold": threshold,
                "finite_sample_order": order,
                "conditional_on_retrievable": conditional_metrics(
                    test_labels[test_retrievable], mask[test_retrievable]
                ),
                "end_to_end": end_to_end_metrics(test_labels, mask),
            }
        bootstrap[alpha_key] = {
            "internal_minus_cosine": paired_bootstrap(
                test_labels[test_retrievable],
                masks["internal_lm_hidden"][test_retrievable],
                masks["jina_v4_cosine"][test_retrievable],
                seed=args.seed + alpha_index,
                samples=args.bootstrap_samples,
            ),
            "reranker_internal_minus_reranker": paired_bootstrap(
                test_labels[test_retrievable],
                masks["reranker_internal"][test_retrievable],
                masks["jina_m0_reranker"][test_retrievable],
                seed=args.seed + 10 + alpha_index,
                samples=args.bootstrap_samples,
            ),
        }

    output = {
        "status": "complete",
        "dataset": args.dataset,
        "protocol": (
            "disjoint_training_subset_conformal_calibration_"
            "official_test_fixed_jina_v4_top30"
        ),
        "split": {
            "probe_train_total": len(train_labels),
            "probe_train_retrievable": int(train_retrievable.sum()),
            "calibration_total": len(cal_labels),
            "calibration_retrievable": int(cal_retrievable.sum()),
            "test_total": len(test_labels),
            "test_retrievable": int(test_retrievable.sum()),
            "retrieval_query_coverage_ceiling": float(test_retrievable.mean()),
            "top_l": int(test_labels.shape[1]),
        },
        "probe": {
            "selected_layer": layers[layer_index],
            "selected_c": selected_c,
            "selection": "5-fold query-grouped OOF mean AP",
            "candidate_audit": audit,
        },
        "selected_weights": weights,
        "train_oof_ranking": {
            name: query_metrics(train_labels[train_retrievable], score[train_retrievable])
            for name, score in train_scores.items()
        },
        "test_ranking": {
            name: fixed_k_metrics(test_labels, score) for name, score in test_scores.items()
        },
        "conformal": conformal,
        "paired_bootstrap": bootstrap,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.predictions_output,
        qids=test["qids"],
        chunk_ids=test["chunk_ids"],
        modalities=test["modalities"],
        labels=test_labels,
        **{f"score_{name}": score for name, score in test_scores.items()},
        **masks_to_save,
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        main()
