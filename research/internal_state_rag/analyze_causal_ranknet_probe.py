"""Distill leave-one-out causal chunk value into a query-wise internal RankNet."""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import rankdata, spearmanr

from research.internal_state_rag.analyze_hidden_chunk_probe import (
    make_probe,
    query_metrics,
    query_z,
)
from research.internal_state_rag.analyze_pairwise_conformal_clean_split import (
    conditional_metrics,
    conformal_threshold,
    fixed_k_metrics,
    keep_mask,
)


SEEDS = (17, 29, 43)
FUSION_WEIGHTS = (0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0)


@dataclass(frozen=True)
class ProbeConfig:
    name: str
    layers: tuple[int, ...]
    include_scalars: bool
    hidden_width: int


CONFIGS = (
    ProbeConfig("layer30_linear", (30,), False, 0),
    ProbeConfig("layer30_mlp", (30,), False, 128),
    ProbeConfig("late4_mlp", (18, 24, 30, 35), False, 128),
    ProbeConfig("late4_mlp_plus_bge_lm_rank", (18, 24, 30, 35), True, 128),
)


class RankNet(torch.nn.Module):
    def __init__(self, dimension: int, hidden_width: int) -> None:
        super().__init__()
        if hidden_width == 0:
            self.network = torch.nn.Linear(dimension, 1)
        else:
            self.network = torch.nn.Sequential(
                torch.nn.Linear(dimension, hidden_width),
                torch.nn.GELU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(hidden_width, 1),
            )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(-1)


def within_query_ranks(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            (rankdata(row, method="average") - 1) / (len(row) - 1) - 0.5
            for row in values
        ],
        dtype=np.float32,
    )


def causal_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    correlations, reciprocal_ranks, top1, top3 = [], [], [], []
    for truth, score in zip(target, prediction, strict=True):
        rho = float(spearmanr(truth, score).statistic)
        correlations.append(0.0 if np.isnan(rho) else rho)
        best = int(np.argmax(truth))
        order = np.argsort(-score, kind="stable")
        position = int(np.flatnonzero(order == best)[0]) + 1
        reciprocal_ranks.append(1 / position)
        top1.append(position == 1)
        top3.append(position <= 3)
    return {
        "mean_query_spearman": float(np.mean(correlations)),
        "top_causal_chunk_mrr": float(np.mean(reciprocal_ranks)),
        "top_causal_chunk_top1": float(np.mean(top1)),
        "top_causal_chunk_top3": float(np.mean(top3)),
    }


def raw_features(data: Any, config: ProbeConfig) -> np.ndarray:
    available_layers = data["layer_ids"].astype(int).tolist()
    indices = [available_layers.index(layer) for layer in config.layers]
    hidden = data["features"][:, :, indices, :].astype(np.float32)
    hidden = hidden - hidden.mean(axis=1, keepdims=True)
    blocks = [hidden.reshape(hidden.shape[0], hidden.shape[1], -1)]
    if config.include_scalars:
        blocks.append(
            np.stack(
                [
                    query_z(data["bge_scores"].astype(np.float64)),
                    query_z(data["lm_relevance_scores"].astype(np.float64)),
                    -query_z(data["ranks"].astype(np.float64)),
                ],
                axis=-1,
            ).astype(np.float32)
        )
    return np.concatenate(blocks, axis=-1)


def standardize(
    train: np.ndarray, *targets: np.ndarray
) -> tuple[np.ndarray, list[np.ndarray], dict[str, list[float]]]:
    mean = train.reshape(-1, train.shape[-1]).mean(axis=0)
    scale = train.reshape(-1, train.shape[-1]).std(axis=0)
    scale = np.maximum(scale, 1e-5)
    transform = lambda values: ((values - mean) / scale).astype(np.float32)
    return transform(train), [transform(values) for values in targets], {
        "mean": mean.astype(float).tolist(),
        "scale": scale.astype(float).tolist(),
    }


def rank_loss(scores: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    score_difference = scores[:, :, None] - scores[:, None, :]
    target_difference = target[:, :, None] - target[:, None, :]
    upper = torch.triu(
        torch.ones(target.shape[1], target.shape[1], dtype=torch.bool), diagonal=1
    ).to(target.device)
    mask = upper[None, :, :].expand_as(target_difference)
    logits = score_difference[mask]
    labels = (target_difference[mask] > 0).float()
    weights = target_difference[mask].abs().clamp_min(1 / (target.shape[1] - 1))
    pairwise = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, labels, reduction="none"
    )
    centered_scores = scores - scores.mean(dim=1, keepdim=True)
    regression = torch.nn.functional.smooth_l1_loss(centered_scores, target)
    return (pairwise * weights).sum() / weights.sum() + 0.05 * regression


def train_one(
    train_x: np.ndarray,
    train_target: np.ndarray,
    validation_x: np.ndarray,
    validation_target: np.ndarray,
    *,
    hidden_width: int,
    seed: int,
    device: torch.device,
    epochs: int = 400,
    patience: int = 60,
) -> tuple[RankNet, int, list[dict[str, float]], np.ndarray]:
    torch.manual_seed(seed)
    model = RankNet(train_x.shape[-1], hidden_width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    x = torch.from_numpy(train_x).to(device)
    y = torch.from_numpy(train_target).to(device)
    validation_tensor = torch.from_numpy(validation_x).to(device)
    best_state = copy.deepcopy(model.state_dict())
    best_epoch, best_score, stale = 0, (-np.inf, -np.inf), 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = rank_loss(model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            prediction = model(validation_tensor).cpu().numpy()
        metrics = causal_metrics(validation_target, prediction)
        score = (
            metrics["mean_query_spearman"],
            metrics["top_causal_chunk_mrr"],
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            history.append({"epoch": epoch, "loss": float(loss.item()), **metrics})
        if epoch >= 60 and stale >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prediction = model(validation_tensor).cpu().numpy()
    return model, best_epoch, history, prediction


def fit_fixed_epochs(
    train_x: np.ndarray,
    target: np.ndarray,
    *,
    hidden_width: int,
    seed: int,
    epochs: int,
    device: torch.device,
) -> RankNet:
    torch.manual_seed(seed)
    model = RankNet(train_x.shape[-1], hidden_width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    x = torch.from_numpy(train_x).to(device)
    y = torch.from_numpy(target).to(device)
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = rank_loss(model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    model.eval()
    return model


def predict(model: RankNet, values: np.ndarray, batch_queries: int = 64) -> np.ndarray:
    rows = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for start in range(0, len(values), batch_queries):
            rows.append(
                model(torch.from_numpy(values[start : start + batch_queries]).to(device))
                .cpu()
                .numpy()
            )
    return np.concatenate(rows, axis=0)


def indices_for(data: Any, qids: list[str]) -> np.ndarray:
    positions = {str(qid): index for index, qid in enumerate(data["qids"].tolist())}
    return np.asarray([positions[qid] for qid in qids], dtype=int)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--causal-labels", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--old-probe-train", type=Path, required=True)
    parser.add_argument("--test-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--causal-train", type=int, default=96)
    parser.add_argument("--alphas", default="0.2,0.1,0.05")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    causal_payload = json.loads(args.causal_labels.read_text(encoding="utf-8"))
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    features = np.load(args.features)
    test = np.load(args.test_features)
    old_train = np.load(args.old_probe_train)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    causal_qids = [str(row["qid"]) for row in causal_payload["queries"]]
    if causal_qids != [str(qid) for qid in plan["causal_teacher_qids"]]:
        raise ValueError("Causal label order differs from the frozen plan")
    if args.causal_train >= len(causal_qids):
        raise ValueError("A causal holdout is required")
    causal_indices = indices_for(features, causal_qids)
    conformal_qids = [str(qid) for qid in plan["conformal_qids"]]
    conformal_indices = indices_for(features, conformal_qids)
    causal_values = np.asarray(
        [
            [float(chunk["gold_logprob_drop"]) for chunk in row["chunks"]]
            for row in causal_payload["queries"]
        ]
    )
    causal_target = within_query_ranks(causal_values)
    validation_slice = slice(args.causal_train, len(causal_qids))

    configuration_results = []
    validation_ensembles = {}
    for config in CONFIGS:
        values = raw_features(features, config)[causal_indices]
        train_x, transformed, _ = standardize(
            values[: args.causal_train], values[validation_slice]
        )
        validation_x = transformed[0]
        seed_predictions, best_epochs, histories = [], [], {}
        for seed in SEEDS:
            _, epoch, history, prediction_values = train_one(
                train_x,
                causal_target[: args.causal_train],
                validation_x,
                causal_target[validation_slice],
                hidden_width=config.hidden_width,
                seed=seed,
                device=device,
            )
            seed_predictions.append(prediction_values)
            best_epochs.append(epoch)
            histories[str(seed)] = history
        ensemble = np.mean(seed_predictions, axis=0)
        validation_ensembles[config.name] = ensemble
        metrics = causal_metrics(causal_target[validation_slice], ensemble)
        configuration_results.append(
            {
                "name": config.name,
                "layers": list(config.layers),
                "include_scalars": config.include_scalars,
                "hidden_width": config.hidden_width,
                "validation": metrics,
                "best_epochs": best_epochs,
                "refit_epochs": int(np.median(best_epochs)),
                "histories": histories,
            }
        )
    selected = max(
        configuration_results,
        key=lambda row: (
            row["validation"]["mean_query_spearman"],
            row["validation"]["top_causal_chunk_mrr"],
        ),
    )
    selected_config = next(config for config in CONFIGS if config.name == selected["name"])
    all_features_raw = raw_features(features, selected_config)
    test_features_raw = raw_features(test, selected_config)
    causal_raw = all_features_raw[causal_indices]
    causal_x, transformed, scaler = standardize(
        causal_raw, all_features_raw, test_features_raw
    )
    all_x, test_x = transformed
    refit_models = [
        fit_fixed_epochs(
            causal_x,
            causal_target,
            hidden_width=selected_config.hidden_width,
            seed=seed,
            epochs=int(selected["refit_epochs"]),
            device=device,
        )
        for seed in SEEDS
    ]
    causal_probe_all = np.mean([predict(model, all_x) for model in refit_models], axis=0)
    causal_probe_test = np.mean([predict(model, test_x) for model in refit_models], axis=0)

    old_layer_index = old_train["layer_ids"].astype(int).tolist().index(30)
    old_x = old_train["features"][:, :, old_layer_index, :].astype(np.float32)
    relevance_model = make_probe(0.1).fit(
        old_x.reshape(-1, old_x.shape[-1]), old_train["labels"].astype(bool).ravel()
    )
    feature_layer_index = features["layer_ids"].astype(int).tolist().index(30)
    relevance_all = relevance_model.predict_proba(
        features["features"][:, :, feature_layer_index, :]
        .astype(np.float32)
        .reshape(-1, features["features"].shape[-1])
    )[:, 1].reshape(features["labels"].shape)
    relevance_test = relevance_model.predict_proba(
        test["features"][:, :, feature_layer_index, :]
        .astype(np.float32)
        .reshape(-1, test["features"].shape[-1])
    )[:, 1].reshape(test["labels"].shape)
    baseline_all = (
        query_z(features["bge_scores"].astype(np.float64))
        + 0.2 * query_z(features["lm_relevance_scores"].astype(np.float64))
        + 0.75 * query_z(relevance_all)
    )
    baseline_test = (
        query_z(test["bge_scores"].astype(np.float64))
        + 0.2 * query_z(test["lm_relevance_scores"].astype(np.float64))
        + 0.75 * query_z(relevance_test)
    )

    holdout_indices = causal_indices[args.causal_train :]
    holdout_labels = features["labels"].astype(bool)[holdout_indices]
    fusion_tuning = []
    for weight in FUSION_WEIGHTS:
        scores = baseline_all[holdout_indices] + weight * query_z(
            validation_ensembles[selected_config.name]
        )
        fusion_tuning.append({"weight": weight, **query_metrics(holdout_labels, scores)})
    selected_weight = max(
        fusion_tuning,
        key=lambda row: (row["mean_query_ap"], row["mrr"], -row["weight"]),
    )
    causal_all_z = query_z(causal_probe_all)
    causal_test_z = query_z(causal_probe_test)
    fused_all = baseline_all + float(selected_weight["weight"]) * causal_all_z
    fused_test = baseline_test + float(selected_weight["weight"]) * causal_test_z
    score_sets = {
        "bge": (
            query_z(features["bge_scores"].astype(np.float64)),
            query_z(test["bge_scores"].astype(np.float64)),
        ),
        "three_signal_baseline": (baseline_all, baseline_test),
        "causal_probe": (causal_probe_all, causal_probe_test),
        "three_signal_plus_causal": (fused_all, fused_test),
    }
    labels_all = features["labels"].astype(bool)
    test_labels = test["labels"].astype(bool)
    conformal_retrievable = labels_all[conformal_indices].any(axis=1)
    test_retrievable = test_labels.any(axis=1)
    conformal_results = {}
    for alpha in [float(value) for value in args.alphas.split(",")]:
        conformal_results[str(alpha)] = {}
        for name, (calibration_scores_all, evaluation_scores) in score_sets.items():
            calibration_scores = calibration_scores_all[conformal_indices]
            threshold, order = conformal_threshold(
                calibration_scores[conformal_retrievable],
                labels_all[conformal_indices][conformal_retrievable],
                alpha=alpha,
                coverage_target="all_support",
            )
            mask = keep_mask(evaluation_scores, threshold)
            conformal_results[str(alpha)][name] = {
                "threshold": threshold,
                "finite_sample_order": order,
                "conditional_on_retrievable": conditional_metrics(
                    test_labels[test_retrievable], mask[test_retrievable]
                ),
            }

    teacher_labels = features["labels"].astype(bool)[causal_indices]
    output = {
        "status": "complete",
        "method": "query_centered_multilayer_causal_ranknet_distillation",
        "split": {
            "causal_train_queries": args.causal_train,
            "causal_holdout_queries": len(causal_qids) - args.causal_train,
            "conformal_queries_total": len(conformal_indices),
            "conformal_queries_retrievable": int(conformal_retrievable.sum()),
            "test_queries_total": len(test_labels),
            "test_queries_retrievable": int(test_retrievable.sum()),
            "causal_conformal_overlap": len(set(causal_qids) & set(conformal_qids)),
        },
        "teacher": {
            "support_mean_causal_value": float(causal_values[teacher_labels].mean()),
            "non_support_mean_causal_value": float(causal_values[~teacher_labels].mean()),
            "support_positive_fraction": float((causal_values[teacher_labels] > 0).mean()),
            "non_support_positive_fraction": float(
                (causal_values[~teacher_labels] > 0).mean()
            ),
            "support_ranking": query_metrics(teacher_labels, causal_values),
            "bge_support_ranking": query_metrics(
                teacher_labels, features["bge_scores"][causal_indices]
            ),
        },
        "configuration_selection": configuration_results,
        "selected_configuration": selected,
        "selected_fusion_weight": selected_weight,
        "fusion_weight_tuning": fusion_tuning,
        "test_ranking": {
            name: fixed_k_metrics(test_labels, scores[1])
            for name, scores in score_sets.items()
        },
        "conformal": conformal_results,
        "feature_scaler": scaler,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.predictions,
        calibration_qids=features["qids"],
        test_qids=test["qids"],
        calibration_labels=labels_all,
        test_labels=test_labels,
        causal_probe_calibration=causal_probe_all.astype(np.float32),
        causal_probe_test=causal_probe_test.astype(np.float32),
        baseline_calibration=baseline_all.astype(np.float32),
        baseline_test=baseline_test.astype(np.float32),
        fused_calibration=fused_all.astype(np.float32),
        fused_test=fused_test.astype(np.float32),
        conformal_indices=conformal_indices,
        selected_weight=np.asarray(float(selected_weight["weight"])),
    )
    print(
        json.dumps(
            {
                "selected_configuration": selected,
                "selected_fusion_weight": selected_weight,
                "test_ranking": output["test_ranking"],
                "conformal": conformal_results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
