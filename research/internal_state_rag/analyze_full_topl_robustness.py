import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from research.internal_state_rag.analyze_full_topl_validation import (
    load_observations,
    retention_metrics,
)
from research.internal_state_rag.analyze_position_control import conformal_threshold
from research.internal_state_rag.ranking import grouped_z_scores

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--root",
    type=Path,
    default=Path("research/internal_state_rag/results/full_top30_n420_seed101"),
)
parser.add_argument("--rows", type=Path)
parser.add_argument("--output", type=Path)
args = parser.parse_args()
ROOT = args.root
rows_path = args.rows or ROOT / "queries.jsonl"
rows = load_observations(rows_path)
for row in rows:
    row["log_chunk_tokens"] = float(np.log1p(row["chunk_token_count"]))

labels = np.asarray([int(r["is_support"]) for r in rows])
roles = np.asarray([r["role"] for r in rows])
train = roles == "scorer_train"
cal_qids = {r["qid"] for r in rows if r["role"] == "conformal_calibration"}

z = {
    name: grouped_z_scores(rows, name)
    for name in (
        "bge_score",
        "mean_attention_mass",
        "mean_attention_fraction",
        "log_chunk_tokens",
    )
}

feature_sets = {
    "bge": ["bge_score"],
    "bge_length": ["bge_score", "log_chunk_tokens"],
    "bge_mass": ["bge_score", "mean_attention_mass"],
    "bge_fraction": ["bge_score", "mean_attention_fraction"],
    "attention_only": ["mean_attention_mass", "mean_attention_fraction"],
    "fusion": ["bge_score", "mean_attention_mass", "mean_attention_fraction"],
    "fusion_length": [
        "bge_score",
        "mean_attention_mass",
        "mean_attention_fraction",
        "log_chunk_tokens",
    ],
}

scores = {"bge_raw": z["bge_score"]}
models = {}
for key, names in feature_sets.items():
    X = np.column_stack([z[n] for n in names])
    scaler = StandardScaler().fit(X[train])
    model = LogisticRegression(class_weight="balanced", random_state=0, max_iter=1000).fit(
        scaler.transform(X[train]), labels[train]
    )
    scores[key] = model.predict_proba(scaler.transform(X))[:, 1]
    models[key] = {"features": names, "coef": model.coef_[0].tolist()}

eval_qids = sorted({r["qid"] for r in rows if r["role"] == "evaluation"})
by_qid = defaultdict(list)
for i, row in enumerate(rows):
    if row["role"] == "evaluation":
        by_qid[row["qid"]].append(i)

def per_query(name):
    aps, rrs, tops = [], [], []
    for qid in eval_qids:
        ix = by_qid[qid]
        y = labels[ix]
        s = scores[name][ix]
        order = np.argsort(-s, kind="stable")
        ranks = np.flatnonzero(y[order]) + 1
        aps.append(average_precision_score(y, s))
        rrs.append(1.0 / ranks[0])
        tops.append(float(ranks[0] == 1))
    return np.asarray(aps), np.asarray(rrs), np.asarray(tops)

def bootstrap_delta(a, b, seed=8128, n=20000):
    rng = np.random.default_rng(seed)
    delta = a - b
    samples = np.empty(n)
    for j in range(n):
        samples[j] = delta[rng.integers(0, len(delta), len(delta))].mean()
    return {
        "mean_delta": float(delta.mean()),
        "ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
        "p_delta_gt_0": float(np.mean(samples > 0)),
    }

ranking = {}
perq = {}
for name in scores:
    ap, rr, top = per_query(name)
    perq[name] = (ap, rr, top)
    ranking[name] = {
        "mean_query_ap": float(ap.mean()),
        "mrr": float(rr.mean()),
        "top1": float(top.mean()),
    }

paired = {}
for other in ("bge_raw", "bge_length"):
    paired[f"fusion_minus_{other}"] = {
        metric: bootstrap_delta(perq["fusion"][j], perq[other][j])
        for j, metric in enumerate(("query_ap", "mrr", "top1"))
    }

retention = {}
paired_retention = {}
for alpha in (0.05, 0.1):
    retention[str(alpha)] = {}
    thresholds = {}
    for name in scores:
        threshold = conformal_threshold(rows, scores[name], cal_qids, alpha)
        thresholds[name] = threshold
        retention[str(alpha)][name] = retention_metrics(
            rows, scores[name], role="evaluation", threshold=threshold
        )
    query_stats = {}
    for qid in eval_qids:
        indices = by_qid[qid]
        support = [index for index in indices if labels[index]]
        query_stats[qid] = {
            "stratum": rows[indices[0]]["stratum"],
            **{
                name: {
                    "coverage": bool(np.all(scores[name][support] >= thresholds[name])),
                    "n_kept": int(np.sum(scores[name][indices] >= thresholds[name])),
                }
                for name in ("bge_raw", "fusion")
            },
        }
    paired_retention[str(alpha)] = {}
    for stratum in ("all", "easy", "hard"):
        selected = [
            stats
            for stats in query_stats.values()
            if stratum == "all" or stats["stratum"] == stratum
        ]
        paired_retention[str(alpha)][stratum] = {
            "n_queries": len(selected),
            "bge_coverage": float(
                np.mean([stats["bge_raw"]["coverage"] for stats in selected])
            ),
            "fusion_coverage": float(
                np.mean([stats["fusion"]["coverage"] for stats in selected])
            ),
            "bge_only_coverage": int(
                sum(
                    stats["bge_raw"]["coverage"]
                    and not stats["fusion"]["coverage"]
                    for stats in selected
                )
            ),
            "fusion_only_coverage": int(
                sum(
                    stats["fusion"]["coverage"]
                    and not stats["bge_raw"]["coverage"]
                    for stats in selected
                )
            ),
            "both_fail": int(
                sum(
                    not stats["bge_raw"]["coverage"]
                    and not stats["fusion"]["coverage"]
                    for stats in selected
                )
            ),
            "mean_chunk_delta_fusion_minus_bge": float(
                np.mean(
                    [
                        stats["fusion"]["n_kept"] - stats["bge_raw"]["n_kept"]
                        for stats in selected
                    ]
                )
            ),
        }

position_rhos = []
for qid in eval_qids:
    ix = by_qid[qid]
    original = [rows[i]["original_attention_mass"] for i in ix]
    reversed_ = [rows[i]["reversed_attention_mass"] for i in ix]
    position_rhos.append(float(spearmanr(original, reversed_).statistic))

feature_corr = np.corrcoef(
    np.column_stack(
        [z["mean_attention_mass"], z["mean_attention_fraction"], z["log_chunk_tokens"]]
    ).T
).tolist()

out = {
    "models": models,
    "ranking": ranking,
    "paired_bootstrap": paired,
    "retention": retention,
    "paired_retention": paired_retention,
    "position_spearman": {
        "mean": float(np.mean(position_rhos)),
        "median": float(np.median(position_rhos)),
        "positive_rate": float(np.mean(np.asarray(position_rhos) > 0)),
    },
    "feature_corr_mass_fraction_log_tokens": feature_corr,
}
out["seed"] = 8128
out["n_bootstrap_resamples"] = 20000
output_path = args.output or ROOT / "robustness_analysis.json"
output_path.write_text(
    json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps(out, indent=2))
