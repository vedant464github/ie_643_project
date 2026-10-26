"""
Backbone bake-off (source-only, no target labels used for choosing).

Domains for selection: climate, energy, finance_vol (raw finance excluded: everything ties naive there).
Seeds: climate and finance_vol 0,1,2 with full training; energy 0,1 on every 4th training window, max 15 epochs.
Batch size and lr (1e-3) identical across backbones within a domain. Hard cap 50k trainable parameters.

Selection rule (fixed before any results):
1. Per domain, rank backbones by mean SOURCE val MSE (normalized scale) across seeds (rank 1 = lowest;
   exact ties share the average rank).
2. Order backbones by mean rank across the three domains; the top two are A (best) and B.
3. Tie-break: if |mean_A - mean_B| <= max(std_A, std_B) (sample std, ddof=1) in at least 2 of 3 domains,
   or A and B have equal mean rank, pick the one with fewer trainable parameters (mean over the three
   domains); if those counts are within 10% of each other, pick the one with lower total source-training
   wall-clock over all bake-off runs. Otherwise pick A.
Target-test numbers are reported but never used to choose.
"""
import json
import time
from pathlib import Path

import numpy as np

from src.experiment import run_transfer
from src.models.backbone import BACKBONES, build_model, count_params
from src.run_experiments import DOMAIN_CONFIGS

OUT_DIR = Path("experiments/bakeoff")
PARAM_CAP = 50_000
SELECTION_DOMAINS = ["climate", "energy", "finance_vol"]
PLAN = {
    "climate": {"seeds": [0, 1, 2], "train_stride": 1, "max_epochs": None},
    "finance_vol": {"seeds": [0, 1, 2], "train_stride": 1, "max_epochs": None},
    "energy": {"seeds": [0, 1], "train_stride": 4, "max_epochs": 15},
}

def summarize(runs):
    """Per domain/backbone: mean ± std (ddof=1) of val MSE/MAE, target-test MAE/RMSE, params, wall-clock."""
    summary = {}
    for d in SELECTION_DOMAINS:
        summary[d] = {}
        for b in BACKBONES:
            rs = [r for r in runs if r["domain"] == d and r["backbone"] == b]
            if not rs:
                continue
            def ms(vals):
                vals = np.asarray(vals, dtype=float)
                return {"mean": float(vals.mean()), "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0}
            summary[d][b] = {
                "n_seeds": len(rs),
                "params": rs[0]["params"],
                "val_mse_norm": ms([r["source_val"]["mse_norm"] for r in rs]),
                "val_mae_norm": ms([r["source_val"]["mae_norm"] for r in rs]),
                "val_mae_orig": ms([r["source_val"]["orig_scale"]["MAE"] for r in rs]),
                "best_epoch": [r["source_training"]["best_epoch"] for r in rs],
                "train_seconds": ms([r["source_training"]["train_seconds"] for r in rs]),
                "target_test_source_stats_MAE": ms([r["source_only_on_target_test"]["source_stats"]["MAE"] for r in rs]),
                "target_test_source_stats_RMSE": ms([r["source_only_on_target_test"]["source_stats"]["RMSE"] for r in rs]),
                "target_test_target_stats_MAE": ms([r["source_only_on_target_test"]["target_stats"]["MAE"] for r in rs]),
                "target_test_target_stats_RMSE": ms([r["source_only_on_target_test"]["target_stats"]["RMSE"] for r in rs]),
            }
    return summary

def select(summary, runs):
    backbones = [b for b in BACKBONES if all(b in summary[d] for d in SELECTION_DOMAINS)]
    ranks = {b: [] for b in backbones}
    for d in SELECTION_DOMAINS:
        means = {b: summary[d][b]["val_mse_norm"]["mean"] for b in backbones}
        for b in backbones:
            better = sum(means[o] < means[b] for o in backbones)
            equal = sum(means[o] == means[b] for o in backbones)
            ranks[b].append(better + (equal + 1) / 2)  # average rank for exact ties
    mean_rank = {b: float(np.mean(r)) for b, r in ranks.items()}
    order = sorted(backbones, key=lambda b: mean_rank[b])
    a, b = order[0], order[1]

    close_domains = [d for d in SELECTION_DOMAINS
                     if abs(summary[d][a]["val_mse_norm"]["mean"] - summary[d][b]["val_mse_norm"]["mean"])
                     <= max(summary[d][a]["val_mse_norm"]["std"], summary[d][b]["val_mse_norm"]["std"])]
    tie = len(close_domains) >= 2 or mean_rank[a] == mean_rank[b]
    params = {x: float(np.mean([summary[d][x]["params"] for d in SELECTION_DOMAINS])) for x in (a, b)}
    wall = {x: float(sum(r["source_training"]["train_seconds"] for r in runs if r["backbone"] == x)) for x in (a, b)}

    if not tie:
        chosen, reason = a, "best mean rank; no tie with runner-up"
    elif abs(params[a] - params[b]) / max(params[a], params[b]) > 0.10:
        chosen = min((a, b), key=lambda x: params[x])
        reason = "tie with runner-up -> fewer trainable parameters"
    else:
        chosen = min((a, b), key=lambda x: wall[x])
        reason = "tie with runner-up, params within 10% -> lower total wall-clock"
    return {"ranks_per_domain": {x: dict(zip(SELECTION_DOMAINS, ranks[x])) for x in backbones},
            "mean_rank": mean_rank, "order": order, "top_two": [a, b],
            "close_domains_within_1std": close_domains, "tie": tie,
            "mean_params": params, "total_train_seconds": wall, "chosen": chosen, "reason": reason}

if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # parameter counts and cap check, before any training
    param_counts = {d: {b: count_params(build_model(b, DOMAIN_CONFIGS[d]["input_len"], DOMAIN_CONFIGS[d]["horizon"]))
                        for b in BACKBONES} for d in SELECTION_DOMAINS}
    print("Trainable parameters:", json.dumps(param_counts, indent=2))
    over = [(d, b, n) for d in param_counts for b, n in param_counts[d].items() if n > PARAM_CAP]
    if over:
        raise SystemExit(f"Over the {PARAM_CAP} parameter cap, shrink or drop before running: {over}")

    with open(OUT_DIR / "selection_rule.json", "w") as f:
        json.dump({"written_at": time.strftime("%Y-%m-%d %H:%M:%S"), "rule": __doc__.strip(),
                   "plan": PLAN, "backbones": list(BACKBONES), "param_cap": PARAM_CAP,
                   "param_counts": param_counts}, f, indent=2)

    runs = []
    for d in SELECTION_DOMAINS:
        cfg, plan = DOMAIN_CONFIGS[d], PLAN[d]
        for b in BACKBONES:
            for seed in plan["seeds"]:
                print(f"\n=== {d} | {b} | seed {seed} ===")
                r = run_transfer(cfg, b, seed, train_stride=plan["train_stride"],
                                 max_epochs=plan["max_epochs"], target_only=False, verbose=False)
                r["domain"] = d
                runs.append(r)
                print(f"  val_mse_norm={r['source_val']['mse_norm']:.5f} best_epoch={r['source_training']['best_epoch']}"
                      f" train_s={r['source_training']['train_seconds']}")
                with open(OUT_DIR / "runs.json", "w") as f:  # rewrite after each run so partial results survive
                    json.dump(runs, f, indent=2)

    summary = summarize(runs)
    selection = select(summary, runs)
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump({"summary": summary, "selection": selection}, f, indent=2)
    print(json.dumps(selection, indent=2))
    print(f"\nSaved: {OUT_DIR / 'runs.json'}, {OUT_DIR / 'summary.json'}")
