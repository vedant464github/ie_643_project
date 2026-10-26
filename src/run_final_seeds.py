"""
3-seed runs with the chosen backbone, all domains, full training (no subsampling):
source-only (evaluated on target test with source stats and with target train stats) and target-only
(fixed epochs = that seed's source best epoch). Results per (domain, seed) go to separate files so runs can
be split across processes; `--summarize` aggregates them into mean ± std (ddof=1) of MAE and RMSE.

  python -m src.run_final_seeds --backbone gru --domains climate finance finance_vol --seeds 0 1 2
  python -m src.run_final_seeds --backbone gru --summarize
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.experiment import run_transfer, target_baselines
from src.run_experiments import DOMAIN_CONFIGS
from src.train import device_info

ALL_DOMAINS = ["finance", "finance_vol", "climate", "energy"]
SEEDS = [0, 1, 2]
SETTINGS = {
    "source_only_source_stats": lambda r: r["source_only_on_target_test"]["source_stats"],
    "source_only_target_stats": lambda r: r["source_only_on_target_test"]["target_stats"],
    "target_only": lambda r: r["target_only_on_target_test"]["target_stats"],
    "source_only_on_source_test": lambda r: r["source_test"],
}

def out_dir(backbone):
    return Path("experiments/final_3seed") / backbone

def run(backbone, domains, seeds):
    d_out = out_dir(backbone)
    (d_out / "runs").mkdir(parents=True, exist_ok=True)
    (d_out / "checkpoints").mkdir(parents=True, exist_ok=True)
    for d in domains:
        cfg = DOMAIN_CONFIGS[d]
        for seed in seeds:
            print(f"\n=== {backbone} | {d} | seed {seed} ===")
            r = run_transfer(cfg, backbone, seed,
                             checkpoint_prefix=str(d_out / "checkpoints" / f"{d}_seed{seed}"))
            r["domain"] = d
            with open(d_out / "runs" / f"{d}_seed{seed}.json", "w") as f:
                json.dump(r, f, indent=2)

def summarize(backbone):
    d_out = out_dir(backbone)
    summary = {"backbone": backbone, "std": "sample std across seeds (ddof=1)", "device_info": device_info(),
               "domains": {}}
    for d in ALL_DOMAINS:
        runs = [json.load(open(p)) for p in sorted((d_out / "runs").glob(f"{d}_seed*.json"))]
        if not runs:
            continue
        entry = {"seeds": [r["seed"] for r in runs], "params": runs[0]["params"],
                 "best_epochs": [r["source_training"]["best_epoch"] for r in runs],
                 "baselines": target_baselines(DOMAIN_CONFIGS[d])}
        for name, get in SETTINGS.items():
            entry[name] = {}
            for metric in ("MAE", "RMSE"):
                vals = np.array([get(r)[metric] for r in runs], dtype=float)
                entry[name][metric] = {"mean": float(vals.mean()),
                                       "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                                       "per_seed": vals.tolist()}
        summary["domains"][d] = entry
    with open(d_out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Saved: {d_out / 'summary.json'}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--domains", nargs="+", default=ALL_DOMAINS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads for this process")
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.summarize:
        summarize(args.backbone)
    else:
        run(args.backbone, args.domains, args.seeds)
