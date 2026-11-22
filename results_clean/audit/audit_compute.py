"""
Audit computations for results_clean/audit/. Inference only with saved checkpoints; no training; nothing is written
outside results_clean/audit/.

  set OMP_NUM_THREADS=2 && venv\\Scripts\\python -m results_clean.audit.audit_compute
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.data.prepare import chronological_split, apply_norm
from src.evaluate import denorm, load_model_checkpoint, predict, seasonal_naive
from src.experiment import prepare_pair, load_series
from src.run_experiments import DOMAIN_CONFIGS
from src.utils.metrics import evaluate_all

torch.set_num_threads(2)
OUT = Path("results_clean/audit")
EXP = Path("experiments")
FINAL = EXP / "final_3seed" / "tcn"
DOMAINS = ["energy", "climate", "finance", "finance_vol"]
SEEDS = [0, 1, 2]
TF, VF = 0.7, 0.15  # prepare.chronological_split defaults

# canonical row -> (checkpoint path fn, stats used at the input end / output end, JSON path fn, JSON getter)
MODEL_ROWS = {
    "source-only, source stats": (lambda d, s: FINAL / "checkpoints" / f"{d}_seed{s}_source_only.pt", "source",
                                  lambda d, s: FINAL / "runs" / f"{d}_seed{s}.json",
                                  lambda r: r["source_only_on_target_test"]["source_stats"]),
    "source-only, target-train stats": (lambda d, s: FINAL / "checkpoints" / f"{d}_seed{s}_source_only.pt", "target",
                                        lambda d, s: FINAL / "runs" / f"{d}_seed{s}.json",
                                        lambda r: r["source_only_on_target_test"]["target_stats"]),
    "CORAL lambda=10": (lambda d, s: EXP / "coral" / d / f"seed{s}.pt", "target",
                        lambda d, s: EXP / "coral" / d / f"seed{s}.json",
                        lambda r: r["coral_on_target_test"]["target_stats"]),
    "CORAL lambda=1 (sensitivity)": (lambda d, s: EXP / "coral_lambda1" / d / f"seed{s}.pt", "target",
                                     lambda d, s: EXP / "coral_lambda1" / d / f"seed{s}.json",
                                     lambda r: r["coral_on_target_test"]["target_stats"]),
    "AdaRNN-style lambda=10": (lambda d, s: EXP / "adarnn" / d / f"seed{s}.pt", "target",
                               lambda d, s: EXP / "adarnn" / d / f"seed{s}.json",
                               lambda r: r["adarnn_on_target_test"]["target_stats"]),
    "target-only (labels, reference)": (lambda d, s: FINAL / "checkpoints" / f"{d}_seed{s}_target_only.pt", "target",
                                        lambda d, s: FINAL / "runs" / f"{d}_seed{s}.json",
                                        lambda r: r["target_only_on_target_test"]["target_stats"]),
}


def raw_test_truth(series, L, H):
    """Original-unit targets straight from the raw series (float64), aligned with prepare_domain test windows."""
    v = np.asarray(series, dtype=np.float64)
    n = len(v)
    test_start = int(n * (TF + VF))
    nw = n - test_start - L - H + 1
    idx = test_start + np.arange(nw)[:, None] + L + np.arange(H)[None, :]
    return v[idx], idx


def m64(y, p):
    e = p - y
    return float(np.mean(np.abs(e))), float(np.sqrt(np.mean(e ** 2)))


def load_canonical():
    rows = {}
    with open("results_clean/canonical.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[(r["domain"], r["setting"])] = r
    return rows


CANON = load_canonical()
RES = {"round_trip": {}, "recompute": {}, "deviations": {}, "z_share": {}, "error_structure": {}, "bakeoff": {}}
PREDS = {}  # (domain, row, seed) -> original-unit predictions (float64)

for d in DOMAINS:
    cfg = DOMAIN_CONFIGS[d]
    L, H = cfg["input_len"], cfg["horizon"]
    src, tgt_src, tgt_tgt, tseries = prepare_pair(cfg)
    s_st, t_st = src["norm_stats"], tgt_tgt["norm_stats"]
    sseries = load_series(cfg, "source")

    # ---------------- 1. round trip: normalise -> denormalise, per split, per stats pairing
    rt = {}
    for name, ser, st in (("source series, source-train stats", sseries, s_st),
                          ("target series, source-train stats", tseries, s_st),
                          ("target series, target-train stats", tseries, t_st)):
        for split, raw in zip(("train", "val", "test"), chronological_split(np.asarray(ser, dtype=np.float64))):
            back64 = denorm(apply_norm(raw, st), st)
            back32 = denorm(apply_norm(raw, st).astype(np.float32), st)  # the pipeline windows are float32
            rt[f"{name} | {split}"] = {
                "n": int(len(raw)), "max_abs_err_float64": float(np.max(np.abs(back64 - raw))),
                "max_abs_err_float32_windows": float(np.max(np.abs(back32 - raw))),
                "max_abs_value": float(np.max(np.abs(raw))),
            }
    # through the actual window arrays used for evaluation
    Y_raw, idx = raw_test_truth(tseries, L, H)
    rt["target test windows Y, target-train stats (pipeline arrays)"] = {
        "n": int(Y_raw.size), "max_abs_err_float32_windows": float(np.max(np.abs(denorm(tgt_tgt["Y_test"].astype(np.float64), t_st) - Y_raw)))}
    rt["target test windows Y, source-train stats (pipeline arrays)"] = {
        "n": int(Y_raw.size), "max_abs_err_float32_windows": float(np.max(np.abs(denorm(tgt_src["Y_test"].astype(np.float64), s_st) - Y_raw)))}
    RES["round_trip"][d] = {"stats": {"source_train": s_st, "target_train": t_st}, "checks": rt}

    # ---------------- 2. recompute seed-0 (and all seeds, for later sections) metrics from raw predictions
    rec = {}
    # deterministic baselines
    zero = np.zeros_like(tgt_tgt["Y_test"], dtype=np.float64)
    base = {"naive train-mean (target-train stats)": denorm(zero, t_st)}
    for per in cfg["seasonal_periods"]:
        _, p = seasonal_naive(tseries, L, H, per)
        lab = {1: "persistence (last observed value repeated)",
               24: "seasonal-naive lag 24 (same hour yesterday)",
               168: "seasonal-naive lag 168 (same hour last week)"}.get(per)
        if lab:
            base[lab] = p
    if d in ("finance", "finance_vol"):
        if d == "finance_vol":
            _, p = seasonal_naive(tseries, L, H, 1)
            base["persistence (last observed value repeated)"] = p
        Xt_raw = denorm(tgt_tgt["X_test"].astype(np.float64), t_st)
        base["mean of last 5 inputs (repeated)"] = np.repeat(Xt_raw[:, -5:].mean(1, keepdims=True), H, axis=1)
    for lab, p in base.items():
        mae, rmse = m64(Y_raw, p)
        c = CANON.get((d, lab))
        rec[lab] = {"seed": "deterministic", "stats_in": "n/a (no model)" if "naive train" not in lab else "target-train",
                    "stats_out": "target-train" if "naive train" in lab else "none (raw units)",
                    "MAE_recomputed": mae, "RMSE_recomputed": rmse,
                    "MAE_json_or_csv": float(c["MAE_mean"]) if c else None,
                    "RMSE_json_or_csv": float(c["RMSE_mean"]) if c else None,
                    "reference": "results_clean/canonical.csv (recomputed from prepare_pair, cross-checked vs summary.json)"}
    for row, (ckf, stats, jf, get) in MODEL_ROWS.items():
        for s in SEEDS:
            model, ck = load_model_checkpoint(ckf(d, s))
            st = s_st if stats == "source" else t_st
            X = tgt_src["X_test"] if stats == "source" else tgt_tgt["X_test"]
            P = denorm(predict(model, X).astype(np.float64), st)
            PREDS[(d, row, s)] = P
            if s == 0:
                mae, rmse = m64(Y_raw, P)
                Ypipe = tgt_src["Y_test"] if stats == "source" else tgt_tgt["Y_test"]
                pipe = evaluate_all(denorm(Ypipe, st), denorm(predict(model, X), st))  # = src.evaluate.evaluate_model
                j = get(json.load(open(jf(d, s))))
                rec[row] = {"seed": 0, "checkpoint": str(ckf(d, s)).replace("\\", "/"), "json": str(jf(d, s)).replace("\\", "/"),
                            "stats_in": f"{stats}-train", "stats_out": f"{stats}-train",
                            "mean_used": st["mean"], "std_used": st["std"],
                            "MAE_recomputed": mae, "RMSE_recomputed": rmse,
                            "MAE_pipeline_float32": pipe["MAE"], "RMSE_pipeline_float32": pipe["RMSE"],
                            "MAE_json_or_csv": j["MAE"], "RMSE_json_or_csv": j["RMSE"],
                            "abs_diff_MAE": abs(mae - j["MAE"]), "abs_diff_RMSE": abs(rmse - j["RMSE"])}
    RES["recompute"][d] = rec

    # ---------------- 3. share of target-test z-scored inputs with |z|>3 (seed-independent), both stats
    RES["z_share"][d] = {
        "source-train stats": float(np.mean(np.abs(tgt_src["X_test"]) > 3)),
        "target-train stats": float(np.mean(np.abs(tgt_tgt["X_test"]) > 3)),
        "source-train stats, max |z|": float(np.max(np.abs(tgt_src["X_test"]))),
        "target-train stats, max |z|": float(np.max(np.abs(tgt_tgt["X_test"]))),
        "source-train stats, mean z": float(np.mean(tgt_src["X_test"])),
        "source-train input z range": [float(src["X_train"].min()), float(src["X_train"].max())],
        "share outside source-train z range (source stats)": float(np.mean(
            (tgt_src["X_test"] < src["X_train"].min()) | (tgt_src["X_test"] > src["X_train"].max()))),
        "share outside source-train z range (target stats)": float(np.mean(
            (tgt_tgt["X_test"] < src["X_train"].min()) | (tgt_tgt["X_test"] > src["X_train"].max()))),
        "n_input_values": int(tgt_src["X_test"].size),
        "unique test-split steps |z|>3 (source stats)": float(np.mean(np.abs(apply_norm(
            chronological_split(np.asarray(tseries, dtype=np.float64))[2], s_st)) > 3)),
        "unique test-split steps |z|>3 (target stats)": float(np.mean(np.abs(apply_norm(
            chronological_split(np.asarray(tseries, dtype=np.float64))[2], t_st)) > 3)),
    }

    # ---------------- per-seed training facts for every model row (deviations section)
    dev = {}
    for row, (ckf, stats, jf, get) in MODEL_ROWS.items():
        per = []
        for s in SEEDS:
            r = json.load(open(jf(d, s)))
            if row.startswith("source-only"):
                tr = r["source_training"]; be, se = tr["best_epoch"], tr["stopped_epoch"]
                sv = r["source_val"]["mse_norm"]
            elif row.startswith("target-only"):
                t = r["target_only_on_target_test"]; be, se = t["epochs"], t["training"]["stopped_epoch"]
                sv = None
            else:
                tr = r["training"]; be, se = tr["best_epoch"], tr["stopped_epoch"]
                sv = r["source_val"]["mse_norm"]
            m = get(r)
            per.append({"seed": s, "MAE": m["MAE"], "RMSE": m["RMSE"], "best_epoch": be, "stopped_epoch": se,
                        "source_val_mse_norm": sv, "max_epochs": r.get("max_epochs") or r.get("protocol", {}).get("max_epochs")})
        dev[row] = per
    RES["deviations"][d] = dev

    # ---------------- 4. error structure (energy by hour of day, climate by month)
    if d in ("energy", "climate"):
        dates = pd.read_csv(cfg["target"]["csv_path"], index_col=0, parse_dates=True).index
        assert len(dates) == len(tseries)
        ts = pd.DatetimeIndex(dates.values[idx.ravel()])
        g = (ts.hour if d == "energy" else ts.month).values.reshape(idx.shape)
        groups = list(range(24)) if d == "energy" else list(range(1, 13))
        es = {"group": "hour of day of the forecast target" if d == "energy" else "calendar month of the forecast target",
              "groups": groups, "count_per_group": [int((g == k).sum()) for k in groups], "methods": {}}
        for row in ("source-only, source stats", "source-only, target-train stats", "CORAL lambda=10",
                    "target-only (labels, reference)"):
            mae_g, share, worst = [], [], []
            for s in SEEDS:
                e = PREDS[(d, row, s)] - Y_raw
                mae_g.append([float(np.abs(e[g == k]).mean()) if (g == k).any() else None for k in groups])
                se_g = np.array([float((e[g == k] ** 2).sum()) for k in groups])
                top = np.argsort(se_g)[::-1][:4]
                share.append(float(se_g[top].sum() / se_g.sum()))
                worst.append([groups[i] for i in top])
            mg = np.array([[np.nan if v is None else v for v in row_] for row_ in mae_g], dtype=float)
            es["methods"][row] = {"MAE_by_group_per_seed": mae_g,
                                  "MAE_by_group_mean": np.nanmean(mg, 0).tolist(),
                                  "MAE_by_group_std": np.nanstd(mg, 0, ddof=1).tolist(),
                                  "worst4_share_sq_err_per_seed": share, "worst4_groups_per_seed": worst,
                                  "uniform_share_of_4": 4 / len(groups)}
        RES["error_structure"][d] = es

    # ---------------- 6. naive val MSE (source val, normalised; train-mean = 0 there)
    RES["bakeoff"][d] = {"naive_val_mse_norm": float(np.mean(src["Y_val"].astype(np.float64) ** 2)),
                         "n_val_windows": int(len(src["Y_val"]))}
    print(f"done {d}", flush=True)

# bake-off ratios
bk = json.load(open(EXP / "bakeoff" / "summary.json"))["summary"]
ratios = {}
for d, v in bk.items():
    nv = RES["bakeoff"][d]["naive_val_mse_norm"]
    ratios[d] = {b: {"val_mse_mean": e["val_mse_norm"]["mean"], "val_mse_std": e["val_mse_norm"]["std"],
                     "n_seeds": e["n_seeds"], "params": e["params"], "ratio": e["val_mse_norm"]["mean"] / nv,
                     "ratio_std": e["val_mse_norm"]["std"] / nv} for b, e in v.items()}
RES["bakeoff"]["ratios"] = ratios

with open(OUT / "audit_numbers.json", "w", encoding="utf-8") as f:
    json.dump(RES, f, indent=1)

# ---------------- figures (Okabe-Ito, same colours per method as results_clean/scripts/make_figures.py)
STY = {"source-only, source stats": dict(color="#E69F00", ls=(0, (6, 2)), marker="o", label="source-only, source stats"),
       "source-only, target-train stats": dict(color="#56B4E9", ls=(0, (2, 1.5)), marker="s", label="source-only, target-train stats"),
       "CORAL lambda=10": dict(color="#0072B2", ls="-", marker="^", label="CORAL λ=10 (target-train stats)"),
       "target-only (labels, reference)": dict(color="#444444", ls="-.", marker="D", label="target-only (labels, reference)")}
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#d9d9d9", "axes.axisbelow": True, "savefig.dpi": 180, "savefig.bbox": "tight"})
for d, unit, fname in (("energy", "MW", "energy_mae_by_hour.png"), ("climate", "°C", "climate_mae_by_month.png")):
    es = RES["error_structure"][d]
    fig, ax = plt.subplots(figsize=(10, 5))
    for row, sty in STY.items():
        m = es["methods"][row]
        ax.errorbar(es["groups"], m["MAE_by_group_mean"], yerr=m["MAE_by_group_std"], lw=2, ms=6, capsize=2, **sty)
    ax.set_xticks(es["groups"])
    if d == "climate":
        ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_xlabel(es["group"])
    ax.set_ylabel(f"MAE ({unit}), mean ± std over seeds 0-2")
    ax.set_ylim(bottom=0)
    ttl = "Energy, COMED target test: MAE by hour of day" if d == "energy" else "Climate, Delhi target test: MAE by month"
    ax.set_title(ttl + " (same windows, all horizon steps)", loc="left")
    ax.legend(frameon=False, fontsize=10)
    fig.savefig(OUT / fname)
    plt.close(fig)
print("wrote", OUT / "audit_numbers.json")
