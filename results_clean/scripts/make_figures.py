"""
Qualitative + summary figures for results_clean/figures/. Inference only with seed-0 checkpoints; nothing written
under experiments/. Bar charts read results_clean/_tables.json (written by build_tables.py).

  set OMP_NUM_THREADS=2 && venv\\Scripts\\python -m results_clean.scripts.make_figures
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.evaluate import denorm, load_model_checkpoint, predict
from src.experiment import prepare_pair
from src.run_experiments import DOMAIN_CONFIGS
from src.utils.metrics import evaluate_all
from results_clean.scripts.build_tables import (ADA, C1, C10, DOMAINS, NAIVE, PERSIST, SO_S, SO_T, TO, UNIT)

torch.set_num_threads(2)
OUT = Path("results_clean/figures")
OUT.mkdir(parents=True, exist_ok=True)
EXP = Path("experiments")
FINAL = EXP / "final_3seed" / "tcn"
T = json.load(open("results_clean/_tables.json"))
ROWS = T["rows"]

# Okabe-Ito (colour-blind safe), validated with the dataviz validator; one colour per method everywhere.
INK, INK2, GRID = "#111111", "#444444", "#d9d9d9"
COLOR = {"actual": INK, NAIVE: "#999999", "persist1": "#CC79A7", "persist2": "#CC79A7",
         SO_S: "#E69F00", SO_T: "#56B4E9", C10: "#0072B2", C1: "#009E73", ADA: "#D55E00", TO: "#444444"}
HATCH = {"persist2": "//"}
LINE = {"actual": dict(ls="-", lw=2.6), SO_S: dict(ls=(0, (6, 2)), lw=2.2), SO_T: dict(ls=(0, (2, 1.5)), lw=2.4),
        C10: dict(ls="-", lw=2.2)}
SHORT = {NAIVE: "naive\ntrain-mean", SO_S: "source-only\nsource stats", SO_T: "source-only\ntarget stats",
         C10: "CORAL\nλ=10", C1: "CORAL λ=1\n(sensitivity)", ADA: "AdaRNN-style\nλ=10", TO: "target-only\n(labels)"}
PSHORT = {"seasonal-naive lag 24 (same hour yesterday)": "lag 24\n(yesterday)",
          "seasonal-naive lag 168 (same hour last week)": "lag 168\n(last week)",
          "persistence (last observed value repeated)": "persistence\n(last value)",
          "mean of last 5 inputs (repeated)": "mean of\nlast 5"}
UNIT_LBL = {"energy": "MW", "climate": "°C", "finance": "log-return", "finance_vol": "|log-return|"}
TITLE = {"energy": "Energy (AEP → COMED)", "climate": "Climate (Shimla → Delhi)",
         "finance": "Finance (Nifty IT → Pharma, log-return)", "finance_vol": "Finance vol (|log-return|)"}

plt.rcParams.update({
    "font.size": 16, "axes.titlesize": 18, "axes.labelsize": 16, "xtick.labelsize": 14, "ytick.labelsize": 14,
    "legend.fontsize": 14, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "savefig.dpi": 200, "savefig.bbox": "tight",
})


def settings(d):
    """Canonical order with the role used for colour."""
    out = [(NAIVE, NAIVE)]
    for i, (lab, _, _) in enumerate(PERSIST[d]):
        out.append((lab, "persist1" if i == 0 else "persist2"))
    return out + [(s, s) for s in (SO_S, SO_T, C10, C1, ADA, TO)]


def dates_of(d):
    cfg = DOMAIN_CONFIGS[d]
    return pd.read_csv(cfg["target"]["csv_path"], index_col=0, parse_dates=True).index


# ------------------------------------------------------------- inference (seed 0)
PRED, CHECK = {}, []


def infer(d):
    cfg = DOMAIN_CONFIGS[d]
    src, tgt_src, tgt_tgt, series = prepare_pair(cfg)
    s_st, t_st = src["norm_stats"], tgt_tgt["norm_stats"]
    so, _ = load_model_checkpoint(FINAL / "checkpoints" / f"{d}_seed0_source_only.pt")
    co, ck = load_model_checkpoint(EXP / "coral" / d / "seed0.pt")
    assert ck["lambda"] == 10.0 and ck["seed"] == 0
    Y = denorm(tgt_tgt["Y_test"].astype(np.float64), t_st)
    P = {SO_S: denorm(predict(so, tgt_src["X_test"]).astype(np.float64), s_st),
         SO_T: denorm(predict(so, tgt_tgt["X_test"]).astype(np.float64), t_st),
         C10: denorm(predict(co, tgt_tgt["X_test"]).astype(np.float64), t_st)}
    # check: recomputed seed-0 metrics must reproduce the JSONs (same checkpoints, same windows)
    jf = json.load(open(FINAL / "runs" / f"{d}_seed0.json"))
    jc = json.load(open(EXP / "coral" / d / "seed0.json"))
    ref = {SO_S: jf["source_only_on_target_test"]["source_stats"], SO_T: jf["source_only_on_target_test"]["target_stats"],
           C10: jc["coral_on_target_test"]["target_stats"]}
    for k, p in P.items():
        m = evaluate_all(Y, p)
        CHECK.append(f"[{d}] seed0 {k}: recomputed MAE {m['MAE']} RMSE {m['RMSE']} | JSON MAE {ref[k]['MAE']} RMSE {ref[k]['RMSE']}"
                     f" | n windows {len(Y)}")
    n = len(series)
    test_start = int(n * 0.85)
    idx = test_start + np.arange(len(Y))[:, None] + cfg["input_len"] + np.arange(cfg["horizon"])[None, :]
    PRED[d] = dict(Y=Y, P=P, t=dates_of(d).values[idx])  # t[i, j] = timestamp of window i, horizon step j


# ------------------------------------------------------------- (a) forecast vs actual
SPAN = {"energy": (168, "7 days, hourly"), "climate": (60, "60 days"), "finance": (60, "60 trading days"),
        "finance_vol": (60, "60 trading days")}


def fig_a(d):
    r = PRED[d]
    n, desc = SPAN[d]
    t = r["t"][:n, 0]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(t, r["Y"][:n, 0], color=COLOR["actual"], label="actual", **LINE["actual"])
    for k, lab in [(SO_S, "source-only, source stats"), (SO_T, "source-only, target-train stats"), (C10, "CORAL λ=10")]:
        ax.plot(t, r["P"][k][:n, 0], color=COLOR[k], label=lab, **LINE[k])
    ax.set_ylabel(f"{UNIT_LBL[d]}")
    ax.set_xlabel("date (target test set)" if d != "energy" else "date-time (target test set, hourly)")
    ax.set_title(f"{TITLE[d]}\n1-step-ahead forecast, first {desc} of the target test set, seed 0", loc="left")
    fig.autofmt_xdate()
    ax.legend(ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.42 if d == "energy" else -0.36))
    fig.savefig(OUT / f"a_forecast_{d}.png")
    plt.close(fig)


# ------------------------------------------------------------- (b) per-domain bars
def bar(ax, x, h, err, role, width=0.8, label=None):
    ax.bar(x, h, width, yerr=err if err and not np.isnan(err) else None, color=COLOR[role],
           hatch=HATCH.get(role), edgecolor="white" if role not in HATCH else INK2, linewidth=0.8,
           error_kw=dict(ecolor=INK, capsize=4, lw=1.4), label=label)


def fig_b(d):
    rows = ROWS[d]
    naive = rows[NAIVE]["MAE"]["mean"]
    S = settings(d)
    fig, ax = plt.subplots(figsize=(11, 0.62 * len(S) + 1.6))
    xmax = 0
    for i, (lab, role) in enumerate(S):
        r = rows[lab]["MAE"]
        h, e = 100 * r["mean"] / naive, 100 * r["std"] / naive
        y = len(S) - 1 - i
        ax.barh(y, h, 0.75, xerr=None if np.isnan(e) else e, color=COLOR[role], hatch=HATCH.get(role),
                edgecolor="white" if role not in HATCH else INK2, linewidth=0.8, error_kw=dict(ecolor=INK, capsize=4, lw=1.4))
        end = h + (0 if np.isnan(e) else e)
        ax.text(end + 1.2, y, f"{h:.1f}" + ("" if np.isnan(e) else f" ± {e:.1f}"), va="center", fontsize=14, color=INK)
        xmax = max(xmax, end)
    ax.axvline(100, color=INK2, lw=1.2, ls="--")
    ax.set_yticks(range(len(S)), [SHORT.get(l, PSHORT.get(l, l)).replace("\n", " ") for l, _ in S][::-1])
    ax.set_xlabel("target-test MAE as % of naive train-mean (lower is better)")
    ax.set_xlim(0, xmax * 1.2)
    ax.set_title(f"{TITLE[d]}: MAE % of naive (mean ± std, 3 seeds)", loc="left")
    ax.grid(axis="y", visible=False)
    fig.savefig(OUT / f"b_bars_{d}.png")
    plt.close(fig)


# ------------------------------------------------------------- (c) energy by hour of day
def fig_c():
    r = PRED["energy"]
    hours = pd.DatetimeIndex(r["t"].ravel()).hour.values.reshape(r["t"].shape)
    fig, ax = plt.subplots(figsize=(10, 5))
    for k, lab in [(SO_S, "source-only, source stats"), (SO_T, "source-only, target-train stats"), (C10, "CORAL λ=10")]:
        ae = np.abs(r["P"][k] - r["Y"])
        mae_h = [ae[hours == h].mean() for h in range(24)]
        ax.plot(range(24), mae_h, color=COLOR[k], marker="o", ms=8, label=lab, **{kk: v for kk, v in LINE[k].items()})
    ax.set_xticks(range(0, 24, 3))
    ax.set_xlabel("hour of day of the forecast target (local time in the PJM data)")
    ax.set_ylabel("MAE (MW)")
    ax.set_ylim(bottom=0)
    ax.set_title("Energy (COMED target test): MAE by hour of day, all 24 horizon steps, seed 0", loc="left")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, fontsize=13)
    fig.savefig(OUT / "c_energy_mae_by_hour.png")
    plt.close(fig)


# ------------------------------------------------------------- (d) one panel across domains
def fig_d():
    S_common = [(NAIVE, NAIVE), ("persist1", "persist1"), (SO_S, SO_S), (SO_T, SO_T), (C10, C10), (C1, C1), (ADA, ADA), (TO, TO)]
    fig, ax = plt.subplots(figsize=(14, 6))
    w = 0.1
    for di, d in enumerate(DOMAINS):
        rows = ROWS[d]
        naive = rows[NAIVE]["MAE"]["mean"]
        for si, (lab, role) in enumerate(S_common):
            key = PERSIST[d][0][0] if lab == "persist1" else lab
            r = rows[key]["MAE"]
            x = di + (si - (len(S_common) - 1) / 2) * w
            bar(ax, x, 100 * r["mean"] / naive, 100 * r["std"] / naive, role, width=w * 0.92,
                label=(SHORT.get(lab, "persistence / seasonal-naive†").replace("\n", " ") if di == 0 else None))
    ax.axhline(100, color=INK2, lw=1.2, ls="--")
    ax.set_xticks(range(4), [TITLE[d].replace(" (", "\n(") for d in DOMAINS], fontsize=13)
    ax.set_ylabel("target-test MAE, % of naive train-mean")
    ax.set_title("Error relative to the naive train-mean baseline, all domains (mean ± std over 3 seeds)", loc="left")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.17), fontsize=13)
    fig.text(0.01, -0.17, "† energy: same hour yesterday (lag 24); climate, finance, finance vol: last observed value repeated",
             fontsize=12, color=INK2)
    fig.savefig(OUT / "d_all_domains_pct_naive.png")
    plt.close(fig)


# ------------------------------------------------------------- (e) scatter
def fig_e():
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))
    for ax, d in zip(axes, ["energy", "climate"]):
        r = PRED[d]
        y, p = r["Y"].ravel(), r["P"][C10].ravel()
        ax.scatter(y, p, s=4 if d == "energy" else 10, alpha=0.06 if d == "energy" else 0.35, color=COLOR[C10],
                   edgecolors="none", rasterized=True)
        lo, hi = min(y.min(), p.min()), max(y.max(), p.max())
        ax.plot([lo, hi], [lo, hi], color=INK, lw=1.8, ls="--", label="y = x")
        ax.set_xlabel(f"actual ({UNIT_LBL[d]})")
        ax.set_ylabel(f"CORAL λ=10 prediction ({UNIT_LBL[d]})")
        ax.set_title(f"{TITLE[d]}\n{len(r['Y'])} windows × {r['Y'].shape[1]} steps, seed 0", loc="left", fontsize=16)
        ax.set_aspect("equal")
        ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "e_scatter_coral_energy_climate.png")
    plt.close(fig)


def main():
    for d in DOMAINS:
        infer(d)
        fig_a(d)
        fig_b(d)
    fig_c()
    fig_d()
    fig_e()
    (OUT / "_inference_check.txt").write_text("\n".join(CHECK) + "\n", encoding="utf-8")
    print("\n".join(CHECK))


if __name__ == "__main__":
    main()
