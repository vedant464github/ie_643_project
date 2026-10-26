"""
Cross-method summary and report figures. Only results that exist on disk are included; missing cells are "not run".

  python -m src.report            # experiments/summary_all.md (+ .json) and report/figures/*.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.evaluate import predict, denorm
from src.experiment import prepare_pair, target_baselines
from src.models.backbone import build_model
from src.run_experiments import DOMAIN_CONFIGS

DOMAINS = ["finance", "finance_vol", "climate", "energy"]
FIG = Path("report/figures")
FINAL = Path("experiments/final_3seed/tcn")

# (label, results dir, json key with target-test metrics, gap key) - adaptation methods, in staircase order
METHODS = [
    ("CORAL λ=10", Path("experiments/coral"), "coral_on_target_test", "after_coral"),
    ("CORAL λ=1 (sensitivity)", Path("experiments/coral_lambda1"), "coral_on_target_test", "after_coral"),
    ("AdaRNN-style λ=10", Path("experiments/adarnn"), "adarnn_on_target_test", "after_adarnn"),
]
STAIRCASE = ["source-only, source stats", "source-only, target-train stats", "CORAL λ=10",
             "CORAL λ=1 (sensitivity)", "AdaRNN-style λ=10", "target-only (uses target labels)"]

# reference palette (dataviz skill, light mode): references in neutral gray, methods in categorical slots 1-3
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
GRAY, GRAY_DARK = "#a3a29c", "#6f6e69"
COLORS = {"source-only, source stats": GRAY, "source-only, target-train stats": GRAY_DARK,
          "CORAL λ=10": "#2a78d6", "CORAL λ=1 (sensitivity)": "#eb6834", "AdaRNN-style λ=10": "#1baf7a",
          "target-only (uses target labels)": GRAY}
SEED_MARKERS = {0: "o", 1: "s", 2: "^"}

plt.rcParams.update({
    "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "lines.linewidth": 1.2, "savefig.dpi": 300,
})
COL_W = 3.4  # inches, single column


def load_runs(path, domain):
    return [json.load(open(p)) for p in sorted((path / domain).glob("seed*.json"))] if path.exists() else []


def final_runs(domain):
    return [json.load(open(p)) for p in sorted(FINAL.glob(f"runs/{domain}_seed*.json"))]


def ms(vals):
    v = np.array(vals, dtype=float)
    if len(v) == 0:
        return None
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else None, "n": int(len(v)),
            "per_seed": v.tolist()}


def collect():
    out = {}
    for d in DOMAINS:
        fin = final_runs(d)
        rows = {
            "source-only, source stats": (fin, lambda r: r["source_only_on_target_test"]["source_stats"]),
            "source-only, target-train stats": (fin, lambda r: r["source_only_on_target_test"]["target_stats"]),
        }
        gaps = {"source-only": ms([r["gap_diagnostic"]["before_source_only"]["mmd2_rbf"]
                                   for r in load_runs(METHODS[0][1], d)])}
        gap_per_seed = {"source-only": {r["seed"]: r["gap_diagnostic"]["before_source_only"]["mmd2_rbf"]
                                        for r in load_runs(METHODS[0][1], d)}}
        method_runs = {}
        for label, path, key, gkey in METHODS:
            runs = load_runs(path, d)
            method_runs[label] = runs
            rows[label] = (runs, lambda r, key=key: r[key]["target_stats"])
            gaps[label] = ms([r["gap_diagnostic"][gkey]["mmd2_rbf"] for r in runs])
            gap_per_seed[label] = {r["seed"]: r["gap_diagnostic"][gkey]["mmd2_rbf"] for r in runs}
        rows["target-only (uses target labels)"] = (fin, lambda r: r["target_only_on_target_test"]["target_stats"])
        table = {name: ({m: ms([get(r)[m] for r in runs]) for m in ("MAE", "RMSE")} | {"seeds": [r["seed"] for r in runs]})
                 for name, (runs, get) in rows.items()}
        out[d] = {"table": table, "mmd2": gaps, "mmd2_per_seed": gap_per_seed, "fin": fin, "methods": method_runs,
                  "naive": target_baselines(DOMAIN_CONFIGS[d])["naive_train_mean"]["target_stats"]}
    return out


def cell(m):
    if m is None:
        return "not run"
    sd = "n/a" if m["std"] is None else f"{m['std']:.2g}"
    return f"{m['mean']:.5g} ± {sd} (n={m['n']})"


def paired_diffs(e, methods=("CORAL λ=10", "AdaRNN-style λ=10"), metrics=("MAE", "RMSE")):
    """Per-seed method minus source-only (target-train stats), matched by seed. Descriptive only."""
    base = {r["seed"]: r["source_only_on_target_test"]["target_stats"] for r in e["fin"]}
    out = {}
    for label in methods:
        runs = {r["seed"]: r for r in e["methods"][label]}
        key = next(k for k in next(iter(runs.values())) if k.endswith("_on_target_test")) if runs else None
        seeds = sorted(set(base) & set(runs))
        out[label] = {m: {"seeds": seeds, "diff": [runs[s][key]["target_stats"][m] - base[s][m] for s in seeds]}
                      for m in metrics}
        for m in metrics:
            out[label][m] |= ms(out[label][m]["diff"]) or {}
    return out


def write_summary(data):
    status = Path("experiments/adarnn/status.json")
    st = json.load(open(status)) if status.exists() else None
    md = ["# Summary: TCN adaptation staircase, all domains", "",
          "Target test set, original scale, mean ± sample std (ddof=1) over seeds; n = seeds in the cell; "
          "'not run' = no result on disk. All adaptation rows normalise target inputs with target-train stats and "
          "use target-train INPUTS only (no target labels). Target-only uses target labels (reference only). "
          "CORAL λ=10 is the pre-registered final; CORAL λ=1 is a sensitivity check and does not replace it. "
          "finance is raw log-returns (no learnable signal) and finance_vol is a negative control: lead with RMSE "
          "there. MAPE/sMAPE are not reported.", ""]
    nr_p = Path("experiments/not_run.json")
    if nr_p.exists():
        nr = json.load(open(nr_p, encoding="utf-8"))
        md += ["**Not run:** " + "; ".join(f"{k}: {', '.join(v)}" for k, v in nr["not_run"].items()) + ".",
               f"Reason: {nr['reason']} Context: {nr['context']}.", ""]
    elif st:
        not_run = [k for k, v in st["runs"].items() if v != "done"]
        md += [f"AdaRNN-style hard stop {st['deadline']}: "
               + (f"not run: {', '.join(not_run)}." if not_run else "all 12 runs finished before the stop."), ""]
    for d in DOMAINS:
        e = data[d]
        cols = ("RMSE", "MAE") if d == "finance_vol" else ("MAE", "RMSE")
        md += [f"## {d}", "", f"| setting | {cols[0]} | {cols[1]} |", "|---|---|---|",
               f"| naive train-mean (target-train stats) | {e['naive'][cols[0]]:.5g} (n=1) | "
               f"{e['naive'][cols[1]]:.5g} (n=1) |"]
        for name in STAIRCASE:
            row = e["table"][name]
            md.append(f"| {name} | {cell(row[cols[0]])} | {cell(row[cols[1]])} |")
        md += ["", "MMD² (RBF, median heuristic) between source-val and target-val features, label-free:", "",
               "| model | MMD² |", "|---|---|"]
        md += [f"| {k}{' (before adaptation)' if k == 'source-only' else ''} | {cell(v)} |"
               for k, v in e["mmd2"].items()]
        md.append("")
    pd_energy = paired_diffs(data["energy"])
    md += ["## energy: paired differences vs source-only (target-train stats)", "",
           "Descriptive only; not a success criterion. Per seed, method minus source-only (target-train stats) on "
           "the same seed's target test set (negative = lower error than the target-stats row); mean and sample "
           "std (ddof=1) of the per-seed differences.", "",
           "| method | metric | seed 0 | seed 1 | seed 2 | mean ± std of diff |", "|---|---|---|---|---|---|"]
    for label, mets in pd_energy.items():
        for m, v in mets.items():
            per = {s: f"{d:+.4g}" for s, d in zip(v["seeds"], v["diff"])}
            sd = "n/a" if v.get("std") is None else f"{v['std']:.3g}"
            md.append(f"| {label} | {m} | " + " | ".join(per.get(s, "not run") for s in (0, 1, 2))
                      + (f" | {v['mean']:+.4g} ± {sd} (n={v['n']}) |" if v.get("n") else " | not run |"))
    md.append("")
    md += ["Figures: report/figures/. Per-method details: experiments/coral/summary.md (λ=10), "
           "experiments/coral/lambda_selection.json, experiments/adarnn/tdc.json, experiments/adarnn/status.json.", ""]
    Path("experiments/summary_all.md").write_text("\n".join(md), encoding="utf-8")
    js = {d: {"table": data[d]["table"], "mmd2": data[d]["mmd2"], "naive_train_mean": data[d]["naive"]}
          for d in DOMAINS}
    js["energy"]["paired_diff_vs_target_stats_descriptive"] = pd_energy
    Path("experiments/summary_all.json").write_text(json.dumps(js, indent=2), encoding="utf-8")


def fig_staircase(data):
    for d in DOMAINS:
        e = data[d]
        names = [n for n in STAIRCASE if e["table"][n]["MAE"] is not None]
        fig, axes = plt.subplots(1, 2, figsize=(COL_W, 0.32 * len(names) + 0.9), sharey=True)
        y = np.arange(len(names))[::-1]
        for ax, m in zip(axes, ("MAE", "RMSE")):
            means = [e["table"][n][m]["mean"] for n in names]
            stds = [e["table"][n][m]["std"] or 0 for n in names]
            ax.barh(y, means, xerr=stds, height=0.62, color=[COLORS[n] for n in names], edgecolor=SURFACE,
                    linewidth=1, error_kw={"elinewidth": 0.8, "capsize": 2, "ecolor": INK})
            ax.set_title(m, color=INK)
            ax.grid(axis="y", visible=False)
            ax.set_xlim(0, max(np.array(means) + np.array(stds)) * 1.12)
        axes[0].set_yticks(y, [n.replace("(sensitivity)", "(sens.)").replace(" (uses target labels)", "*")
                               for n in names])
        n_txt = ", ".join(sorted({str(e["table"][n]["MAE"]["n"]) for n in names}))
        fig.suptitle(f"{d}: target test, mean ± std (n={n_txt} seeds)", fontsize=7.5, color=INK, x=0.02, ha="left")
        fig.text(0.02, 0.01, f"naive train-mean: MAE {e['naive']['MAE']:.4g}, RMSE {e['naive']['RMSE']:.4g}. "
                 "*target-only uses target labels.", fontsize=6, color=INK2)
        fig.tight_layout(rect=(0, 0.05, 1, 0.95))
        fig.savefig(FIG / f"staircase_{d}.png")
        plt.close(fig)


def _load_ckpt(path):
    ck = torch.load(path, map_location="cpu")
    m = build_model(ck["backbone"], ck["input_len"], ck["horizon"], **ck.get("model_kwargs", {}))
    m.load_state_dict(ck["state_dict"])
    return m.eval()


def fig_forecast():
    spans = {"finance": 90, "finance_vol": 90, "climate": 120, "energy": 168}
    units = {"finance": "log-return", "finance_vol": "|log-return|", "climate": "°C", "energy": "MW"}
    fig, axes = plt.subplots(4, 1, figsize=(COL_W, 6.2))
    for ax, d in zip(axes, DOMAINS):
        coral_ck = Path("experiments/coral") / d / "seed0.pt"
        so_ck = FINAL / "checkpoints" / f"{d}_seed0_source_only.pt"
        if not coral_ck.exists() or not so_ck.exists():
            ax.set_visible(False)
            continue
        _, _, tgt, _ = prepare_pair(DOMAIN_CONFIGS[d])
        st, n = tgt["norm_stats"], spans[d]
        X, Y = tgt["X_test"][:n], tgt["Y_test"][:n]
        h = DOMAIN_CONFIGS[d]["horizon"]
        t = np.arange(n) + DOMAIN_CONFIGS[d]["input_len"] + h - 1  # index of window i's last target in the test split
        ax.plot(t, denorm(Y[:, -1], st), color=INK, lw=1.0, label="actual")
        ax.plot(t, denorm(predict(_load_ckpt(so_ck), X)[:, -1], st), color=GRAY_DARK, lw=1.0, ls="--",
                label="source-only, target-train stats")
        ax.plot(t, denorm(predict(_load_ckpt(coral_ck), X)[:, -1], st), color=COLORS["CORAL λ=10"], lw=1.0,
                label="CORAL λ=10")
        step = "hour" if d == "energy" else "trading day" if d.startswith("finance") else "day"
        ax.set_title(f"{d} ({DOMAIN_CONFIGS[d]['target']['name']}): {h}-step-ahead, seed 0", loc="left", color=INK)
        ax.set_ylabel(units[d])
        ax.set_xlabel(f"{step} from start of target test", labelpad=1)
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.55), ncol=3, frameon=False, handlelength=1.5)
    fig.tight_layout(h_pad=0.6)
    fig.savefig(FIG / "forecast_vs_actual.png")
    plt.close(fig)


def fig_lambda(data):
    sel_p = Path("experiments/coral/lambda_selection.json")
    if not sel_p.exists():
        return
    sel = json.load(open(sel_p))
    lams = [0, 0.1, 1, 10]
    fig, axes = plt.subplots(2, 2, figsize=(COL_W, 3.2), sharex=True)
    for ax, d in zip(axes.flat, DOMAINS):
        r = [sel["table"][d][f"{l:g}"]["ratio_to_lambda0"] for l in lams]
        ax.axhline(1.05, color=INK2, lw=0.8, ls="--")
        ax.plot(range(4), r, color=INK, marker="o", ms=4, lw=1.2)
        ax.plot([3], [r[3]], marker="o", ms=6, color=COLORS["CORAL λ=10"], ls="none")
        ax.set_xticks(range(4), [f"{l:g}" for l in lams])
        ax.set_title(d + (" (cheap protocol)" if d == "energy" else ""), loc="left", color=INK)
        ax.set_ylim(min(0.98, min(r) - 0.005), max(1.06, max(r) + 0.005))
    for ax in axes[1]:
        ax.set_xlabel("λ")
    for ax in axes[:, 0]:
        ax.set_ylabel("src-val MSE / λ=0")
    fig.suptitle("λ selection, seed 0: dashed = 5% limit; blue = chosen λ=10", fontsize=7, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG / "lambda_selection.png")
    plt.close(fig)

    labels = ["source-only, target-train stats", "CORAL λ=1 (sensitivity)", "CORAL λ=10"]
    fig, axes = plt.subplots(2, 2, figsize=(COL_W, 3.2))
    for ax, d in zip(axes.flat, DOMAINS):
        m = "RMSE"
        rows = [(l, data[d]["table"][l][m]) for l in labels if data[d]["table"][l][m] is not None]
        x = np.arange(len(rows))
        for i, (l, v) in enumerate(rows):
            ax.errorbar(i, v["mean"], yerr=v["std"] or 0, fmt="o", ms=5, color=COLORS[l], ecolor=INK,
                        elinewidth=0.8, capsize=2)
        tick = {labels[0]: "target\nstats", labels[1]: "λ=1", labels[2]: "λ=10"}
        ax.set_xticks(x, [tick[l] for l, _ in rows])
        ax.set_title(d, loc="left", color=INK)
        ax.margins(x=0.25)
    fig.suptitle("Target-test RMSE, mean ± std over seeds\n(λ sensitivity; not used for selection)", fontsize=7,
                 x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG / "lambda_sensitivity.png")
    plt.close(fig)


def fig_mmd(data):
    fig, axes = plt.subplots(2, 2, figsize=(COL_W, 3.4))
    order = ["source-only"] + [m[0] for m in METHODS]
    short = {"source-only": "SO", "CORAL λ=10": "C10", "CORAL λ=1 (sensitivity)": "C1", "AdaRNN-style λ=10": "Ada"}
    for ax, d in zip(axes.flat, DOMAINS):
        ps = data[d]["mmd2_per_seed"]
        present = [k for k in order if ps.get(k)]
        for seed, mk in SEED_MARKERS.items():
            xs = [i for i, k in enumerate(present) if seed in ps[k]]
            ys = [ps[present[i]][seed] for i in xs]
            ax.plot(xs, ys, color=GRID, lw=0.8, zorder=1)
            for i, yv in zip(xs, ys):
                k = present[i]
                ax.scatter(i, yv, marker=mk, s=16, color=COLORS.get(k, GRAY_DARK), edgecolor=SURFACE,
                           linewidth=0.6, zorder=2, label=f"seed {seed}" if i == 0 else None)
        ax.set_xticks(range(len(present)), [short[k] for k in present])
        ax.set_title(d, loc="left", color=INK)
        ax.set_ylabel("MMD²")
    h = [plt.Line2D([], [], marker=m, ls="none", color=INK2, ms=4, label=f"seed {s}") for s, m in SEED_MARKERS.items()]
    fig.legend(handles=h, loc="upper left", ncol=3, frameon=False, bbox_to_anchor=(0.0, 0.955))
    fig.suptitle("MMD², source-val vs target-val features (inputs only)", fontsize=7, x=0.02, ha="left")
    fig.text(0.02, 0.005, "SO = source-only (before); C10/C1 = CORAL λ=10/1; Ada = AdaRNN-style",
             fontsize=6, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.89))
    fig.savefig(FIG / "mmd_per_seed.png")
    plt.close(fig)


def fig_training(data):
    fig, axes = plt.subplots(2, 2, figsize=(COL_W, 3.4))
    for ax, d in zip(axes.flat, DOMAINS):
        fin = [r for r in data[d]["fin"] if r["seed"] == 0]
        if fin:
            v = fin[0]["source_training"]["val_mse"]
            ax.plot(range(1, len(v) + 1), v, color=GRAY_DARK, label="source-only")
        for label, *_ in METHODS:
            runs = [r for r in data[d]["methods"][label] if r["seed"] == 0]
            if runs:
                v = runs[0]["training"]["val_mse"]
                ax.plot(range(1, len(v) + 1), v, color=COLORS[label], label=label.replace(" (sensitivity)", ""),
                        marker="o", ms=2)
                if label.startswith("AdaRNN"):
                    ax.axvline(runs[0]["training"]["pretrain_epochs"] + 0.5, color=COLORS[label], lw=0.6, ls=":")
        ax.set_title(d, loc="left", color=INK)
        ax.set_xlabel("epoch")
        ax.set_ylabel("source-val MSE (norm)")
    seen = {}
    for ax in axes.flat:
        for hd, lb in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(lb, hd)
    fig.legend(seen.values(), seen.keys(), loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0.02, 1, 0.9))
    fig.text(0.02, 0.005, "seed 0; dotted line = end of AdaRNN-style MSE-only pretraining", fontsize=6, color=INK2)
    fig.savefig(FIG / "training_curves.png")
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    data = collect()
    write_summary(data)
    fig_staircase(data)
    fig_forecast()
    fig_lambda(data)
    fig_mmd(data)
    fig_training(data)
    print("wrote experiments/summary_all.md, report/figures/")
