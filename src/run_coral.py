"""
CORAL on the TCN backbone (see CONSTRAINTS.md Amendments, 2026-10-25).

  python -m src.run_coral --grid                     # lambda in {0, 0.1, 1, 10}, seed 0, all domains
  python -m src.run_coral --select                   # writes experiments/coral/lambda_selection.json
  python -m src.run_coral --final --seeds 0 1 2      # chosen lambda; finance/finance_vol/climate seed 0 reused
  python -m src.run_coral --summarize                # experiments/coral/summary.{json,md}

Protocols: finance, finance_vol, climate full training (all windows, <=20 epochs, batch 64) throughout. Energy grid
uses the bake-off protocol (every 4th training window, <=15 epochs) for selection only; energy final seeds use the
full protocol (all windows, <=40 epochs, batch 128). Early stopping on SOURCE val MSE everywhere. Each result JSON is
written as soon as its run finishes, and existing JSONs are skipped, so a crashed invocation can be resumed.
"""
import argparse
import datetime
import gc
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from src.adaptation.coral import fit_coral, gap_diagnostic, TimeLimitExceeded
from src.evaluate import predict, denorm, evaluate_model
from src.experiment import prepare_pair, target_baselines
from src.models.backbone import build_model, count_params
from src.run_experiments import DOMAIN_CONFIGS
from src.train import DEVICE, device_info, set_seed, rss_mb
from src.utils.metrics import evaluate_all
from src.utils.status import log

BACKBONE = "tcn"
DOMAINS = ["finance", "finance_vol", "climate", "energy"]
LAMBDAS = [0.0, 0.1, 1.0, 10.0]
TOLERANCE = 0.05
ENERGY_MAX_SECONDS = 3600
CORAL_DIR = Path("experiments/coral")  # grid and lambda selection live here
OUT = CORAL_DIR  # final runs and summary; --out redirects (e.g. the lambda=1 sensitivity run)
FINAL = Path("experiments/final_3seed") / BACKBONE


def protocol(domain, stage):
    cfg = DOMAIN_CONFIGS[domain]
    if domain == "energy" and stage == "grid":
        return {"train_stride": 4, "max_epochs": 15, "name": "bake-off (every 4th window, <=15 epochs)"}
    return {"train_stride": 1, "max_epochs": cfg["epochs"], "name": f"full (all windows, <={cfg['epochs']} epochs)"}


def lam_tag(lam):
    return f"{lam:g}".replace(".", "p")


def load_source_only(domain, seed):
    ckpt = torch.load(FINAL / "checkpoints" / f"{domain}_seed{seed}_source_only.pt", map_location="cpu")
    m = build_model(ckpt["backbone"], ckpt["input_len"], ckpt["horizon"], **ckpt["model_kwargs"])
    m.load_state_dict(ckpt["state_dict"])
    return m.to(DEVICE).eval()


def run_coral(domain, seed, lam, stage, ckpt_path):
    cfg = DOMAIN_CONFIGS[domain]
    L, H = cfg["input_len"], cfg["horizon"]
    prot = protocol(domain, stage)
    k = prot["train_stride"]
    src, tgt_src, tgt_tgt, _ = prepare_pair(cfg)
    src_stats, tgt_stats = src["norm_stats"], tgt_tgt["norm_stats"]

    set_seed(seed)
    model = build_model(BACKBONE, L, H).to(DEVICE)
    with torch.no_grad():  # the joint loss uses head(get_features(x)); it must equal forward(x)
        xb = torch.tensor(src["X_val"][:8], device=DEVICE)
        assert torch.allclose(model(xb), model.head(model.get_features(xb)))
    t0 = time.perf_counter()
    hist = fit_coral(model, src["X_train"][::k], src["Y_train"][::k], tgt_tgt["X_train"][::k],
                     src["X_val"], src["Y_val"], lam, epochs=prot["max_epochs"], batch_size=cfg["batch_size"],
                     max_seconds=ENERGY_MAX_SECONDS if domain == "energy" else None,
                     tag=f"[coral {domain} lam={lam:g} s{seed}] ")

    val_pred = predict(model, src["X_val"])
    before = load_source_only(domain, seed)
    res = {
        "method": "coral", "backbone": BACKBONE, "domain": domain, "seed": seed, "lambda": lam, "stage": stage,
        "protocol": prot, "device_info": device_info(), "params": count_params(model),
        "batch_size": cfg["batch_size"], "lr": 1e-3,
        "n_source_train_windows": int(len(src["X_train"][::k])),
        "n_target_train_inputs": int(len(tgt_tgt["X_train"][::k])),
        "target_data_used_in_training": "target-train inputs only (target-train stats); no labels, no val/test",
        "features": "TCNForecaster.get_features(): 32 channels at the last time step; CORAL covariance over the batch",
        "training": hist,
        "source_val": {
            "mse_norm": float(np.mean((val_pred - src["Y_val"]) ** 2)),
            "mae_norm": float(np.mean(np.abs(val_pred - src["Y_val"]))),
            "orig_scale": evaluate_all(denorm(src["Y_val"], src_stats), denorm(val_pred, src_stats)),
        },
        "source_test": evaluate_model(model, src["X_test"], src["Y_test"], src_stats),
        "coral_on_target_test": {
            "target_stats": evaluate_model(model, tgt_tgt["X_test"], tgt_tgt["Y_test"], tgt_stats),
            "source_stats": evaluate_model(model, tgt_src["X_test"], tgt_src["Y_test"], src_stats),
        },
        "gap_diagnostic": {
            "before_source_only": gap_diagnostic(before, src["X_val"], tgt_tgt["X_val"], seed=seed),
            "after_coral": gap_diagnostic(model, src["X_val"], tgt_tgt["X_val"], seed=seed),
            "note": "label-free: inputs only; before = experiments/final_3seed/tcn source-only checkpoint, same seed",
        },
    }
    torch.save({"state_dict": model.state_dict(), "backbone": BACKBONE, "model_kwargs": {}, "lambda": lam,
                "input_len": L, "horizon": H, "norm_stats": {"source_train": src_stats, "target_train": tgt_stats},
                "inference_stats": "target_train (normalise target inputs with target-train stats)",
                "seed": seed, "best_epoch": hist["best_epoch"], "protocol": prot}, ckpt_path)
    res["wall_clock_seconds"] = round(time.perf_counter() - t0, 2)
    res["rss_mb_end"] = rss_mb()
    del model, before, src, tgt_src, tgt_tgt
    gc.collect()
    if domain == "energy" and res["wall_clock_seconds"] > ENERGY_MAX_SECONDS:
        raise TimeLimitExceeded(f"energy run took {res['wall_clock_seconds']}s > {ENERGY_MAX_SECONDS}s")
    return res


def _short(res):
    t = res["coral_on_target_test"]["target_stats"]
    g = res["gap_diagnostic"]
    return (f"src-val MSE {res['source_val']['mse_norm']:.5f}, best epoch {res['training']['best_epoch']}"
            f"/{res['training']['stopped_epoch']}, target-test (tgt stats) MAE {t['MAE']:.5g} RMSE {t['RMSE']:.5g}, "
            f"MMD2 {g['before_source_only']['mmd2_rbf']:.4f} -> {g['after_coral']['mmd2_rbf']:.4f}, "
            f"{res['wall_clock_seconds']:.0f}s, RSS end {res['rss_mb_end']:.0f} MB")


def grid(domains):
    d_out = CORAL_DIR / "grid"
    (d_out / "checkpoints").mkdir(parents=True, exist_ok=True)
    for d in domains:
        for lam in LAMBDAS:
            name = f"{d}_lam{lam_tag(lam)}_seed0"
            if (d_out / f"{name}.json").exists():
                print(f"skip existing {name}")
                continue
            log(f"CORAL grid start: {d}, λ={lam:g}, seed 0, protocol {protocol(d, 'grid')['name']}.")
            res = run_coral(d, 0, lam, "grid", d_out / "checkpoints" / f"{name}.pt")
            with open(d_out / f"{name}.json", "w") as f:
                json.dump(res, f, indent=2)
            log(f"CORAL grid done: {d}, λ={lam:g}, seed 0: {_short(res)}.")


def select():
    d_out = CORAL_DIR / "grid"
    table = {}
    for d in DOMAINS:
        table[d] = {}
        for lam in LAMBDAS:
            r = json.load(open(d_out / f"{d}_lam{lam_tag(lam)}_seed0.json"))
            table[d][f"{lam:g}"] = {"source_val_mse_norm": r["source_val"]["mse_norm"],
                                    "best_epoch": r["training"]["best_epoch"], "protocol": r["protocol"]["name"]}
        ref = table[d]["0"]["source_val_mse_norm"]
        for lam in LAMBDAS:
            e = table[d][f"{lam:g}"]
            e["ratio_to_lambda0"] = e["source_val_mse_norm"] / ref
            e["within_5pct"] = e["ratio_to_lambda0"] <= 1 + TOLERANCE
    qualifying = [lam for lam in LAMBDAS[1:] if all(table[d][f"{lam:g}"]["within_5pct"] for d in DOMAINS)]
    chosen = max(qualifying) if qualifying else None
    out = {"rule": "largest lambda in {0.1, 1, 10} whose seed-0 SOURCE val MSE (normalised) is at most 5% above the "
                   "same domain's lambda=0 run, in every domain (finance, finance_vol, climate, energy). "
                   "Target labels/val/test not used.",
           "domains": DOMAINS, "seed": 0, "tolerance": TOLERANCE, "table": table,
           "qualifying_lambdas": qualifying, "chosen_lambda": chosen, "device_info": device_info(),
           "written": datetime.datetime.now().isoformat(timespec="seconds")}
    with open(CORAL_DIR / "lambda_selection.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    return chosen


def final(domains, seeds, lam=None):
    if lam is None:
        lam = json.load(open(CORAL_DIR / "lambda_selection.json"))["chosen_lambda"]
    if lam is None:
        raise SystemExit("No lambda qualified; not running final seeds.")
    for d in domains:
        (OUT / d).mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            js, ck = OUT / d / f"seed{seed}.json", OUT / d / f"seed{seed}.pt"
            if js.exists():
                print(f"skip existing {js}")
                continue
            if d != "energy" and seed == 0:  # same protocol as the grid: reuse the grid run
                name = f"{d}_lam{lam_tag(lam)}_seed0"
                r = json.load(open(CORAL_DIR / "grid" / f"{name}.json"))
                r["reused_from"] = str(CORAL_DIR / "grid" / f"{name}.json")
                r["stage"] = "final"
                shutil.copy(CORAL_DIR / "grid" / "checkpoints" / f"{name}.pt", ck)
                with open(js, "w") as f:
                    json.dump(r, f, indent=2)
                log(f"CORAL final: {d} seed 0 reused from grid run {name} (same full protocol).")
                continue
            log(f"CORAL final start: {d}, λ={lam:g}, seed {seed}, protocol {protocol(d, 'final')['name']}.")
            r = run_coral(d, seed, lam, "final", ck)
            with open(js, "w") as f:
                json.dump(r, f, indent=2)
            log(f"CORAL final done: {d}, λ={lam:g}, seed {seed}: {_short(r)}.")


def _ms(vals):
    v = np.array(vals, dtype=float)
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else None, "n": int(len(v)),
            "per_seed": v.tolist()}


def summarize():
    lam = json.load(open(next((OUT / DOMAINS[0]).glob("seed*.json"))))["lambda"]
    base_rows = {
        "source-only (source stats)": lambda r: r["source_only_on_target_test"]["source_stats"],
        "source-only (target-train stats)": lambda r: r["source_only_on_target_test"]["target_stats"],
        "target-only (upper reference, uses target labels)": lambda r: r["target_only_on_target_test"]["target_stats"],
    }
    summ = {"backbone": BACKBONE, "chosen_lambda": lam, "std": "sample std across seeds (ddof=1)",
            "device_info": device_info(), "domains": {}}
    for d in DOMAINS:
        fin = [json.load(open(p)) for p in sorted(FINAL.glob(f"runs/{d}_seed*.json"))]
        cor = [json.load(open(p)) for p in sorted((OUT / d).glob("seed*.json"))]
        e = {"rows": {}, "baselines": target_baselines(DOMAIN_CONFIGS[d])}
        for name, get in base_rows.items():
            e["rows"][name] = {m: _ms([get(r)[m] for r in fin]) for m in ("MAE", "RMSE")}
            e["rows"][name]["seeds"] = [r["seed"] for r in fin]
        e["rows"][f"CORAL λ={lam:g} (target-train stats)"] = {
            **{m: _ms([r["coral_on_target_test"]["target_stats"][m] for r in cor]) for m in ("MAE", "RMSE")},
            "seeds": [r["seed"] for r in cor]}
        e["gap"] = {k: {"mmd2_rbf": _ms([r["gap_diagnostic"][k]["mmd2_rbf"] for r in cor]),
                        "coral_distance": _ms([r["gap_diagnostic"][k]["coral_distance"] for r in cor])}
                    for k in ("before_source_only", "after_coral")}
        e["source_val_mse_norm"] = {"coral": _ms([r["source_val"]["mse_norm"] for r in cor]),
                                    "source_only": _ms([r["source_val"]["mse_norm"] for r in fin])}
        e["best_epochs"] = [r["training"]["best_epoch"] for r in cor]
        e["wall_clock_seconds"] = [r["wall_clock_seconds"] for r in cor]
        summ["domains"][d] = e
    with open(OUT / "summary.json", "w") as f:
        json.dump(summ, f, indent=2)
    write_markdown(summ)
    return summ


def _cell(m):
    sd = "n/a" if m["std"] is None else f"{m['std']:.2g}"
    return f"{m['mean']:.5g} ± {sd} (n={m['n']})"


def write_markdown(summ):
    lam = summ["chosen_lambda"]
    coral_row = f"CORAL λ={lam:g} (target-train stats)"
    sel = json.load(open(CORAL_DIR / "lambda_selection.json"))
    md = [f"# CORAL (TCN), λ = {lam:g}", "",
          "Target test set, original scale. mean ± sample std (ddof=1) over seeds; n = number of seeds in the cell. "
          "Naive train-mean is deterministic (n=1). Per-window metrics are rounded to 5 decimals before averaging "
          "(src.utils.metrics), so finance stds near 1e-5 are at the rounding floor. MAPE/sMAPE are not reported. "
          "finance_vol is a negative control: lead with RMSE there (naive median beats every model on MAE).", "",
          "CORAL used target-TRAIN inputs only (target-train stats), no target labels, val or test. λ chosen on "
          "seed-0 SOURCE val MSE (experiments/coral/lambda_selection.json). Target-only uses target labels and is "
          "a reference, not an adaptation method.", ""]
    for d, e in summ["domains"].items():
        cols = ("RMSE", "MAE") if d == "finance_vol" else ("MAE", "RMSE")
        md += [f"## {d}", "", f"| setting | {cols[0]} | {cols[1]} |", "|---|---|---|"]
        nm = e["baselines"]["naive_train_mean"]["target_stats"]
        md.append(f"| naive train-mean (target-train stats) | {nm[cols[0]]:.5g} (n=1) | {nm[cols[1]]:.5g} (n=1) |")
        for name, row in e["rows"].items():
            md.append(f"| {name} | {_cell(row[cols[0]])} | {_cell(row[cols[1]])} |")
        ts, co = e["rows"]["source-only (target-train stats)"], e["rows"][coral_row]
        md.append("")
        for m in cols:
            delta = co[m]["mean"] - ts[m]["mean"]
            noise = max(ts[m]["std"] or 0, co[m]["std"] or 0)
            verdict = ("better by more than seed std" if delta < -noise else
                       "worse by more than seed std" if delta > noise else "within seed std")
            md.append(f"- {m}: CORAL − target-stats = {delta:+.4g} (max seed std {noise:.2g}) → {verdict}")
        g = e["gap"]
        md += ["", "Label-free gap, source-val vs target-val features (MMD² RBF, median heuristic; CORAL distance):", "",
               "| model | MMD² | CORAL distance |", "|---|---|---|",
               f"| source-only | {_cell(g['before_source_only']['mmd2_rbf'])} | "
               f"{_cell(g['before_source_only']['coral_distance'])} |",
               f"| CORAL λ={lam:g} | {_cell(g['after_coral']['mmd2_rbf'])} | {_cell(g['after_coral']['coral_distance'])} |",
               "", f"Source-val MSE (norm): source-only {_cell(e['source_val_mse_norm']['source_only'])}, "
               f"CORAL {_cell(e['source_val_mse_norm']['coral'])}. CORAL best epochs {e['best_epochs']}, "
               f"wall clock {e['wall_clock_seconds']} s.", ""]
    md += ["## λ selection (seed 0, source-val MSE ratio to λ=0)", "",
           "| domain | protocol | " + " | ".join(f"λ={l:g}" for l in LAMBDAS[1:]) + " |", "|---|---|---|---|---|"]
    for d, t in sel["table"].items():
        md.append(f"| {d} | {t['0']['protocol']} | " +
                  " | ".join(f"{t[f'{l:g}']['ratio_to_lambda0']:.4f}" + ("" if t[f"{l:g}"]["within_5pct"] else " ✗")
                             for l in LAMBDAS[1:]) + " |")
    md += ["", f"Qualifying: {sel['qualifying_lambdas']}; chosen λ = {lam:g}.", ""]
    md += ["## Staircase criteria (CONSTRAINTS.md Decisions)", "",
           "CORAL must beat the source-only target-stats row; energy improvement must exceed seed std; climate and "
           "finance must be no worse than seed noise. Seed std = max of the two rows' sample stds. Paired deltas are "
           "per seed (CORAL − target-stats, same seed).", ""]
    for d, e in summ["domains"].items():
        ts, co = e["rows"]["source-only (target-train stats)"], e["rows"][coral_row]
        parts = []
        for m in ("MAE", "RMSE"):
            if ts["seeds"] != co["seeds"]:
                parts.append(f"{m}: seeds differ, no paired delta")
                continue
            paired = np.array(co[m]["per_seed"]) - np.array(ts[m]["per_seed"])
            delta, noise = co[m]["mean"] - ts[m]["mean"], max(ts[m]["std"] or 0, co[m]["std"] or 0)
            ok = (delta < -noise) if d == "energy" else (delta <= noise)
            parts.append(f"{m} Δ {delta:+.4g} vs std {noise:.2g}, paired per seed {np.round(paired, 6).tolist()} → "
                         f"{'met' if ok else 'NOT met'}")
        md.append(f"- **{d}** ({'improvement > seed std' if d == 'energy' else 'no worse than seed noise'}): "
                  + "; ".join(parts))
    md.append("")
    with open(OUT / "summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--domains", nargs="+", default=DOMAINS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--lam", type=float, default=None, help="--final only: override the selected lambda")
    ap.add_argument("--out", default=None, help="--final/--summarize output dir (default experiments/coral)")
    args = ap.parse_args()
    if args.out:
        OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.grid:
        grid(args.domains)
    if args.select:
        select()
    if args.final:
        final(args.domains, args.seeds, args.lam)
    if args.summarize:
        print(json.dumps(summarize(), indent=2))
