"""
AdaRNN-style runs on TCN (design pre-registered in CONSTRAINTS.md Amendments 2026-10-25 (b)).

  python -m src.run_adarnn --tdc                       # writes experiments/adarnn/tdc.json (inputs only)
  python -m src.run_adarnn --run --seeds 0 1 2         # finance, finance_vol, climate, energy in that order

Hard stop at 22:00 local today (was 16:00; CONSTRAINTS.md 2026-10-25 (d)): no run starts after it, and a run still training at that point is aborted. No per-run time cap (removed, 2026-10-25 (e)). Every
(domain, seed) without a results JSON is listed as "not run" in experiments/adarnn/status.json. Existing JSONs are
skipped, so a crashed invocation can be resumed.
"""
import argparse
import datetime
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.adaptation.adarnn import tdc_split, fit_adarnn, DeadlineExceeded
from src.adaptation.coral import gap_diagnostic, TimeLimitExceeded
from src.data.prepare import prepare_domain
from src.evaluate import predict, denorm, evaluate_model
from src.experiment import prepare_pair, load_series
from src.models.backbone import build_model, count_params
from src.run_coral import load_source_only, protocol, BACKBONE
from src.run_experiments import DOMAIN_CONFIGS
from src.train import DEVICE, device_info, set_seed, rss_mb
from src.utils.metrics import evaluate_all
from src.utils.status import log

DOMAINS = ["finance", "finance_vol", "climate", "energy"]  # cheapest first
LAMBDA = 10.0
PRETRAIN_EPOCHS = 3
ALL_SEEDS = [0, 1, 2]
DEADLINE = datetime.datetime(2026, 10, 25, 22, 0, 0)
OUT = Path("experiments/adarnn")


def compute_tdc():
    out = {"note": "computed once on source-train input windows (source stats), before any AdaRNN-style training"}
    for d in DOMAINS:
        cfg = DOMAIN_CONFIGS[d]
        src = prepare_domain(load_series(cfg, "source"), cfg["input_len"], cfg["horizon"])
        out[d] = tdc_split(src["X_train"])
        c = out[d]["chosen"]
        log(f"AdaRNN TDC {d}: split fractions {c['split_fracs']} (window idx {c['split_idx']} of "
            f"{out[d]['n_windows']}), score {c['score']:.5f}, {len(out[d]['candidates'])} candidates.")
    out["written"] = datetime.datetime.now().isoformat(timespec="seconds")
    with open(OUT / "tdc.json", "w") as f:
        json.dump(out, f, indent=2)


def run_adarnn(domain, seed, split_idx, ckpt_path):
    cfg = DOMAIN_CONFIGS[domain]
    L, H = cfg["input_len"], cfg["horizon"]
    prot = protocol(domain, "final")
    src, tgt_src, tgt_tgt, _ = prepare_pair(cfg)
    src_stats, tgt_stats = src["norm_stats"], tgt_tgt["norm_stats"]

    set_seed(seed)
    model = build_model(BACKBONE, L, H).to(DEVICE)
    t0 = time.perf_counter()
    hist = fit_adarnn(model, src["X_train"], src["Y_train"], split_idx, tgt_tgt["X_train"], src["X_val"],
                      src["Y_val"], LAMBDA, epochs=prot["max_epochs"], batch_size=cfg["batch_size"],
                      pretrain_epochs=PRETRAIN_EPOCHS,
                      max_seconds=None, deadline=DEADLINE,  # no per-run cap (CONSTRAINTS.md 2026-10-25 (e))
                      tag=f"[adarnn {domain} s{seed}] ")
    val_pred = predict(model, src["X_val"])
    before = load_source_only(domain, seed)
    res = {
        "method": "adarnn_style", "backbone": BACKBONE, "domain": domain, "seed": seed, "lambda": LAMBDA,
        "protocol": prot, "pretrain_epochs": PRETRAIN_EPOCHS, "tdc_split_idx": list(split_idx),
        "device_info": device_info(), "params": count_params(model), "batch_size": cfg["batch_size"], "lr": 1e-3,
        "n_source_train_windows": int(len(src["X_train"])), "n_target_train_inputs": int(len(tgt_tgt["X_train"])),
        "target_data_used_in_training": "target-train inputs only (target-train stats); no labels, no val/test",
        "loss": "MSE(period batches) + lambda * mean of CORAL over the 6 pairs of {P1,P2,P3,target}; no boosting",
        "training": hist,
        "source_val": {
            "mse_norm": float(np.mean((val_pred - src["Y_val"]) ** 2)),
            "mae_norm": float(np.mean(np.abs(val_pred - src["Y_val"]))),
            "orig_scale": evaluate_all(denorm(src["Y_val"], src_stats), denorm(val_pred, src_stats)),
        },
        "source_test": evaluate_model(model, src["X_test"], src["Y_test"], src_stats),
        "adarnn_on_target_test": {
            "target_stats": evaluate_model(model, tgt_tgt["X_test"], tgt_tgt["Y_test"], tgt_stats),
            "source_stats": evaluate_model(model, tgt_src["X_test"], tgt_src["Y_test"], src_stats),
        },
        "gap_diagnostic": {
            "before_source_only": gap_diagnostic(before, src["X_val"], tgt_tgt["X_val"], seed=seed),
            "after_adarnn": gap_diagnostic(model, src["X_val"], tgt_tgt["X_val"], seed=seed),
            "note": "label-free: inputs only; before = experiments/final_3seed/tcn source-only checkpoint, same seed",
        },
    }
    torch.save({"state_dict": model.state_dict(), "backbone": BACKBONE, "model_kwargs": {}, "lambda": LAMBDA,
                "input_len": L, "horizon": H, "norm_stats": {"source_train": src_stats, "target_train": tgt_stats},
                "inference_stats": "target_train (normalise target inputs with target-train stats)",
                "seed": seed, "best_epoch": hist["best_epoch"], "tdc_split_idx": list(split_idx)}, ckpt_path)
    res["wall_clock_seconds"] = round(time.perf_counter() - t0, 2)
    res["rss_mb_end"] = rss_mb()
    del model, before, src, tgt_src, tgt_tgt
    gc.collect()
    return res


def write_status(seeds, stopped_reason=None):
    st = {"deadline": DEADLINE.isoformat(), "runs": {}, "stopped_reason": stopped_reason,
          "written": datetime.datetime.now().isoformat(timespec="seconds")}
    for d in DOMAINS:
        for s in seeds:
            st["runs"][f"{d}_seed{s}"] = "done" if (OUT / d / f"seed{s}.json").exists() else "not run"
    with open(OUT / "status.json", "w") as f:
        json.dump(st, f, indent=2)
    return st


def run(seeds):
    tdc = json.load(open(OUT / "tdc.json"))
    reason = None
    try:
        for d in DOMAINS:
            (OUT / d).mkdir(parents=True, exist_ok=True)
            for seed in seeds:
                js = OUT / d / f"seed{seed}.json"
                if js.exists():
                    print(f"skip existing {js}")
                    continue
                if datetime.datetime.now() >= DEADLINE:
                    raise DeadlineExceeded(f"hard stop {DEADLINE:%H:%M} reached before {d} seed {seed} started")
                log(f"AdaRNN-style start: {d}, seed {seed}, λ={LAMBDA:g}, protocol {protocol(d, 'final')['name']}.")
                r = run_adarnn(d, seed, tdc[d]["chosen"]["split_idx"], OUT / d / f"seed{seed}.pt")
                with open(js, "w") as f:
                    json.dump(r, f, indent=2)
                t, g = r["adarnn_on_target_test"]["target_stats"], r["gap_diagnostic"]
                log(f"AdaRNN-style done: {d}, seed {seed}: src-val MSE {r['source_val']['mse_norm']:.5f}, best "
                    f"epoch {r['training']['best_epoch']}/{r['training']['stopped_epoch']}, target-test (tgt stats) "
                    f"MAE {t['MAE']:.5g} RMSE {t['RMSE']:.5g}, MMD2 {g['before_source_only']['mmd2_rbf']:.4f} -> "
                    f"{g['after_adarnn']['mmd2_rbf']:.4f}, {r['wall_clock_seconds']:.0f}s, RSS end {r['rss_mb_end']:.0f} MB.")
    except DeadlineExceeded as e:
        reason = str(e)
        log(f"AdaRNN-style hard stop: {reason}. Remaining runs marked 'not run'.")
    st = write_status(ALL_SEEDS, reason)  # parallel per-seed processes must not drop each other's seeds
    print(json.dumps(st, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tdc", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.tdc:
        compute_tdc()
    if args.run:
        run(args.seeds)
