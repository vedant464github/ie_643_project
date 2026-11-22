"""
Novelty assessment runner (CONSTRAINTS.md Amendments 2026-11-15 (h)). GRU backbone, methods:
  source_only -> rows (a) source stats and (b) target-train stats from the same model
  coral       -> (c) GRU + CORAL on the last-layer final hidden state
  adarnn      -> (d) GRU + faithful AdaRNN (per-step TDM, boosting importance weights)

  python -m src.adarnn_full.run grid   --domain climate --method coral          # seed 0, lambda in {0,0.1,1,10}
  python -m src.adarnn_full.run select --domain climate                         # both methods, per-domain rule
  python -m src.adarnn_full.run final  --domain climate --method adarnn --seeds 1 2
  python -m src.adarnn_full.run final  --domain energy --method adarnn --seeds 0 --probe   # energy protocol probe

Results: experiments/novelty_adarnn/. Existing JSONs are skipped (resume). Hard stop 2026-11-19 22:00.
"""
import argparse
import datetime
import json
import os
import shutil
import threading
import time
from pathlib import Path

import numpy as np
import torch

from src.adarnn_full.adarnn import fit_adarnn_full, DeadlineExceeded, EpochTooSlow
from src.adarnn_full.model import build_gru, check_equivalence
from src.adarnn_full.train_timed import fit_timed
from src.evaluate import predict, denorm, evaluate_model
from src.experiment import prepare_pair
from src.models.backbone import count_params
from src.run_experiments import DOMAIN_CONFIGS
from src.train import DEVICE, device_info, set_seed, rss_mb
from src.utils.metrics import evaluate_all
from src.utils.status import log

OUT = Path("experiments/novelty_adarnn")
DEADLINE = datetime.datetime(2026, 11, 19, 22, 0, 0)
LAMBDAS = [0.0, 0.1, 1.0, 10.0]
TOLERANCE = 0.05
PRETRAIN_EPOCHS = 3
ENERGY_EPOCH_LIMIT = 300.0
ENERGY_FULL_LIMIT = 150.0
PROTOCOLS = {"full": {"train_stride": 1, "name": "full (all windows)"},
             "cheap": {"train_stride": 4, "max_epochs": 15, "name": "cheap (every 4th window, <=15 epochs)"}}
TDC = json.load(open("experiments/adarnn/tdc.json"))


# ------------------------------------------------------------------ helpers
def now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)  # atomic: a killed process never leaves a half-written result


def start_watchdog():
    """At the hard stop, log and terminate this process; finished JSONs are already on disk."""
    def _kill():
        log(f"novelty: HARD STOP {DEADLINE:%Y-%m-%d %H:%M} reached in PID {os.getpid()}; process terminated, "
            "its current run is 'not run'.")
        os._exit(3)
    secs = (DEADLINE - datetime.datetime.now()).total_seconds()
    if secs <= 0:
        raise SystemExit("hard stop already passed; nothing started")
    t = threading.Timer(secs, _kill)
    t.daemon = True
    t.start()


def protocol(domain, stage):
    cfg = DOMAIN_CONFIGS[domain]
    if domain == "energy" and stage == "grid":
        p = dict(PROTOCOLS["cheap"])
    elif domain == "energy":
        f = OUT / "energy_protocol.json"
        choice = json.load(open(f))["protocol"] if f.exists() else "full"
        p = dict(PROTOCOLS[choice])
    else:
        p = dict(PROTOCOLS["full"])
    p.setdefault("max_epochs", cfg["epochs"])
    p["name"] = (f"{'cheap' if p['train_stride'] > 1 else 'full'} ("
                 f"{'every 4th window' if p['train_stride'] > 1 else 'all windows'}, <={p['max_epochs']} epochs)")
    return p


def lam_tag(lam):
    return "none" if lam is None else f"{lam:g}".replace(".", "p")


def result_path(domain, method, seed, lam, stage):
    if stage == "grid":
        return OUT / "grid" / f"{domain}_{method}_lam{lam_tag(lam)}_seed{seed}.json"
    return OUT / domain / method / f"seed{seed}.json"


def periods_for(X, Y, split_idx, k):
    idx = np.arange(len(X))
    out = []
    for s, e in [(0, split_idx[0]), (split_idx[0], split_idx[1]), (split_idx[1], len(X))]:
        sel = idx[s:e]
        sel = sel[sel % k == 0]
        out.append((X[sel], Y[sel]))
    return out


# ------------------------------------------------------------------ one run
def run_one(domain, method, seed, lam, stage, probe=False):
    cfg = DOMAIN_CONFIGS[domain]
    L, H, B = cfg["input_len"], cfg["horizon"], cfg["batch_size"]
    prot = protocol(domain, stage)
    k = prot["train_stride"]
    src, tgt_src, tgt_tgt, _ = prepare_pair(cfg)
    s_st, t_st = src["norm_stats"], tgt_tgt["norm_stats"]
    tag = f"[{method} {domain} lam={lam_tag(lam)} s{seed} {stage}] "
    max_ep_s = ENERGY_EPOCH_LIMIT if domain == "energy" else None

    def on_epoch(epoch, secs):
        if epoch == 1:
            log(f"novelty: first epoch {domain} {method} seed {seed} λ={lam_tag(lam)} ({stage}, {prot['name']}): "
                f"{secs:.1f} s/epoch.")
            if probe:
                energy_probe_gate(seed, secs)

    set_seed(seed)
    model = build_gru(L, H).to(DEVICE)
    check_equivalence(model, torch.tensor(src["X_val"][:8], device=DEVICE))
    started, t0 = now(), time.perf_counter()
    Xs, Ys, Xt = src["X_train"][::k], src["Y_train"][::k], tgt_tgt["X_train"][::k]
    extra = {}
    if method == "adarnn":
        split = TDC[domain]["chosen"]["split_idx"]
        per = periods_for(src["X_train"], src["Y_train"], split, k)
        hist = fit_adarnn_full(model, per, Xt, src["X_val"], src["Y_val"], lam, prot["max_epochs"], B, len(Xs),
                               pretrain_epochs=PRETRAIN_EPOCHS, deadline=DEADLINE, max_epoch_seconds=max_ep_s,
                               epoch_callback=on_epoch, tag=tag)
        extra = {"tdc_split_idx": split, "tdc_source": "experiments/adarnn/tdc.json",
                 "period_sizes": [len(p[0]) for p in per], "pretrain_epochs": PRETRAIN_EPOCHS}
    else:
        hist = fit_timed(model, Xs, Ys, Xt, src["X_val"], src["Y_val"], lam=lam if method == "coral" else None,
                         epochs=prot["max_epochs"], batch_size=B, deadline=DEADLINE, max_epoch_seconds=max_ep_s,
                         epoch_callback=on_epoch, tag=tag)
    vp = predict(model, src["X_val"])
    res = {
        "method": method, "backbone": "gru (GRUForecaster dims: 2 layers, hidden 32)", "domain": domain, "seed": seed,
        "lambda": lam, "stage": stage, "protocol": prot, "device_info": device_info(), "params": count_params(model),
        "batch_size": B, "lr": 1e-3, "n_source_train_windows": int(len(Xs)), "n_target_train_inputs": int(len(Xt)),
        "target_data_used_in_training": ("none" if method == "source_only" else
                                         "target-train inputs only (target-train stats); no labels, no val/test"),
        **extra, "training": hist,
        "source_val": {"mse_norm": float(np.mean((vp - src["Y_val"]) ** 2)),
                       "orig_scale": evaluate_all(denorm(src["Y_val"], s_st), denorm(vp, s_st))},
        "source_test": evaluate_model(model, src["X_test"], src["Y_test"], s_st),
        "target_test": {"target_stats": evaluate_model(model, tgt_tgt["X_test"], tgt_tgt["Y_test"], t_st),
                        "source_stats": evaluate_model(model, tgt_src["X_test"], tgt_src["Y_test"], s_st)},
        "n_target_test_windows": int(len(tgt_tgt["Y_test"])),
        "started": started, "finished": now(), "wall_clock_seconds": round(time.perf_counter() - t0, 2),
        "rss_mb_end": rss_mb(),
    }
    path = result_path(domain, method, seed, lam, stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "input_len": L, "horizon": H, "method": method, "lambda": lam,
                "seed": seed, "norm_stats": {"source_train": s_st, "target_train": t_st}}, path.with_suffix(".pt"))
    write_json(path, res)
    t = res["target_test"]["target_stats"]
    log(f"novelty done: {domain} {method} λ={lam_tag(lam)} seed {seed} ({stage}, {prot['name']}): src-val MSE "
        f"{res['source_val']['mse_norm']:.5f}, best epoch {hist['best_epoch']}/{hist['stopped_epoch']}, target test "
        f"(tgt stats) MAE {t['MAE']:.5g} RMSE {t['RMSE']:.5g}, {res['wall_clock_seconds']:.0f} s.")
    return res


# ------------------------------------------------------------------ energy protocol probe
class ProtocolSwitch(RuntimeError):
    pass


def energy_probe_gate(seed, secs):
    """Each of the 3 parallel AdaRNN energy processes reports its first epoch; the decision uses the maximum."""
    d = OUT / "energy_probe"
    write_json(d / f"seed{seed}.json", {"seed": seed, "first_epoch_seconds": secs, "at": now()})
    while len(list(d.glob("seed*.json"))) < 3:
        time.sleep(5)
    times = {int(p.stem[4:]): json.load(open(p))["first_epoch_seconds"] for p in d.glob("seed*.json")}
    worst = max(times.values())
    choice = "full" if worst <= ENERGY_FULL_LIMIT else "cheap"
    if seed == 0:
        write_json(OUT / "energy_protocol.json", {
            "protocol": choice, "first_epoch_seconds": times, "max_first_epoch_seconds": worst,
            "rule": f"full if max first-epoch time of the 3 parallel AdaRNN energy runs <= {ENERGY_FULL_LIMIT} s, else "
                    "cheap for all four GRU methods; any epoch > 300 s stops energy", "decided": now()})
        log(f"novelty: energy protocol probe, first-epoch seconds {times} (max {worst:.1f}) -> protocol '{choice}'.")
    else:
        while not (OUT / "energy_protocol.json").exists():
            time.sleep(2)
    if choice != "full":
        raise ProtocolSwitch(f"first epoch {worst:.1f}s > {ENERGY_FULL_LIMIT}s: cheap protocol, full-protocol run aborted")


# ------------------------------------------------------------------ stages
def select(domain):
    out = {"rule": "per method and domain: largest lambda in {0.1, 1, 10} whose seed-0 SOURCE val MSE (normalised) is at "
                   "most 5% above the same method's lambda=0 run; none -> final seeds not run. Target data not used.",
           "domain": domain, "written": now()}
    for method in ["coral", "adarnn"]:
        tab = {}
        for lam in LAMBDAS:
            r = json.load(open(result_path(domain, method, 0, lam, "grid")))
            tab[f"{lam:g}"] = {"source_val_mse_norm": r["source_val"]["mse_norm"], "best_epoch": r["training"]["best_epoch"],
                               "protocol": r["protocol"]["name"]}
        ref = tab["0"]["source_val_mse_norm"]
        for e in tab.values():
            e["ratio_to_lambda0"] = e["source_val_mse_norm"] / ref
            e["within_5pct"] = e["ratio_to_lambda0"] <= 1 + TOLERANCE
        q = [lam for lam in LAMBDAS[1:] if tab[f"{lam:g}"]["within_5pct"]]
        out[method] = {"table": tab, "qualifying": q, "chosen_lambda": max(q) if q else None}
        log(f"novelty λ selection {domain} {method}: ratios "
            f"{ {k: round(v['ratio_to_lambda0'], 4) for k, v in tab.items()} } -> chosen {out[method]['chosen_lambda']}.")
    write_json(OUT / f"lambda_selection_{domain}.json", out)


def chosen_lambda(domain, method):
    if method == "source_only":
        return None
    return json.load(open(OUT / f"lambda_selection_{domain}.json"))[method]["chosen_lambda"]


def final(domain, method, seeds, probe=False):
    lam = chosen_lambda(domain, method)
    if method != "source_only" and lam is None:
        log(f"novelty: {domain} {method}: no λ qualified; final seeds not run.")
        return
    for seed in seeds:
        path = result_path(domain, method, seed, lam, "final")
        if path.exists():
            print(f"skip existing {path}")
            continue
        if datetime.datetime.now() >= DEADLINE:
            log(f"novelty: hard stop passed; {domain} {method} seed {seed} not run.")
            return
        grid_p = result_path(domain, method, 0, lam, "grid")
        if (method != "source_only" and seed == 0 and grid_p.exists()
                and json.load(open(grid_p))["protocol"] == protocol(domain, "final")):
            r = json.load(open(grid_p))
            r.update({"stage": "final", "reused_from_grid": str(grid_p)})
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(grid_p.with_suffix(".pt"), path.with_suffix(".pt"))
            write_json(path, r)
            log(f"novelty: {domain} {method} seed 0 final = grid run at λ={lam:g} (same protocol), reused.")
            continue
        log(f"novelty start: {domain} {method} λ={lam_tag(lam)} seed {seed} ({protocol(domain, 'final')['name']}).")
        try:
            run_one(domain, method, seed, lam, "final", probe=probe)
        except ProtocolSwitch as e:
            log(f"novelty: {domain} {method} seed {seed}: {e}.")
            return
        except EpochTooSlow as e:
            log(f"novelty: ENERGY STOPPED: {e}.")
            write_json(OUT / "energy_stopped.json", {"reason": str(e), "at": now()})
            return


def grid(domain, method):
    for lam in LAMBDAS:
        path = result_path(domain, method, 0, lam, "grid")
        if path.exists():
            print(f"skip existing {path}")
            continue
        if datetime.datetime.now() >= DEADLINE:
            log(f"novelty: hard stop passed; grid {domain} {method} λ={lam:g} not run.")
            return
        log(f"novelty start: grid {domain} {method} λ={lam:g} seed 0 ({protocol(domain, 'grid')['name']}).")
        try:
            run_one(domain, method, 0, lam, "grid")
        except EpochTooSlow as e:
            log(f"novelty: ENERGY STOPPED: {e}.")
            write_json(OUT / "energy_stopped.json", {"reason": str(e), "at": now()})
            return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["grid", "select", "final"])
    ap.add_argument("--domain", required=True)
    ap.add_argument("--method", choices=["source_only", "coral", "adarnn"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--probe", action="store_true")
    a = ap.parse_args()
    if a.domain == "energy" and (OUT / "energy_stopped.json").exists() and a.stage != "select":
        # designed refusal, not a crash: exit 0 so the queue driver moves on (fixed 2026-11-15 23:10, see STATUS.md)
        log(f"novelty: energy stopped (an epoch exceeded 300 s); '{a.stage} {a.method} seeds {a.seeds}' not run.")
        return
    start_watchdog()
    if a.stage == "grid":
        grid(a.domain, a.method)
    elif a.stage == "select":
        select(a.domain)
    else:
        final(a.domain, a.method, a.seeds, probe=a.probe)


if __name__ == "__main__":
    main()
