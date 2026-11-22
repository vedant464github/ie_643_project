"""
Novelty fix runner, climate only (CONSTRAINTS.md Amendments 2026-11-20 (j)). One job = one (method, seed):
  source_only -> (a) GRU source-only, evaluated with target-train stats (source stats kept for reference)
  coral       -> (b) GRU + CORAL lambda=10 on the last-layer final hidden state
  adarnn10    -> (c) GRU + AdaRNN step-matched, lambda=10
  adarnn0     -> (d) GRU + AdaRNN step-matched, lambda=0 (schedule-only control)

  python -m src.adarnn_fix.run --method adarnn10 --seed 0

Training code is reused unchanged: src.adarnn_full.train_timed.fit_timed (a, b) and
src.adarnn_full.adarnn.fit_adarnn_full (c, d). Step matching: fit_adarnn_full uses
steps = n_source_train // (3 * batch); passing n_source_train = S_BASE * 3 * batch gives S_BASE steps per epoch
(pretrain and joint), each step still one batch of 64 from each period (+ target in joint epochs).
Cap: 60 joint epochs; the 3 pretrain epochs are extra (epochs = 3 + 60) for (c)/(d); (a)/(b) epochs = 60.
Exit codes: 0 done/skipped, 3 hard stop, 4 sleep/resume detected inside an epoch > 300 s (driver reruns the job).
"""
import argparse
import datetime
import json
import math
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

from src.adarnn_fix.power import keep_awake, sleep_events
from src.adarnn_full.adarnn import fit_adarnn_full, DeadlineExceeded
from src.adarnn_full.model import build_gru, check_equivalence
from src.adarnn_full.run import periods_for, write_json
from src.adarnn_full.train_timed import fit_timed
from src.evaluate import predict, denorm, evaluate_model
from src.experiment import prepare_pair
from src.models.backbone import count_params
from src.run_experiments import DOMAIN_CONFIGS
from src.train import DEVICE, device_info, set_seed, rss_mb
from src.utils.metrics import evaluate_all
from src.utils.status import log

DOMAIN = "climate"
OUT = Path("experiments/novelty_fix")
DEADLINE = datetime.datetime(2026, 11, 21, 12, 0, 0)
JOINT_CAP = 60
PRETRAIN_EPOCHS = 3
LAM = {"source_only": None, "coral": 10.0, "adarnn10": 10.0, "adarnn0": 0.0}
LABEL = {"source_only": "(a) GRU source-only, target-train stats", "coral": "(b) GRU + CORAL lambda=10",
         "adarnn10": "(c) GRU + AdaRNN step-matched lambda=10", "adarnn0": "(d) GRU + AdaRNN step-matched lambda=0"}
SLOW_EPOCH_S = 300.0
TDC = json.load(open("experiments/adarnn/tdc.json"))


class SleepDetected(RuntimeError):
    pass


def now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def result_path(method, seed):
    return OUT / method / f"seed{seed}.json"


def start_watchdog():
    def _kill():
        log(f"novelty_fix: HARD STOP {DEADLINE:%Y-%m-%d %H:%M} reached in PID {os.getpid()}; current run aborted, 'not run'.")
        os._exit(3)
    secs = (DEADLINE - datetime.datetime.now()).total_seconds()
    if secs <= 0:
        log("novelty_fix: hard stop already passed; nothing started.")
        sys.exit(3)
    t = threading.Timer(secs, _kill)
    t.daemon = True
    t.start()


def projection(t_c):
    """Worst case (every run to its cap). Joint/pretrain epoch-time ratio taken from the old climate AdaRNN seed-0 log."""
    old = json.load(open("experiments/novelty_adarnn/climate/adarnn/seed0.json"))["training"]
    ph, es = old["phase"], old["epoch_seconds"]
    ratio = float(np.mean([s for p, s in zip(ph, es) if p == "joint"]) / np.mean([s for p, s in zip(ph, es) if p == "pretrain"]))
    fe = {}
    for _ in range(24):  # wait up to 2 min for the concurrent seed-0 baselines' first epochs
        fe = {p.stem: json.load(open(p))["seconds"] for p in (OUT / "first_epoch").glob("*_seed0.json")}
        if {"source_only_seed0", "coral_seed0"} <= fe.keys():
            break
        time.sleep(5)
    t_so, t_co = fe.get("source_only_seed0", t_c), fe.get("coral_seed0", t_c)
    per_seed = {"source_only": JOINT_CAP * t_so, "coral": JOINT_CAP * t_co,
                "adarnn10": PRETRAIN_EPOCHS * t_c + JOINT_CAP * ratio * t_c}
    per_seed["adarnn0"] = per_seed["adarnn10"]
    serial = 3 * sum(per_seed.values())
    wall = serial / 3  # 3 parallel processes
    msg = (f"novelty_fix PROJECTION: first epoch of (c) adarnn10 seed 0 (pretrain, 47 steps) = {t_c:.1f} s; "
           f"first epochs source_only {t_so:.1f} s, coral {t_co:.1f} s; joint/pretrain ratio {ratio:.2f} (old log). "
           f"Worst case all runs to the cap: {serial / 3600:.2f} h serial, ~{wall / 3600:.2f} h wall-clock with 3 "
           f"processes -> {'EXCEEDS' if wall > 3 * 3600 else 'does not exceed'} 3 hours; "
           f"{'passes' if datetime.datetime.now() + datetime.timedelta(seconds=wall) > DEADLINE else 'within'} the hard stop. Continuing.")
    log(msg)
    write_json(OUT / "projection.json", {"first_epoch_c_seconds": t_c, "first_epoch_source_only": t_so,
                                         "first_epoch_coral": t_co, "joint_pretrain_ratio": ratio,
                                         "per_seed_worst_seconds": per_seed, "serial_hours": serial / 3600,
                                         "wall_hours_3proc": wall / 3600, "exceeds_3h": bool(wall > 3 * 3600), "at": now()})


def run(method, seed):
    cfg = DOMAIN_CONFIGS[DOMAIN]
    L, H, B = cfg["input_len"], cfg["horizon"], cfg["batch_size"]
    lam = LAM[method]
    src, tgt_src, tgt_tgt, _ = prepare_pair(cfg)
    s_st, t_st = src["norm_stats"], tgt_tgt["norm_stats"]
    n_src = len(src["X_train"])
    s_base = math.ceil(n_src / B)  # batches per epoch of make_loader (no drop_last)
    tag = f"[fix {method} s{seed}] "
    epoch_log = []

    def on_epoch(epoch, secs):
        end = datetime.datetime.now()
        epoch_log.append({"epoch": epoch, "seconds": round(secs, 2), "end": end.isoformat(timespec="seconds")})
        if epoch == 1:
            write_json(OUT / "first_epoch" / f"{method}_seed{seed}.json", {"seconds": secs, "at": now()})
            log(f"novelty_fix: first epoch {method} seed {seed}: {secs:.1f} s.")
            if method == "adarnn10" and seed == 0:
                print(f"FIRST EPOCH OF (c): {secs:.1f} s", flush=True)
                projection(secs)
        if secs > SLOW_EPOCH_S:
            start = end - datetime.timedelta(seconds=secs + 5)
            ev = sleep_events(start, end + datetime.timedelta(seconds=5))
            log(f"novelty_fix: {method} seed {seed} epoch {epoch} took {secs:.0f} s (> {SLOW_EPOCH_S:.0f} s); "
                f"System log sleep/resume events {start:%H:%M:%S}-{end:%H:%M:%S}: {ev or 'none'}.")
            if ev:
                raise SleepDetected(f"sleep/resume {ev} inside epoch {epoch}")

    set_seed(seed)
    model = build_gru(L, H).to(DEVICE)
    check_equivalence(model, torch.tensor(src["X_val"][:8], device=DEVICE))
    started, t0 = now(), time.perf_counter()
    Xt = tgt_tgt["X_train"]
    extra = {}
    if method.startswith("adarnn"):
        split = TDC[DOMAIN]["chosen"]["split_idx"]
        per = periods_for(src["X_train"], src["Y_train"], split, 1)
        hist = fit_adarnn_full(model, per, Xt, src["X_val"], src["Y_val"], lam, PRETRAIN_EPOCHS + JOINT_CAP, B,
                               s_base * 3 * B, pretrain_epochs=PRETRAIN_EPOCHS, deadline=DEADLINE,
                               epoch_callback=on_epoch, tag=tag)
        assert hist["steps_per_epoch"] == s_base
        extra = {"tdc_split_idx": split, "tdc_source": "experiments/adarnn/tdc.json",
                 "period_sizes": [len(p[0]) for p in per], "pretrain_epochs": PRETRAIN_EPOCHS,
                 "steps_per_epoch": s_base, "joint_epochs_run": hist["stopped_epoch"] - PRETRAIN_EPOCHS}
    else:
        hist = fit_timed(model, src["X_train"], src["Y_train"], Xt, src["X_val"], src["Y_val"], lam=lam,
                         epochs=JOINT_CAP, batch_size=B, deadline=DEADLINE, epoch_callback=on_epoch, tag=tag)
        extra = {"steps_per_epoch": s_base}
    hist["epoch_wallclock"] = epoch_log
    vp = predict(model, src["X_val"])
    res = {
        "method": method, "label": LABEL[method], "backbone": "gru (GRUForecaster dims: 2 layers, hidden 32)",
        "domain": DOMAIN, "seed": seed, "lambda": lam, "amendment": "CONSTRAINTS.md 2026-11-20 (j)",
        "epoch_cap_joint": JOINT_CAP, "device_info": device_info(), "params": count_params(model), "batch_size": B,
        "lr": 1e-3, "n_source_train_windows": int(n_src), "n_target_train_inputs": int(len(Xt)),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"), "torch_threads": torch.get_num_threads(),
        "target_data_used_in_training": ("none" if method == "source_only" else
                                         "target-train inputs only (target-train stats); no labels, no val/test"),
        **extra, "training": hist,
        "source_val": {"mse_norm": float(np.mean((vp - src["Y_val"]) ** 2)),
                       "orig_scale": evaluate_all(denorm(src["Y_val"], s_st), denorm(vp, s_st))},
        "target_test": {"target_stats": evaluate_model(model, tgt_tgt["X_test"], tgt_tgt["Y_test"], t_st),
                        "source_stats": evaluate_model(model, tgt_src["X_test"], tgt_src["Y_test"], s_st)},
        "n_target_test_windows": int(len(tgt_tgt["Y_test"])),
        "started": started, "finished": now(), "wall_clock_seconds": round(time.perf_counter() - t0, 2),
        "rss_mb_end": rss_mb(),
    }
    path = result_path(method, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "input_len": L, "horizon": H, "method": method, "lambda": lam,
                "seed": seed, "norm_stats": {"source_train": s_st, "target_train": t_st}}, path.with_suffix(".pt"))
    write_json(path, res)
    t = res["target_test"]["target_stats"]
    log(f"novelty_fix done: {method} seed {seed}: src-val MSE {res['source_val']['mse_norm']:.5f}, best epoch "
        f"{hist['best_epoch']}/{hist['stopped_epoch']}, target test MAE {t['MAE']:.4f} RMSE {t['RMSE']:.4f}, "
        f"{res['wall_clock_seconds']:.0f} s.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=list(LAM), required=True)
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args()
    if result_path(a.method, a.seed).exists():
        print(f"skip existing {result_path(a.method, a.seed)}")
        return
    keep_awake()
    start_watchdog()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    log(f"novelty_fix start: {a.method} seed {a.seed} (PID {os.getpid()}).")
    try:
        run(a.method, a.seed)
    except SleepDetected as e:
        log(f"novelty_fix: {a.method} seed {a.seed}: {e}; job will be rerun from scratch.")
        sys.exit(4)
    except DeadlineExceeded as e:
        log(f"novelty_fix: {e}; not run.")
        sys.exit(3)


if __name__ == "__main__":
    main()
