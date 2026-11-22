"""
Ablation (CONSTRAINTS.md Amendments 2026-11-21 (k)): (c) adarnn10 and (d) adarnn0 rerun with the JOINT-epoch cap at 20
instead of 60 (3 MSE-only pretrain epochs unchanged and outside the cap -> at most 3 + 20 epochs), climate, seeds 0-2.

Same code path as the novelty_fix runs: this module calls src.adarnn_fix.run.run() unchanged and only overrides
that module's constants OUT (-> experiments/novelty_fix_cap20), JOINT_CAP (-> 20) and DEADLINE (-> 2026-11-21 18:00),
its log prefix ('novelty_fix' -> 'ablation_cap20'), and its projection() (a cap-60 queue estimate; replaced by a
no-op). Training (fit_adarnn_full), TDC periods, seeds, early stopping and evaluation are untouched.

  python -m src.adarnn_fix.ablation_cap20 --method adarnn10 --seed 0     # one job
  python -m src.adarnn_fix.ablation_cap20 --queue                        # all 6 jobs, 3 in parallel

Job exit codes as in run.py: 0 done/skipped, 3 hard stop, 4 sleep/resume inside an epoch > 300 s (rerun the job).
Queue: crash -> resume (rerun) once; a second crash on the same job stops the queue (no new jobs start).
"""
import argparse
import datetime
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import src.adarnn_fix.run as R
from src.adarnn_fix.power import keep_awake
from src.utils.status import log as _log

OUT = Path("experiments/novelty_fix_cap20")
JOINT_CAP = 20
DEADLINE = datetime.datetime(2026, 11, 21, 18, 0, 0)
METHODS = ["adarnn10", "adarnn0"]
SEEDS = [0, 1, 2]
WORKERS = 3
MAX_SLEEP_RERUNS = 3


def log(msg):
    _log(msg.replace("novelty_fix", "ablation_cap20", 1))


def patch():
    R.OUT, R.JOINT_CAP, R.DEADLINE = OUT, JOINT_CAP, DEADLINE
    R.log = log
    R.projection = lambda t_c: log(f"ablation_cap20: first epoch of (c) adarnn10 seed 0 = {t_c:.1f} s (no projection; 6 short runs).")


def run_job(method, seed):
    patch()
    path = R.result_path(method, seed)
    if path.exists():
        print(f"skip existing {path}")
        return
    keep_awake()
    R.start_watchdog()
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    log(f"ablation_cap20 start: {method} seed {seed} (PID {os.getpid()}), joint cap {JOINT_CAP}.")
    try:
        R.run(method, seed)
    except R.SleepDetected as e:
        log(f"ablation_cap20: {method} seed {seed}: {e}; job will be rerun from scratch.")
        sys.exit(4)
    except R.DeadlineExceeded as e:
        log(f"ablation_cap20: {e}; not run.")
        sys.exit(3)
    # label the JSON as the ablation (run() writes the novelty_fix amendment tag); written immediately
    res = json.load(open(path))
    res.update({"amendment": "CONSTRAINTS.md 2026-11-21 (k) (ablation of 2026-11-20 (j))",
                "ablation": "joint-epoch cap 20 instead of 60; everything else as experiments/novelty_fix",
                "reference_cap60_json": str(Path("experiments/novelty_fix") / method / f"seed{seed}.json")})
    R.write_json(path, res)


def queue():
    keep_awake()
    env = dict(os.environ, OMP_NUM_THREADS="2", TORCH_THREADS="2", PYTHONIOENCODING="utf-8")
    jobs = [(m, s) for s in SEEDS for m in METHODS]
    lock, stop = threading.Lock(), threading.Event()
    outcome = {}
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    log(f"ablation_cap20 queue start (PID {os.getpid()}), {WORKERS} workers, seed-major order: {jobs}")

    def worker(wid):
        while True:
            with lock:
                if stop.is_set() or not jobs:
                    return
                m, s = jobs.pop(0)
            if datetime.datetime.now() >= DEADLINE:
                log(f"ablation_cap20 queue: hard stop passed; {m} seed {s} not run.")
                outcome[(m, s)] = "not run (hard stop)"
                stop.set()
                return
            crashes = sleeps = 0
            with open(OUT / "logs" / f"{m}_seed{s}.log", "a", encoding="utf-8") as logf:
                while True:
                    rc = subprocess.call([sys.executable, "-u", "-m", "src.adarnn_fix.ablation_cap20",
                                          "--method", m, "--seed", str(s)], env=env, stdout=logf, stderr=subprocess.STDOUT)
                    if rc == 0:
                        outcome[(m, s)] = "done"
                        break
                    if rc == 3:
                        log(f"ablation_cap20 queue (worker {wid}): hard stop during {m} seed {s}; no further jobs start.")
                        outcome[(m, s)] = "not run (hard stop)"
                        stop.set()
                        break
                    if rc == 4 and sleeps < MAX_SLEEP_RERUNS:
                        sleeps += 1
                        log(f"ablation_cap20 queue (worker {wid}): sleep detected in {m} seed {s}; rerun {sleeps}/{MAX_SLEEP_RERUNS}.")
                        continue
                    crashes += 1
                    if crashes >= 2:
                        log(f"ablation_cap20 queue (worker {wid}): second crash (rc {rc}) in {m} seed {s}; stopping the queue.")
                        outcome[(m, s)] = f"not run (second crash, rc {rc})"
                        stop.set()
                        break
                    log(f"ablation_cap20 queue (worker {wid}): crash (rc {rc}) in {m} seed {s}; resuming once (rerun from scratch).")

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(WORKERS)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    for m, s in jobs:
        outcome[(m, s)] = "not run"
    log(f"ablation_cap20 queue finished: { {f'{m} s{s}': v for (m, s), v in outcome.items()} }")


def report():
    """Writes sections (i)-(iv) of results_clean/novelty2/ablation_cap20.md from the JSONs; (v) is hand-written below."""
    import numpy as np
    from src.adarnn_fix.summarize import baselines
    from src.adarnn_full.summarize import mstd, verdict
    REF = Path("experiments/novelty_fix")
    rows = [("source_only", REF, "(a) GRU source-only, target-train stats"),
            ("adarnn10", REF, "(c) AdaRNN step-matched λ=10, cap 60"),
            ("adarnn10", OUT, "(c) AdaRNN step-matched λ=10, cap 20"),
            ("adarnn0", REF, "(d) AdaRNN step-matched λ=0, cap 60"),
            ("adarnn0", OUT, "(d) AdaRNN step-matched λ=0, cap 20")]
    J = {}
    for m, d, lab in rows:
        J[lab] = {s: json.load(open(d / m / f"seed{s}.json")) for s in SEEDS if (d / m / f"seed{s}.json").exists()}
    base, n_test = baselines()
    for lab, runs in J.items():
        for r in runs.values():
            assert r["n_target_test_windows"] == n_test
    met = lambda r, k: r["target_test"]["target_stats"][k]
    pre = lambda r: r["training"].get("pretrain_epochs", 0) if r["method"].startswith("adarnn") else 0

    L = ["# Ablation: AdaRNN step-matched at joint-epoch cap 20 (climate Shimla → Delhi, GRU)", "",
         "Pre-registered in CONSTRAINTS.md Amendments 2026-11-21 (k). Same code path as `experiments/novelty_fix` "
         "(`src.adarnn_fix.run.run()` called unchanged by `src/adarnn_fix/ablation_cap20.py`, which only sets the output "
         "dir, the joint cap 20 and the hard stop). Target test set (" + str(n_test) + " windows, the same as "
         "results_clean/canonical.csv), target-train-stat normalisation, MAE/RMSE in °C via `src.evaluate.evaluate_model`. "
         "Std is the sample std (ddof=1) over seeds 0, 1, 2. Target labels, target val and target test were not used in training or selection.", "",
         "## (i) Epoch counting", "",
         "`src/adarnn_full/adarnn.py::fit_adarnn_full` uses a single counter `for epoch in range(1, epochs + 1)`. "
         "Epochs 1–3 are the MSE-only pretrain phase and later epochs are joint. `best_epoch` and `stopped_epoch` come "
         "from that counter, so **the best/stopped epochs reported for (c) and (d) in STATUS.md and summary.md include the "
         "3 pretrain epochs**. Early stopping is checked only from the first joint epoch. `run.py` passes "
         "`epochs = 3 + JOINT_CAP`, so the cap counts joint epochs only. Every JSON's `training.phase` list shows 3 "
         "`pretrain` entries, then the joint entries. (a) has no pretrain, so its epochs are joint epochs. "
         "Note: the JSON field `joint_epochs_run` is stopped − 3, the number of joint epochs trained, not the best joint epoch.", "",
         "| run | best (total) | stopped (total) | best (joint) | stopped (joint) | joint cap |", "|---|---|---|---|---|---|"]
    for lab in list(J)[1:]:
        for s, r in sorted(J[lab].items()):
            h, p = r["training"], pre(r)
            L.append(f"| {lab} seed {s} | {h['best_epoch']} | {h['stopped_epoch']} | {h['best_epoch'] - p} | "
                     f"{h['stopped_epoch'] - p} | {r['epoch_cap_joint']} |")
    L += ["", "## (ii) Target test, °C", "",
          "| row | n | MAE s0; s1; s2 | MAE mean ± std | RMSE s0; s1; s2 | RMSE mean ± std |", "|---|---|---|---|---|---|"]
    for k, v in base.items():
        L.append(f"| baseline: {k} | – | | {v['MAE']:.3f} | | {v['RMSE']:.3f} |")
    stats = {}
    for lab, runs in J.items():
        ss = sorted(runs)
        cell = []
        for k in ["MAE", "RMSE"]:
            vals = [met(runs[s], k) for s in ss]
            mu, sd = mstd(vals)
            stats[(lab, k)] = (mu, sd, dict(zip(ss, vals)))
            cell += ["; ".join(f"{x:.4f}" for x in vals), f"{mu:.4f} ± {sd:.4f}"]
        miss = [s for s in SEEDS if s not in runs]
        L.append(f"| {lab}{' (not run: seeds ' + str(miss) + ')' if miss else ''} | {len(ss)} | " + " | ".join(cell) + " |")
    L += ["", "## (iii) Paired per-seed differences, cap 20 − cap 60 (same method, same seed)", "",
          "| method | metric | seed 0 | seed 1 | seed 2 | max abs | identical weights? |", "|---|---|---|---|---|---|---|"]
    ident = {}
    for m, l60, l20 in [("adarnn10", rows[1][2], rows[2][2]), ("adarnn0", rows[3][2], rows[4][2])]:
        for k in ["MAE", "RMSE"]:
            diffs = [stats[(l20, k)][2][s] - stats[(l60, k)][2][s] if s in stats[(l20, k)][2] else None for s in SEEDS]
            same = []
            for s in SEEDS:
                if s in J[l20]:
                    a, b = J[l20][s], J[l60][s]
                    same.append("yes" if (a["training"]["val_mse"] == b["training"]["val_mse"][:len(a["training"]["val_mse"])]
                                          and a["training"]["best_epoch"] == b["training"]["best_epoch"]
                                          and a["source_val"]["mse_norm"] == b["source_val"]["mse_norm"]) else "no")
                else:
                    same.append("not run")
            ident[m] = same
            ok = [abs(x) for x in diffs if x is not None]
            L.append(f"| {m} | {k} | " + " | ".join("not run" if x is None else f"{x:+.5f}" for x in diffs) +
                     f" | {max(ok):.5f} | {'; '.join(same)} |")
    L += ["", "'Identical weights' = the per-epoch source-val MSE curve matches the cap-60 run's curve over the same epochs, "
          "and the restored epoch and final source-val MSE are the same (exact float equality).", "",
          "Pre-registered interpretation: the cap did not matter if every seed's cap-20 MAE is within 0.005 °C of its cap-60 MAE.", ""]
    for m, l60, l20 in [("adarnn10", rows[1][2], rows[2][2]), ("adarnn0", rows[3][2], rows[4][2])]:
        d = {s: stats[(l20, "MAE")][2][s] - stats[(l60, "MAE")][2][s] for s in stats[(l20, "MAE")][2]}
        allin = len(d) == 3 and all(abs(x) <= 0.005 for x in d.values())
        L.append(f"- {m}: " + ("all three seeds within 0.005 °C MAE → **the cap did not matter**." if allin else
                 "differs (>0.005 °C) or incomplete on seeds " + str([s for s in SEEDS if s not in d or abs(d[s]) > 0.005]) +
                 " → the cap contributed on those seeds."))
    L += ["", "## (iv) Noise rule (results_clean/noise_rule.md): (c) cap 20 vs (a)", "",
          "| metric | mean(M) | mean(R) | d | paired diffs (s0; s1; s2) | s_R | s_M | direction | verdict |", "|---|---|---|---|---|---|---|---|---|"]
    for k in ["MAE", "RMSE"]:
        mM, sM, vM = stats[(rows[2][2], k)]
        mR, sR, vR = stats[(rows[0][2], k)]
        d = mM - mR
        pd = "; ".join(f"{vM[s] - vR[s]:+.4f}" for s in SEEDS if s in vM)
        L.append(f"| {k} | {mM:.4f} | {mR:.4f} | {d:+.4f} | {pd} | {sR:.4f} | {sM:.4f} | "
                 f"{'lower error' if d < 0 else 'higher error'} | {verdict(d, sR, sM)} |")
    L += ["", "Counted as a gain only if (c) has lower error and the verdict is 'exceeds both stds' (STATUS.md 2026-11-20 03:46:51).", ""]
    md = "\n".join(L)
    Path("results_clean/novelty2/ablation_cap20.md").write_text(md + "\n", encoding="utf-8")
    print(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=METHODS)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--queue", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        report()
    elif a.queue:
        queue()
    else:
        run_job(a.method, a.seed)


if __name__ == "__main__":
    main()
