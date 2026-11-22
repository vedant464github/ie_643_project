"""
Seed-major queue for the novelty fix (CONSTRAINTS.md Amendments 2026-11-20 (j)): (a, b, c, d) for seed 0, then seed 1,
then seed 2, up to 3 processes in parallel. Per job: rc 0 done; rc 3 hard stop (no further jobs start);
rc 4 sleep/resume inside a >300 s epoch -> rerun from scratch (at most 3 times); any other rc = crash ->
resume (rerun) once; a second crash on the same job is logged and the job is skipped.

  python -m src.adarnn_fix.driver
"""
import datetime
import os
import subprocess
import sys
import threading

from src.adarnn_fix.power import keep_awake
from src.adarnn_fix.run import DEADLINE
from src.utils.status import log

METHODS = ["source_only", "coral", "adarnn10", "adarnn0"]
SEEDS = [0, 1, 2]
WORKERS = 3
MAX_SLEEP_RERUNS = 3


def main():
    keep_awake()
    env = dict(os.environ, OMP_NUM_THREADS="2", TORCH_THREADS="2", PYTHONIOENCODING="utf-8")
    jobs = [(m, s) for s in SEEDS for m in METHODS]
    lock, stop = threading.Lock(), threading.Event()
    outcome = {}
    log(f"novelty_fix queue start (PID {os.getpid()}), {WORKERS} workers, seed-major order: {jobs}")

    def worker(wid):
        while not stop.is_set():
            with lock:
                if not jobs:
                    return
                m, s = jobs.pop(0)
            if datetime.datetime.now() >= DEADLINE:
                log(f"novelty_fix queue: hard stop passed; {m} seed {s} not run.")
                outcome[(m, s)] = "not run (hard stop)"
                stop.set()
                return
            crashes = sleeps = 0
            logf = open(f"experiments/novelty_fix/logs/{m}_seed{s}.log", "a", encoding="utf-8")
            while True:
                rc = subprocess.call([sys.executable, "-u", "-m", "src.adarnn_fix.run", "--method", m, "--seed", str(s)],
                                     env=env, stdout=logf, stderr=subprocess.STDOUT)
                if rc == 0:
                    outcome[(m, s)] = "done"
                    break
                if rc == 3:
                    log(f"novelty_fix queue (worker {wid}): hard stop during {m} seed {s}; no further jobs start.")
                    outcome[(m, s)] = "not run (hard stop)"
                    stop.set()
                    break
                if rc == 4 and sleeps < MAX_SLEEP_RERUNS:
                    sleeps += 1
                    log(f"novelty_fix queue (worker {wid}): sleep detected in {m} seed {s}; rerun {sleeps}/{MAX_SLEEP_RERUNS}.")
                    continue
                crashes += 1
                if crashes >= 2:
                    log(f"novelty_fix queue (worker {wid}): second crash (rc {rc}) in {m} seed {s}; job skipped, continuing.")
                    outcome[(m, s)] = f"skipped after second crash (rc {rc})"
                    break
                log(f"novelty_fix queue (worker {wid}): crash (rc {rc}) in {m} seed {s}; resuming once (rerun from scratch).")
            logf.close()

    os.makedirs("experiments/novelty_fix/logs", exist_ok=True)
    ts = [threading.Thread(target=worker, args=(i,)) for i in range(WORKERS)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    log(f"novelty_fix queue finished: { {f'{m} s{s}': v for (m, s), v in outcome.items()} }")


if __name__ == "__main__":
    main()
