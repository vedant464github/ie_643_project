"""
Runs a queue of `src.adarnn_full.run` invocations sequentially in one process slot (CONSTRAINTS.md 2026-11-15 (h)):
on a crash the same invocation is resumed once (finished JSONs are skipped); a second crash stops the queue.
Exit code 3 from a child = hard stop reached; the queue ends.

  python -m src.adarnn_full.driver --name climate_A "final --domain climate --method source_only" "grid --domain ..."
"""
import argparse
import os
import shlex
import subprocess
import sys

from src.utils.status import log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("jobs", nargs="+")
    a = ap.parse_args()
    env = dict(os.environ, OMP_NUM_THREADS="2", PYTHONIOENCODING="utf-8")
    crashes = 0
    log(f"novelty queue {a.name} (PID {os.getpid()}) start: {a.jobs}")
    for job in a.jobs:
        while True:
            rc = subprocess.call([sys.executable, "-u", "-m", "src.adarnn_full.run"] + shlex.split(job), env=env)
            if rc == 0:
                break
            if rc == 3:
                log(f"novelty queue {a.name}: hard stop during '{job}'; queue ends, remaining jobs not run.")
                return
            crashes += 1
            if crashes >= 2:
                log(f"novelty queue {a.name}: second crash (rc {rc}) in '{job}'; queue stopped per failure rules.")
                sys.exit(1)
            log(f"novelty queue {a.name}: crash (rc {rc}) in '{job}'; resuming once.")
    log(f"novelty queue {a.name}: all jobs finished.")


if __name__ == "__main__":
    main()
