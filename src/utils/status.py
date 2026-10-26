"""Timestamped entries in experiments/STATUS.md:  python -m src.utils.status "message" """
import datetime
import sys
from pathlib import Path

STATUS = Path("experiments/STATUS.md")

def log(msg):
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(STATUS, "a", encoding="utf-8") as f:
        f.write(f"\n- **{stamp}** — {msg}\n")
    line = f"[{stamp}] {msg}"
    enc = sys.stdout.encoding or "utf-8"
    print(line.encode(enc, errors="replace").decode(enc), flush=True)  # the file copy keeps the exact text

if __name__ == "__main__":
    log(" ".join(sys.argv[1:]))
