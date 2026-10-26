"""
Packages the Streamlit app's inputs; no training. Copies seed-0 TCN checkpoints (source-only from
experiments/final_3seed/tcn, CORAL λ=10 from experiments/coral) into app/checkpoints/ unchanged, writes
app/checkpoints/manifest.json (domain-level preprocessing only; input_len, horizon and normalisation stats are read
from each checkpoint), and builds one upload-format sample per domain in app/samples/ from data/processed.

  python -m src.build_app_assets
"""
import json
import shutil
from pathlib import Path

import pandas as pd
import torch

APP = Path("app")
CKPT = APP / "checkpoints"
SAMPLES = APP / "samples"

DOMAINS = {
    "climate": {"label": "climate (Shimla → Delhi, daily mean temp)", "frequency": "D", "preprocess": "fill_daily",
                "transform": None, "units": "°C", "value_name": "temp",
                "sample": ("data/processed/climate/delhi_clean.csv", "date", "temp", None)},
    "energy": {"label": "energy (AEP → COMED, hourly load)", "frequency": "h", "preprocess": "fill_hourly",
               "transform": None, "units": "MW", "value_name": "load_mw",
               "sample": ("data/processed/energy/comed_clean.csv", "datetime", "load_mw", 3000)},
    "finance_vol": {"label": "finance_vol (Nifty IT → Pharma, |log-return|)", "frequency": "B",
                    "preprocess": "close_to_log_return", "transform": "abs", "units": "|log-return|",
                    "value_name": "Close", "sample": ("data/processed/finance/nifty_pharma_clean.csv", "date", "Close", None)},
    "finance": {"label": "finance: no learnable signal (negative control)", "frequency": "B",
                "preprocess": "close_to_log_return", "transform": None, "units": "log-return", "value_name": "Close",
                "sample": ("data/processed/finance/nifty_pharma_clean.csv", "date", "Close", None)},
}
SOURCES = {
    "source_only": lambda d: Path("experiments/final_3seed/tcn/checkpoints") / f"{d}_seed0_source_only.pt",
    "coral": lambda d: Path("experiments/coral") / d / "seed0.pt",
}


def main():
    CKPT.mkdir(parents=True, exist_ok=True)
    SAMPLES.mkdir(parents=True, exist_ok=True)
    manifest = {"note": "domain-level preprocessing; input_len, horizon and norm stats come from each checkpoint",
                "domains": {}}
    for d, meta in DOMAINS.items():
        files = {}
        for kind, src in SOURCES.items():
            src = src(d)
            dst = CKPT / f"{d}_{kind}_seed0.pt"
            shutil.copy2(src, dst)
            ck = torch.load(dst, map_location="cpu")
            assert {"state_dict", "backbone", "input_len", "horizon", "norm_stats"} <= set(ck), (dst, list(ck))
            files[kind] = {"file": dst.name, "copied_from": str(src).replace("\\", "/")}
        path, date_col, value_col, last_n = meta["sample"]
        df = pd.read_csv(path, index_col=0)
        out = pd.DataFrame({date_col: df.index, value_col: df[value_col].values})
        if last_n:
            out = out.tail(last_n)
        sample = SAMPLES / f"{d}_sample.csv"
        out.to_csv(sample, index=False)
        manifest["domains"][d] = {k: v for k, v in meta.items() if k != "sample"} | {
            "checkpoints": files, "sample": sample.name, "sample_from": path}
        print(d, files, sample, len(out))
    (CKPT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
