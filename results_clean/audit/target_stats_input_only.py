"""
Target-train stats from input-window values only (drops the final H label-only values of the target-train split),
and seed-0 "source-only, target-train stats" re-evaluated with them. Inference only.

  set OMP_NUM_THREADS=2 && venv\\Scripts\\python -m results_clean.audit.target_stats_input_only
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import json
from pathlib import Path

import numpy as np
import torch

from src.data.prepare import chronological_split, prepare_domain
from src.evaluate import evaluate_model, load_model_checkpoint
from src.experiment import load_series, prepare_pair
from src.run_experiments import DOMAIN_CONFIGS

torch.set_num_threads(2)
FINAL = Path("experiments/final_3seed/tcn")
out = {}
for d in ["energy", "climate", "finance", "finance_vol"]:
    cfg = DOMAIN_CONFIGS[d]
    L, H = cfg["input_len"], cfg["horizon"]
    _, _, tgt_tgt, _ = prepare_pair(cfg)
    old = tgt_tgt["norm_stats"]
    series = load_series(cfg, "target")
    train_raw = chronological_split(series)[0]
    inp = train_raw[: len(train_raw) - H]  # window i uses inputs i..i+L-1; last input index is n_train-H-1
    # check: exactly the set of positions covered by target-train input windows
    n_win = len(tgt_tgt["X_train"])
    assert len(inp) == n_win + L - 1
    new = {"mean": float(inp.mean()), "std": float(inp.std())}  # ddof=0, as src.data.prepare.normalize
    # recomputing the old stats the same way must reproduce the pipeline's values
    assert np.isclose(train_raw.mean(), old["mean"], rtol=0, atol=1e-9 * max(1, abs(old["mean"])))
    assert np.isclose(train_raw.std(), old["std"], rtol=1e-12)

    model, ck = load_model_checkpoint(FINAL / "checkpoints" / f"{d}_seed0_source_only.pt")
    assert ck["seed"] == 0
    t_new = prepare_domain(series, L, H, stats=new)
    m_old = evaluate_model(model, tgt_tgt["X_test"], tgt_tgt["Y_test"], old)
    m_new = evaluate_model(model, t_new["X_test"], t_new["Y_test"], new)
    stored = json.load(open(FINAL / "runs" / f"{d}_seed0.json"))["source_only_on_target_test"]["target_stats"]
    out[d] = {"H": H, "n_train_values": int(len(train_raw)), "n_input_only_values": int(len(inp)),
              "n_label_only_values_dropped": int(H), "old_stats": old, "new_stats": new,
              "rel_diff_mean": (new["mean"] - old["mean"]) / abs(old["mean"]),
              "rel_diff_std": (new["std"] - old["std"]) / old["std"],
              "stored_seed0": {"MAE": stored["MAE"], "RMSE": stored["RMSE"]},
              "recomputed_old_stats": {"MAE": m_old["MAE"], "RMSE": m_old["RMSE"]},
              "recomputed_input_only_stats": {"MAE": m_new["MAE"], "RMSE": m_new["RMSE"]}}
    print(d, json.dumps(out[d]), flush=True)
with open("results_clean/audit/target_stats_input_only.json", "w") as f:
    json.dump(out, f, indent=1)
