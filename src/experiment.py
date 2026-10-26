"""
One source->target transfer run for a given backbone and seed, plus seed-independent target baselines.
No target labels are used for early stopping or model selection: source-only is selected on SOURCE val;
target-only trains a fixed number of epochs (= that seed's source best epoch).
"""
import time

import numpy as np
import torch

from src.data.prepare import prepare_domain
from src.data.windowing import load_domain_series
from src.evaluate import predict, denorm, evaluate_model, naive_mean_metrics, mean_last_k_baseline, seasonal_naive
from src.models.backbone import build_model, count_params
from src.train import DEVICE, device_info, set_seed, fit
from src.utils.metrics import evaluate_all

def load_series(cfg, side):
    return load_domain_series(cfg[side]["csv_path"], cfg["value_col"], transform=cfg.get("transform"))

def prepare_pair(cfg):
    """Source split (source stats) and target split under source stats and under target train stats."""
    L, H = cfg["input_len"], cfg["horizon"]
    src = prepare_domain(load_series(cfg, "source"), L, H)
    tgt_series = load_series(cfg, "target")
    tgt_src = prepare_domain(tgt_series, L, H, stats=src["norm_stats"])
    tgt_tgt = prepare_domain(tgt_series, L, H)
    # both normalisations must describe the same original-scale test targets
    assert np.allclose(denorm(tgt_src["Y_test"], src["norm_stats"]),
                       denorm(tgt_tgt["Y_test"], tgt_tgt["norm_stats"]), rtol=1e-4, atol=1e-4)
    return src, tgt_src, tgt_tgt, tgt_series

def target_baselines(cfg):
    """Naive train-mean (both stats) and seasonal-naive on the target test windows."""
    src, tgt_src, tgt_tgt, tgt_series = prepare_pair(cfg)
    Y_true = denorm(tgt_tgt["Y_test"], tgt_tgt["norm_stats"])
    out = {
        "naive_train_mean": {
            "source_stats": naive_mean_metrics(tgt_src["Y_test"], src["norm_stats"]),
            "target_stats": naive_mean_metrics(tgt_tgt["Y_test"], tgt_tgt["norm_stats"]),
        },
        "seasonal_naive": {},
        "device_info": device_info(),
        "n_target_test_windows": int(len(Y_true)),
        "norm_stats": {"source_train": src["norm_stats"], "target_train": tgt_tgt["norm_stats"]},
    }
    if cfg.get("mean_last_k"):
        k = cfg["mean_last_k"]
        out[f"mean_of_last_{k}_inputs"] = {
            "source_stats": mean_last_k_baseline(tgt_src["X_test"], tgt_src["Y_test"], src["norm_stats"], k),
            "target_stats": mean_last_k_baseline(tgt_tgt["X_test"], tgt_tgt["Y_test"], tgt_tgt["norm_stats"], k),
        }
    for period in cfg["seasonal_periods"]:
        sn_true, sn_pred = seasonal_naive(tgt_series, cfg["input_len"], cfg["horizon"], period)
        assert np.allclose(sn_true, Y_true, rtol=1e-4, atol=1e-3), "seasonal-naive windows misaligned"
        out["seasonal_naive"][f"period_{period}"] = evaluate_all(sn_true, sn_pred)
    return out

def run_transfer(cfg, backbone, seed, train_stride=1, max_epochs=None, target_only=True,
                 model_kwargs=None, checkpoint_prefix=None, verbose=True):
    """
    train_stride: use every k-th training window (source and target-only alike).
    max_epochs: overrides cfg["epochs"] as the epoch cap.
    """
    L, H = cfg["input_len"], cfg["horizon"]
    model_kwargs = model_kwargs or {}
    epochs = max_epochs or cfg["epochs"]
    src, tgt_src, tgt_tgt, _ = prepare_pair(cfg)
    src_stats, tgt_stats = src["norm_stats"], tgt_tgt["norm_stats"]

    set_seed(seed)
    model = build_model(backbone, L, H, **model_kwargs).to(DEVICE)
    t0 = time.perf_counter()
    hist = fit(model, src["X_train"][::train_stride], src["Y_train"][::train_stride],
               src["X_val"], src["Y_val"], epochs=epochs, batch_size=cfg["batch_size"],
               verbose=verbose, tag=f"[{backbone} s{seed} source] ")

    val_pred = predict(model, src["X_val"])
    res = {
        "backbone": backbone, "seed": seed, "device_info": device_info(), "params": count_params(model), "model_kwargs": model_kwargs,
        "train_stride": train_stride, "max_epochs": epochs, "batch_size": cfg["batch_size"], "lr": 1e-3,
        "n_source_train_windows": int(len(src["X_train"][::train_stride])),
        "source_training": hist,
        "source_val": {
            "mse_norm": float(np.mean((val_pred - src["Y_val"]) ** 2)),
            "mae_norm": float(np.mean(np.abs(val_pred - src["Y_val"]))),
            "orig_scale": evaluate_all(denorm(src["Y_val"], src_stats), denorm(val_pred, src_stats)),
        },
        "source_test": evaluate_model(model, src["X_test"], src["Y_test"], src_stats),
        "source_only_on_target_test": {
            "source_stats": evaluate_model(model, tgt_src["X_test"], tgt_src["Y_test"], src_stats),
            "target_stats": evaluate_model(model, tgt_tgt["X_test"], tgt_tgt["Y_test"], tgt_stats),
        },
    }
    if checkpoint_prefix:
        torch.save({"state_dict": model.state_dict(), "backbone": backbone, "model_kwargs": model_kwargs,
                    "input_len": L, "horizon": H, "norm_stats": src_stats, "seed": seed,
                    "best_epoch": hist["best_epoch"]}, f"{checkpoint_prefix}_source_only.pt")

    if target_only:
        set_seed(seed)
        tmodel = build_model(backbone, L, H, **model_kwargs).to(DEVICE)
        thist = fit(tmodel, tgt_tgt["X_train"][::train_stride], tgt_tgt["Y_train"][::train_stride],
                    epochs=hist["best_epoch"], batch_size=cfg["batch_size"],
                    verbose=verbose, tag=f"[{backbone} s{seed} target_only] ")
        res["target_only_on_target_test"] = {
            "target_stats": evaluate_model(tmodel, tgt_tgt["X_test"], tgt_tgt["Y_test"], tgt_stats),
            "epochs": hist["best_epoch"],
            "epoch_rule": "fixed = this seed's source best epoch (source val); no target val used",
            "training": thist,
        }
        if checkpoint_prefix:
            torch.save({"state_dict": tmodel.state_dict(), "backbone": backbone, "model_kwargs": model_kwargs,
                        "input_len": L, "horizon": H, "norm_stats": tgt_stats, "seed": seed,
                        "epochs": hist["best_epoch"]}, f"{checkpoint_prefix}_target_only.pt")

    res["wall_clock_seconds"] = round(time.perf_counter() - t0, 2)
    return res
