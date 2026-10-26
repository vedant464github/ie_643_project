import numpy as np
import pandas as pd
from src.data.windowing import make_windows

def chronological_split(series, train_frac=0.7, val_frac=0.15):
    """Splits a 1D series into train/val/test by time order. Returns 3 arrays."""
    n = len(series)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    train = series[:train_end]
    val = series[train_end:val_end]
    test = series[val_end:]
    return train, val, test

def apply_norm(x, stats):
    """Z-score with given {"mean", "std"} stats (inverse: src.evaluate.denorm)."""
    return (x - stats["mean"]) / stats["std"]

def normalize(train, val, test, stats=None):
    """Z-score normalization. Mean/std computed on TRAIN ONLY, applied to all three.
    If stats is given (e.g. another domain's train stats), those are used instead."""
    if stats is None:
        mean = train.mean()
        std = train.std()
        if std == 0:
            std = 1.0  # avoid divide-by-zero on a constant series
    else:
        mean, std = stats["mean"], stats["std"]
    stats = {"mean": float(mean), "std": float(std)}
    return apply_norm(train, stats), apply_norm(val, stats), apply_norm(test, stats), stats

def prepare_domain(series, input_len, horizon, train_frac=0.7, val_frac=0.15, stats=None):
    """
    Full pipeline for one domain's series:
    chronological split -> normalize (train stats only, or the given stats) -> window each split.
    Returns dict with X/Y for train/val/test, plus normalization stats for inverse-transform later.
    """
    train_raw, val_raw, test_raw = chronological_split(series, train_frac, val_frac)
    train_n, val_n, test_n, stats = normalize(train_raw, val_raw, test_raw, stats)

    X_train, Y_train = make_windows(train_n, input_len, horizon)
    X_val, Y_val = make_windows(val_n, input_len, horizon)
    X_test, Y_test = make_windows(test_n, input_len, horizon)

    return {
        "X_train": X_train, "Y_train": Y_train,
        "X_val": X_val, "Y_val": Y_val,
        "X_test": X_test, "Y_test": Y_test,
        "norm_stats": stats,
    }

if __name__ == "__main__":
    from src.data.windowing import load_domain_series

    configs = [
        ("data/processed/finance/nifty_it_clean.csv", "log_return", 30, 5),
        ("data/processed/climate/shimla_clean.csv", "temp", 30, 5),
        ("data/processed/energy/aep_clean.csv", "load_mw", 168, 24),
    ]
    for path, col, in_len, hor in configs:
        series = load_domain_series(path, col)
        d = prepare_domain(series, in_len, hor)
        print(f"{path}")
        print(f"  train X{d['X_train'].shape} Y{d['Y_train'].shape}")
        print(f"  val   X{d['X_val'].shape} Y{d['Y_val'].shape}")
        print(f"  test  X{d['X_test'].shape} Y{d['Y_test'].shape}")
        print(f"  norm_stats: {d['norm_stats']}")