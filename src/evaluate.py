import math

import numpy as np
import torch

from src.models.backbone import GRUForecaster
from src.utils.metrics import evaluate_all

def load_checkpoint(path):
    """Rebuilds a GRUForecaster from a checkpoint saved by train_source_only. Returns (model, ckpt)."""
    ckpt = torch.load(path, map_location="cpu")
    model = GRUForecaster(**ckpt["model_kwargs"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt

def load_model_checkpoint(path):
    """Any backbone checkpoint saved by run_final_seeds / run_coral / run_adarnn. CPU. Returns (model, ckpt)."""
    from src.models.backbone import build_model
    ckpt = torch.load(path, map_location="cpu")
    model = build_model(ckpt["backbone"], ckpt["input_len"], ckpt["horizon"], **ckpt.get("model_kwargs", {}))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt

def predict(model, X, batch_size=1024):
    """Runs the model over X (normalized space) in batches. Returns numpy predictions."""
    model.eval()
    device = next(model.parameters()).device
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i:i + batch_size], dtype=torch.float32, device=device)
            preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds)

def denorm(x, stats):
    return x * stats["std"] + stats["mean"]

def evaluate_model(model, X, Y, stats):
    """Metrics in original scale; X/Y must have been normalized with the same stats."""
    return evaluate_all(denorm(Y, stats), denorm(predict(model, X), stats))

def naive_mean_metrics(Y, stats):
    """Predict-train-mean baseline (= zero in normalized space), original scale."""
    return evaluate_all(denorm(Y, stats), denorm(np.zeros_like(Y), stats))

def mean_last_k_baseline(X, Y, stats, k):
    """Predict the mean of the last k inputs for every horizon step, original scale."""
    pred = np.repeat(X[:, -k:].mean(axis=1, keepdims=True), Y.shape[1], axis=1)
    return evaluate_all(denorm(Y, stats), denorm(pred, stats))

def seasonal_naive(series, input_len, horizon, period, train_frac=0.7, val_frac=0.15):
    """
    Seasonal-naive forecast for every test window: y_hat[t+j] = y[t + j - period*ceil(j/period)],
    where t is the last observed step of the window. period=1 is persistence (repeat last value).
    Uses only past observations (may reach back before the test split), never future ones.
    Returns (Y_true, Y_pred) in original scale, aligned with prepare_domain's test windows.
    """
    values = np.asarray(series, dtype=np.float64)
    n = len(values)
    test_start = int(n * (train_frac + val_frac))
    n_windows = n - test_start - input_len - horizon + 1
    t = test_start + np.arange(n_windows)[:, None] + input_len - 1  # last observed index per window
    j = np.arange(1, horizon + 1)[None, :]
    lag_idx = t + j - period * np.ceil(j / period).astype(int)
    if lag_idx.min() < 0:
        raise ValueError(f"Not enough history for seasonal period {period}")
    return values[t + j], values[lag_idx]
