"""
Deep CORAL (Sun & Saenko, 2016) for the forecasting backbones, plus a label-free MMD gap diagnostic.

Joint training: loss = MSE(source) + lambda * CORAL(get_features(source batch), get_features(target batch)).
Target batches are target-TRAIN inputs only (normalised with target-train stats); no target labels, no target
val or test. Early stopping on SOURCE val MSE, same rule as src.train.fit (patience 3, min improvement 1e-5,
best-val weights restored). lambda=0 runs through the same code (target batches are still drawn) so the
reference differs from the CORAL runs only in lambda.
"""
import copy
import time

import numpy as np
import torch
import torch.nn as nn

from src.evaluate import predict
from src.train import rss_mb
from src.train_source_only import make_loader


class TimeLimitExceeded(RuntimeError):
    pass


def coral_loss(fs, ft):
    """||C_s - C_t||_F^2 / (4 d^2), covariances over the batch dimension (unbiased)."""
    d = fs.shape[1]
    def cov(f):
        f = f - f.mean(dim=0, keepdim=True)
        return f.T @ f / (f.shape[0] - 1)
    return ((cov(fs) - cov(ft)) ** 2).sum() / (4 * d * d)


def _cycle(loader):
    while True:
        for (xb,) in loader:
            yield xb


def fit_coral(model, X_train, Y_train, Xt_train, X_val, Y_val, lam, epochs=20, batch_size=64, lr=1e-3,
              patience=3, max_seconds=None, verbose=True, tag=""):
    """
    Xt_train: target-train inputs (no labels). One target batch of the same size is drawn per source batch.
    max_seconds: wall-clock cap for training; raises TimeLimitExceeded when exceeded.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    src_loader = make_loader(X_train, Y_train, batch_size, shuffle=True)
    tgt_iter = _cycle(torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(Xt_train, dtype=torch.float32)),
        batch_size=batch_size, shuffle=True, drop_last=True))
    device = next(model.parameters()).device

    history = {"train_loss": [], "train_mse": [], "train_coral": [], "val_mse": [], "rss_mb": []}
    best_val, best_state, best_epoch, patience_counter = float("inf"), None, epochs, 0
    start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        losses, mses, corals = [], [], []
        for X_batch, Y_batch in src_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            Xt_batch = next(tgt_iter).to(device)
            optimizer.zero_grad()
            fs = model.get_features(X_batch)
            ft = model.get_features(Xt_batch)
            l_mse = mse(model.head(fs), Y_batch)
            # a final source batch of size 1 has no covariance; it contributes MSE only
            l_coral = coral_loss(fs, ft) if len(X_batch) > 1 else torch.zeros((), device=device)
            loss = l_mse + lam * l_coral
            loss.backward()
            optimizer.step()
            losses.append(loss.item()); mses.append(l_mse.item()); corals.append(l_coral.item())
        history["train_loss"].append(float(np.mean(losses)))
        history["train_mse"].append(float(np.mean(mses)))
        history["train_coral"].append(float(np.mean(corals)))

        val_mse = float(np.mean((predict(model, X_val) - Y_val) ** 2))
        history["val_mse"].append(val_mse)
        history["rss_mb"].append(rss_mb())
        elapsed = time.perf_counter() - start
        if verbose:
            print(f"  {tag}Epoch {epoch:2d} | loss={history['train_loss'][-1]:.5f} | mse={history['train_mse'][-1]:.5f}"
                  f" | coral={history['train_coral'][-1]:.6f} | val_mse={val_mse:.5f}"
                  f" | rss={history['rss_mb'][-1]:.0f}MB | {elapsed:.0f}s", flush=True)
        if max_seconds and elapsed > max_seconds:
            raise TimeLimitExceeded(f"{tag}training exceeded {max_seconds}s at epoch {epoch}")

        if val_mse < best_val - 1e-5:
            best_val, best_state, best_epoch, patience_counter = val_mse, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"  {tag}Early stopping at epoch {epoch}; restoring epoch {best_epoch}")
                break

    model.load_state_dict(best_state)
    model.eval()
    history.update({"best_epoch": best_epoch, "stopped_epoch": epoch, "best_val_mse": best_val,
                    "train_seconds": round(time.perf_counter() - start, 2)})
    return history


def features(model, X, batch_size=1024):
    model.eval()
    device = next(model.parameters()).device
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            out.append(model.get_features(torch.tensor(X[i:i + batch_size], dtype=torch.float32,
                                                       device=device)).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def mmd2_rbf(A, B, scales=(0.25, 0.5, 1.0, 2.0, 4.0)):
    """
    Unbiased multi-kernel RBF MMD^2. Bandwidth = median pairwise distance of the pooled sample times each scale
    (median heuristic per model, so the value does not depend on the feature scale of that model).
    """
    Z = np.concatenate([A, B])
    sq = (Z ** 2).sum(1)
    D2 = np.maximum(sq[:, None] + sq[None, :] - 2 * Z @ Z.T, 0)
    med = np.median(D2[np.triu_indices(len(Z), 1)])
    n, m = len(A), len(B)
    K = sum(np.exp(-D2 / (2 * (s ** 2) * med)) for s in scales)
    Kaa, Kbb, Kab = K[:n, :n], K[n:, n:], K[:n, n:]
    return float((Kaa.sum() - np.trace(Kaa)) / (n * (n - 1)) + (Kbb.sum() - np.trace(Kbb)) / (m * (m - 1))
                 - 2 * Kab.mean()), float(np.sqrt(med))


def gap_diagnostic(model, Xs_val, Xt_val, max_n=2000, seed=0):
    """Label-free domain gap between source-val and target-val input features (inputs only, no targets)."""
    rng = np.random.default_rng(seed)
    fs, ft = features(model, Xs_val), features(model, Xt_val)
    a = fs[rng.choice(len(fs), min(max_n, len(fs)), replace=False)]
    b = ft[rng.choice(len(ft), min(max_n, len(ft)), replace=False)]
    mmd2, bw = mmd2_rbf(a, b)
    coral = float(coral_loss(torch.tensor(fs), torch.tensor(ft)))
    return {"mmd2_rbf": mmd2, "median_bandwidth": bw, "coral_distance": coral,
            "n_source_val": int(len(a)), "n_target_val": int(len(b)),
            "feature_norm_mean": {"source": float(np.linalg.norm(fs, axis=1).mean()),
                                  "target": float(np.linalg.norm(ft, axis=1).mean())},
            "features": "get_features() on source-val inputs (source stats) and target-val inputs (target-train stats)"}
