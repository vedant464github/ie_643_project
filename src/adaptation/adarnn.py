"""
AdaRNN-style adaptation (simplified, pre-registered in CONSTRAINTS.md Amendments 2026-10-25 (b)).

TDC: split the source-train input windows into K=3 contiguous periods, split points from a 10% grid with every
period >= 20% of windows, chosen to maximise the summed pairwise distance between periods' input-window mean and
covariance. Inputs only, computed once before training.
TDM (simplified): MSE on the period batches + lambda * mean over all pairs among {P1, P2, P3, target} of CORAL on
get_features(). No boosting / importance weights. 3 MSE-only pretrain epochs, then joint training with equal
sampling (one batch per period and one target-train batch per step); early stopping on SOURCE val MSE from the
first joint epoch only.
"""
import copy
import datetime
import itertools
import time

import numpy as np
import torch
import torch.nn as nn

from src.adaptation.coral import coral_loss, TimeLimitExceeded
from src.evaluate import predict
from src.train import rss_mb
from src.train_source_only import make_loader


class DeadlineExceeded(RuntimeError):
    pass


def period_distance(Xa, Xb):
    """||mu_a - mu_b||^2 / L + ||C_a - C_b||_F^2 / L^2 on L-dim input windows."""
    L = Xa.shape[1]
    Xa, Xb = Xa.astype(np.float64), Xb.astype(np.float64)
    dmu = ((Xa.mean(0) - Xb.mean(0)) ** 2).sum() / L
    dcov = ((np.cov(Xa, rowvar=False) - np.cov(Xb, rowvar=False)) ** 2).sum() / L ** 2
    return float(dmu + dcov)


def tdc_split(X, grid=tuple(np.arange(1, 10) / 10), min_frac=0.2):
    """Returns the chosen split indices and every candidate's score."""
    n = len(X)
    candidates = []
    for a, b in itertools.combinations(grid, 2):
        if a < min_frac - 1e-9 or b - a < min_frac - 1e-9 or 1 - b < min_frac - 1e-9:
            continue
        i, j = int(round(a * n)), int(round(b * n))
        parts = [X[:i], X[i:j], X[j:]]
        score = sum(period_distance(parts[p], parts[q]) for p, q in itertools.combinations(range(3), 2))
        candidates.append({"split_fracs": [round(float(a), 2), round(float(b), 2)], "split_idx": [i, j], "score": score})
    best = max(candidates, key=lambda c: c["score"])
    return {"chosen": best, "n_windows": n, "candidates": candidates,
            "distance": "||mu_i-mu_j||^2/L + ||C_i-C_j||_F^2/L^2, summed over the 3 period pairs"}


def _cycle(loader):
    while True:
        for batch in loader:
            yield batch


def fit_adarnn(model, X_train, Y_train, split_idx, Xt_train, X_val, Y_val, lam, epochs=20, batch_size=64, lr=1e-3,
               pretrain_epochs=3, patience=3, max_seconds=None, deadline=None, verbose=True, tag=""):
    """epochs is the total cap, pretrain epochs included. deadline: datetime after which a DeadlineExceeded is raised."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    device = next(model.parameters()).device
    i, j = split_idx
    bounds = [(0, i), (i, j), (j, len(X_train))]
    period_iters = [_cycle(make_loader_dl(X_train[s:e], Y_train[s:e], batch_size)) for s, e in bounds]
    tgt_iter = _cycle(torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(Xt_train, dtype=torch.float32)),
        batch_size=batch_size, shuffle=True, drop_last=True))
    steps_per_epoch = len(X_train) // (3 * batch_size)
    pretrain_loader = make_loader(X_train, Y_train, batch_size, shuffle=True)

    history = {"phase": [], "train_loss": [], "train_mse": [], "train_coral": [], "val_mse": [], "rss_mb": [],
               "pretrain_epochs": pretrain_epochs, "joint_steps_per_epoch": steps_per_epoch}
    best_val, best_state, best_epoch, patience_counter = float("inf"), None, None, 0
    start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        losses, mses, corals = [], [], []
        if epoch <= pretrain_epochs:
            phase = "pretrain"
            for X_batch, Y_batch in pretrain_loader:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                optimizer.zero_grad()
                loss = mse(model(X_batch), Y_batch)
                loss.backward()
                optimizer.step()
                losses.append(loss.item()); mses.append(loss.item()); corals.append(0.0)
        else:
            phase = "joint"
            for _ in range(steps_per_epoch):
                batches = [next(it) for it in period_iters]
                Xs = [b[0].to(device) for b in batches]
                Ys = torch.cat([b[1] for b in batches]).to(device)
                Xt_batch = next(tgt_iter)[0].to(device)
                optimizer.zero_grad()
                feats = [model.get_features(x) for x in Xs]
                l_mse = mse(model.head(torch.cat(feats)), Ys)
                feats.append(model.get_features(Xt_batch))
                l_coral = torch.stack([coral_loss(a, b) for a, b in itertools.combinations(feats, 2)]).mean()
                loss = l_mse + lam * l_coral
                loss.backward()
                optimizer.step()
                losses.append(loss.item()); mses.append(l_mse.item()); corals.append(l_coral.item())
        history["phase"].append(phase)
        history["train_loss"].append(float(np.mean(losses)))
        history["train_mse"].append(float(np.mean(mses)))
        history["train_coral"].append(float(np.mean(corals)))
        val_mse = float(np.mean((predict(model, X_val) - Y_val) ** 2))
        history["val_mse"].append(val_mse)
        history["rss_mb"].append(rss_mb())
        elapsed = time.perf_counter() - start
        if verbose:
            print(f"  {tag}Epoch {epoch:2d} [{phase}] | loss={history['train_loss'][-1]:.5f} | "
                  f"mse={history['train_mse'][-1]:.5f} | coral={history['train_coral'][-1]:.6f} | "
                  f"val_mse={val_mse:.5f} | rss={history['rss_mb'][-1]:.0f}MB | {elapsed:.0f}s", flush=True)
        if max_seconds and elapsed > max_seconds:
            raise TimeLimitExceeded(f"{tag}training exceeded {max_seconds}s at epoch {epoch}")
        if deadline and datetime.datetime.now() >= deadline:
            raise DeadlineExceeded(f"{tag}hard stop {deadline:%H:%M} reached at epoch {epoch}")

        if phase == "pretrain":
            continue
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


def make_loader_dl(X, Y, batch_size):
    """Shuffled period loader with drop_last, so every CORAL batch has the full batch size."""
    ds = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32))
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
