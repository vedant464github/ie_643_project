"""
Copies of src.train.fit (MSE, source-only) and src.adaptation.coral.fit_coral with identical training logic
(same loaders, batch order, loss, early stopping), plus per-epoch timing, the hard-stop deadline and the
energy 300 s/epoch rule (CONSTRAINTS.md 2026-11-15 (h)). The originals are not modified.
"""
import copy
import datetime
import time

import numpy as np
import torch
import torch.nn as nn

from src.adaptation.coral import coral_loss
from src.adarnn_full.adarnn import DeadlineExceeded, EpochTooSlow
from src.evaluate import predict
from src.train import rss_mb
from src.train_source_only import make_loader


def _cycle(loader):
    while True:
        for (xb,) in loader:
            yield xb


def fit_timed(model, X_train, Y_train, Xt_train, X_val, Y_val, lam=None, epochs=20, batch_size=64, lr=1e-3,
              patience=3, deadline=None, max_epoch_seconds=None, epoch_callback=None, verbose=True, tag=""):
    """lam=None: plain MSE (== src.train.fit with val). lam=float: CORAL on get_features (== fit_coral)."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    src_loader = make_loader(X_train, Y_train, batch_size, shuffle=True)
    coral = lam is not None
    if coral:
        tgt_iter = _cycle(torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.tensor(Xt_train, dtype=torch.float32)),
            batch_size=batch_size, shuffle=True, drop_last=True))
    device = next(model.parameters()).device
    hist = {"train_loss": [], "train_mse": [], "train_coral": [], "val_mse": [], "epoch_seconds": [], "rss_mb": []}
    best_val, best_state, best_epoch, wait = float("inf"), None, epochs, 0
    start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        t_ep = time.perf_counter()
        model.train()
        losses, mses, corals = [], [], []
        for X_batch, Y_batch in src_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            if coral:
                Xt_batch = next(tgt_iter).to(device)
            optimizer.zero_grad()
            if coral:
                fs, ft = model.get_features(X_batch), model.get_features(Xt_batch)
                l_mse = mse(model.head(fs), Y_batch)
                l_coral = coral_loss(fs, ft) if len(X_batch) > 1 else torch.zeros((), device=device)
                loss = l_mse + lam * l_coral
            else:
                l_mse = loss = mse(model(X_batch), Y_batch)
                l_coral = torch.zeros(())
            loss.backward()
            optimizer.step()
            losses.append(loss.item()); mses.append(l_mse.item()); corals.append(float(l_coral))
        val_mse = float(np.mean((predict(model, X_val) - Y_val) ** 2))
        ep_s = time.perf_counter() - t_ep
        for k, v in [("train_loss", float(np.mean(losses))), ("train_mse", float(np.mean(mses))),
                     ("train_coral", float(np.mean(corals))), ("val_mse", val_mse), ("epoch_seconds", round(ep_s, 1)),
                     ("rss_mb", rss_mb())]:
            hist[k].append(v)
        if verbose:
            print(f"  {tag}Epoch {epoch:2d} | loss={hist['train_loss'][-1]:.5f} | mse={hist['train_mse'][-1]:.5f}"
                  f" | coral={hist['train_coral'][-1]:.6f} | val_mse={val_mse:.5f} | {ep_s:.0f}s/epoch"
                  f" | rss={hist['rss_mb'][-1]:.0f}MB", flush=True)
        if epoch_callback:
            epoch_callback(epoch, ep_s)
        if max_epoch_seconds and ep_s > max_epoch_seconds:
            raise EpochTooSlow(f"{tag}epoch {epoch} took {ep_s:.0f}s > {max_epoch_seconds}s")
        if deadline and datetime.datetime.now() >= deadline:
            raise DeadlineExceeded(f"{tag}hard stop {deadline} reached at epoch {epoch}")
        if val_mse < best_val - 1e-5:
            best_val, best_state, best_epoch, wait = val_mse, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"  {tag}Early stopping at epoch {epoch}; restoring epoch {best_epoch}")
                break
    model.load_state_dict(best_state)
    model.eval()
    hist.update({"best_epoch": best_epoch, "stopped_epoch": epoch, "best_val_mse": best_val,
                 "train_seconds": round(time.perf_counter() - start, 2)})
    return hist
