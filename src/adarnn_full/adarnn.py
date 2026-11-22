"""
Faithful AdaRNN training (Du et al. 2021, arXiv 2108.04443, Sec. 4.2, Algorithm 1), settings in CONSTRAINTS.md
Amendments 2026-11-15 (h).

  pretrain (lambda = 0, Eq. 2 with period-balanced batches)  ->  joint epochs:
  L = MSE(P1, P2, P3 batches) + lambda * mean_{6 pairs (i,j) of P1,P2,P3,target} sum_t alpha_ij^t CORAL(h_i^t, h_j^t)
  alpha_ij initialised to 1/V at the first joint epoch; after each joint epoch n >= 2 (Eq. 6, 7):
      alpha_ij^t <- alpha_ij^t * (1 + sigmoid(d_ij^t(n) - d_ij^t(n-1)))   where d_ij^t(n) >= d_ij^t(n-1)
      alpha_ij   <- alpha_ij / sum_t alpha_ij^t
  d_ij^t(n): mean over the epoch's joint steps of the batch CORAL distance at step t (detached).
"""
import copy
import datetime
import itertools
import time

import numpy as np
import torch
import torch.nn as nn

from src.evaluate import predict
from src.train import rss_mb

PAIRS = list(itertools.combinations(range(4), 2))  # (P1,P2) (P1,P3) (P1,T) (P2,P3) (P2,T) (P3,T)
PAIR_NAMES = ["P1-P2", "P1-P3", "P1-T", "P2-P3", "P2-T", "P3-T"]


class DeadlineExceeded(RuntimeError):
    pass


class EpochTooSlow(RuntimeError):
    pass


def coral_per_step(ha, hb):
    """ha, hb: (batch, V, d). Returns (V,) CORAL distances ||C_a^t - C_b^t||_F^2 / (4 d^2), unbiased covariances."""
    d = ha.shape[2]

    def cov(h):
        h = h - h.mean(dim=0, keepdim=True)
        return torch.einsum("bvi,bvj->vij", h, h) / (h.shape[0] - 1)
    return ((cov(ha) - cov(hb)) ** 2).sum(dim=(1, 2)) / (4 * d * d)


def boosting_update(alpha, d_new, d_old):
    """Eq. (6)-(7) with the paper's >= test, then normalisation over t. alpha, d_*: (pairs, V) tensors."""
    grow = d_new >= d_old
    alpha = torch.where(grow, alpha * (1 + torch.sigmoid(d_new - d_old)), alpha)
    return alpha / alpha.sum(dim=1, keepdim=True)


def _cycle(loader):
    while True:
        for batch in loader:
            yield batch


def _loader(tensors, batch_size):
    ds = torch.utils.data.TensorDataset(*[torch.tensor(t, dtype=torch.float32) for t in tensors])
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)


def fit_adarnn_full(model, periods, Xt_train, X_val, Y_val, lam, epochs, batch_size, n_source_train,
                    lr=1e-3, pretrain_epochs=3, patience=3, deadline=None, max_epoch_seconds=None,
                    epoch_callback=None, verbose=True, tag=""):
    """
    periods: [(X_P1, Y_P1), (X_P2, Y_P2), (X_P3, Y_P3)] source-train windows (source stats).
    Xt_train: target-train inputs only (target-train stats). epochs: total cap, pretrain included.
    epoch_callback(epoch, seconds): called after every epoch (used to print/log the first-epoch time).
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    device = next(model.parameters()).device
    period_iters = [_cycle(_loader(p, batch_size)) for p in periods]
    tgt_iter = _cycle(_loader([Xt_train], batch_size))
    steps = n_source_train // (3 * batch_size)
    V = periods[0][0].shape[1]
    alpha = torch.full((len(PAIRS), V), 1.0 / V, device=device)
    d_prev = None

    hist = {"phase": [], "train_loss": [], "train_mse": [], "train_tdm": [], "val_mse": [], "epoch_seconds": [],
            "rss_mb": [], "alpha_used": [], "dist_per_step": [], "alpha_updated": [],
            "pretrain_epochs": pretrain_epochs, "steps_per_epoch": steps, "pairs": PAIR_NAMES, "V": V}
    best_val, best_state, best_epoch, best_alpha, wait = float("inf"), None, None, None, 0
    start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        t_ep = time.perf_counter()
        model.train()
        phase = "pretrain" if epoch <= pretrain_epochs else "joint"
        losses, mses, tdms = [], [], []
        dist_sum = torch.zeros(len(PAIRS), V, device=device)
        for _ in range(steps):
            batches = [next(it) for it in period_iters]
            Xs = [b[0].to(device) for b in batches]
            Ys = torch.cat([b[1] for b in batches]).to(device)
            optimizer.zero_grad()
            if phase == "pretrain":  # Algorithm 1 step 2: Eq. (2) with lambda = 0
                l_mse = mse(model(torch.cat(Xs)), Ys)
                loss, l_tdm = l_mse, torch.zeros((), device=device)
            else:
                Xt = next(tgt_iter)[0].to(device)
                pred, hs = model.forward_with_states(torch.cat(Xs + [Xt]))
                B = batch_size
                l_mse = mse(pred[:3 * B], Ys)
                H = [hs[k * B:(k + 1) * B] for k in range(4)]  # P1, P2, P3, target
                d = torch.stack([coral_per_step(H[i], H[j]) for i, j in PAIRS])  # (6, V)
                l_tdm = (alpha * d).sum(dim=1).mean()  # Eq. (3) per pair, mean over the 6 pairs (Eq. 5)
                loss = l_mse + lam * l_tdm
                dist_sum += d.detach()
            loss.backward()
            optimizer.step()
            losses.append(loss.item()); mses.append(l_mse.item()); tdms.append(l_tdm.item())

        hist["alpha_used"].append(alpha.cpu().tolist() if phase == "joint" else None)
        updated = False
        if phase == "joint":
            d_epoch = dist_sum / steps
            hist["dist_per_step"].append(d_epoch.cpu().tolist())
            if d_prev is not None:  # Eq. (6)-(7); no d^(n-1) after the first joint epoch
                alpha = boosting_update(alpha, d_epoch, d_prev)
                updated = True
            d_prev = d_epoch
        else:
            hist["dist_per_step"].append(None)
        hist["alpha_updated"].append(updated)

        val_mse = float(np.mean((predict(model, X_val) - Y_val) ** 2))
        ep_s = time.perf_counter() - t_ep
        for k, v in [("phase", phase), ("train_loss", float(np.mean(losses))), ("train_mse", float(np.mean(mses))),
                     ("train_tdm", float(np.mean(tdms))), ("val_mse", val_mse), ("epoch_seconds", round(ep_s, 1)),
                     ("rss_mb", rss_mb())]:
            hist[k].append(v)
        if verbose:
            print(f"  {tag}Epoch {epoch:2d} [{phase}] | loss={hist['train_loss'][-1]:.5f} | mse={hist['train_mse'][-1]:.5f}"
                  f" | tdm={hist['train_tdm'][-1]:.6f} | val_mse={val_mse:.5f} | {ep_s:.0f}s/epoch"
                  f" | rss={hist['rss_mb'][-1]:.0f}MB", flush=True)
        if epoch_callback:
            epoch_callback(epoch, ep_s)
        if max_epoch_seconds and ep_s > max_epoch_seconds:
            raise EpochTooSlow(f"{tag}epoch {epoch} took {ep_s:.0f}s > {max_epoch_seconds}s")
        if deadline and datetime.datetime.now() >= deadline:
            raise DeadlineExceeded(f"{tag}hard stop {deadline} reached at epoch {epoch}")

        if phase == "pretrain":
            continue
        if val_mse < best_val - 1e-5:
            best_val, best_state, best_epoch, wait = val_mse, copy.deepcopy(model.state_dict()), epoch, 0
            best_alpha = hist["alpha_used"][-1]
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"  {tag}Early stopping at epoch {epoch}; restoring epoch {best_epoch}")
                break

    model.load_state_dict(best_state)
    model.eval()
    hist.update({"best_epoch": best_epoch, "stopped_epoch": epoch, "best_val_mse": best_val,
                 "alpha_at_best_epoch": best_alpha, "alpha_final": alpha.cpu().tolist(),
                 "train_seconds": round(time.perf_counter() - start, 2)})
    return hist
