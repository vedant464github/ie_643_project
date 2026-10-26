import copy
import random
import time

import numpy as np
import torch
import torch.nn as nn

from src.evaluate import predict
from src.train_source_only import make_loader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TORCH_THREADS = 2  # run control (CONSTRAINTS.md 2026-10-25 (d)): 14 threads contended on small batches
torch.set_num_threads(TORCH_THREADS)

def device_info():
    """Recorded in every results JSON."""
    return {"device": str(DEVICE), "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cpu_threads": torch.get_num_threads()}

def rss_mb():
    """Resident set size of this process in MB (Windows: GetProcessMemoryInfo; else ru_maxrss peak)."""
    try:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                                               "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                                               "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        k32 = ctypes.WinDLL("kernel32")
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        gpmi = ctypes.WinDLL("psapi").GetProcessMemoryInfo
        gpmi.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        gpmi(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
        return round(pmc.WorkingSetSize / 2**20, 1)
    except (AttributeError, OSError):
        import resource
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def fit(model, X_train, Y_train, X_val=None, Y_val=None, epochs=20, batch_size=64, lr=1e-3,
        patience=3, verbose=True, tag=""):
    """
    Trains with MSE loss. With validation data: early stopping (patience, 1e-5 min improvement) and the
    best-val weights are restored. Without validation data: fixed number of epochs, last weights kept.
    Val MSE is computed exactly over the full val set (normalized space).
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    train_loader = make_loader(X_train, Y_train, batch_size, shuffle=True)
    use_val = X_val is not None
    device = next(model.parameters()).device

    history = {"train_loss": [], "val_mse": [], "rss_mb": []}
    best_val, best_state, best_epoch, patience_counter = float("inf"), None, epochs, 0
    start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(X_batch), Y_batch)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        history["train_loss"].append(float(np.mean(losses)))

        msg = f"  {tag}Epoch {epoch:2d} | train_loss={history['train_loss'][-1]:.5f}"
        if use_val:
            val_mse = float(np.mean((predict(model, X_val) - Y_val) ** 2))
            history["val_mse"].append(val_mse)
            msg += f" | val_mse={val_mse:.5f}"
        history["rss_mb"].append(rss_mb())
        msg += f" | rss={history['rss_mb'][-1]:.0f}MB"
        if verbose:
            print(msg)

        if use_val:
            if val_mse < best_val - 1e-5:
                best_val, best_state, best_epoch, patience_counter = val_mse, copy.deepcopy(model.state_dict()), epoch, 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"  {tag}Early stopping at epoch {epoch}; restoring epoch {best_epoch}")
                    break

    if use_val:
        model.load_state_dict(best_state)
    model.eval()
    history.update({
        "best_epoch": best_epoch,
        "stopped_epoch": epoch,
        "best_val_mse": best_val if use_val else None,
        "train_seconds": round(time.perf_counter() - start, 2),
    })
    return history
