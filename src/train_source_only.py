import copy
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from src.data.windowing import load_domain_series
from src.data.prepare import prepare_domain
from src.models.backbone import GRUForecaster
from src.utils.metrics import evaluate_all

def make_loader(X, Y, batch_size=64, shuffle=False):
    X_t = torch.tensor(X, dtype=torch.float32)
    Y_t = torch.tensor(Y, dtype=torch.float32)
    return DataLoader(TensorDataset(X_t, Y_t), batch_size=batch_size, shuffle=shuffle)

def train_source_only(csv_path, value_col, input_len, horizon,
                       hidden_size=32, num_layers=2, epochs=20, lr=1e-3, batch_size=64,
                       domain_name=None, checkpoint_dir="experiments/checkpoints", transform=None):
    series = load_domain_series(csv_path, value_col, transform=transform)
    data = prepare_domain(series, input_len, horizon)

    train_loader = make_loader(data["X_train"], data["Y_train"], batch_size, shuffle=True)
    val_loader = make_loader(data["X_val"], data["Y_val"], batch_size, shuffle=False)
    test_loader = make_loader(data["X_test"], data["Y_test"], batch_size, shuffle=False)

    model = GRUForecaster(input_len, horizon, hidden_size, num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    patience = 3
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for X_batch, Y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = loss_fn(pred, Y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                pred = model(X_batch)
                val_losses.append(loss_fn(pred, Y_batch).item())
        val_loss = np.mean(val_losses)

        print(f"Epoch {epoch:2d} | train_loss={np.mean(train_losses):.5f} | val_loss={val_loss:.5f}")

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch} (no val improvement for {patience} epochs)")
                break

    # restore best-val weights before evaluating (early stopping otherwise leaves last-epoch weights)
    model.load_state_dict(best_state)
    print(f"Restored best weights from epoch {best_epoch} (val_loss={best_val_loss:.5f})")

    if domain_name is not None:
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        ckpt_path = Path(checkpoint_dir) / f"{domain_name}_source_only.pt"
        torch.save({
            "state_dict": best_state,
            "model_kwargs": {"input_len": input_len, "horizon": horizon,
                             "hidden_size": hidden_size, "num_layers": num_layers},
            "norm_stats": data["norm_stats"],
            "best_epoch": best_epoch,
            "best_val_loss": float(best_val_loss),
        }, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

    # final test evaluation, in normalized space first
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            pred = model(X_batch)
            preds.append(pred.numpy())
            trues.append(Y_batch.numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)

    # de-normalize back to original scale before computing metrics
    mean, std = data["norm_stats"]["mean"], data["norm_stats"]["std"]
    preds_denorm = preds * std + mean
    trues_denorm = trues * std + mean

    metrics = evaluate_all(trues_denorm, preds_denorm)
    print("Test metrics (original scale):", metrics)

    # naive baseline: predict the train-mean (= zero in normalized space) for every step
    naive_pred_norm = np.zeros_like(trues)
    naive_pred_denorm = naive_pred_norm * std + mean  # same denorm as the model's predictions
    naive_metrics = evaluate_all(trues_denorm, naive_pred_denorm)
    print("Naive (predict-train-mean) baseline, original scale:", naive_metrics)

    return model, metrics, naive_metrics

if __name__ == "__main__":
    train_source_only(
        csv_path="data/processed/finance/nifty_it_clean.csv",
        value_col="log_return",
        input_len=30,
        horizon=5,
        epochs=20,
        domain_name="finance",
    )