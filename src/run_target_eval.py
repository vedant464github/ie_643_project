"""
TODO 3: source-only models evaluated on the TARGET test set, plus target-side baselines.

Protocol (no target labels for early stopping / model selection):
- source_only: checkpoint from run_experiments (selected on SOURCE val). Evaluated on target test under
  (a) source train stats and (b) target train stats (unlabelled target statistics, allowed in UDA).
- target_only: trained on target train for a FIXED number of epochs = source model's best epoch.
  No target val is used, so no early stopping / checkpoint selection on target labels.
- naive train-mean (source stats and target stats) and seasonal-naive on the same target test windows.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.data.prepare import prepare_domain
from src.data.windowing import load_domain_series
from src.evaluate import load_checkpoint, evaluate_model, naive_mean_metrics, seasonal_naive, denorm
from src.models.backbone import GRUForecaster
from src.run_experiments import DOMAIN_CONFIGS
from src.train_source_only import make_loader
from src.utils.metrics import evaluate_all

CHECKPOINT_DIR = Path("experiments/checkpoints")
RESULTS_PATH = Path("experiments/target_eval_results.json")
SEED = 0

def train_fixed_epochs(X_train, Y_train, model_kwargs, epochs, batch_size, lr=1e-3, seed=SEED):
    """Plain training loop with no validation: returns the last-epoch model and its train-loss history."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = GRUForecaster(**model_kwargs)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    train_loader = make_loader(X_train, Y_train, batch_size, shuffle=True)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for X_batch, Y_batch in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(X_batch), Y_batch)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        history.append(float(np.mean(losses)))
        print(f"  [target_only] Epoch {epoch:2d} | train_loss={history[-1]:.5f}")
    model.eval()
    return model, history

def run_domain(name, cfg, source_only_on_source):
    input_len, horizon = cfg["input_len"], cfg["horizon"]
    model, ckpt = load_checkpoint(CHECKPOINT_DIR / f"{name}_source_only.pt")
    src_stats = ckpt["norm_stats"]

    tgt_series = load_domain_series(cfg["target"]["csv_path"], cfg["value_col"], transform=cfg.get("transform"))
    tgt_srcnorm = prepare_domain(tgt_series, input_len, horizon, stats=src_stats)
    tgt_tgtnorm = prepare_domain(tgt_series, input_len, horizon)
    tgt_stats = tgt_tgtnorm["norm_stats"]

    # both normalisations must describe the same original-scale test targets
    Y_true = denorm(tgt_tgtnorm["Y_test"], tgt_stats)
    assert np.allclose(denorm(tgt_srcnorm["Y_test"], src_stats), Y_true, rtol=1e-4, atol=1e-4)

    res = {
        "source": cfg["source"]["name"], "target": cfg["target"]["name"],
        "input_len": input_len, "horizon": horizon,
        "n_target_test_windows": int(len(Y_true)),
        "norm_stats": {"source_train": src_stats, "target_train": tgt_stats},
        "source_only_on_source_test": source_only_on_source,
        "source_only_on_target_test": {
            "source_stats": evaluate_model(model, tgt_srcnorm["X_test"], tgt_srcnorm["Y_test"], src_stats),
            "target_stats": evaluate_model(model, tgt_tgtnorm["X_test"], tgt_tgtnorm["Y_test"], tgt_stats),
            "checkpoint_best_epoch": ckpt["best_epoch"],
        },
        "naive_train_mean_on_target_test": {
            "source_stats": naive_mean_metrics(tgt_srcnorm["Y_test"], src_stats),
            "target_stats": naive_mean_metrics(tgt_tgtnorm["Y_test"], tgt_stats),
        },
        "seasonal_naive_on_target_test": {},
    }

    for period in cfg["seasonal_periods"]:
        sn_true, sn_pred = seasonal_naive(tgt_series, input_len, horizon, period)
        assert np.allclose(sn_true, Y_true, rtol=1e-4, atol=1e-3), "seasonal-naive windows misaligned"
        res["seasonal_naive_on_target_test"][f"period_{period}"] = evaluate_all(sn_true, sn_pred)

    epochs = ckpt["best_epoch"]
    print(f"  training target_only for {epochs} fixed epochs (= source best epoch)")
    tgt_model, history = train_fixed_epochs(tgt_tgtnorm["X_train"], tgt_tgtnorm["Y_train"],
                                            ckpt["model_kwargs"], epochs, cfg["batch_size"])
    torch.save({"state_dict": tgt_model.state_dict(), "model_kwargs": ckpt["model_kwargs"],
                "norm_stats": tgt_stats, "epochs": epochs, "seed": SEED},
               CHECKPOINT_DIR / f"{name}_target_only.pt")
    res["target_only_on_target_test"] = {
        "target_stats": evaluate_model(tgt_model, tgt_tgtnorm["X_test"], tgt_tgtnorm["Y_test"], tgt_stats),
        "epochs": epochs,
        "epoch_rule": "fixed = source_only best epoch (source val); no target val used",
        "seed": SEED,
        "train_loss_history": history,
    }
    return res

if __name__ == "__main__":
    with open("experiments/source_only_results.json") as f:
        source_results = json.load(f)

    results = {"protocol": __doc__.strip(), "domains": {}}
    for name, cfg in DOMAIN_CONFIGS.items():
        print(f"\n=== Target eval: {name} ({cfg['source']['name']} -> {cfg['target']['name']}) ===")
        results["domains"][name] = run_domain(name, cfg, source_results[name])
        with open(RESULTS_PATH, "w") as f:  # rewrite after each domain so partial results survive
            json.dump(results, f, indent=2)
        print(json.dumps(results["domains"][name], indent=2))

    print(f"\nSaved: {RESULTS_PATH}")
