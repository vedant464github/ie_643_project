import json
from pathlib import Path
from src.train_source_only import train_source_only

# Shared settings per domain (window sizes, epochs, batch_size) plus the source and target series.
# Source and target use the same value_col so one pipeline/model applies to both.
# seasonal_periods: lags for the seasonal-naive baseline (1 = persistence).
DOMAIN_CONFIGS = {
    "finance": {
        "value_col": "log_return", "input_len": 30, "horizon": 5, "epochs": 20, "batch_size": 64,
        "seasonal_periods": [1, 5],
        "source": {"name": "nifty_it", "csv_path": "data/processed/finance/nifty_it_clean.csv"},
        "target": {"name": "nifty_pharma", "csv_path": "data/processed/finance/nifty_pharma_clean.csv"},
    },
    # volatility proxy: same pair and windows as finance, target series is |log_return|
    "finance_vol": {
        "value_col": "log_return", "transform": "abs", "input_len": 30, "horizon": 5, "epochs": 20, "batch_size": 64,
        "seasonal_periods": [], "mean_last_k": 5,
        "source": {"name": "nifty_it", "csv_path": "data/processed/finance/nifty_it_clean.csv"},
        "target": {"name": "nifty_pharma", "csv_path": "data/processed/finance/nifty_pharma_clean.csv"},
    },
    "climate": {
        "value_col": "temp", "input_len": 30, "horizon": 5, "epochs": 20, "batch_size": 64,
        "seasonal_periods": [1, 365],
        "source": {"name": "shimla", "csv_path": "data/processed/climate/shimla_clean.csv"},
        "target": {"name": "delhi", "csv_path": "data/processed/climate/delhi_clean.csv"},
    },
    "energy": {
        "value_col": "load_mw", "input_len": 168, "horizon": 24, "epochs": 40, "batch_size": 128,
        "seasonal_periods": [24, 168],
        "source": {"name": "aep", "csv_path": "data/processed/energy/aep_clean.csv"},
        "target": {"name": "comed", "csv_path": "data/processed/energy/comed_clean.csv"},
    },
}

if __name__ == "__main__":
    Path("experiments").mkdir(exist_ok=True)
    results = {}

    for name, cfg in DOMAIN_CONFIGS.items():
        print(f"\n=== Source-only: {name} ({cfg['source']['name']} -> {cfg['target']['name']}) ===")
        _, metrics, naive_metrics = train_source_only(
            csv_path=cfg["source"]["csv_path"],
            value_col=cfg["value_col"],
            input_len=cfg["input_len"],
            horizon=cfg["horizon"],
            epochs=cfg["epochs"],
            batch_size=cfg["batch_size"],
            domain_name=name,
            transform=cfg.get("transform"),
        )
        results[name] = {
            "method": "source_only",
            "eval_split": f"source_test ({cfg['source']['name']})",
            "metrics": metrics,
            "naive_train_mean": naive_metrics,
        }

    with open("experiments/source_only_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved: experiments/source_only_results.json")
    print(json.dumps(results, indent=2))
