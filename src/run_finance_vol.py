"""finance_vol (|log_return|) first look: GRU, seed 0, full training, plus target baselines."""
import json
from pathlib import Path

from src.experiment import run_transfer, target_baselines
from src.run_experiments import DOMAIN_CONFIGS

OUT_DIR = Path("experiments/finance_vol")

if __name__ == "__main__":
    (OUT_DIR / "checkpoints").mkdir(parents=True, exist_ok=True)
    cfg = DOMAIN_CONFIGS["finance_vol"]
    results = {
        "domain": "finance_vol", "config": cfg,
        "baselines": target_baselines(cfg),
        "gru_seed0": run_transfer(cfg, "gru", seed=0, checkpoint_prefix=str(OUT_DIR / "checkpoints" / "gru_seed0")),
    }
    with open(OUT_DIR / "gru_1seed_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {OUT_DIR / 'gru_1seed_results.json'}")
