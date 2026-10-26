import numpy as np
import pandas as pd

def make_windows(series, input_len, horizon):
    """
    series: 1D array-like of values (already cleaned, single column)
    input_len: number of past steps used as input
    horizon: number of future steps to predict
    Returns: X of shape (n_samples, input_len), Y of shape (n_samples, horizon)
    """
    values = np.asarray(series, dtype=np.float32)
    n = len(values)
    n_samples = n - input_len - horizon + 1
    if n_samples <= 0:
        raise ValueError(f"Series too short for input_len={input_len}, horizon={horizon}")

    X = np.zeros((n_samples, input_len), dtype=np.float32)
    Y = np.zeros((n_samples, horizon), dtype=np.float32)
    for i in range(n_samples):
        X[i] = values[i : i + input_len]
        Y[i] = values[i + input_len : i + input_len + horizon]
    return X, Y

SERIES_TRANSFORMS = {"abs": np.abs}

def load_domain_series(csv_path, value_col, date_col=None, transform=None):
    """Loads a processed CSV and returns a single 1D series for windowing.
    transform: optional name from SERIES_TRANSFORMS (e.g. "abs" for |log_return| volatility proxy)."""
    df = pd.read_csv(csv_path, index_col=0 if date_col is None else date_col, parse_dates=True)
    values = df[value_col].values
    return SERIES_TRANSFORMS[transform](values) if transform else values

if __name__ == "__main__":
    # quick sanity check across all 3 domains
    configs = [
        ("data/processed/finance/nifty_it_clean.csv", "log_return", 30, 5),
        ("data/processed/climate/shimla_clean.csv", "temp", 30, 5),
        ("data/processed/energy/aep_clean.csv", "load_mw", 168, 24),  # 1 week in, 1 day out (hourly)
    ]
    for path, col, in_len, hor in configs:
        series = load_domain_series(path, col)
        X, Y = make_windows(series, in_len, hor)
        print(f"{path}: series_len={len(series)} -> X{X.shape} Y{Y.shape}")