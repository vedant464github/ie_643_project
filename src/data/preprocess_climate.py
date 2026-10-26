import pandas as pd

def fill_daily(df, col="temp"):
    """Daily calendar index from first to last date; small gaps forward filled, then back fill any leading gap."""
    full_range = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df = df.reindex(full_range)
    df[col] = df[col].ffill().bfill()
    df.index.name = "time"
    return df

def load_and_clean(path):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df.sort_index()
    df = df[["temperature_2m_mean"]].copy()
    df = df.rename(columns={"temperature_2m_mean": "temp"})
    return fill_daily(df, "temp")

if __name__ == "__main__":
    shimla = load_and_clean("data/raw/climate/shimla.csv")
    delhi = load_and_clean("data/raw/climate/delhi.csv")

    shimla.to_csv("data/processed/climate/shimla_clean.csv")
    delhi.to_csv("data/processed/climate/delhi_clean.csv")

    print("Shimla:", shimla.shape, shimla.index.min(), "to", shimla.index.max(), "| mean:", shimla["temp"].mean().round(2))
    print("Delhi:", delhi.shape, delhi.index.min(), "to", delhi.index.max(), "| mean:", delhi["temp"].mean().round(2))