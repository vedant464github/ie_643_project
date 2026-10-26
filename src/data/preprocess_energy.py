import pandas as pd

def fill_hourly(df, col="load_mw"):
    """Hourly index from first to last timestamp; gaps linearly interpolated, then ffill/bfill at the ends."""
    full_range = pd.date_range(df.index.min(), df.index.max(), freq="h")
    df = df.reindex(full_range)
    df[col] = df[col].interpolate(method="linear").ffill().bfill()
    df.index.name = "datetime"
    return df

def load_and_clean(path, col_name):
    df = pd.read_csv(path)
    df.columns = ["datetime", "load_mw"]
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.drop_duplicates(subset="datetime")
    df = df.sort_values("datetime").set_index("datetime")
    df = fill_hourly(df, "load_mw")
    df = df.rename(columns={"load_mw": col_name})
    return df

if __name__ == "__main__":
    aep = load_and_clean("data/raw/energy/AEP_hourly.csv", "load_mw")
    comed = load_and_clean("data/raw/energy/COMED_hourly.csv", "load_mw")

    # trim to the overlapping window so source/target are comparable
    start = max(aep.index.min(), comed.index.min())
    end = min(aep.index.max(), comed.index.max())
    aep = aep.loc[start:end]
    comed = comed.loc[start:end]

    aep.to_csv("data/processed/energy/aep_clean.csv")
    comed.to_csv("data/processed/energy/comed_clean.csv")

    print("AEP:", aep.shape, aep.index.min(), "to", aep.index.max(), "| mean:", aep["load_mw"].mean().round(1))
    print("COMED:", comed.shape, comed.index.min(), "to", comed.index.max(), "| mean:", comed["load_mw"].mean().round(1))