import pandas as pd
import numpy as np

def close_to_log_return(df):
    """Trading-day Close prices -> Close and log_return columns; rows without a price or a return are dropped."""
    df = df.sort_index()
    df = df[["Close"]].copy()  # keep only Close — dropping Volume per our decision
    df = df.dropna()
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df = df.dropna()  # first row has no return
    return df

def load_and_clean(path):
    df = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)
    df.columns = df.columns.get_level_values(0)  # flatten Price/Ticker multi-header
    return close_to_log_return(df)

if __name__ == "__main__":
    it = load_and_clean("data/raw/finance/nifty_it.csv")
    pharma = load_and_clean("data/raw/finance/nifty_pharma.csv")

    it.to_csv("data/processed/finance/nifty_it_clean.csv")
    pharma.to_csv("data/processed/finance/nifty_pharma_clean.csv")

    print("IT:", it.shape, it.index.min(), "to", it.index.max())
    print("Pharma:", pharma.shape, pharma.index.min(), "to", pharma.index.max())