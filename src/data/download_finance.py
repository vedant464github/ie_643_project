import yfinance as yf

it = yf.download("^CNXIT", start="2015-01-01", end="2026-10-01")
pharma = yf.download("^CNXPHARMA", start="2015-01-01", end="2026-10-01")

it.to_csv("data/raw/finance/nifty_it.csv")
pharma.to_csv("data/raw/finance/nifty_pharma.csv")