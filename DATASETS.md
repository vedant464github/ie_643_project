# Datasets

Six series form three source → target pairs. `finance_vol` adds no files: it uses the two finance series with the
`abs` transform applied in the pipeline (`src/run_experiments.py`, `DOMAIN_CONFIGS`). The raw files in
`data/raw/` are the exact inputs to the preprocessing scripts, which write `data/processed/`. `data/raw/` is not in
this repository; the download steps below recreate it, and `data/processed/` is included. Run the scripts from the
repo root with `python -m src.data.<script>`.

| series | role | raw file | processed file | rows raw → processed | date range (processed) |
|---|---|---|---|---|---|
| Nifty IT index | finance source | `data/raw/finance/nifty_it.csv` | `data/processed/finance/nifty_it_clean.csv` | 2,897 → 2,896 trading days | 2015-01-02 to 2026-09-30 |
| Nifty Pharma index | finance target | `data/raw/finance/nifty_pharma.csv` | `data/processed/finance/nifty_pharma_clean.csv` | 2,896 → 2,895 trading days | 2015-01-02 to 2026-09-30 |
| Shimla daily mean temperature | climate source | `data/raw/climate/shimla.csv` | `data/processed/climate/shimla_clean.csv` | 4,291 → 4,291 days | 2015-01-01 to 2026-09-30 |
| Delhi daily mean temperature | climate target | `data/raw/climate/delhi.csv` | `data/processed/climate/delhi_clean.csv` | 4,291 → 4,291 days | 2015-01-01 to 2026-09-30 |
| AEP hourly load (MW) | energy source | `data/raw/energy/AEP_hourly.csv` | `data/processed/energy/aep_clean.csv` | 121,273 → 66,504 hours | 2011-01-01 01:00 to 2018-08-03 00:00 |
| COMED hourly load (MW) | energy target | `data/raw/energy/COMED_hourly.csv` | `data/processed/energy/comed_clean.csv` | 66,497 → 66,504 hours | 2011-01-01 01:00 to 2018-08-03 00:00 |

Row counts exclude header lines.

## Finance: Nifty IT and Nifty Pharma (NSE sector indices)

- **Source:** Yahoo Finance via the `yfinance` package. The tickers are `^CNXIT` (Nifty IT) and `^CNXPHARMA` (Nifty Pharma).
- **How obtained:** `python -m src.data.download_finance`, which calls
  `yf.download(ticker, start="2015-01-01", end="2026-10-01")`. The end date is exclusive, so the last day is 2026-09-30.
  The CSVs keep yfinance's two-row Price/Ticker header and contain Close, High, Low, Open and Volume.
- **Raw date range:** 2015-01-01 to 2026-09-30, with no missing Close values.
- **Preprocessing:** `python -m src.data.preprocess_finance`. It flattens the header, keeps Close, drops rows
  without a price, and computes `log_return = log(Close_t / Close_{t-1})`. The first row has no return and is
  dropped, which is the one-row difference in the counts above.
- **Licence:** no open licence. Use is subject to Yahoo's terms of use, and the index values belong to NSE Indices
  Ltd. The raw files are not redistributed here; the processed series are included for reproducibility of this
  coursework only.

## Climate: Shimla and Delhi (Open-Meteo historical weather archive)

- **Source:** the Open-Meteo Historical Weather API (`https://archive-api.open-meteo.com/v1/archive`), daily
  `temperature_2m_max`, `temperature_2m_min` and `temperature_2m_mean`, with timezone `Asia/Kolkata`.
- **How obtained:** `python -m src.data.download_climate`, with these coordinates:
  - Shimla: latitude 31.1048, longitude 77.1734
  - Delhi: latitude 28.7041, longitude 77.1025
  - date range 2015-01-01 to 2026-09-30 for both
- **Preprocessing:** `python -m src.data.preprocess_climate`. It keeps `temperature_2m_mean` as `temp` on a daily
  index and fills gaps forward, then backward. The raw files have no missing values, so the row counts are
  unchanged.
- **Licence:** CC BY 4.0 (Open-Meteo API data). Required attribution: "Weather data by Open-Meteo.com"
  (https://open-meteo.com/), with a link to the licence and a note of any changes.

## Energy: AEP and COMED (PJM hourly load, Kaggle)

- **Source:** the Kaggle dataset `robikscube/hourly-energy-consumption` (PJM Interconnection hourly load in
  megawatts, one CSV per region). Only `AEP_hourly.csv` and `COMED_hourly.csv` are used. The other files in the
  dataset (other PJM regions, `pjm_hourly_est.csv`, `est_hourly.paruqet`) are not used.
- **How obtained:** with the Kaggle CLI (kaggle 2.2.4; needs a Kaggle API token), from the repo root:

  ```
  kaggle datasets download robikscube/hourly-energy-consumption -p data/raw/energy --unzip
  ```

  Older Kaggle CLI versions take the dataset with `-d`:
  `kaggle datasets download -d robikscube/hourly-energy-consumption -p data/raw/energy --unzip`.
- **Raw date ranges:** AEP 2004-10-01 01:00 to 2018-08-03 00:00. COMED 2011-01-01 01:00 to 2018-08-03 00:00.
  Each file has 4 duplicated timestamps.
- **Preprocessing:** `python -m src.data.preprocess_energy`. It renames the value column to `load_mw`, parses the
  timestamps, drops duplicate timestamps, sorts the rows and reindexes to a continuous hourly index. Gaps are
  filled by linear interpolation, then forward and backward fill at the ends. Both series are then trimmed to
  their overlapping window, 2011-01-01 01:00 to 2018-08-03 00:00 (66,504 hours). Within that window AEP had
  13 missing hours and COMED had 11, all filled by interpolation.
- **Licence:** CC0: Public Domain, as listed in the Kaggle API metadata for the dataset (checked 2026-10-26;
  dataset creator Rob Mulla). The underlying data is published by PJM Interconnection.

## Splits

All splits are made later, inside the pipeline (`src/data/prepare.py`): a chronological 70/15/15 train/val/test split
first, then normalisation with train statistics only, then windowing. The processed files are full series, not
split.
