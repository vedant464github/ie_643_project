# Unsupervised domain adaptation for time-series forecasting (IE643)

One forecasting pipeline (same preprocessing, TCN model and loss everywhere) is trained on a labelled **source** series
and applied to a related **target** series using only the target's unlabelled inputs. Source-only, Deep CORAL, an
AdaRNN-style variant and target-only training are compared on three source → target pairs, with three seeds each.

| domain | source → target | series | input_len → horizon |
|---|---|---|---|
| climate | Shimla → Delhi | daily mean temperature (°C) | 30 → 5 |
| energy | AEP → COMED | hourly load (MW) | 168 → 24 |
| finance | Nifty IT → Nifty Pharma | daily log-return (negative control: no learnable signal) | 30 → 5 |
| finance_vol | Nifty IT → Nifty Pharma | daily \|log-return\| (negative control) | 30 → 5 |

Backbone: a small dilated TCN (35k parameters), chosen in a backbone bake-off on source validation data. Adaptation
uses target-train **inputs** only; target-only training is an upper reference that uses target labels. The
pre-registered decisions and every later amendment are in [`CONSTRAINTS.md`](CONSTRAINTS.md) (called `CLAUDE.md` in
the run logs under `experiments/`), and a timestamped run log is in `experiments/STATUS.md`.

## Setup

Windows, Python 3.12. From the repo root:

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` is a full `pip freeze` of the working environment. The experiments ran on the **CPU build** of
PyTorch 2.14.1 (installed version string `2.14.1+cpu`); it is pinned as plain `torch==2.14.1`. On Windows the
default PyPI wheel is the CPU build. The training code uses CUDA when it is available and otherwise falls back to
CPU (`src/train.py`, device recorded in every results JSON). The app always runs on CPU.

Threads: `src/train.py` sets 2 torch threads, and the parallel energy runs used `OMP_NUM_THREADS=2` (see
CONSTRAINTS.md, Amendments 2026-10-25 (d) and (e)). Before long runs, set it in the shell:

```
set OMP_NUM_THREADS=2
```

## Streamlit app

```
streamlit run app/app.py
```

- **Sidebar:** domain, model (Source-only | CORAL-adapted, TCN seed 0) and normalisation (stats of the first 70%
  of the uploaded series, mirroring target-train stats, which is the default | the checkpoint's source stats). The
  active choice, its stats, input_len and horizon are shown in the sidebar.
- **Data:** a bundled sample per domain (`app/samples/`, built from `data/processed`) or an uploaded CSV. The app
  picks default date and value columns (the first column that parses as dates, and the first numeric column),
  which you can change. For finance and finance_vol, upload index **closing prices**: the app computes log-returns
  (plus the absolute value for finance_vol) with the training preprocessing code. Climate must be daily and energy
  hourly; missing steps are filled exactly as in preprocessing (daily forward or backward fill; hourly linear
  interpolation). Finance must be trading-day data and rows without a price are dropped, as in preprocessing.
- **Forecast tab:** forecasts `horizon` steps from the last `input_len` points. By default the final `horizon`
  points are held out as actuals and scored (MAE, RMSE). Untick that to forecast past the end.
- **Backtest tab:** every sliding window of the series is forecast. The tab plots step-1 forecasts against actuals
  and shows MAE/RMSE for step 1 and for all steps, next to the train-mean baseline (the mean of the series' first
  70%).
- Bad inputs give a plain error message: too short, missing or non-numeric columns, unparsable, duplicate or
  unsorted dates, wrong frequency, or fewer than input_len + horizon points.

The backtest is a diagnostic, not a held-out evaluation. The first-70% stats and the baseline mean overlap the
windows, and each bundled sample is the full target series, whose first 70% CORAL saw as unlabelled inputs. For
real results see below.

The app reads only `app/checkpoints/`. Each checkpoint carries its normalisation stats, input_len and horizon, and
`app/checkpoints/manifest.json` holds the domain-level preprocessing (frequency, transform, units). Rebuild the
checkpoints and samples with `python -m src.build_app_assets`, which copies the files from `experiments/` without
modifying them. App tests: `python app/test_app.py` (Streamlit AppTest, no browser, no training).

## Reproducing the experiments

Run from the repo root as modules. Raw data is not in the repo: the download steps write `data/raw/`, and the
preprocessing steps write `data/processed/` (already included). Sources, dates and licences are in
[`DATASETS.md`](DATASETS.md). All CPU timings are in `experiments/STATUS.md`.

```
python -m src.data.download_climate          # Open-Meteo archive -> data/raw/climate
python -m src.data.download_finance          # yfinance ^CNXIT, ^CNXPHARMA -> data/raw/finance
kaggle datasets download robikscube/hourly-energy-consumption -p data/raw/energy --unzip   # needs a Kaggle API token
python -m src.data.preprocess_climate        # -> data/processed/climate
python -m src.data.preprocess_energy         # -> data/processed/energy
python -m src.data.preprocess_finance        # -> data/processed/finance

python -m src.run_bakeoff                    # backbone bake-off (source val selection)
python -m src.run_final_seeds --backbone tcn --domains finance finance_vol climate energy --seeds 0 1 2
python -m src.run_final_seeds --backbone tcn --summarize
python -m src.run_coral --grid               # λ grid, seed 0
python -m src.run_coral --select             # -> experiments/coral/lambda_selection.json
python -m src.run_coral --final --seeds 0 1 2
python -m src.run_coral --summarize
python -m src.run_coral --final --lam 1 --out experiments/coral_lambda1 --seeds 0 1 2   # λ sensitivity
python -m src.run_adarnn --tdc               # period split, inputs only
python -m src.run_adarnn --run --seeds 0 1 2
python -m src.report                         # experiments/summary_all.md (+ .json), report/figures/
python -m src.build_app_assets               # app/checkpoints/, app/samples/
```

`src/run_adarnn.py` contains the run-control deadline of the original runs (`DEADLINE = 2026-10-25 22:00`); change
it before re-running, or no run will start.

Earlier GRU baselines: `python -m src.run_experiments` and `python -m src.run_target_eval`
(results in `experiments/reference_gru_cpu_1seed/`, kept unchanged). First finance_vol look (GRU, seed 0):
`python -m src.run_finance_vol` (results in `experiments/finance_vol/`).

## Results

- `experiments/summary_all.md`: the full staircase (source stats, target stats, CORAL λ=10, CORAL λ=1,
  AdaRNN-style, target-only), MAE/RMSE mean ± std with n per cell, MMD before/after.
- `experiments/coral/summary.md`: CORAL λ=10 against the pre-registered success criteria.
- `report/figures/`: staircase bars, forecast vs actual, λ selection and sensitivity, MMD per seed, training
  curves.

Status (2026-10-25): all energy runs are complete. AdaRNN-style λ=10 and CORAL λ=1 energy seeds 0-2 finished on 2026-10-25, so all 12 AdaRNN-style runs and all 12 CORAL λ=1 runs (4 domains × 3 seeds each) are done, and every cell of the staircase has n=3. No adaptation method beats source-only with target-train normalisation by more than seed noise. The pre-registered criteria are not met on energy (CORAL λ=10 gain smaller than the seed std) or on climate RMSE. Descriptive per-seed paired differences on energy are at the end of `experiments/summary_all.md`.

## Layout

```
CONSTRAINTS.md          pre-registered decisions and dated amendments
DATASETS.md             sources, download commands, date ranges, row counts and licences of the six series
requirements.txt        pip freeze of the working environment
app/                    Streamlit app: app.py (UI), core.py (logic), test_app.py (AppTest), checkpoints/, samples/
data/processed/         finance/, climate/, energy/: the six preprocessed series (data/raw/ is not included)
src/data/               download_*, preprocess_*, loaders.py, windowing.py, prepare.py (split -> normalise -> window)
src/models/             backbone.py (GRU, DLinear, TCN, PatchTST), revin.py
src/adaptation/         coral.py (CORAL loss, joint training, MMD diagnostic), adarnn.py (TDC split, joint training)
src/utils/              metrics.py, status.py (STATUS.md logger)
src/                    run_*.py experiment drivers, train.py, train_source_only.py, evaluate.py, experiment.py,
                        report.py (summary + figures), build_app_assets.py
experiments/            results JSON, checkpoints, summaries and STATUS.md (run logs not included)
report/figures/         PNG figures written by src/report.py
diagnostics/            bench.py, bench2.py: the thread-contention benchmarks behind CONSTRAINTS.md Amendment (d)
```
