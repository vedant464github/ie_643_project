# Unsupervised domain adaptation for time-series forecasting (IE643)

One forecasting pipeline (same preprocessing, TCN model and loss everywhere) is trained on a labelled **source** series
and applied to a related **target** series using only the target's unlabelled inputs. Source-only, Deep CORAL, an
AdaRNN-style variant and target-only training are compared on three source → target pairs, with three seeds each.
A separate novelty assessment tests a faithful AdaRNN on a GRU backbone (see [Novelty assessment](#novelty-assessment)).

| domain | source → target | series | input_len → horizon |
|---|---|---|---|
| climate | Shimla → Delhi | daily mean temperature (°C) | 30 → 5 |
| energy | AEP → COMED | hourly load (MW) | 168 → 24 |
| finance | Nifty IT → Nifty Pharma | daily log-return (negative control: no learnable signal) | 30 → 5 |
| finance_vol | Nifty IT → Nifty Pharma | daily \|log-return\| (negative control) | 30 → 5 |

Backbone: a small dilated TCN (35k parameters), chosen in a backbone bake-off on source validation data. Adaptation
uses target-train **inputs** only; target-only training is an upper reference that uses target labels. The
pre-registered decisions and every later amendment are in [`CONSTRAINTS.md`](CONSTRAINTS.md) (run logs and `results_clean/` write-ups call it `CLAUDE.md`), and a timestamped run log is in `experiments/STATUS.md`.

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

## Datasets

Six series (two per domain). Sources, download commands, date ranges, row counts and licences are in
[`DATASETS.md`](DATASETS.md). `data/raw/` is not in the repository; the download steps recreate it.
`data/processed/` is included, so the app and the experiments run without downloading anything.

## Streamlit app

```
set OMP_NUM_THREADS=2
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
preprocessing steps write `data/processed/` (already included). All CPU timings are in `experiments/STATUS.md`.

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

Clean results tables, figures and the audit are regenerated from the JSONs (no training):

```
python -m results_clean.scripts.build_tables        # canonical.md/.csv, headline.md, noise_rule.md
python -m results_clean.scripts.make_figures        # results_clean/figures/
python -m results_clean.scripts.consistency_scan    # results_clean/CONSISTENCY.md
python -m results_clean.audit.audit_compute         # results_clean/audit/
```

## Results

- [`results_clean/canonical.md`](results_clean/canonical.md) (and `canonical.csv`): one table per domain, target
  test set, every row regenerated from the run JSONs, with the naive train-mean and persistence/seasonal-naive
  baselines.
- [`results_clean/headline.md`](results_clean/headline.md): source-only vs adapted, with the split of the drop into
  the normalisation part and the method part, and the gap to target-only.
- [`results_clean/noise_rule.md`](results_clean/noise_rule.md): the single rule used to label a difference as
  "within both", "exceeds the target-stats std only" or "exceeds both stds".
- `results_clean/APPROVED_CLAIMS.md` lists every claim with its number and source row; `PROVENANCE.md` and
  `CONSISTENCY.md` trace the numbers; `results_clean/audit/` holds the definition, scale and deviation audits.
- Older summaries: `experiments/summary_all.md` (full staircase) and `experiments/coral/summary.md` (CORAL λ=10
  against the pre-registered success criteria), with figures in `report/figures/`.

**Headline** (from `results_clean/headline.md`, target test, mean over seeds 0-2): adapting the source-trained TCN
(target-train-stat normalisation + CORAL λ=10) lowers COMED energy MAE from 802.88 to 554.78 MW (30.9%) and Delhi
climate MAE from 2.945 to 1.698 °C (42.3%). Almost all of that drop comes from the normalisation: on energy
241.27 MW of the drop (97.2%) is normalisation and 6.84 MW (2.8%) is CORAL, and on climate normalisation accounts
for more than the whole drop while CORAL adds 0.025 °C of error.

Against the pre-registered criteria: on energy, CORAL λ=10 vs source-only with target-train stats is −6.84 MW MAE
("within both" stds, 8.75 and 10.35 MW), so the energy criterion is not met; on climate CORAL λ=10 is worse than
that row (+0.025 °C MAE, "exceeds the target-stats std only"; +0.015 °C RMSE, "exceeds both stds"). Raw finance and
finance_vol are negative controls: on source validation no TCN seed beats the train-mean.

## How to read the numbers

- All reported errors (MAE, RMSE) are in **original units**: MW for energy, °C for climate, log-return fractions
  for finance and finance_vol. They are computed after inverting the z-score.
- Training runs on **z-scored** data, with statistics from the training split only.
- Each table row uses **one set of statistics at both ends** (normalising the inputs and de-normalising the
  forecasts): source-train stats for the "source-only, source stats" row, and target-train stats (from target-train
  inputs only, never target labels) for every other row.
- The backbone bake-off table (`experiments/bakeoff/`) is the exception: it reports source-validation MSE in
  z-scored space, which is unitless.
- MAPE/sMAPE are not reported for the finance domains (meaningless for log-returns).

## Pre-registration

The design decisions were fixed before any result was seen and are in `CONSTRAINTS.md` (section "Decisions").
Every later change, including each post-hoc step, is a dated entry in its "Amendments" sections, and each entry says
whether it was fixed before the runs it governs or reports an outcome. The post-hoc steps are: finance_vol dropped from the backbone selection
rule and TCN picked over DLinear by judgment (2026-10-23), and the headline reframed as source-only vs adapted
with a new persistence/seasonal baseline and one canonical table (2026-11-15 (g)), with no re-runs and no new
training.

## Novelty assessment

**What was tried.** A faithful AdaRNN (Du et al. 2021, arXiv 2108.04443: per-time-step temporal distribution
matching with boosting-based importance weights, CONSTRAINTS.md 2026-11-15 (h)) on the project's GRU (2 layers,
hidden 32, about 10k parameters), against the same GRU trained source-only (source stats and target-train stats) and
with plain CORAL on the final hidden state. TDC periods are reused from `experiments/adarnn/tdc.json`; λ was chosen
per domain and method by the TCN rule on seed-0 source validation (λ=10 everywhere).

**Documented failure (climate).** Faithful AdaRNN was worse than the same GRU source-only with target-train stats:
1.792 ± 0.006 vs 1.678 ± 0.014 °C MAE, and also worse than persistence (1.733 °C). The λ=0 run showed the same gap
(1.786 °C), so the cause was the training schedule, not the matching term: AdaRNN got 15 optimiser steps per epoch
vs 47 for the baseline, and every run stopped at the 20-epoch cap. Energy was stopped by the pre-registered
300 s/epoch rule (the laptop had gone into Modern Standby); only AdaRNN seed 2 finished there, so no energy
conclusion is drawn. Finance and finance_vol behave as negative controls. Details: `results_clean/novelty/`.

**Step-matched fix and λ=0 control (CONSTRAINTS.md 2026-11-20 (j)).** The AdaRNN sampler was step-matched (47
optimiser steps per epoch) and the cap raised to 60 joint epochs for all four methods; (d) AdaRNN with λ=0 isolates
the schedule from the matching term. AdaRNN λ=10 falls to 1.676 ± 0.022 °C MAE and the λ=0 control to
1.667 ± 0.016, against 1.678 ± 0.014 for source-only with target-train stats. AdaRNN λ=10 vs that baseline is
"within both" seed stds on MAE and RMSE, so it ties it and the pre-registered success criterion is not met; AdaRNN
λ=10 vs λ=0 is also "within both", so the matching term adds nothing measurable. The comparison is step-matched,
not sample-matched. Details: `results_clean/novelty2/summary.md` and `narrative.md`.

**Cap-20 ablation (CONSTRAINTS.md 2026-11-21 (k)).** Rerunning (c) and (d) with a joint-epoch cap of 20 gives
bit-identical results to cap 60 on every seed, so the earlier gap comes from the step-matched schedule, not from
the higher cap (a 17-joint-epoch cap, the old protocol's effective limit, was not tested). Details:
`results_clean/novelty2/ablation_cap20.md`. All of this is climate only, GRU only, three seeds.

**Where things live.** Code: `src/adarnn_full/` (faithful AdaRNN, GRU wrapper, runner, queue driver, summary) and
`src/adarnn_fix/` (step-matched runner, seed-major driver, keep-awake helper, cap-20 ablation, summary). Results:
`experiments/novelty_adarnn/`, `experiments/novelty_fix/`, `experiments/novelty_fix_cap20/`. Summaries:
`results_clean/novelty/`, `results_clean/novelty2/`. Diagram: `python -m src.adarnn_full.diagram` writes
`results_clean/novelty/architecture.png`.

**Commands used** (from the queue start lines in `experiments/STATUS.md` and the module docstrings; each driver
runs its jobs with `OMP_NUM_THREADS=2`):

```
# Faithful AdaRNN (2026-11-15 (h)). Each driver queue runs `python -m src.adarnn_full.run <job>` jobs in turn.
python -m src.adarnn_full.driver --name climA "final --domain climate --method source_only --seeds 0 1 2"
python -m src.adarnn_full.driver --name climB "grid --domain climate --method coral"
python -m src.adarnn_full.driver --name climC "grid --domain climate --method adarnn"
python -m src.adarnn_full.run select --domain climate          # λ rule, both methods
python -m src.adarnn_full.driver --name climD "final --domain climate --method coral --seeds 0 1 2"
python -m src.adarnn_full.driver --name climE "final --domain climate --method adarnn --seeds 0 1 2"
python -m src.adarnn_full.driver --name enB "grid --domain energy --method coral"
python -m src.adarnn_full.driver --name enC "grid --domain energy --method adarnn"
python -m src.adarnn_full.driver --name finB "grid --domain finance --method coral" "grid --domain finance_vol --method coral"
python -m src.adarnn_full.driver --name finC "grid --domain finance --method adarnn" "grid --domain finance_vol --method adarnn"
python -m src.adarnn_full.run select --domain energy           # likewise finance, finance_vol
python -m src.adarnn_full.driver --name enP0 "final --domain energy --method adarnn --seeds 0 --probe"   # enP1, enP2: seeds 1, 2
python -m src.adarnn_full.driver --name finX "final --domain finance --method source_only --seeds 0 1 2" "final --domain finance_vol --method source_only --seeds 0 1 2"
python -m src.adarnn_full.driver --name finY "final --domain finance --method coral --seeds 0 1 2" "final --domain finance_vol --method coral --seeds 0 1 2"
python -m src.adarnn_full.driver --name finZ "final --domain finance --method adarnn --seeds 0 1 2" "final --domain finance_vol --method adarnn --seeds 0 1 2"
python -m src.adarnn_full.summarize                             # results_clean/novelty/

# Step-matched fix (2026-11-20 (j)): 4 methods x seeds 0-2, seed-major, 3 in parallel
python -m src.adarnn_fix.driver                                 # or one job: python -m src.adarnn_fix.run --method adarnn10 --seed 0
python -m src.adarnn_fix.summarize                              # results_clean/novelty2/summary.md

# Cap-20 ablation (2026-11-21 (k))
python -m src.adarnn_fix.ablation_cap20 --queue                 # all 6 jobs, 3 in parallel
python -m src.adarnn_fix.ablation_cap20 --report                # results_clean/novelty2/ablation_cap20.md
```

The energy queues that were cut short by the 300 s/epoch stop (enQ0, enQ1, enQ2) are listed in
`experiments/STATUS.md`. Like `src/run_adarnn.py`, these runners carry the run-control deadlines of the original
runs (`DEADLINE` in `src/adarnn_full/run.py`, `src/adarnn_fix/run.py` and `src/adarnn_fix/ablation_cap20.py`);
change them before re-running, or no run will start.

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
src/adarnn_full/        faithful AdaRNN on a GRU (novelty assessment)
src/adarnn_fix/         step-matched AdaRNN fix and cap-20 ablation
src/utils/              metrics.py, status.py (STATUS.md logger)
src/                    run_*.py experiment drivers, train.py, train_source_only.py, evaluate.py, experiment.py,
                        report.py (summary + figures), build_app_assets.py
experiments/            results JSON, checkpoints, summaries and STATUS.md (run logs not included), including
                        novelty_adarnn/, novelty_fix/, novelty_fix_cap20/
results_clean/          canonical tables, headline, noise rule, approved claims, provenance, consistency scan,
                        figures/, scripts/, audit/, novelty/, novelty2/
architecture/           IE643_Model_Architectures.pptx, png/ renders, README.md, CHECKS.md
report/figures/         PNG figures written by src/report.py
diagnostics/            bench.py, bench2.py: the thread-contention benchmarks behind CONSTRAINTS.md Amendment (d)
```
