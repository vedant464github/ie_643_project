# Documented failures, with evidence

This file lists every documented failure. It does not pick one as the main failure. All numbers are copied from the
evidence files named in each entry. Target test, original units, mean ± sample std over seeds 0-2 unless stated
otherwise. "SO-tgt" is source-only evaluated with target-train stats.

## A. Result failures

### A1. Faithful AdaRNN (GRU) on climate is worse than the same GRU with target-train stats

- **Result.** MAE 1.792 ± 0.006 °C vs 1.678 ± 0.014 °C (+0.114 °C, worse on every seed: +0.118, +0.104, +0.122).
  RMSE 2.325 ± 0.007 vs 2.211 ± 0.008. It is also worse than persistence (1.733 °C MAE).
- **Comparison at λ=0** (seed 0, same AdaRNN code and schedule, matching term switched off):

  | run | λ | target-test MAE | RMSE | source-val MSE (norm) | best / stopping epoch |
  |---|---|---|---|---|---|
  | GRU + faithful AdaRNN grid | 0 | 1.78634 | 2.31793 | 0.07747 | 20 / 20 |
  | GRU + faithful AdaRNN grid | 0.1 | 1.78632 | 2.31791 | 0.07746 | 20 / 20 |
  | GRU + faithful AdaRNN grid | 1 | 1.78616 | 2.31773 | 0.07744 | 20 / 20 |
  | GRU + faithful AdaRNN grid (= final seed 0) | 10 | 1.78496 | 2.31634 | 0.07720 | 20 / 20 |
  | GRU source-only, SO-tgt, seed 0 | – | 1.66725 | 2.20372 | 0.07092 | 19 / 20 |

  At λ=0 the AdaRNN run is already 0.119 °C MAE worse than source-only. Going from λ=0 to λ=10 changes MAE by
  −0.0014 °C.
- **Steps per epoch.** Climate has 2,969 source-train windows and batch size 64.
  - AdaRNN: one step draws one batch from each of the three TDC periods (sizes 1,188 / 593 / 1,188), so
    floor(2969 / (3·64)) = **15 steps per epoch**, covering 15·192 = 2,880 source windows. This count is recorded as
    `training.steps_per_epoch = 15` in the JSON.
  - Source-only GRU: one ordinary shuffled loader of batch 64 (`src/train_source_only.py:make_loader`, no
    `drop_last`), so ceil(2969/64) = **47 steps per epoch**.
  - Both runs have a 20-epoch cap, and the AdaRNN cap includes its 3 pretrain epochs.
- **Training state at the cap.** All three AdaRNN finals restored epoch 20 of 20, so they hit the cap.
  - Seed 0's last four val MSEs were 0.07881, 0.07854, 0.07725, 0.07720, so it was still decreasing.
  - Source-val MSE for AdaRNN seeds 0-2: 0.07720, 0.07675, 0.07738. For GRU source-only: 0.07092, 0.07107, 0.07090.
- **Evidence:**
  - `experiments/novelty_adarnn/grid/climate_adarnn_lam{0,0p1,1,10}_seed0.json`
  - `experiments/novelty_adarnn/climate/adarnn/seed{0,1,2}.json`
  - `experiments/novelty_adarnn/climate/source_only/seed{0,1,2}.json`
  - `experiments/novelty_adarnn/lambda_selection_climate.json`
  - `results_clean/novelty/summary.md`, `results_clean/novelty/narrative.md`
  - Settings: CLAUDE.md 2026-11-15 (h)
  - Code: `src/adarnn_full/adarnn.py:fit_adarnn_full`, `src/adarnn_full/train_timed.py:fit_timed`

### A2. CORAL (TCN, λ=10) on climate is worse than SO-tgt; the pre-registered climate criterion is not met on RMSE

- **Result.**
  - MAE 1.698 ± 0.031 vs 1.673 ± 0.003 °C (+0.025; paired per seed +0.019, +0.061, −0.005). Noise rule: "exceeds the
    target-stats std only".
  - RMSE 2.211 ± 0.010 vs 2.196 ± 0.009 (+0.015; paired +0.008, +0.037, −0.000). Noise rule: "exceeds both stds".
  - Pre-registered staircase check: MAE "met", RMSE "**NOT met**".
- **Related results:**
  - The CORAL λ=1 sensitivity run is lower than SO-tgt (1.667 MAE, 2.183 RMSE). It is not a replacement for λ=10
    (CLAUDE.md 2026-10-25 (a)).
  - GRU + CORAL in the novelty study is within noise of its own SO-tgt row (+0.004 °C MAE).
  - Per-seed epochs and source-val MSE are in `DEVIATIONS.md`.
- **Evidence:**
  - `experiments/coral/summary.md` (staircase lines) and `experiments/coral/climate/seed{0,1,2}.json`
  - `experiments/coral/lambda_selection.json`
  - `results_clean/noise_rule.md` (climate CORAL rows) and `results_clean/canonical.csv`
  - CLAUDE.md 2026-10-25 (a)

### A3. Energy: residual gap to target-only; the pre-registered energy criterion is not met

- **Result (MAE / RMSE, MW):**

  | row | MAE | RMSE |
  |---|---|---|
  | source-only, source stats | 802.88 ± 32.60 | 1061.17 ± 34.08 |
  | SO-tgt | 561.61 ± 8.75 | 824.40 ± 10.23 |
  | CORAL λ=10 | 554.78 ± 10.35 | 810.45 ± 14.58 |
  | AdaRNN-style λ=10 | 560.19 ± 2.00 | 811.97 ± 1.09 |
  | target-only (labels) | 524.44 ± 10.02 | 794.52 ± 6.95 |

- **Remaining gap.** CORAL to target-only is 30.34 MW MAE and 15.93 MW RMSE.
  - Of the SO-tgt → target-only gap (37.17 MW MAE, 29.88 MW RMSE), CORAL closes 6.84 MW MAE (18.4%) and 13.96 MW
    RMSE (46.7%). These are derived from the canonical means above.
- **Pre-registered criterion** (energy gain over SO-tgt larger than the seed std):
  - Not met on MAE: |−6.84| ≤ max std 10.35 (`experiments/coral/summary.md` uses max std 10 and reports MAE and
    RMSE "NOT met").
  - Under the later noise rule (`results_clean/noise_rule.md`), CORAL MAE is "within both". RMSE "exceeds the
    target-stats std only".
- **Where the gap sits** (`ERROR_STRUCTURE.md`): CORAL minus target-only per hour is 36-62 MW at hours 0-10 and 7-12 MW
  at hours 16-19.
- **Evidence:**
  - `results_clean/canonical.csv` (energy rows)
  - `experiments/coral/summary.md`, `results_clean/noise_rule.md`
  - `experiments/final_3seed/tcn/runs/energy_seed*.json`, `experiments/coral/energy/seed*.json`,
    `experiments/adarnn/energy/seed*.json`
  - CLAUDE.md 2026-10-25 (a)

### A4. Finance (raw log-returns): no model beats the naive train-mean

- **Source validation.** No TCN seed beats the train-mean. Val MSE (norm) is 0.80531, 0.81047, 0.80480 vs naive
  0.79838 (naive recomputed in `audit_numbers.json` → `bakeoff.finance`).
  - Best source epoch is 1 for every seed, and training stopped at epoch 4.
- **Target test.** Every source-trained or adapted row sits at 100.7-100.9% of the naive train-mean MAE (0.00712).
  Target-only with labels is at 100.5%.
  - The GRU rows in the novelty study sit at 99.9-100.1%.
- **Status.** Raw finance was excluded from backbone selection before the bake-off for this reason. It is reported
  as a negative control.
- **Evidence:**
  - `results_clean/canonical.csv` (finance rows)
  - `experiments/final_3seed/tcn/runs/finance_seed*.json`
  - `results_clean/APPROVED_CLAIMS.md` item 11
  - `results_clean/novelty/summary.md` (finance)
  - `experiments/bakeoff/amendment.md` §2

### A5. finance_vol: no backbone beats the naive train-mean on source validation (negative control)

- **Source validation (bake-off).** Naive train-mean val MSE is 0.80510. The best backbone is GRU at 0.80768 ±
  0.00369, and the TCN finals are at 0.818-0.825. The train-median beats every backbone on MAE.
- **Target test.** Model MAE is 102.6-106.8% of the mean-of-last-5 baseline (0.00497).
- **Faithful AdaRNN.** It has lower target-test error than its GRU SO-tgt row (MAE 0.005073 vs 0.005433), but its
  source-val MSE (0.823-0.827) is worse than source-only's and worse than the naive train-mean's. The narrative says
  not to claim this as a gain.
- **Evidence:**
  - `experiments/bakeoff/finance_vol_val_baselines.json`, `experiments/bakeoff/amendment.md` §2
  - `results_clean/canonical.csv`
  - `results_clean/novelty/narrative.md`
  - CLAUDE.md 2026-10-23

## B. Process and protocol failures

| # | failure | outcome | evidence |
|---|---|---|---|
| B1 | Bake-off selection rule picked GRU; the rule was amended after results (finance_vol excluded), then the parameter tie-break (DLinear) was overridden by judgement in favour of TCN. | TCN used for all finals; deviation recorded as post hoc. | `experiments/bakeoff/amendment.md`, `experiments/bakeoff/selection_rule.json`, CLAUDE.md 2026-10-23 |
| B2 | TCN energy seed 2 killed by system memory pressure during target-only training. | Rerun completed 2026-10-25 07:27; superseded. | CLAUDE.md 2026-10-24 status, 2026-10-25 (a); `experiments/STATUS.md` (07:27:42 entry); `experiments/logs/tcn_energy_seed2.log` |
| B3 | Machine ran about 10× slower from about 12:00 on 2026-10-25 (14-thread contention found later). AdaRNN-style energy seed 0 aborted at the 60-minute cap (epoch 25, still improving), and a speed gate failed (9.44 s vs 3 s limit). | Rerun with 2 threads, cap removed; all energy runs finished by 20:35; superseded. Thread count differs between runs before and after 19:06. | CLAUDE.md 2026-10-25 (c), (d), (e), (f); `experiments/not_run_history.json`; `experiments/logs/adarnn_energy_seed*.log` |
| B4 | CORAL λ=1 energy seed 1 ran about 2× slower from about 20:01 (cause not investigated). | Completed; no settings changed. | CLAUDE.md 2026-10-25 (f); `experiments/STATUS.md` (20:36:14 entry) |
| B5 | Novelty-study energy stopped by the pre-registered 300 s/epoch rule at 23:09 on 2026-11-15. Epochs of 805-812 s coincided with Modern Standby 22:56:03-23:08:54. | Applied as written; only GRU + AdaRNN seed 2 exists for energy (700.51 MW MAE, 953.23 RMSE, best epoch 8 of 11); every other GRU energy cell is "not run". | `experiments/novelty_adarnn/energy_stopped.json`, `experiments/novelty_adarnn/energy/adarnn/seed2.json`, `results_clean/novelty/narrative.md`, CLAUDE.md 2026-11-15 (i) |
| B6 | Orchestration bug: `run.py`'s post-stop refusal exited with rc 1 and was counted as a crash, blocking the queued finance and finance_vol novelty finals. | Fixed to exit 0; those finals (never started) then ran once. | CLAUDE.md 2026-11-15 (i); `experiments/STATUS.md` (2026-11-15 entries); `results_clean/novelty/narrative.md` |
