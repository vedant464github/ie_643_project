# IE643 Project: Unsupervised Domain Adaptation for Time-Series Forecasting

## Goal
One general adaptation strategy, validated on 3 intra-domain source->target pairs:
- Finance: Nifty IT -> Nifty Pharma (daily log-returns, input_len=30, horizon=5)
- Climate: Shimla -> Delhi (daily mean temp, input_len=30, horizon=5)
- Energy: AEP -> COMED (hourly load, input_len=168, horizon=24)
Source-only baseline, CORAL, AdaRNN (primary), fine-tune upper bound, RevIN. No target labels during adaptation
(target labels used only for evaluation; fine-tune is the deliberate exception as an upper bound).

## Rules
- Same pipeline/model/loss for all 3 domains; only window sizes, epochs, batch_size are per-domain config.
- Order: chronological split -> normalize (train stats only) -> window. Never leak.
- Windows Windows/cmd, venv at ./venv. Run modules as `python -m src.<module>` from repo root.
- Metrics: MAE, RMSE, MAPE, sMAPE. MAPE/sMAPE are meaningless for finance log-returns; report MAE/RMSE there.
- Always report the naive predict-train-mean baseline in original scale alongside the model.
- Model exposes get_features() on GRUForecaster for CORAL/AdaRNN.

## Decisions (fixed before results)
finance_vol: target = abs(log_return) via a per-domain transform: "abs" field applied before the chronological split. Baselines: train-mean, mean-of-last-5-inputs. MAE/RMSE only. Raw finance stays as a separate negative-control domain.
Backbones: GRU, DLinear, TCN, PatchTST (2 layers, d_model 32, optional RevIN flag, mean-pooled get_features()). Hard cap 50k trainable parameters each; print counts, shrink or drop any config over the cap, never raise it.
Selection: SOURCE val MSE in normalised scale on climate, energy and finance_vol only. Rank per domain by mean over seeds, pick best mean rank. If the top two are within one seed std in most domains, pick fewer parameters; use wall-clock only if parameter counts are within 10%. Target-test numbers are never used to choose.
Seeds: climate and finance_vol 3 seeds (0,1,2), full training. Energy 2 seeds (0,1), every 4th training window, max 15 epochs. Same batch size and lr across backbones within a domain.
Staircase for later: source stats → + target-stat normalisation → + CORAL → + AdaRNN-style → target-only. CORAL and AdaRNN must beat the target-stats row. Success criteria: energy improvement over that row larger than the seed std; climate and finance no worse than seed noise.
Device: CUDA if available, recorded in every results JSON; keep the reference GRU CPU results untouched.

## Amendments (after results)
2026-10-23: Selection rule excluded finance_vol post hoc, since no backbone beat the naive val MSE (same reason raw finance was excluded). Recomputed ranks over climate and energy gave a TCN/DLinear tie. TCN chosen over DLinear by judgment, overriding the parameter tie-break, because DLinear has no learned feature layer for CORAL/TDM. Details in experiments/bakeoff/amendment.md.
Final runs: TCN, seeds 0-2, on finance, finance_vol, climate and energy. Energy protocol: full training set (all 46,361 source train windows, no subsampling, unlike the bake-off's every-4th-window / max-15-epoch protocol), max 40 epochs, early stopping on source val MSE (patience 3, min improvement 1e-5) with best-val weights restored, batch size 128, Adam lr 1e-3, CPU. Target-only: all target train windows, batch 128, fixed epochs = that seed's source best epoch (no target val). Other domains: full training, max 20 epochs, batch 64, same early stopping. Written to experiments/final_3seed/tcn/. Status 2026-10-24: finance, finance_vol, climate seeds 0-2 and energy seeds 0-1 complete; energy seed 2 incomplete (run killed by system memory pressure during target-only training; its source checkpoint exists but no results JSON). summary.json not yet written. To finish: `python -m src.run_final_seeds --backbone tcn --domains energy --seeds 2`, then `python -m src.run_final_seeds --backbone tcn --summarize`.
finance_vol is a negative-control domain (no backbone beats train-mean on source val). Naive-median MAE is lower than any model's, so lead with RMSE for this variant.
2026-10-25: Energy λ grid (λ=0 reference and λ in {0.1,1,10}) uses the bake-off protocol (every 4th training window, ≤15 epochs), for selection only; energy seeds 0-2 at the chosen λ use the full protocol (all windows, ≤40 epochs, patience 3). Finance, finance_vol and climate use full training throughout.
2026-10-25 (a): Energy seed 2 completed (TCN final_3seed now has seeds 0-2 in all four domains; summary.json written). CORAL λ=10 results are final (experiments/coral/). Pre-registered criteria not met on energy (gain over the target-stats row smaller than seed std) and on climate RMSE (worse than target-stats by more than seed std). No settings were changed after seeing results. A CORAL λ=1 run (seeds 0-2, same protocol, experiments/coral_lambda1/) is a sensitivity check only and does not replace λ=10.
2026-10-25 (b), AdaRNN-style design, written before any run: TCN backbone, unchanged. Simplified TDC, computed once on source-train INPUT windows only (source stats), before training: split the source-train windows into K=3 contiguous periods with two split points from the grid {10%,20%,...,90%} of the window count, each period >=20% of windows. Pick the pair that maximises the sum over the 3 period pairs of d(i,j) = ||mu_i-mu_j||^2/L + ||C_i-C_j||_F^2/L^2 (mu, C = mean vector and covariance of the L-dim input windows). Windows are assigned by start index; windows next to a split point share raw steps, and this is accepted. Loss = MSE on the period batches + λ · mean over all 6 pairs among {P1,P2,P3,target} of CORAL(get_features()), with λ=10 (as CORAL, no new grid). No boosting/importance weights. Training: 3 pretrain epochs of MSE only (ordinary shuffled source batches over all source-train windows), then joint training. Each joint step draws one batch of the domain batch size from each of P1, P2, P3 and target-train inputs (equal sampling; smaller periods cycle more often), and one joint epoch = floor(n_source_train / (3·batch)) steps. The total epoch cap stays the domain cap (20; energy 40), with pretrain epochs counted in it. Early stopping on source val MSE (patience 3, min improvement 1e-5) is tracked from the first joint epoch only, and the best joint-epoch weights are restored. Target-train inputs only, normalised with target-train stats; no target labels, val or test. Same protocol as the CORAL finals: full training, energy all windows, batch 64 (energy 128), Adam lr 1e-3. Seeds 0-2, order finance, finance_vol, climate, energy. The MMD diagnostic is the same as for CORAL. Hard stop for all AdaRNN-style runs at 2026-10-25 16:00 local, and any run not finished by then is reported as "not run". Output in experiments/adarnn/.
2026-10-25 (c), run control, not experiment settings: The AdaRNN-style runs had a 16:00 hard stop. From about 12:00 the machine ran roughly 10× slower (climate CORAL epoch ~10 s vs ~1 s in the morning; no competing process; cause not identified; no system settings changed). The CORAL λ=1 run was stopped by design before its first energy epoch so AdaRNN-style (deadline-bound) could go first; λ=1 finance, finance_vol and climate seeds 0-2 are complete. AdaRNN-style finance, finance_vol and climate seeds 0-2 are complete. AdaRNN-style energy seed 0 was aborted at the 60-minute energy cap at epoch 25 (3712 s, still improving) with nothing saved, and per the failure rules no further runs were started: AdaRNN-style energy seeds 0-2 and CORAL λ=1 energy seeds 0-2 are "not run". A speed gate at 16:46 (one climate CORAL epoch, limit 3 s) measured 9.44 s and failed, so nothing was launched. The 2026-10-24 status line saying energy seed 2 is incomplete is superseded (see 2026-10-25 (a)).
2026-10-25 (d): rerun of AdaRNN-style energy and CORAL λ=1 energy with deadline 22:00 and torch threads reduced from 14 to 2 after a benchmark showed 14-thread contention (3.81 s vs 0.39 s per 300 small-batch steps); thread count can change floating-point summation order, so numbers may differ slightly from earlier runs; no experiment setting changed.
2026-10-25 (e): the 60-minute per-run cap for AdaRNN-style energy runs is removed; the 22:00 deadline is the only limit, because the cap was set when 14-thread contention slowed runs about 10× and would now abort a healthy run; energy seeds 0-2 run as three parallel processes of 2 threads each (OMP_NUM_THREADS=2); no experiment setting changed.
2026-10-25 (f): all pending energy runs completed by 20:35: AdaRNN-style λ=10 and CORAL λ=1, seeds 0-2, no crash and no cap hit. Entry (c)’s ‘not run’ status is superseded. Thread counts: runs before 19:06 used the default 14 torch threads, runs after used 2; energy seed 1 of CORAL λ=1 ran about 2× slower from ~20:01 (cause not investigated).

## Layout
data/raw, data/processed/{finance,climate,energy}; src/data/{windowing,prepare}.py; src/models/backbone.py;
src/utils/metrics.py; src/train_source_only.py; src/run_experiments.py; experiments/ (results JSON, checkpoints).

## Status
Done: data download + preprocessing (all 3), windowing/split/normalize, GRU source-only baselines (best-val weights
restored, checkpoints in experiments/checkpoints/), per-domain DOMAIN_CONFIGS with source+target series (TODO 1-2),
target-domain eval of source-only + target-only + naive/seasonal-naive baselines (TODO 3, src/run_target_eval.py).
Source-test results: finance ~= naive (no signal), climate MAE 1.31 vs naive 4.97, energy MAE 557 vs naive 2016.
Target-only baseline trains for a FIXED epoch count (= source best epoch); no target val labels used for selection.

## TODO (in order)
1. [done] train_source_only restores best-val weights; checkpoints at experiments/checkpoints/<domain>_source_only.pt;
   energy epochs 40 (early stopping, patience 3), batch 128.
2. [done] DOMAIN_CONFIGS carries per-domain epochs/batch_size/seasonal_periods and source AND target series.
3. [done] Source-only on TARGET test set (source stats and target train stats), target-only, seasonal-naive.
4. CORAL, then AdaRNN (TDC+TDM), then RevIN, fine-tune upper bound.
5. Domain-gap diagnostic (MMD/Wasserstein on features), results JSON + plots.
6. Streamlit app (CSV upload, >=2 domains), requirements.txt, README, report.