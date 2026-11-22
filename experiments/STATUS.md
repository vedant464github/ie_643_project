# STATUS log (unattended CORAL session)

Timestamps are local machine time.

- **2026-10-25 07:15:59** — Session start. Surveyed repo: TCN final runs present for finance/finance_vol/climate seeds 0-2 and energy seeds 0-1; energy seed 2 has a source checkpoint but no results JSON; no summary.json. src/adaptation/coral.py is empty. No Python processes running; RAM 16.4 GB total, 5.3 GB free. psutil not installed -> RSS will be read via Windows GetProcessMemoryInfo (ctypes), no new packages installed.

- **2026-10-25 07:16:37** — Added 2026-10-25 energy λ-grid entry to CLAUDE.md Amendments (Decisions untouched). Added rss_mb() to src/train.py; fit() now records/prints RSS (MB) at the end of every epoch (history['rss_mb']); no training setting changed. Launching energy seed 2 rerun: python -m src.run_final_seeds --backbone tcn --domains energy --seeds 2 (log: experiments/logs/tcn_energy_seed2.log).

- **2026-10-25 07:19:01** — Implemented CORAL: src/adaptation/coral.py (coral_loss = ||Cs-Ct||_F^2/(4d^2), joint fit_coral with MSE + λ·CORAL, source-val-MSE early stopping identical to fit(): patience 3, min improv 1e-5, best weights restored; unbiased multi-kernel RBF MMD² with median-heuristic bandwidth, ≤2000 windows per side) and src/run_coral.py (--grid/--select/--final/--summarize; resumable; 60-min cap on energy runs). Interpretation logged: 'pooled get_features()' = TCNForecaster.get_features() unchanged (32 channels at the last time step, the head's input; changing the backbone would alter the source-only reference); CORAL covariances are taken over each batch. One target-train batch (same batch size, drop_last) drawn per source batch; λ=0 runs the same code path incl. target batches. In the energy cheap protocol the every-4th stride is applied to target-train inputs too (as the bake-off did for target-only). Smoke test on climate (1 epoch, not saved) passed.

- **2026-10-25 07:27:42** — Energy seed 2 rerun finished (exit 0, 637 s; source early-stopped at epoch 23, best epoch 20; target-only 20 fixed epochs). RSS per epoch: source 549–686 MB, target-only 529–609 MB, no upward trend → no memory growth, so no del/gc.collect() fix was applied and no settings changed. The rerun overwrote experiments/final_3seed/tcn/checkpoints/energy_seed2_source_only.pt (from the killed run) and wrote energy_seed2_target_only.pt and runs/energy_seed2.json. Ran `python -m src.run_final_seeds --backbone tcn --summarize` → experiments/final_3seed/tcn/summary.json (exit 0). Best source epochs per seed: {'finance': [1, 1, 1], 'finance_vol': [1, 2, 2], 'climate': [6, 7, 5], 'energy': [20, 14, 20]}. TCN summary (sample std, ddof=1; finance MAPE/sMAPE not reported):

  | domain | setting | MAE mean ± std | RMSE mean ± std | n |
  |---|---|---|---|---|
  | finance | source-only, SOURCE test | 0.011163 ± 4.5e-05 | 0.01545 ± 6.6e-05 | 3 |
  | finance | source-only, target test, source stats | 0.00717 ± 0 | 0.0096167 ± 1.2e-05 | 3 |
  | finance | source-only, target test, target-train stats | 0.0071733 ± 5.8e-06 | 0.00962 ± 1e-05 | 3 |
  | finance | target-only, target test | 0.0071567 ± 3.8e-05 | 0.0096033 ± 4.9e-05 | 3 |
  | finance | naive train-mean, target test (target-train stats) | 0.00712 | 0.00954 | 1 (deterministic) |
  | finance_vol | source-only, SOURCE test | 0.0074333 ± 9.5e-05 | 0.01079 ± 6.1e-05 | 3 |
  | finance_vol | source-only, target test, source stats | 0.0051 ± 0.0002 | 0.0063833 ± 0.00011 | 3 |
  | finance_vol | source-only, target test, target-train stats | 0.0051567 ± 0.0002 | 0.00641 ± 0.00012 | 3 |
  | finance_vol | target-only, target test | 0.0052667 ± 0.00012 | 0.00646 ± 7.5e-05 | 3 |
  | finance_vol | naive train-mean, target test (target-train stats) | 0.00572 | 0.00678 | 1 (deterministic) |
  | climate | source-only, SOURCE test | 1.2945 ± 0.018 | 1.7941 ± 0.0073 | 3 |
  | climate | source-only, target test, source stats | 2.9452 ± 0.6 | 3.7093 ± 0.65 | 3 |
  | climate | source-only, target test, target-train stats | 1.6731 ± 0.0027 | 2.196 ± 0.0094 | 3 |
  | climate | target-only, target test | 1.6736 ± 0.025 | 2.205 ± 0.026 | 3 |
  | climate | naive train-mean, target test (target-train stats) | 6.1367 | 6.964 | 1 (deterministic) |
  | energy | source-only, SOURCE test | 545 ± 19 | 738.47 ± 24 | 3 |
  | energy | source-only, target test, source stats | 802.88 ± 33 | 1061.2 ± 34 | 3 |
  | energy | source-only, target test, target-train stats | 561.61 ± 8.7 | 824.4 ± 10 | 3 |
  | energy | target-only, target test | 524.44 ± 10 | 794.52 ± 7 | 3 |
  | energy | naive train-mean, target test (target-train stats) | 1720.8 | 2268.2 | 1 (deterministic) |

- **2026-10-25 07:27:50** — CORAL grid start: finance, λ=0, seed 0, protocol full (all windows, <=20 epochs).

- **2026-10-25 07:27:54** — CORAL grid done: finance, λ=0, seed 0: src-val MSE 0.80531, best epoch 1/4, target-test (tgt stats) MAE 0.00717 RMSE 0.00963, MMD2 0.0492 -> 0.0492, 4s, RSS end 350 MB.

- **2026-10-25 07:27:54** — CORAL grid start: finance, λ=0.1, seed 0, protocol full (all windows, <=20 epochs).

- **2026-10-25 07:27:56** — CORAL grid done: finance, λ=0.1, seed 0: src-val MSE 0.80565, best epoch 1/4, target-test (tgt stats) MAE 0.00717 RMSE 0.00963, MMD2 0.0492 -> 0.0490, 3s, RSS end 351 MB.

- **2026-10-25 07:27:56** — CORAL grid start: finance, λ=1, seed 0, protocol full (all windows, <=20 epochs).

- **2026-10-25 07:27:59** — CORAL grid done: finance, λ=1, seed 0: src-val MSE 0.80551, best epoch 1/4, target-test (tgt stats) MAE 0.00717 RMSE 0.00963, MMD2 0.0492 -> 0.0490, 3s, RSS end 351 MB.

- **2026-10-25 07:27:59** — CORAL grid start: finance, λ=10, seed 0, protocol full (all windows, <=20 epochs).

- **2026-10-25 07:28:02** — CORAL grid done: finance, λ=10, seed 0: src-val MSE 0.80441, best epoch 1/4, target-test (tgt stats) MAE 0.00716 RMSE 0.00962, MMD2 0.0492 -> 0.0420, 3s, RSS end 351 MB.

- **2026-10-25 07:28:02** — CORAL grid start: finance_vol, λ=0, seed 0, protocol full (all windows, <=20 epochs).

- **2026-10-25 07:28:05** — CORAL grid done: finance_vol, λ=0, seed 0: src-val MSE 0.81799, best epoch 1/4, target-test (tgt stats) MAE 0.00522 RMSE 0.00642, MMD2 0.0709 -> 0.0709, 3s, RSS end 352 MB.

- **2026-10-25 07:28:05** — CORAL grid start: finance_vol, λ=0.1, seed 0, protocol full (all windows, <=20 epochs).

- **2026-10-25 07:28:08** — CORAL grid done: finance_vol, λ=0.1, seed 0: src-val MSE 0.81783, best epoch 1/4, target-test (tgt stats) MAE 0.00522 RMSE 0.00642, MMD2 0.0709 -> 0.0712, 3s, RSS end 352 MB.

- **2026-10-25 07:28:08** — CORAL grid start: finance_vol, λ=1, seed 0, protocol full (all windows, <=20 epochs).

- **2026-10-25 07:28:11** — CORAL grid done: finance_vol, λ=1, seed 0: src-val MSE 0.81737, best epoch 1/4, target-test (tgt stats) MAE 0.00523 RMSE 0.00643, MMD2 0.0709 -> 0.0687, 3s, RSS end 352 MB.

- **2026-10-25 07:28:11** — CORAL grid start: finance_vol, λ=10, seed 0, protocol full (all windows, <=20 epochs).

- **2026-10-25 07:28:14** — CORAL grid done: finance_vol, λ=10, seed 0: src-val MSE 0.81408, best epoch 1/4, target-test (tgt stats) MAE 0.00524 RMSE 0.00644, MMD2 0.0709 -> 0.0521, 3s, RSS end 352 MB.

- **2026-10-25 07:28:14** — CORAL grid start: climate, λ=0, seed 0, protocol full (all windows, <=20 epochs).

- **2026-10-25 07:28:21** — CORAL grid done: climate, λ=0, seed 0: src-val MSE 0.07244, best epoch 4/7, target-test (tgt stats) MAE 1.6885 RMSE 2.2012, MMD2 0.0562 -> 0.0576, 8s, RSS end 356 MB.

- **2026-10-25 07:28:21** — CORAL grid start: climate, λ=0.1, seed 0, protocol full (all windows, <=20 epochs).

- **2026-10-25 07:28:28** — CORAL grid done: climate, λ=0.1, seed 0: src-val MSE 0.07295, best epoch 3/6, target-test (tgt stats) MAE 1.6863 RMSE 2.2231, MMD2 0.0562 -> 0.0614, 7s, RSS end 358 MB.

- **2026-10-25 07:28:28** — CORAL grid start: climate, λ=1, seed 0, protocol full (all windows, <=20 epochs).

- **2026-10-25 07:28:39** — CORAL grid done: climate, λ=1, seed 0: src-val MSE 0.07204, best epoch 7/10, target-test (tgt stats) MAE 1.6785 RMSE 2.196, MMD2 0.0562 -> 0.0510, 10s, RSS end 358 MB.

- **2026-10-25 07:28:39** — CORAL grid start: climate, λ=10, seed 0, protocol full (all windows, <=20 epochs).

- **2026-10-25 07:28:49** — CORAL grid done: climate, λ=10, seed 0: src-val MSE 0.07258, best epoch 7/10, target-test (tgt stats) MAE 1.6917 RMSE 2.209, MMD2 0.0562 -> 0.0430, 10s, RSS end 365 MB.

- **2026-10-25 07:28:49** — CORAL grid start: energy, λ=0, seed 0, protocol bake-off (every 4th window, <=15 epochs).

- **2026-10-25 07:31:07** — CORAL grid done: energy, λ=0, seed 0: src-val MSE 0.11030, best epoch 14/15, target-test (tgt stats) MAE 598.36 RMSE 856.19, MMD2 0.0215 -> 0.0277, 137s, RSS end 906 MB.

- **2026-10-25 07:31:07** — CORAL grid start: energy, λ=0.1, seed 0, protocol bake-off (every 4th window, <=15 epochs).

- **2026-10-25 07:32:40** — CORAL grid done: energy, λ=0.1, seed 0: src-val MSE 0.11582, best epoch 7/10, target-test (tgt stats) MAE 619.56 RMSE 874.8, MMD2 0.0215 -> 0.0322, 93s, RSS end 908 MB.

- **2026-10-25 07:32:40** — CORAL grid start: energy, λ=1, seed 0, protocol bake-off (every 4th window, <=15 epochs).

- **2026-10-25 07:34:52** — CORAL grid done: energy, λ=1, seed 0: src-val MSE 0.10988, best epoch 13/15, target-test (tgt stats) MAE 609.02 RMSE 866.46, MMD2 0.0215 -> 0.0245, 132s, RSS end 880 MB.

- **2026-10-25 07:34:52** — CORAL grid start: energy, λ=10, seed 0, protocol bake-off (every 4th window, <=15 epochs).

- **2026-10-25 07:37:08** — CORAL grid done: energy, λ=10, seed 0: src-val MSE 0.11317, best epoch 14/15, target-test (tgt stats) MAE 622.6 RMSE 878.54, MMD2 0.0215 -> 0.0227, 135s, RSS end 833 MB.

- **2026-10-25 07:37:29** — λ grid finished (exit 0, 16 runs, all seed 0; per-run lines above). Wrote experiments/coral/lambda_selection.json. Source-val MSE ratio to λ=0 (λ=0.1 / 1 / 10): finance 1.0004 / 1.0003 / 0.9989; finance_vol 0.9998 / 0.9992 / 0.9952; climate 1.0071 / 0.9945 / 1.0019; energy (cheap protocol) 1.0500 / 0.9962 / 1.0260. λ=0.1 fails only on energy (1.05001 > 1.05). Qualifying: {1, 10}. **Chosen λ = 10** (largest qualifying). Target-test numbers appear in the per-run lines because every run is evaluated, but the rule used source-val MSE only. Note: finance/finance_vol best epoch is 1 for every λ (no learnable signal, as in source-only), and energy grid runs mostly hit the 15-epoch cap. Launching final seeds: python -m src.run_coral --final --domains finance finance_vol climate energy --seeds 0 1 2 (log experiments/logs/coral_final.log).

- **2026-10-25 07:37:30** — CORAL final: finance seed 0 reused from grid run finance_lam10_seed0 (same full protocol).

- **2026-10-25 07:37:30** — CORAL final start: finance, λ=10, seed 1, protocol full (all windows, <=20 epochs).

- **2026-10-25 07:37:36** — CORAL final done: finance, λ=10, seed 1: src-val MSE 0.80955, best epoch 4/7, target-test (tgt stats) MAE 0.00721 RMSE 0.00965, MMD2 0.0450 -> 0.0526, 6s, RSS end 351 MB.

- **2026-10-25 07:37:36** — CORAL final start: finance, λ=10, seed 2, protocol full (all windows, <=20 epochs).

- **2026-10-25 07:37:40** — CORAL final done: finance, λ=10, seed 2: src-val MSE 0.80271, best epoch 2/5, target-test (tgt stats) MAE 0.00716 RMSE 0.00961, MMD2 0.0283 -> 0.0282, 3s, RSS end 352 MB.

- **2026-10-25 07:37:40** — CORAL final: finance_vol seed 0 reused from grid run finance_vol_lam10_seed0 (same full protocol).

- **2026-10-25 07:37:40** — CORAL final start: finance_vol, λ=10, seed 1, protocol full (all windows, <=20 epochs).

- **2026-10-25 07:37:44** — CORAL final done: finance_vol, λ=10, seed 1: src-val MSE 0.82211, best epoch 2/5, target-test (tgt stats) MAE 0.00518 RMSE 0.00642, MMD2 0.0680 -> 0.0695, 3s, RSS end 352 MB.

- **2026-10-25 07:37:44** — CORAL final start: finance_vol, λ=10, seed 2, protocol full (all windows, <=20 epochs).

- **2026-10-25 07:37:47** — CORAL final done: finance_vol, λ=10, seed 2: src-val MSE 0.81974, best epoch 1/4, target-test (tgt stats) MAE 0.0052 RMSE 0.00643, MMD2 0.0976 -> 0.0768, 3s, RSS end 352 MB.

- **2026-10-25 07:37:47** — CORAL final: climate seed 0 reused from grid run climate_lam10_seed0 (same full protocol).

- **2026-10-25 07:37:47** — CORAL final start: climate, λ=10, seed 1, protocol full (all windows, <=20 epochs).

- **2026-10-25 07:37:54** — CORAL final done: climate, λ=10, seed 1: src-val MSE 0.07434, best epoch 4/7, target-test (tgt stats) MAE 1.7316 RMSE 2.2226, MMD2 0.0565 -> 0.0531, 8s, RSS end 360 MB.

- **2026-10-25 07:37:54** — CORAL final start: climate, λ=10, seed 2, protocol full (all windows, <=20 epochs).

- **2026-10-25 07:38:01** — CORAL final done: climate, λ=10, seed 2: src-val MSE 0.07374, best epoch 3/6, target-test (tgt stats) MAE 1.6709 RMSE 2.2021, MMD2 0.0656 -> 0.0596, 6s, RSS end 365 MB.

- **2026-10-25 07:38:01** — CORAL final start: energy, λ=10, seed 0, protocol full (all windows, <=40 epochs).

- **2026-10-25 07:44:54** — CORAL final done: energy, λ=10, seed 0: src-val MSE 0.08172, best epoch 10/13, target-test (tgt stats) MAE 551.6 RMSE 804.08, MMD2 0.0215 -> 0.0260, 413s, RSS end 878 MB.

- **2026-10-25 07:44:54** — CORAL final start: energy, λ=10, seed 1, protocol full (all windows, <=40 epochs).

- **2026-10-25 07:57:32** — CORAL final done: energy, λ=10, seed 1: src-val MSE 0.07636, best epoch 21/24, target-test (tgt stats) MAE 566.34 RMSE 827.13, MMD2 0.0330 -> 0.0216, 757s, RSS end 825 MB.

- **2026-10-25 07:57:32** — CORAL final start: energy, λ=10, seed 2, protocol full (all windows, <=40 epochs).

- **2026-10-25 08:05:03** — CORAL final done: energy, λ=10, seed 2: src-val MSE 0.08196, best epoch 11/14, target-test (tgt stats) MAE 546.39 RMSE 800.13, MMD2 0.0384 -> 0.0304, 451s, RSS end 883 MB.

- **2026-10-25 08:05:41** — Final CORAL seeds finished (exit 0, no crashes, no resumes needed). λ=10. finance/finance_vol/climate seed 0 reused from grid (same full protocol); seeds 1-2 new. Energy seeds 0-2 new under the full protocol: 413 s, 757 s and 451 s (all < 60 min), best epochs 10/21/11, RSS end ≤ 883 MB. Results JSON + checkpoints (state_dict, source- and target-train norm stats, input_len, horizon, λ) in experiments/coral/<domain>/seed<k>.{json,pt}. Ran --summarize → experiments/coral/summary.{md,json}. Headline (target test, target-train stats, mean ± std, n=3): energy CORAL MAE 554.8 ± 10 / RMSE 810.4 ± 15 vs target-stats 561.6 ± 8.7 / 824.4 ± 10 vs target-only 524.4 ± 10 / 794.5 ± 7. Better on every seed (paired MAE Δ −9.7/−4.2/−6.6), but the mean gain is smaller than the seed std, so the energy success criterion is NOT met. Climate RMSE +0.015 vs std 0.010 → 'no worse than noise' NOT met on RMSE (met on MAE). finance and finance_vol: within noise (met). MMD² source-val vs target-val: falls on finance_vol, climate and energy means (energy 0.031 → 0.026), flat on finance; CORAL distance falls in every domain (energy 0.0058 → 0.0001). Per-seed MMD² can rise (energy seed 0: 0.0215 → 0.0260). Not started: AdaRNN-style, app. Note: CLAUDE.md Amendments 'Status 2026-10-24' line (energy seed 2 incomplete) is now stale; left unedited, since only the requested entry was added.

- **2026-10-25 11:59:45** — Session 2 start (11:59). Appended two Amendments entries to CLAUDE.md, (a) the CORAL λ=10 outcome and (b) the pre-registered AdaRNN-style design, before any run in this session. Earlier entries and the Decisions section are unedited. The λ=1 grid seed-0 runs for finance/finance_vol/climate already use the full protocol, so they will be reused as λ=1 seed 0 (as was done for λ=10). Energy λ=1 seed 0 is a new full-protocol run.

- **2026-10-25 12:00:11** — Moved the status logger to src/utils/status.py (the session scratchpad helper is gone). run_coral.py now takes --lam/--out; grid and lambda_selection paths stay in experiments/coral. Launching the λ=1 sensitivity run: python -m src.run_coral --final --lam 1 --out experiments/coral_lambda1 --domains finance finance_vol climate energy --seeds 0 1 2 (log experiments/logs/coral_lambda1.log).

- **2026-10-25 12:01:02** — The λ=1 launch at 12:00:11 did not start: the status logger's console print raised UnicodeEncodeError (cp1252, 'λ') before the run. No training ran. Fixed the print to replace unencodable characters and relaunched the same command.

- **2026-10-25 12:01:16** — CORAL final: finance seed 0 reused from grid run finance_lam1_seed0 (same full protocol).

- **2026-10-25 12:01:16** — CORAL final start: finance, λ=1, seed 1, protocol full (all windows, <=20 epochs).

- **2026-10-25 12:01:51** — CORAL final done: finance, λ=1, seed 1: src-val MSE 0.81048, best epoch 1/4, target-test (tgt stats) MAE 0.00718 RMSE 0.00962, MMD2 0.0450 -> 0.0441, 35s, RSS end 349 MB.

- **2026-10-25 12:01:51** — CORAL final start: finance, λ=1, seed 2, protocol full (all windows, <=20 epochs).

- **2026-10-25 12:02:28** — AdaRNN TDC finance: split fractions [0.6, 0.8] (window idx [1196, 1594] of 1993), score 0.16786, 15 candidates.

- **2026-10-25 12:02:28** — AdaRNN TDC finance_vol: split fractions [0.6, 0.8] (window idx [1196, 1594] of 1993), score 0.91504, 15 candidates.

- **2026-10-25 12:02:28** — AdaRNN TDC climate: split fractions [0.4, 0.6] (window idx [1188, 1781] of 2969), score 0.17652, 15 candidates.

- **2026-10-25 12:02:30** — CORAL final done: finance, λ=1, seed 2: src-val MSE 0.80316, best epoch 2/5, target-test (tgt stats) MAE 0.00716 RMSE 0.00962, MMD2 0.0283 -> 0.0323, 38s, RSS end 350 MB.

- **2026-10-25 12:02:30** — CORAL final: finance_vol seed 0 reused from grid run finance_vol_lam1_seed0 (same full protocol).

- **2026-10-25 12:02:30** — CORAL final start: finance_vol, λ=1, seed 1, protocol full (all windows, <=20 epochs).

- **2026-10-25 12:03:19** — AdaRNN TDC energy: split fractions [0.2, 0.8] (window idx [9272, 37089] of 46361), score 0.51186, 15 candidates.

- **2026-10-25 12:03:28** — Implemented AdaRNN-style per the pre-registered design: src/adaptation/adarnn.py (tdc_split, fit_adarnn) and src/run_adarnn.py (resumable; 16:00 hard stop checked before each run and after each epoch; 60-min energy cap; status.json lists done / not run). Smoke test on climate (4 epochs, not saved) passed. TDC computed once on source-train inputs → experiments/adarnn/tdc.json: finance and finance_vol splits at 60%/80% (idx 1196/1594 of 1993); climate 40%/60% (1188/1781 of 2969); energy 20%/80% (9272/37089 of 46361). 15 candidate pairs per domain. AdaRNN runs will start after the λ=1 runs finish (sequential, so the two don't compete for CPU).

- **2026-10-25 12:03:44** — CORAL final done: finance_vol, λ=1, seed 1: src-val MSE 0.82218, best epoch 2/5, target-test (tgt stats) MAE 0.0052 RMSE 0.00643, MMD2 0.0680 -> 0.0755, 73s, RSS end 350 MB.

- **2026-10-25 12:03:44** — CORAL final start: finance_vol, λ=1, seed 2, protocol full (all windows, <=20 epochs).

- **2026-10-25 12:04:14** — CORAL final done: finance_vol, λ=1, seed 2: src-val MSE 0.82155, best epoch 1/4, target-test (tgt stats) MAE 0.00511 RMSE 0.00638, MMD2 0.0976 -> 0.0897, 29s, RSS end 350 MB.

- **2026-10-25 12:04:14** — CORAL final: climate seed 0 reused from grid run climate_lam1_seed0 (same full protocol).

- **2026-10-25 12:04:14** — CORAL final start: climate, λ=1, seed 1, protocol full (all windows, <=20 epochs).

- **2026-10-25 12:04:37** — Installed matplotlib 3.11.2 into ./venv for report/figures and added it to requirements.txt.

- **2026-10-25 12:06:54** — CORAL final done: climate, λ=1, seed 1: src-val MSE 0.07107, best epoch 13/16, target-test (tgt stats) MAE 1.6711 RMSE 2.1749, MMD2 0.0565 -> 0.0509, 160s, RSS end 354 MB.

- **2026-10-25 12:06:54** — CORAL final start: climate, λ=1, seed 2, protocol full (all windows, <=20 epochs).

- **2026-10-25 12:08:37** — Throughput dropped about 10× versus this morning. CORAL climate epochs take ~10 s (were ~1 s), and a 50-step TCN micro-benchmark takes 5.4–8.0 s at 1, 4 or 14 threads. No competing process; CPU load 10%; on AC power at 100% battery; WMI reports clock 2300/2300 MHz. Likely system-level throttling (thermal or vendor power mode). No system settings changed. Results are unaffected, only wall-clock. At this speed an energy run could approach the 60-min cap, and the 16:00 AdaRNN stop becomes binding. Decision: AdaRNN-style (the primary method, with a deadline) runs before the λ=1 energy seeds. The λ=1 process will be stopped as soon as climate seed 2 finishes, before any λ=1 energy run starts. λ=1 energy seeds 0-2 resume after AdaRNN (the runner skips finished seeds). Nothing is dropped; this is an order change only.

- **2026-10-25 12:09:02** — CORAL final done: climate, λ=1, seed 2: src-val MSE 0.07149, best epoch 9/12, target-test (tgt stats) MAE 1.6522 RMSE 2.179, MMD2 0.0656 -> 0.0535, 128s, RSS end 363 MB.

- **2026-10-25 12:09:02** — CORAL final start: energy, λ=1, seed 0, protocol full (all windows, <=40 epochs).

- **2026-10-25 12:09:15** — Stopped the λ=1 process by design, just after 'energy λ=1 seed 0' started (no energy epoch completed; nothing written for energy). λ=1 finance, finance_vol and climate seeds 0-2 are complete in experiments/coral_lambda1/. Launching AdaRNN-style: python -m src.run_adarnn --run --seeds 0 1 2 (log experiments/logs/adarnn.log).

- **2026-10-25 12:09:17** — AdaRNN-style start: finance, seed 0, λ=10, protocol full (all windows, <=20 epochs).

- **2026-10-25 12:09:58** — AdaRNN-style done: finance, seed 0: src-val MSE 0.80493, best epoch 6/9, target-test (tgt stats) MAE 0.00719 RMSE 0.00966, MMD2 0.0492 -> 0.0523, 41s, RSS end 362 MB.

- **2026-10-25 12:09:58** — AdaRNN-style start: finance, seed 1, λ=10, protocol full (all windows, <=20 epochs).

- **2026-10-25 12:10:31** — AdaRNN-style done: finance, seed 1: src-val MSE 0.80904, best epoch 4/7, target-test (tgt stats) MAE 0.00719 RMSE 0.00964, MMD2 0.0450 -> 0.0550, 33s, RSS end 339 MB.

- **2026-10-25 12:10:31** — AdaRNN-style start: finance, seed 2, λ=10, protocol full (all windows, <=20 epochs).

- **2026-10-25 12:11:04** — AdaRNN-style done: finance, seed 2: src-val MSE 0.80633, best epoch 5/8, target-test (tgt stats) MAE 0.00717 RMSE 0.0096, MMD2 0.0283 -> 0.0336, 33s, RSS end 351 MB.

- **2026-10-25 12:11:04** — AdaRNN-style start: finance_vol, seed 0, λ=10, protocol full (all windows, <=20 epochs).

- **2026-10-25 12:11:34** — AdaRNN-style done: finance_vol, seed 0: src-val MSE 0.81551, best epoch 4/7, target-test (tgt stats) MAE 0.00543 RMSE 0.00656, MMD2 0.0709 -> 0.0599, 30s, RSS end 338 MB.

- **2026-10-25 12:11:34** — AdaRNN-style start: finance_vol, seed 1, λ=10, protocol full (all windows, <=20 epochs).

- **2026-10-25 12:12:02** — AdaRNN-style done: finance_vol, seed 1: src-val MSE 0.82506, best epoch 4/7, target-test (tgt stats) MAE 0.00533 RMSE 0.00651, MMD2 0.0680 -> 0.0723, 28s, RSS end 358 MB.

- **2026-10-25 12:12:02** — AdaRNN-style start: finance_vol, seed 2, λ=10, protocol full (all windows, <=20 epochs).

- **2026-10-25 12:12:33** — AdaRNN-style done: finance_vol, seed 2: src-val MSE 0.80897, best epoch 4/7, target-test (tgt stats) MAE 0.00517 RMSE 0.0064, MMD2 0.0976 -> 0.0587, 30s, RSS end 353 MB.

- **2026-10-25 12:12:33** — AdaRNN-style start: climate, seed 0, λ=10, protocol full (all windows, <=20 epochs).

- **2026-10-25 12:12:52** — Wrote src/report.py, which builds experiments/summary_all.{md,json} and report/figures/*.png from whatever results exist (missing cells read 'not run'). Figures use the dataviz reference palette (light mode; categorical slots 1-3 validated all-pairs; the aqua contrast WARN is relieved by visible axis labels) at 3.4 in single-column width, 300 dpi. Test render on partial results; fixed label collisions, an incomplete legend, and wrong λ tick labels when a row is missing. The forecast figure now shows the last horizon step, where the methods differ. Final render will follow the remaining runs.

- **2026-10-25 12:13:41** — AdaRNN-style done: climate, seed 0: src-val MSE 0.07083, best epoch 8/11, target-test (tgt stats) MAE 1.6765 RMSE 2.2123, MMD2 0.0562 -> 0.0478, 67s, RSS end 376 MB.

- **2026-10-25 12:13:41** — AdaRNN-style start: climate, seed 1, λ=10, protocol full (all windows, <=20 epochs).

- **2026-10-25 12:14:54** — AdaRNN-style done: climate, seed 1: src-val MSE 0.07159, best epoch 9/12, target-test (tgt stats) MAE 1.6753 RMSE 2.1959, MMD2 0.0565 -> 0.0492, 73s, RSS end 358 MB.

- **2026-10-25 12:14:54** — AdaRNN-style start: climate, seed 2, λ=10, protocol full (all windows, <=20 epochs).

- **2026-10-25 12:15:36** — AdaRNN-style done: climate, seed 2: src-val MSE 0.07140, best epoch 4/7, target-test (tgt stats) MAE 1.6853 RMSE 2.201, MMD2 0.0656 -> 0.0636, 42s, RSS end 370 MB.

- **2026-10-25 12:15:36** — AdaRNN-style start: energy, seed 0, λ=10, protocol full (all windows, <=40 epochs).

- **2026-10-25 13:18:08** — AdaRNN-style process exited at about 13:17. Energy seed 0 hit the 60-min cap: training passed 3600 s at epoch 25 (3712 s). Source-val MSE per epoch: pretrain 0.1304/0.1184/0.1195, then joint 0.1043 → 0.0794 (epoch 24) → 0.0816 (epoch 25). Still improving, no early stop; RSS 579-718 MB. Aborted before saving, so no JSON or checkpoint. This is the 'single energy run exceeds 60 min' failure rule, not a crash, so no resume: all training stops. Not run: AdaRNN-style energy seeds 0-2 and CORAL λ=1 energy seeds 0-2 (postponed earlier for the deadline). No setting was changed to make a run fit (no stride, epoch cap or timeout change). Wrote experiments/adarnn/status.json and experiments/not_run.json. Next, report only: summary_all.md and figures from existing results.

- **2026-10-25 13:18:59** — Final report render: experiments/summary_all.md (+ .json). Staircase of source stats, target stats, CORAL λ=10, CORAL λ=1, AdaRNN-style and target-only, plus naive train-mean; MAE/RMSE mean ± std with n per cell; MMD² before/after per method; not-run cells marked with the reason. report/figures/: staircase_{finance,finance_vol,climate,energy}.png, forecast_vs_actual.png (seed 0, last horizon step, actual vs source-only target-stats vs CORAL λ=10), lambda_selection.png, lambda_sensitivity.png, mmd_per_seed.png, training_curves.png (seed 0). Energy panels show only existing methods. Headline: no adaptation method beats the target-stats row beyond seed noise in any domain. Climate: CORAL λ=1 MAE 1.667 ± 0.014 vs target-stats 1.673 ± 0.003; AdaRNN-style 1.679 ± 0.006. finance and finance_vol: all within noise; AdaRNN-style slightly worse on finance_vol RMSE (0.00649 vs 0.00641). Session end: AdaRNN-style energy and CORAL λ=1 energy not run (60-min cap rule). AdaRNN-style is not complete in the pre-registered sense.

- **2026-10-25 16:47:11** — Session 3 speed gate (16:46): timed one climate CORAL epoch (TCN, λ=10, seed 0, batch 64, full climate train windows, 14 threads, nothing saved) at 9.44 s. Limit 3 s; this morning ~1 s (the 07:28 grid climate runs took 7-10 s for 7-10 epochs). Gate FAILED, so nothing launched and nothing else done this session. CLAUDE.md not amended, AdaRNN deadline not changed, no AdaRNN or CORAL λ=1 energy run started. summary_all.md and report/figures unchanged. Energy cells for AdaRNN-style and CORAL λ=1 remain 'not run'.

- **2026-10-25 16:50:59** — Appended CLAUDE.md Amendments entry 2026-10-25 (c) covering run control only: 16:00 hard stop, ~10x slowdown from ~12:00, λ=1 stopped before its first energy epoch so AdaRNN-style went first, AdaRNN-style energy seed 0 aborted at the 60-min cap at epoch 25 with nothing saved, 16:46 speed gate failure (9.44 s > 3 s), and the 2026-10-24 seed-2 line superseded. No rerun clause; earlier entries unedited. Nothing launched. Pending, on the user's word that the machine is fixed: re-run the speed gate; if it passes, set the AdaRNN deadline to 22:00 (only change), run src.run_adarnn --run (energy seeds 0-2), then src.run_coral --final --lam 1 --out experiments/coral_lambda1 --domains energy, regenerating summary_all.md and figures after each seed.

- **2026-10-25 18:32:24** — Session 4: building the Streamlit app (TODO 6). Nothing is training; no training will be started, and experiments/ is read only (files copied out, nothing written there except this log). app/ was empty. Installing streamlit into ./venv.

- **2026-10-25 18:41:37** — Streamlit app built and tested; no training started, nothing under experiments/ modified (only read/copied; this log appended). Steps: (1) pip install streamlit 1.66.1; requirements.txt rewritten from pip freeze (77 pins, torch==2.14.1 without +cpu). (2) Refactored preprocessing so the app can reuse it on uploads: fill_daily (preprocess_climate), fill_hourly (preprocess_energy), close_to_log_return (preprocess_finance); load_and_clean now calls them. Verified the refactored functions reproduce all six data/processed series exactly (in memory, nothing rewritten). Added apply_norm (prepare.py, used by normalize) and load_model_checkpoint (evaluate.py, any backbone). (3) src/build_app_assets.py copied 8 seed-0 checkpoints (source-only from final_3seed/tcn, CORAL λ=10 from coral/<d>/seed0.pt) to app/checkpoints/ (byte-identical, cmp-verified; each has norm_stats, input_len, horizon), wrote manifest.json (frequency/transform/units only), built app/samples (climate date,temp = Delhi 4291 rows; energy datetime,load_mw = last 3000 COMED rows; finance and finance_vol date,Close = Nifty Pharma 2895 rows). (4) app/core.py (validation + src reuse) and app/app.py (lazy cached model loading, sidebar, Forecast and Backtest tabs, st.error for bad inputs). (5) app/test_app.py (AppTest) 30/30 PASS: 16 domain×model×norm sample runs, climate first-70% stats == training target-train stats, upload with picked columns, forecast without holdout, 11 bad uploads (too short, single column, non-numeric, unparsable dates, duplicate, unsorted, below input_len+horizon, wrong frequency ×2, non-positive price, binary file). Output in app/test_results.txt. Fix during testing: use_container_width → width='stretch' (deprecation). (6) Headless server (--server.headless true, port 8599): /_stcore/health 'ok', page HTTP 200, no errors in log; server killed, port verified closed. (7) README.md written (was empty). Stopping here.

- **2026-10-25 18:54:45** — Session 5 speed gate (18:54, after the user rebooted the machine): timed one climate CORAL epoch (TCN, λ=10, seed 0, batch 64, full climate train windows, 14 threads, CPU, nothing saved; fit_coral with epochs=1, including its val pass) at 7.25 s. Limit 3 s (morning ~1 s; 16:46 gate 9.44 s). Gate FAILED, so nothing launched. CLAUDE.md not amended (no 2026-10-25 (d) entry), AdaRNN deadline not changed, no AdaRNN-style or CORAL λ=1 energy run started. summary_all.md and report/figures unchanged. Energy cells for AdaRNN-style and CORAL λ=1 remain not run.

- **2026-10-25 19:06:10** — Session 6 speed gate (19:05): one climate CORAL epoch (TCN, λ=10, seed 0, batch 64, full climate train windows, CPU, nothing saved; fit_coral epochs=1 incl. val pass) with torch threads = 2 (OMP_NUM_THREADS=2, MKL_NUM_THREADS=2 in the shell; src/train.py now calls torch.set_num_threads(2), and device_info() records cpu_threads in every results JSON) took 2.35 s. Limit 3 s: gate PASSED. Changes, run control only: src/train.py torch.set_num_threads(2); src/run_adarnn.py DEADLINE 16:00 -> 22:00 (docstring updated). CLAUDE.md Amendments 2026-10-25 (d) appended. Next: python -m src.run_adarnn --run (energy seeds 0-2; others skipped), then python -m src.run_coral --final --lam 1 --out experiments/coral_lambda1 --domains energy; python -m src.report after each seed. Failure rules unchanged (resume once after a crash, stop after a second; stop if any energy run > 60 min).

- **2026-10-25 19:06:23** — AdaRNN-style start: energy, seed 0, λ=10, protocol full (all windows, <=40 epochs).

- **2026-10-25 19:09:25** — Session 7 (19:09): previous Claude session was closed with the terminal, but the AdaRNN-style process it launched survived (PIDs 19548 launcher / 6728, started 19:06:21, energy seed 0 epoch 1 logged at 103 s). It was not killed or relaunched, and no half-written files exist (experiments/adarnn/energy and experiments/coral_lambda1/energy are empty). Checked: CLAUDE.md amendment (d) present once; DEADLINE 22:00; src/train.py torch.set_num_threads(2). The speed gate was not rerun now because a concurrent gate would contend with the running job and distort both measurements. It will be rerun before CORAL λ=1 energy is launched. Reattached the per-seed src.report watcher.

- **2026-10-25 19:13:35** — Session 7 (user instruction): stopped the single AdaRNN-style process (PID 6728) at pretrain epoch 2 (epoch 1 103 s, epoch 2 166 s), nothing saved. src/run_adarnn.py: --seeds already existed; per-run 60-min cap removed for AdaRNN-style (max_seconds=None, post-run check deleted); status.json now always lists seeds 0-2 so parallel processes don't drop each other's entries. CLAUDE.md Amendments 2026-10-25 (e) appended. Launching AdaRNN-style energy seeds 0, 1, 2 as three parallel 2-thread processes, logs experiments/logs/adarnn_energy_seed{0,1,2}.log. Guard: stop seed 2 if any pretrain epoch > 200 s or free RAM < 2 GB (free now 6.41 of 15.69 GB). CORAL λ=1 energy not started; it runs only if the AdaRNN seeds finish before 21:00.

- **2026-10-25 19:13:37** — AdaRNN-style start: energy, seed 1, λ=10, protocol full (all windows, <=40 epochs).

- **2026-10-25 19:13:37** — AdaRNN-style start: energy, seed 0, λ=10, protocol full (all windows, <=40 epochs).

- **2026-10-25 19:13:37** — AdaRNN-style start: energy, seed 2, λ=10, protocol full (all windows, <=40 epochs).

- **2026-10-25 19:29:18** — AdaRNN-style done: energy, seed 1: src-val MSE 0.08821, best epoch 14/17, target-test (tgt stats) MAE 558.25 RMSE 812.97, MMD2 0.0330 -> 0.0280, 940s, RSS end 816 MB.

- **2026-10-25 19:31:34** — AdaRNN-style done: energy, seed 0: src-val MSE 0.08251, best epoch 17/20, target-test (tgt stats) MAE 562.24 RMSE 812.13, MMD2 0.0215 -> 0.0277, 1076s, RSS end 824 MB.

- **2026-10-25 19:33:30** — AdaRNN-style done: energy, seed 2: src-val MSE 0.08274, best epoch 20/23, target-test (tgt stats) MAE 560.08 RMSE 810.81, MMD2 0.0384 -> 0.0381, 1193s, RSS end 796 MB.

- **2026-10-25 19:34:35** — AdaRNN-style energy seeds 0-2 all finished (seed 1 19:29, 940 s; seed 0 19:31, 1076 s; seed 2 19:33, 1193 s). No crash, guard never triggered (pretrain epoch 1 55 s each, free RAM >= 4.1 GB). Report regenerated after each seed; not_run.json updated. Finished before 21:00, so CORAL λ=1 energy proceeds. Speed gate (19:33, idle machine, 2 threads): 2.02 s < 3 s, PASSED. Launching python -m src.run_coral --final --lam 1 --out experiments/coral_lambda1 --domains energy (one process, seeds 0-2 sequential, 60-min CORAL energy cap unchanged), log experiments/logs/coral_lambda1.log.

- **2026-10-25 19:34:37** — CORAL final start: energy, λ=1, seed 0, protocol full (all windows, <=40 epochs).

- **2026-10-25 19:52:07** — CORAL final done: energy, λ=1, seed 0: src-val MSE 0.07774, best epoch 17/20, target-test (tgt stats) MAE 557.76 RMSE 812.08, MMD2 0.0215 -> 0.0218, 1049s, RSS end 854 MB.

- **2026-10-25 19:52:07** — CORAL final start: energy, λ=1, seed 1, protocol full (all windows, <=40 epochs).

- **2026-10-25 20:21:18** — CORAL final done: energy, λ=1, seed 1: src-val MSE 0.07673, best epoch 21/24, target-test (tgt stats) MAE 569.6 RMSE 830.38, MMD2 0.0330 -> 0.0218, 1751s, RSS end 873 MB.

- **2026-10-25 20:21:18** — CORAL final start: energy, λ=1, seed 2, protocol full (all windows, <=40 epochs).

- **2026-10-25 20:35:18** — CORAL final done: energy, λ=1, seed 2: src-val MSE 0.08106, best epoch 13/16, target-test (tgt stats) MAE 555.94 RMSE 808.27, MMD2 0.0384 -> 0.0310, 839s, RSS end 907 MB.

- **2026-10-25 20:36:14** — CORAL λ=1 energy seeds 0-2 finished (one process, rc 0, no crash, no cap hit): seed 0 19:52 (1049 s, best epoch 17/20, MAE 557.76 RMSE 812.08), seed 1 20:21 (1751 s, best 21/24, MAE 569.6 RMSE 830.38), seed 2 20:35 (839 s, best 13/16, MAE 555.94 RMSE 808.27). Seed 1 epochs 11-20 ran ~2x slower (64-123 s vs ~50 s) from ~20:01 with no competing process, normal memory and no visible throttling in a 20:15 snapshot; speed returned for seed 2. Energy (target test, tgt stats, n=3): target-stats 561.61±8.7 / 824.4±10; CORAL λ=10 554.78±10 / 810.45±15; CORAL λ=1 561.1±7.4 / 816.91±12; AdaRNN-style 560.19±2 / 811.97±1.1; target-only 524.44±10 / 794.52±7. All pre-registered energy cells are now filled. experiments/not_run.json moved to experiments/not_run_history.json (nothing outstanding); summary_all.md and report/figures regenerated at 20:36. Nothing else running.

- **2026-11-15 21:55:28** — Novelty assessment (CLAUDE.md 2026-11-15 (h)) begins. Code src/adarnn_full/ (model, adarnn, train_timed, run, driver); smoke test on synthetic data only (GRU params 9861 / 10488 energy; per-step CORAL == coral_loss; boosting update checked by hand). Climate phase 1: three parallel queues (OMP_NUM_THREADS=2): source-only seeds 0-2; CORAL grid seed 0 λ∈{0,0.1,1,10}; AdaRNN grid seed 0 λ∈{0,0.1,1,10}. Logs experiments/novelty_adarnn/logs/.

- **2026-11-15 21:55:28** — novelty queue climA (PID 9968) start: ['final --domain climate --method source_only --seeds 0 1 2']

- **2026-11-15 21:55:28** — novelty queue climB (PID 9632) start: ['grid --domain climate --method coral']

- **2026-11-15 21:55:28** — novelty queue climC (PID 26888) start: ['grid --domain climate --method adarnn']

- **2026-11-15 21:55:32** — novelty start: grid climate coral λ=0 seed 0 (full (all windows, <=20 epochs)).

- **2026-11-15 21:55:32** — novelty start: climate source_only λ=none seed 0 (full (all windows, <=20 epochs)).

- **2026-11-15 21:55:32** — novelty start: grid climate adarnn λ=0 seed 0 (full (all windows, <=20 epochs)).

- **2026-11-15 21:55:40** — novelty: first epoch climate adarnn seed 0 λ=0 (grid, full (all windows, <=20 epochs)): 1.8 s/epoch.

- **2026-11-15 21:55:42** — novelty: first epoch climate source_only seed 0 λ=none (final, full (all windows, <=20 epochs)): 3.8 s/epoch.

- **2026-11-15 21:55:45** — novelty: first epoch climate coral seed 0 λ=0 (grid, full (all windows, <=20 epochs)): 7.5 s/epoch.

- **2026-11-15 21:56:26** — novelty done: climate adarnn λ=0 seed 0 (grid, full (all windows, <=20 epochs)): src-val MSE 0.07747, best epoch 20/20, target test (tgt stats) MAE 1.7863 RMSE 2.3179, 54 s.

- **2026-11-15 21:56:26** — novelty start: grid climate adarnn λ=0.1 seed 0 (full (all windows, <=20 epochs)).

- **2026-11-15 21:56:28** — novelty: first epoch climate adarnn seed 0 λ=0p1 (grid, full (all windows, <=20 epochs)): 1.7 s/epoch.

- **2026-11-15 21:56:59** — novelty done: climate source_only λ=none seed 0 (final, full (all windows, <=20 epochs)): src-val MSE 0.07092, best epoch 19/20, target test (tgt stats) MAE 1.6672 RMSE 2.2037, 87 s.

- **2026-11-15 21:56:59** — novelty start: climate source_only λ=none seed 1 (full (all windows, <=20 epochs)).

- **2026-11-15 21:57:03** — novelty: first epoch climate source_only seed 1 λ=none (final, full (all windows, <=20 epochs)): 4.0 s/epoch.

- **2026-11-15 21:57:18** — novelty done: climate adarnn λ=0p1 seed 0 (grid, full (all windows, <=20 epochs)): src-val MSE 0.07746, best epoch 20/20, target test (tgt stats) MAE 1.7863 RMSE 2.3179, 51 s.

- **2026-11-15 21:57:18** — novelty start: grid climate adarnn λ=1 seed 0 (full (all windows, <=20 epochs)).

- **2026-11-15 21:57:20** — novelty: first epoch climate adarnn seed 0 λ=1 (grid, full (all windows, <=20 epochs)): 2.0 s/epoch.

- **2026-11-15 21:57:54** — novelty done: climate coral λ=0 seed 0 (grid, full (all windows, <=20 epochs)): src-val MSE 0.07123, best epoch 14/17, target test (tgt stats) MAE 1.681 RMSE 2.2203, 142 s.

- **2026-11-15 21:57:54** — novelty start: grid climate coral λ=0.1 seed 0 (full (all windows, <=20 epochs)).

- **2026-11-15 21:58:03** — novelty: first epoch climate coral seed 0 λ=0p1 (grid, full (all windows, <=20 epochs)): 9.3 s/epoch.

- **2026-11-15 21:58:07** — novelty done: climate source_only λ=none seed 1 (final, full (all windows, <=20 epochs)): src-val MSE 0.07107, best epoch 13/16, target test (tgt stats) MAE 1.6937 RMSE 2.2191, 68 s.

- **2026-11-15 21:58:07** — novelty start: climate source_only λ=none seed 2 (full (all windows, <=20 epochs)).

- **2026-11-15 21:58:12** — novelty done: climate adarnn λ=1 seed 0 (grid, full (all windows, <=20 epochs)): src-val MSE 0.07744, best epoch 20/20, target test (tgt stats) MAE 1.7862 RMSE 2.3177, 54 s.

- **2026-11-15 21:58:12** — novelty start: grid climate adarnn λ=10 seed 0 (full (all windows, <=20 epochs)).

- **2026-11-15 21:58:13** — novelty: first epoch climate source_only seed 2 λ=none (final, full (all windows, <=20 epochs)): 5.6 s/epoch.

- **2026-11-15 21:58:14** — novelty: first epoch climate adarnn seed 0 λ=10 (grid, full (all windows, <=20 epochs)): 2.1 s/epoch.

- **2026-11-15 21:59:08** — novelty done: climate adarnn λ=10 seed 0 (grid, full (all windows, <=20 epochs)): src-val MSE 0.07720, best epoch 20/20, target test (tgt stats) MAE 1.785 RMSE 2.3163, 55 s.

- **2026-11-15 21:59:09** — novelty queue climC: all jobs finished.

- **2026-11-15 21:59:38** — novelty done: climate source_only λ=none seed 2 (final, full (all windows, <=20 epochs)): src-val MSE 0.07090, best epoch 17/20, target test (tgt stats) MAE 1.6717 RMSE 2.2093, 91 s.

- **2026-11-15 21:59:39** — novelty queue climA: all jobs finished.

- **2026-11-15 21:59:43** — novelty: climate AdaRNN grid finished (all λ hit the 20-epoch cap, best epoch 20/20). Free slot: starting energy AdaRNN grid (cheap protocol, selection only), queue enC.

- **2026-11-15 21:59:44** — novelty queue enC (PID 28436) start: ['grid --domain energy --method adarnn']

- **2026-11-15 21:59:48** — novelty start: grid energy adarnn λ=0 seed 0 (cheap (every 4th window, <=15 epochs)).

- **2026-11-15 22:00:15** — novelty done: climate coral λ=0p1 seed 0 (grid, full (all windows, <=20 epochs)): src-val MSE 0.07123, best epoch 14/17, target test (tgt stats) MAE 1.681 RMSE 2.2203, 141 s.

- **2026-11-15 22:00:15** — novelty start: grid climate coral λ=1 seed 0 (full (all windows, <=20 epochs)).

- **2026-11-15 22:00:17** — novelty: first epoch climate coral seed 0 λ=1 (grid, full (all windows, <=20 epochs)): 1.5 s/epoch.

- **2026-11-15 22:00:23** — novelty: first epoch energy adarnn seed 0 λ=0 (grid, cheap (every 4th window, <=15 epochs)): 27.1 s/epoch.

- **2026-11-15 22:01:52** — novelty done: climate coral λ=1 seed 0 (grid, full (all windows, <=20 epochs)): src-val MSE 0.07121, best epoch 14/17, target test (tgt stats) MAE 1.6809 RMSE 2.2202, 97 s.

- **2026-11-15 22:01:52** — novelty start: grid climate coral λ=10 seed 0 (full (all windows, <=20 epochs)).

- **2026-11-15 22:01:56** — novelty: first epoch climate coral seed 0 λ=10 (grid, full (all windows, <=20 epochs)): 4.0 s/epoch.

- **2026-11-15 22:04:00** — novelty done: climate coral λ=10 seed 0 (grid, full (all windows, <=20 epochs)): src-val MSE 0.07109, best epoch 14/17, target test (tgt stats) MAE 1.6809 RMSE 2.2203, 128 s.

- **2026-11-15 22:04:01** — novelty queue climB: all jobs finished.

- **2026-11-15 22:04:20** — novelty λ selection climate coral: ratios {'0': 1.0, '0.1': 1.0, '1': 0.9997, '10': 0.998} -> chosen 10.0.

- **2026-11-15 22:04:20** — novelty λ selection climate adarnn: ratios {'0': 1.0, '0.1': 1.0, '1': 0.9996, '10': 0.9966} -> chosen 10.0.

- **2026-11-15 22:04:21** — novelty queue climD (PID 29144) start: ['final --domain climate --method coral --seeds 0 1 2']

- **2026-11-15 22:04:21** — novelty queue climE (PID 20900) start: ['final --domain climate --method adarnn --seeds 0 1 2']

- **2026-11-15 22:04:25** — novelty: climate coral seed 0 final = grid run at λ=10 (same protocol), reused.

- **2026-11-15 22:04:25** — novelty start: climate coral λ=10 seed 1 (full (all windows, <=20 epochs)).

- **2026-11-15 22:04:26** — novelty: climate adarnn seed 0 final = grid run at λ=10 (same protocol), reused.

- **2026-11-15 22:04:26** — novelty start: climate adarnn λ=10 seed 1 (full (all windows, <=20 epochs)).

- **2026-11-15 22:04:34** — novelty: first epoch climate adarnn seed 1 λ=10 (final, full (all windows, <=20 epochs)): 2.1 s/epoch.

- **2026-11-15 22:04:40** — novelty: first epoch climate coral seed 1 λ=10 (final, full (all windows, <=20 epochs)): 8.5 s/epoch.

- **2026-11-15 22:05:29** — novelty done: climate adarnn λ=10 seed 1 (final, full (all windows, <=20 epochs)): src-val MSE 0.07675, best epoch 20/20, target test (tgt stats) MAE 1.7973 RMSE 2.33, 63 s.

- **2026-11-15 22:05:29** — novelty start: climate adarnn λ=10 seed 2 (full (all windows, <=20 epochs)).

- **2026-11-15 22:05:31** — novelty: first epoch climate adarnn seed 2 λ=10 (final, full (all windows, <=20 epochs)): 1.9 s/epoch.

- **2026-11-15 22:06:22** — novelty done: climate adarnn λ=10 seed 2 (final, full (all windows, <=20 epochs)): src-val MSE 0.07738, best epoch 20/20, target test (tgt stats) MAE 1.7937 RMSE 2.3279, 53 s.

- **2026-11-15 22:06:24** — novelty queue climE: all jobs finished.

- **2026-11-15 22:07:00** — novelty done: climate coral λ=10 seed 1 (final, full (all windows, <=20 epochs)): src-val MSE 0.07018, best epoch 14/17, target test (tgt stats) MAE 1.6845 RMSE 2.2217, 154 s.

- **2026-11-15 22:07:00** — novelty start: climate coral λ=10 seed 2 (full (all windows, <=20 epochs)).

- **2026-11-15 22:07:08** — novelty: first epoch climate coral seed 2 λ=10 (final, full (all windows, <=20 epochs)): 7.9 s/epoch.

- **2026-11-15 22:09:43** — novelty done: climate coral λ=10 seed 2 (final, full (all windows, <=20 epochs)): src-val MSE 0.07116, best epoch 15/18, target test (tgt stats) MAE 1.6787 RMSE 2.2059, 163 s.

- **2026-11-15 22:09:45** — novelty queue climD: all jobs finished.

- **2026-11-15 22:10:08** — novelty: climate complete (source-only, CORAL λ=10, AdaRNN λ=10, seeds 0-2; seed 0 of CORAL/AdaRNN reused from the grid). Starting energy CORAL grid (cheap protocol), queue enB, in parallel with enC (AdaRNN grid).

- **2026-11-15 22:10:08** — novelty queue enB (PID 28652) start: ['grid --domain energy --method coral']

- **2026-11-15 22:10:13** — novelty start: grid energy coral λ=0 seed 0 (cheap (every 4th window, <=15 epochs)).

- **2026-11-15 22:11:09** — novelty done: energy adarnn λ=0 seed 0 (grid, cheap (every 4th window, <=15 epochs)): src-val MSE 0.17139, best epoch 15/15, target test (tgt stats) MAE 747.55 RMSE 1016, 679 s.

- **2026-11-15 22:11:09** — novelty start: grid energy adarnn λ=0.1 seed 0 (cheap (every 4th window, <=15 epochs)).

- **2026-11-15 22:11:10** — novelty: first epoch energy coral seed 0 λ=0 (grid, cheap (every 4th window, <=15 epochs)): 48.6 s/epoch.

- **2026-11-15 22:11:35** — novelty: first epoch energy adarnn seed 0 λ=0p1 (grid, cheap (every 4th window, <=15 epochs)): 25.9 s/epoch.

- **2026-11-15 22:15:13** — novelty done: energy adarnn λ=0p1 seed 0 (grid, cheap (every 4th window, <=15 epochs)): src-val MSE 0.17138, best epoch 15/15, target test (tgt stats) MAE 747.54 RMSE 1016, 243 s.

- **2026-11-15 22:15:13** — novelty start: grid energy adarnn λ=1 seed 0 (cheap (every 4th window, <=15 epochs)).

- **2026-11-15 22:15:37** — novelty: first epoch energy adarnn seed 0 λ=1 (grid, cheap (every 4th window, <=15 epochs)): 24.3 s/epoch.

- **2026-11-15 22:19:14** — novelty done: energy coral λ=0 seed 0 (grid, cheap (every 4th window, <=15 epochs)): src-val MSE 0.13193, best epoch 13/15, target test (tgt stats) MAE 722.79 RMSE 973.69, 540 s.

- **2026-11-15 22:19:14** — novelty start: grid energy coral λ=0.1 seed 0 (cheap (every 4th window, <=15 epochs)).

- **2026-11-15 22:19:16** — novelty done: energy adarnn λ=1 seed 0 (grid, cheap (every 4th window, <=15 epochs)): src-val MSE 0.17131, best epoch 15/15, target test (tgt stats) MAE 747.48 RMSE 1015.9, 242 s.

- **2026-11-15 22:19:16** — novelty start: grid energy adarnn λ=10 seed 0 (cheap (every 4th window, <=15 epochs)).

- **2026-11-15 22:19:41** — novelty: first epoch energy adarnn seed 0 λ=10 (grid, cheap (every 4th window, <=15 epochs)): 24.9 s/epoch.

- **2026-11-15 22:20:30** — novelty: first epoch energy coral seed 0 λ=0p1 (grid, cheap (every 4th window, <=15 epochs)): 74.9 s/epoch.

- **2026-11-15 22:24:09** — novelty done: energy adarnn λ=10 seed 0 (grid, cheap (every 4th window, <=15 epochs)): src-val MSE 0.17061, best epoch 15/15, target test (tgt stats) MAE 746.64 RMSE 1014.9, 293 s.

- **2026-11-15 22:24:10** — novelty queue enC: all jobs finished.

- **2026-11-15 22:28:28** — novelty done: energy coral λ=0p1 seed 0 (grid, cheap (every 4th window, <=15 epochs)): src-val MSE 0.13183, best epoch 13/15, target test (tgt stats) MAE 722.71 RMSE 973.6, 553 s.

- **2026-11-15 22:28:28** — novelty start: grid energy coral λ=1 seed 0 (cheap (every 4th window, <=15 epochs)).

- **2026-11-15 22:29:33** — novelty: first epoch energy coral seed 0 λ=1 (grid, cheap (every 4th window, <=15 epochs)): 64.6 s/epoch.

- **2026-11-15 22:37:41** — novelty done: energy coral λ=1 seed 0 (grid, cheap (every 4th window, <=15 epochs)): src-val MSE 0.13089, best epoch 13/15, target test (tgt stats) MAE 722.13 RMSE 973.01, 552 s.

- **2026-11-15 22:37:41** — novelty start: grid energy coral λ=10 seed 0 (cheap (every 4th window, <=15 epochs)).

- **2026-11-15 22:38:43** — novelty: first epoch energy coral seed 0 λ=10 (grid, cheap (every 4th window, <=15 epochs)): 62.5 s/epoch.

- **2026-11-15 22:41:13** — novelty: energy AdaRNN grid done; energy CORAL grid still running. Two idle slots: finance λ grids (CORAL, AdaRNN; small domain, minutes) so they finish before the energy probe needs all 3 slots. Finance/finance_vol finals stay after energy.

- **2026-11-15 22:41:13** — novelty queue finB (PID 14784) start: ['grid --domain finance --method coral', 'grid --domain finance_vol --method coral']

- **2026-11-15 22:41:13** — novelty queue finC (PID 12320) start: ['grid --domain finance --method adarnn', 'grid --domain finance_vol --method adarnn']

- **2026-11-15 22:41:15** — novelty start: grid finance coral λ=0 seed 0 (full (all windows, <=20 epochs)).

- **2026-11-15 22:41:15** — novelty start: grid finance adarnn λ=0 seed 0 (full (all windows, <=20 epochs)).

- **2026-11-15 22:41:17** — novelty: first epoch finance adarnn seed 0 λ=0 (grid, full (all windows, <=20 epochs)): 0.4 s/epoch.

- **2026-11-15 22:41:18** — novelty: first epoch finance coral seed 0 λ=0 (grid, full (all windows, <=20 epochs)): 1.8 s/epoch.

- **2026-11-15 22:41:21** — novelty done: finance adarnn λ=0 seed 0 (grid, full (all windows, <=20 epochs)): src-val MSE 0.79893, best epoch 4/7, target test (tgt stats) MAE 0.00712 RMSE 0.00954, 5 s.

- **2026-11-15 22:41:21** — novelty start: grid finance adarnn λ=0.1 seed 0 (full (all windows, <=20 epochs)).

- **2026-11-15 22:41:21** — novelty: first epoch finance adarnn seed 0 λ=0p1 (grid, full (all windows, <=20 epochs)): 0.4 s/epoch.

- **2026-11-15 22:41:24** — novelty done: finance adarnn λ=0p1 seed 0 (grid, full (all windows, <=20 epochs)): src-val MSE 0.79893, best epoch 4/7, target test (tgt stats) MAE 0.00712 RMSE 0.00954, 4 s.

- **2026-11-15 22:41:24** — novelty start: grid finance adarnn λ=1 seed 0 (full (all windows, <=20 epochs)).

- **2026-11-15 22:41:25** — novelty: first epoch finance adarnn seed 0 λ=1 (grid, full (all windows, <=20 epochs)): 0.4 s/epoch.

- **2026-11-15 22:41:26** — novelty done: finance coral λ=0 seed 0 (grid, full (all windows, <=20 epochs)): src-val MSE 0.79825, best epoch 2/5, target test (tgt stats) MAE 0.00712 RMSE 0.00954, 11 s.

- **2026-11-15 22:41:26** — novelty start: grid finance coral λ=0.1 seed 0 (full (all windows, <=20 epochs)).

- **2026-11-15 22:41:27** — novelty: first epoch finance coral seed 0 λ=0p1 (grid, full (all windows, <=20 epochs)): 1.8 s/epoch.

- **2026-11-15 22:41:28** — novelty done: finance adarnn λ=1 seed 0 (grid, full (all windows, <=20 epochs)): src-val MSE 0.79893, best epoch 4/7, target test (tgt stats) MAE 0.00712 RMSE 0.00954, 4 s.

- **2026-11-15 22:41:28** — novelty start: grid finance adarnn λ=10 seed 0 (full (all windows, <=20 epochs)).

- **2026-11-15 22:41:28** — novelty: first epoch finance adarnn seed 0 λ=10 (grid, full (all windows, <=20 epochs)): 0.4 s/epoch.

- **2026-11-15 22:41:31** — novelty done: finance adarnn λ=10 seed 0 (grid, full (all windows, <=20 epochs)): src-val MSE 0.79893, best epoch 4/7, target test (tgt stats) MAE 0.00712 RMSE 0.00954, 4 s.

- **2026-11-15 22:41:34** — novelty start: grid finance_vol adarnn λ=0 seed 0 (full (all windows, <=20 epochs)).

- **2026-11-15 22:41:34** — novelty done: finance coral λ=0p1 seed 0 (grid, full (all windows, <=20 epochs)): src-val MSE 0.79825, best epoch 2/5, target test (tgt stats) MAE 0.00712 RMSE 0.00954, 9 s.

- **2026-11-15 22:41:34** — novelty start: grid finance coral λ=1 seed 0 (full (all windows, <=20 epochs)).

- **2026-11-15 22:41:36** — novelty: first epoch finance coral seed 0 λ=1 (grid, full (all windows, <=20 epochs)): 1.7 s/epoch.

- **2026-11-15 22:41:37** — novelty: first epoch finance_vol adarnn seed 0 λ=0 (grid, full (all windows, <=20 epochs)): 0.4 s/epoch.

- **2026-11-15 22:41:43** — novelty done: finance coral λ=1 seed 0 (grid, full (all windows, <=20 epochs)): src-val MSE 0.79825, best epoch 2/5, target test (tgt stats) MAE 0.00712 RMSE 0.00954, 9 s.

- **2026-11-15 22:41:43** — novelty start: grid finance coral λ=10 seed 0 (full (all windows, <=20 epochs)).

- **2026-11-15 22:41:45** — novelty: first epoch finance coral seed 0 λ=10 (grid, full (all windows, <=20 epochs)): 1.7 s/epoch.

- **2026-11-15 22:41:46** — novelty done: finance_vol adarnn λ=0 seed 0 (grid, full (all windows, <=20 epochs)): src-val MSE 0.82455, best epoch 14/17, target test (tgt stats) MAE 0.00514 RMSE 0.00635, 11 s.

- **2026-11-15 22:41:46** — novelty start: grid finance_vol adarnn λ=0.1 seed 0 (full (all windows, <=20 epochs)).

- **2026-11-15 22:41:46** — novelty: first epoch finance_vol adarnn seed 0 λ=0p1 (grid, full (all windows, <=20 epochs)): 0.4 s/epoch.

- **2026-11-15 22:41:52** — novelty done: finance coral λ=10 seed 0 (grid, full (all windows, <=20 epochs)): src-val MSE 0.79825, best epoch 2/5, target test (tgt stats) MAE 0.00712 RMSE 0.00954, 9 s.

- **2026-11-15 22:41:55** — novelty done: finance_vol adarnn λ=0p1 seed 0 (grid, full (all windows, <=20 epochs)): src-val MSE 0.82455, best epoch 14/17, target test (tgt stats) MAE 0.00514 RMSE 0.00635, 9 s.

- **2026-11-15 22:41:55** — novelty start: grid finance_vol adarnn λ=1 seed 0 (full (all windows, <=20 epochs)).

- **2026-11-15 22:41:55** — novelty start: grid finance_vol coral λ=0 seed 0 (full (all windows, <=20 epochs)).

- **2026-11-15 22:41:56** — novelty: first epoch finance_vol adarnn seed 0 λ=1 (grid, full (all windows, <=20 epochs)): 0.4 s/epoch.

- **2026-11-15 22:41:59** — novelty: first epoch finance_vol coral seed 0 λ=0 (grid, full (all windows, <=20 epochs)): 1.8 s/epoch.

- **2026-11-15 22:42:04** — novelty done: finance_vol coral λ=0 seed 0 (grid, full (all windows, <=20 epochs)): src-val MSE 0.80832, best epoch 1/4, target test (tgt stats) MAE 0.00563 RMSE 0.0067, 9 s.

- **2026-11-15 22:42:04** — novelty start: grid finance_vol coral λ=0.1 seed 0 (full (all windows, <=20 epochs)).

- **2026-11-15 22:42:04** — novelty done: finance_vol adarnn λ=1 seed 0 (grid, full (all windows, <=20 epochs)): src-val MSE 0.82453, best epoch 14/17, target test (tgt stats) MAE 0.00514 RMSE 0.00635, 9 s.

- **2026-11-15 22:42:04** — novelty start: grid finance_vol adarnn λ=10 seed 0 (full (all windows, <=20 epochs)).

- **2026-11-15 22:42:05** — novelty: first epoch finance_vol adarnn seed 0 λ=10 (grid, full (all windows, <=20 epochs)): 0.4 s/epoch.

- **2026-11-15 22:42:06** — novelty: first epoch finance_vol coral seed 0 λ=0p1 (grid, full (all windows, <=20 epochs)): 1.7 s/epoch.

- **2026-11-15 22:42:10** — novelty done: finance_vol adarnn λ=10 seed 0 (grid, full (all windows, <=20 epochs)): src-val MSE 0.82701, best epoch 7/10, target test (tgt stats) MAE 0.00499 RMSE 0.00628, 5 s.

- **2026-11-15 22:42:10** — novelty queue finC: all jobs finished.

- **2026-11-15 22:42:11** — novelty done: finance_vol coral λ=0p1 seed 0 (grid, full (all windows, <=20 epochs)): src-val MSE 0.80832, best epoch 1/4, target test (tgt stats) MAE 0.00563 RMSE 0.0067, 7 s.

- **2026-11-15 22:42:11** — novelty start: grid finance_vol coral λ=1 seed 0 (full (all windows, <=20 epochs)).

- **2026-11-15 22:42:13** — novelty: first epoch finance_vol coral seed 0 λ=1 (grid, full (all windows, <=20 epochs)): 1.7 s/epoch.

- **2026-11-15 22:42:18** — novelty done: finance_vol coral λ=1 seed 0 (grid, full (all windows, <=20 epochs)): src-val MSE 0.80832, best epoch 1/4, target test (tgt stats) MAE 0.00563 RMSE 0.0067, 6 s.

- **2026-11-15 22:42:18** — novelty start: grid finance_vol coral λ=10 seed 0 (full (all windows, <=20 epochs)).

- **2026-11-15 22:42:19** — novelty: first epoch finance_vol coral seed 0 λ=10 (grid, full (all windows, <=20 epochs)): 1.6 s/epoch.

- **2026-11-15 22:42:24** — novelty done: finance_vol coral λ=10 seed 0 (grid, full (all windows, <=20 epochs)): src-val MSE 0.80832, best epoch 1/4, target test (tgt stats) MAE 0.00563 RMSE 0.0067, 7 s.

- **2026-11-15 22:42:25** — novelty queue finB: all jobs finished.

- **2026-11-15 22:46:59** — novelty done: energy coral λ=10 seed 0 (grid, cheap (every 4th window, <=15 epochs)): src-val MSE 0.12830, best epoch 13/15, target test (tgt stats) MAE 714.23 RMSE 966.22, 558 s.

- **2026-11-15 22:47:00** — novelty queue enB: all jobs finished.

- **2026-11-15 22:47:13** — novelty λ selection energy coral: ratios {'0': 1.0, '0.1': 0.9992, '1': 0.9921, '10': 0.9725} -> chosen 10.0.

- **2026-11-15 22:47:14** — novelty λ selection energy adarnn: ratios {'0': 1.0, '0.1': 1.0, '1': 0.9996, '10': 0.9954} -> chosen 10.0.

- **2026-11-15 22:47:16** — novelty λ selection finance coral: ratios {'0': 1.0, '0.1': 1.0, '1': 1.0, '10': 1.0} -> chosen 10.0.

- **2026-11-15 22:47:16** — novelty λ selection finance adarnn: ratios {'0': 1.0, '0.1': 1.0, '1': 1.0, '10': 1.0} -> chosen 10.0.

- **2026-11-15 22:47:18** — novelty λ selection finance_vol coral: ratios {'0': 1.0, '0.1': 1.0, '1': 1.0, '10': 1.0} -> chosen 10.0.

- **2026-11-15 22:47:19** — novelty λ selection finance_vol adarnn: ratios {'0': 1.0, '0.1': 1.0, '1': 1.0, '10': 1.003} -> chosen 10.0.

- **2026-11-15 22:47:19** — novelty: launching energy protocol probe: GRU+AdaRNN energy finals seeds 0,1,2 in 3 parallel processes, full protocol; decision on the max first-epoch time (<=150 s full, else cheap).

- **2026-11-15 22:47:19** — novelty queue enP0 (PID 27172) start: ['final --domain energy --method adarnn --seeds 0 --probe']

- **2026-11-15 22:47:19** — novelty queue enP1 (PID 17256) start: ['final --domain energy --method adarnn --seeds 1 --probe']

- **2026-11-15 22:47:19** — novelty queue enP2 (PID 27916) start: ['final --domain energy --method adarnn --seeds 2 --probe']

- **2026-11-15 22:47:21** — novelty start: energy adarnn λ=10 seed 1 (full (all windows, <=40 epochs)).

- **2026-11-15 22:47:21** — novelty start: energy adarnn λ=10 seed 0 (full (all windows, <=40 epochs)).

- **2026-11-15 22:47:21** — novelty start: energy adarnn λ=10 seed 2 (full (all windows, <=40 epochs)).

- **2026-11-15 22:48:06** — novelty: first epoch energy adarnn seed 1 λ=10 (final, full (all windows, <=40 epochs)): 43.0 s/epoch.

- **2026-11-15 22:48:07** — novelty: first epoch energy adarnn seed 2 λ=10 (final, full (all windows, <=40 epochs)): 43.9 s/epoch.

- **2026-11-15 22:48:12** — novelty: first epoch energy adarnn seed 0 λ=10 (final, full (all windows, <=40 epochs)): 48.8 s/epoch.

- **2026-11-15 22:48:12** — novelty: energy protocol probe, first-epoch seconds {0: 48.78181250000125, 1: 42.98037729999851, 2: 43.86662240000078} (max 48.8) -> protocol 'full'.

- **2026-11-15 22:48:28** — novelty: energy protocol = FULL (first-epoch s: seed0 48.8, seed1 43.0, seed2 43.9; max 48.8 <= 150). AdaRNN energy seeds 0-2 continue as finals. Each slot then runs energy source-only seed s and CORAL λ=10 seed s (full protocol), then finance/finance_vol finals.

- **2026-11-15 22:50:46** — novelty: energy protocol = FULL (first-epoch s: seed0 48.8, seed1 43.0, seed2 43.9; max 48.8 <= 150). AdaRNN energy seeds 0-2 continue as finals. Each slot then runs energy source-only seed s and CORAL λ=10 seed s (full protocol); slots 0/1 then run finance / finance_vol finals.

- **2026-11-15 22:54:47** — novelty done: energy adarnn λ=10 seed 2 (final, full (all windows, <=40 epochs)): src-val MSE 0.12247, best epoch 8/11, target test (tgt stats) MAE 700.51 RMSE 953.23, 446 s.

- **2026-11-15 22:54:48** — novelty queue enP2: all jobs finished.

- **2026-11-15 22:55:16** — novelty queue enQ2 (PID 26724) start: ['final --domain energy --method source_only --seeds 2', 'final --domain energy --method coral --seeds 2']

- **2026-11-15 22:55:19** — novelty start: energy source_only λ=none seed 2 (full (all windows, <=40 epochs)).

- **2026-11-15 22:56:00** — novelty: first epoch energy source_only seed 2 λ=none (final, full (all windows, <=40 epochs)): 38.7 s/epoch.

- **2026-11-15 23:09:16** — novelty: ENERGY STOPPED: [adarnn energy lam=10 s0 final] epoch 14 took 806s > 300.0s.

- **2026-11-15 23:09:16** — novelty queue enP0: all jobs finished.

- **2026-11-15 23:09:19** — novelty: ENERGY STOPPED: [adarnn energy lam=10 s1 final] epoch 14 took 805s > 300.0s.

- **2026-11-15 23:09:19** — novelty queue enP1: all jobs finished.

- **2026-11-15 23:09:32** — novelty: ENERGY STOPPED: [source_only energy lam=none s2 final] epoch 2 took 812s > 300.0s.

- **2026-11-15 23:09:34** — novelty queue enQ0 (PID 17004) start: ['final --domain energy --method source_only --seeds 0', 'final --domain energy --method coral --seeds 0', 'final --domain finance --method source_only --seeds 0 1 2', 'final --domain finance --method coral --seeds 0 1 2', 'final --domain finance --method adarnn --seeds 0 1 2']

- **2026-11-15 23:09:34** — novelty queue enQ1 (PID 26652) start: ['final --domain energy --method source_only --seeds 1', 'final --domain energy --method coral --seeds 1', 'final --domain finance_vol --method source_only --seeds 0 1 2', 'final --domain finance_vol --method coral --seeds 0 1 2', 'final --domain finance_vol --method adarnn --seeds 0 1 2']

- **2026-11-15 23:09:34** — novelty queue enQ2: crash (rc 1) in 'final --domain energy --method coral --seeds 2'; resuming once.

- **2026-11-15 23:09:36** — novelty queue enQ0: crash (rc 1) in 'final --domain energy --method source_only --seeds 0'; resuming once.

- **2026-11-15 23:09:36** — novelty queue enQ1: crash (rc 1) in 'final --domain energy --method source_only --seeds 1'; resuming once.

- **2026-11-15 23:09:37** — novelty queue enQ2: second crash (rc 1) in 'final --domain energy --method coral --seeds 2'; queue stopped per failure rules.

- **2026-11-15 23:09:39** — novelty queue enQ0: second crash (rc 1) in 'final --domain energy --method source_only --seeds 0'; queue stopped per failure rules.

- **2026-11-15 23:09:39** — novelty queue enQ1: second crash (rc 1) in 'final --domain energy --method source_only --seeds 1'; queue stopped per failure rules.

- **2026-11-15 23:10:35** — novelty: ENERGY STOPPED per the pre-registered 300 s/epoch rule at 23:09: AdaRNN seeds 0 and 1 epoch 14 (805, 806 s) and source-only seed 2 epoch 2 (812 s). Cause found in the Windows System log: Kernel-Power 506/507, the machine was in Modern Standby 22:56:03-23:08:54 (771 s); epochs before were ~40 s. Rule applied as written, no override. Energy result kept: AdaRNN λ=10 seed 2 (finished 22:54, before the standby). Everything else in energy is 'not run'. Orchestration bug: run.py's refusal to start energy jobs after the stop exited with rc 1, which the queue driver counted as crashes, so queues enQ0/enQ1 stopped before their finance/finance_vol jobs started. Fix: the refusal now logs and exits 0 (src/adarnn_full/run.py). No experiment setting changed; finance/finance_vol finals had not started, so nothing is rerun. Launching finance/finance_vol finals (3 queues).

- **2026-11-15 23:10:35** — novelty queue finX (PID 29516) start: ['final --domain finance --method source_only --seeds 0 1 2', 'final --domain finance_vol --method source_only --seeds 0 1 2']

- **2026-11-15 23:10:35** — novelty queue finY (PID 3228) start: ['final --domain finance --method coral --seeds 0 1 2', 'final --domain finance_vol --method coral --seeds 0 1 2']

- **2026-11-15 23:10:35** — novelty queue finZ (PID 18936) start: ['final --domain finance --method adarnn --seeds 0 1 2', 'final --domain finance_vol --method adarnn --seeds 0 1 2']

- **2026-11-15 23:10:37** — novelty start: finance source_only λ=none seed 0 (full (all windows, <=20 epochs)).

- **2026-11-15 23:10:37** — novelty: finance coral seed 0 final = grid run at λ=10 (same protocol), reused.

- **2026-11-15 23:10:37** — novelty start: finance coral λ=10 seed 1 (full (all windows, <=20 epochs)).

- **2026-11-15 23:10:37** — novelty: finance adarnn seed 0 final = grid run at λ=10 (same protocol), reused.

- **2026-11-15 23:10:37** — novelty start: finance adarnn λ=10 seed 1 (full (all windows, <=20 epochs)).

- **2026-11-15 23:10:39** — novelty: first epoch finance adarnn seed 1 λ=10 (final, full (all windows, <=20 epochs)): 0.4 s/epoch.

- **2026-11-15 23:10:40** — novelty: first epoch finance source_only seed 0 λ=none (final, full (all windows, <=20 epochs)): 1.0 s/epoch.

- **2026-11-15 23:10:41** — novelty: first epoch finance coral seed 1 λ=10 (final, full (all windows, <=20 epochs)): 1.8 s/epoch.

- **2026-11-15 23:10:42** — novelty done: finance source_only λ=none seed 0 (final, full (all windows, <=20 epochs)): src-val MSE 0.79829, best epoch 1/4, target test (tgt stats) MAE 0.00712 RMSE 0.00954, 5 s.

- **2026-11-15 23:10:42** — novelty start: finance source_only λ=none seed 1 (full (all windows, <=20 epochs)).

- **2026-11-15 23:10:42** — novelty done: finance adarnn λ=10 seed 1 (final, full (all windows, <=20 epochs)): src-val MSE 0.80088, best epoch 4/7, target test (tgt stats) MAE 0.00713 RMSE 0.00955, 5 s.

- **2026-11-15 23:10:42** — novelty start: finance adarnn λ=10 seed 2 (full (all windows, <=20 epochs)).

- **2026-11-15 23:10:43** — novelty: first epoch finance adarnn seed 2 λ=10 (final, full (all windows, <=20 epochs)): 0.4 s/epoch.

- **2026-11-15 23:10:43** — novelty: first epoch finance source_only seed 1 λ=none (final, full (all windows, <=20 epochs)): 0.8 s/epoch.

- **2026-11-15 23:10:46** — novelty done: finance coral λ=10 seed 1 (final, full (all windows, <=20 epochs)): src-val MSE 0.79907, best epoch 1/4, target test (tgt stats) MAE 0.00712 RMSE 0.00954, 9 s.

- **2026-11-15 23:10:46** — novelty start: finance coral λ=10 seed 2 (full (all windows, <=20 epochs)).

- **2026-11-15 23:10:46** — novelty done: finance source_only λ=none seed 1 (final, full (all windows, <=20 epochs)): src-val MSE 0.79907, best epoch 1/4, target test (tgt stats) MAE 0.00712 RMSE 0.00954, 4 s.

- **2026-11-15 23:10:46** — novelty start: finance source_only λ=none seed 2 (full (all windows, <=20 epochs)).

- **2026-11-15 23:10:47** — novelty: first epoch finance source_only seed 2 λ=none (final, full (all windows, <=20 epochs)): 0.9 s/epoch.

- **2026-11-15 23:10:47** — novelty done: finance adarnn λ=10 seed 2 (final, full (all windows, <=20 epochs)): src-val MSE 0.80096, best epoch 6/9, target test (tgt stats) MAE 0.00713 RMSE 0.00955, 5 s.

- **2026-11-15 23:10:47** — novelty: first epoch finance coral seed 2 λ=10 (final, full (all windows, <=20 epochs)): 1.8 s/epoch.

- **2026-11-15 23:10:50** — novelty done: finance source_only λ=none seed 2 (final, full (all windows, <=20 epochs)): src-val MSE 0.79959, best epoch 1/4, target test (tgt stats) MAE 0.00712 RMSE 0.00954, 4 s.

- **2026-11-15 23:10:50** — novelty: finance_vol adarnn seed 0 final = grid run at λ=10 (same protocol), reused.

- **2026-11-15 23:10:50** — novelty start: finance_vol adarnn λ=10 seed 1 (full (all windows, <=20 epochs)).

- **2026-11-15 23:10:52** — novelty: first epoch finance_vol adarnn seed 1 λ=10 (final, full (all windows, <=20 epochs)): 0.5 s/epoch.

- **2026-11-15 23:10:52** — novelty start: finance_vol source_only λ=none seed 0 (full (all windows, <=20 epochs)).

- **2026-11-15 23:10:52** — novelty done: finance coral λ=10 seed 2 (final, full (all windows, <=20 epochs)): src-val MSE 0.79959, best epoch 1/4, target test (tgt stats) MAE 0.00712 RMSE 0.00954, 7 s.

- **2026-11-15 23:10:54** — novelty: first epoch finance_vol source_only seed 0 λ=none (final, full (all windows, <=20 epochs)): 0.4 s/epoch.

- **2026-11-15 23:10:55** — novelty: finance_vol coral seed 0 final = grid run at λ=10 (same protocol), reused.

- **2026-11-15 23:10:55** — novelty start: finance_vol coral λ=10 seed 1 (full (all windows, <=20 epochs)).

- **2026-11-15 23:10:55** — novelty done: finance_vol source_only λ=none seed 0 (final, full (all windows, <=20 epochs)): src-val MSE 0.80832, best epoch 1/4, target test (tgt stats) MAE 0.00563 RMSE 0.0067, 3 s.

- **2026-11-15 23:10:55** — novelty start: finance_vol source_only λ=none seed 1 (full (all windows, <=20 epochs)).

- **2026-11-15 23:10:56** — novelty: first epoch finance_vol source_only seed 1 λ=none (final, full (all windows, <=20 epochs)): 0.4 s/epoch.

- **2026-11-15 23:10:56** — novelty done: finance_vol adarnn λ=10 seed 1 (final, full (all windows, <=20 epochs)): src-val MSE 0.82250, best epoch 14/17, target test (tgt stats) MAE 0.0052 RMSE 0.00639, 6 s.

- **2026-11-15 23:10:56** — novelty start: finance_vol adarnn λ=10 seed 2 (full (all windows, <=20 epochs)).

- **2026-11-15 23:10:56** — novelty: first epoch finance_vol adarnn seed 2 λ=10 (final, full (all windows, <=20 epochs)): 0.2 s/epoch.

- **2026-11-15 23:10:57** — novelty done: finance_vol source_only λ=none seed 1 (final, full (all windows, <=20 epochs)): src-val MSE 0.81101, best epoch 1/4, target test (tgt stats) MAE 0.00542 RMSE 0.00653, 2 s.

- **2026-11-15 23:10:57** — novelty start: finance_vol source_only λ=none seed 2 (full (all windows, <=20 epochs)).

- **2026-11-15 23:10:57** — novelty: first epoch finance_vol source_only seed 2 λ=none (final, full (all windows, <=20 epochs)): 0.4 s/epoch.

- **2026-11-15 23:10:58** — novelty: first epoch finance_vol coral seed 1 λ=10 (final, full (all windows, <=20 epochs)): 0.8 s/epoch.

- **2026-11-15 23:10:58** — novelty done: finance_vol adarnn λ=10 seed 2 (final, full (all windows, <=20 epochs)): src-val MSE 0.82629, best epoch 4/7, target test (tgt stats) MAE 0.00503 RMSE 0.00631, 2 s.

- **2026-11-15 23:10:58** — novelty queue finZ: all jobs finished.

- **2026-11-15 23:10:59** — novelty done: finance_vol source_only λ=none seed 2 (final, full (all windows, <=20 epochs)): src-val MSE 0.80372, best epoch 1/4, target test (tgt stats) MAE 0.00525 RMSE 0.00643, 2 s.

- **2026-11-15 23:10:59** — novelty queue finX: all jobs finished.

- **2026-11-15 23:11:00** — novelty done: finance_vol coral λ=10 seed 1 (final, full (all windows, <=20 epochs)): src-val MSE 0.81101, best epoch 1/4, target test (tgt stats) MAE 0.00542 RMSE 0.00653, 4 s.

- **2026-11-15 23:11:00** — novelty start: finance_vol coral λ=10 seed 2 (full (all windows, <=20 epochs)).

- **2026-11-15 23:11:00** — novelty: first epoch finance_vol coral seed 2 λ=10 (final, full (all windows, <=20 epochs)): 0.6 s/epoch.

- **2026-11-15 23:11:02** — novelty done: finance_vol coral λ=10 seed 2 (final, full (all windows, <=20 epochs)): src-val MSE 0.80372, best epoch 1/4, target test (tgt stats) MAE 0.00525 RMSE 0.00643, 2 s.

- **2026-11-15 23:11:03** — novelty queue finY: all jobs finished.

- **2026-11-15 23:12:06** — novelty: finance and finance_vol finals done (4 GRU methods x seeds 0-2). results_clean/novelty/summary.md, summary.json and architecture.png written; CLAUDE.md 2026-11-15 (i) records the outcome. Nothing running.

- **2026-11-20 03:43:25** — novelty_fix: CLAUDE.md amendment 2026-11-20 (j) appended before any run. S_old=15 (floor(2969/192), logged steps_per_epoch), S_base=47 (ceil(2969/64), make_loader without drop_last). Power check: AC, sleep/hibernate on AC = never, DC sleep 300 s; no system setting changed; processes hold SetThreadExecutionState keep-awake.

- **2026-11-20 03:45:52** — novelty_fix queue start (PID 26656), 3 workers, seed-major order: [('source_only', 0), ('coral', 0), ('adarnn10', 0), ('adarnn0', 0), ('source_only', 1), ('coral', 1), ('adarnn10', 1), ('adarnn0', 1), ('source_only', 2), ('coral', 2), ('adarnn10', 2), ('adarnn0', 2)]

- **2026-11-20 03:45:55** — novelty_fix start: source_only seed 0 (PID 1848).

- **2026-11-20 03:45:55** — novelty_fix start: coral seed 0 (PID 12472).

- **2026-11-20 03:45:55** — novelty_fix start: adarnn10 seed 0 (PID 17928).

- **2026-11-20 03:45:57** — novelty_fix: first epoch source_only seed 0: 0.7 s.

- **2026-11-20 03:45:57** — novelty_fix: first epoch adarnn10 seed 0: 1.0 s.

- **2026-11-20 03:45:57** — novelty_fix: first epoch coral seed 0: 1.3 s.

- **2026-11-20 03:46:02** — novelty_fix PROJECTION: first epoch of (c) adarnn10 seed 0 (pretrain, 47 steps) = 1.0 s; first epochs source_only 0.7 s, coral 1.3 s; joint/pretrain ratio 1.39 (old log). Worst case all runs to the cap: 0.25 h serial, ~0.08 h wall-clock with 3 processes -> does not exceed 3 hours; within the hard stop. Continuing.

- **2026-11-20 03:46:11** — novelty_fix done: source_only seed 0: src-val MSE 0.07092, best epoch 19/22, target test MAE 1.6672 RMSE 2.2037, 16 s.

- **2026-11-20 03:46:13** — novelty_fix start: adarnn0 seed 0 (PID 20420).

- **2026-11-20 03:46:15** — novelty_fix: first epoch adarnn0 seed 0: 0.9 s.

- **2026-11-20 03:46:19** — novelty_fix done: coral seed 0: src-val MSE 0.07109, best epoch 14/17, target test MAE 1.6809 RMSE 2.2203, 24 s.

- **2026-11-20 03:46:21** — novelty_fix start: source_only seed 1 (PID 13340).

- **2026-11-20 03:46:22** — novelty_fix done: adarnn10 seed 0: src-val MSE 0.07078, best epoch 12/15, target test MAE 1.6949 RMSE 2.2336, 28 s.

- **2026-11-20 03:46:24** — novelty_fix: first epoch source_only seed 1: 0.6 s.

- **2026-11-20 03:46:25** — novelty_fix start: coral seed 1 (PID 9076).

- **2026-11-20 03:46:27** — novelty_fix: first epoch coral seed 1: 1.4 s.

- **2026-11-20 03:46:34** — novelty_fix done: source_only seed 1: src-val MSE 0.07107, best epoch 13/16, target test MAE 1.6937 RMSE 2.2191, 13 s.

- **2026-11-20 03:46:37** — novelty_fix start: adarnn10 seed 1 (PID 20776).

- **2026-11-20 03:46:39** — novelty_fix: first epoch adarnn10 seed 1: 1.3 s.

- **2026-11-20 03:46:43** — novelty_fix done: adarnn0 seed 0: src-val MSE 0.07042, best epoch 16/19, target test MAE 1.6695 RMSE 2.2027, 30 s.

- **2026-11-20 03:46:46** — novelty_fix start: adarnn0 seed 1 (PID 18464).

- **2026-11-20 03:46:48** — novelty_fix: first epoch adarnn0 seed 1: 1.0 s.

- **2026-11-20 03:46:51** — novelty_fix decision (before any result seen): success criterion counted as met for a metric only if (c) has lower error than (a) AND the noise-rule verdict is 'exceeds both stds'; weaker verdicts are reported but not counted as a win. Summary also reports (c) vs (d) with the same rule as extra information.

- **2026-11-20 03:46:52** — novelty_fix done: coral seed 1: src-val MSE 0.07018, best epoch 14/17, target test MAE 1.6845 RMSE 2.2217, 27 s.

- **2026-11-20 03:46:55** — novelty_fix start: source_only seed 2 (PID 26724).

- **2026-11-20 03:46:57** — novelty_fix: first epoch source_only seed 2: 0.8 s.

- **2026-11-20 03:46:58** — novelty_fix clarification: the criterion interpretation above was written into src/adarnn_fix/summarize.py before I read any result, but seed-0 result lines were already in this log at 03:46:51 (runs take seconds). Timing stated for transparency; the interpretation is unchanged.

- **2026-11-20 03:47:07** — novelty_fix done: adarnn10 seed 1: src-val MSE 0.06997, best epoch 14/17, target test MAE 1.6810 RMSE 2.2176, 30 s.

- **2026-11-20 03:47:10** — novelty_fix start: coral seed 2 (PID 504).

- **2026-11-20 03:47:13** — novelty_fix: first epoch coral seed 2: 1.3 s.

- **2026-11-20 03:47:13** — novelty_fix done: source_only seed 2: src-val MSE 0.07090, best epoch 17/20, target test MAE 1.6717 RMSE 2.2093, 19 s.

- **2026-11-20 03:47:16** — novelty_fix start: adarnn10 seed 2 (PID 16372).

- **2026-11-20 03:47:17** — novelty_fix done: adarnn0 seed 1: src-val MSE 0.07006, best epoch 14/17, target test MAE 1.6813 RMSE 2.2177, 31 s.

- **2026-11-20 03:47:19** — novelty_fix: first epoch adarnn10 seed 2: 1.2 s.

- **2026-11-20 03:47:19** — novelty_fix start: adarnn0 seed 2 (PID 2500).

- **2026-11-20 03:47:22** — novelty_fix: first epoch adarnn0 seed 2: 1.2 s.

- **2026-11-20 03:47:41** — novelty_fix done: coral seed 2: src-val MSE 0.07116, best epoch 15/18, target test MAE 1.6787 RMSE 2.2059, 31 s.

- **2026-11-20 03:48:01** — novelty_fix done: adarnn10 seed 2: src-val MSE 0.06968, best epoch 23/26, target test MAE 1.6515 RMSE 2.1816, 45 s.

- **2026-11-20 03:48:03** — novelty_fix done: adarnn0 seed 2: src-val MSE 0.06954, best epoch 23/26, target test MAE 1.6502 RMSE 2.1784, 44 s.

- **2026-11-20 03:48:03** — novelty_fix queue finished: {'source_only s0': 'done', 'coral s0': 'done', 'adarnn10 s0': 'done', 'source_only s1': 'done', 'adarnn0 s0': 'done', 'coral s1': 'done', 'adarnn10 s1': 'done', 'source_only s2': 'done', 'adarnn0 s1': 'done', 'coral s2': 'done', 'adarnn10 s2': 'done', 'adarnn0 s2': 'done'}

- **2026-11-20 03:49:03** — novelty_fix: results_clean/novelty2/summary.md, summary.json, val_curves.png, narrative.md written. All 12 runs complete (4 methods x seeds 0-2), no crash/sleep/slow epoch. Climate MAE: (a) 1.678±0.014, (b) CORAL 1.681±0.003, (c) AdaRNN step-matched λ=10 1.676±0.022, (d) λ=0 1.667±0.016. Success criterion ((c) beats (a)) NOT met: within both stds (MAE d=-0.002, RMSE d=+0.000). The gap to the 20-epoch AdaRNN (1.792) is closed by the schedule; (c) vs (d) within both. Nothing running; stopped here per task.

- **2026-11-21 13:08:44** — ablation_cap20 STEP 1: epoch counting read from src/adarnn_full/adarnn.py (single epoch counter over pretrain+joint) and confirmed in experiments/novelty_fix JSONs (training.phase = 3 pretrain + joint). best/stopped epochs reported for (c)/(d) INCLUDE the 3 pretrain epochs. Joint terms: (c) s0 9/12, s1 11/14, s2 20/23; (d) s0 13/16, s1 11/14, s2 20/23. Written to results_clean/novelty2/ablation_cap20.md.

- **2026-11-21 13:09:34** — ablation_cap20 STEP 2: pre-registration appended to CLAUDE.md as amendment 2026-11-21 (k) (CLAUDE.md had no trailing newline, so the append first joined TODO item 6; I split my own appended text onto a new line under '## Amendments (continued...)'; no earlier text changed). Run-control decisions (before any run): code src/adarnn_fix/ablation_cap20.py calls src.adarnn_fix.run.run() unchanged, overriding only OUT, JOINT_CAP=20, DEADLINE=18:00, the log prefix and the cap-60 projection (no-op). Queue seed-major (adarnn10, adarnn0) x seeds 0-2, 3 workers, OMP_NUM_THREADS=2. 'Stop after a second crash' read as: no further jobs start. Sleep reruns capped at 3 per job as in novelty_fix. Expectation from the code, written before running: seeds 0 and 1 stopped at joint epochs <=16 < 20, and seed 2's best joint epoch is 20 = the new cap, so if training is reproducible per seed the cap-20 runs should restore the same weights as the cap-60 runs; the run tests this.

- **2026-11-21 13:09:55** — ablation_cap20 queue start (PID 31576), 3 workers, seed-major order: [('adarnn10', 0), ('adarnn0', 0), ('adarnn10', 1), ('adarnn0', 1), ('adarnn10', 2), ('adarnn0', 2)]

- **2026-11-21 13:09:59** — ablation_cap20 start: adarnn10 seed 1 (PID 24868), joint cap 20.

- **2026-11-21 13:10:00** — ablation_cap20 start: adarnn10 seed 0 (PID 28380), joint cap 20.

- **2026-11-21 13:10:00** — ablation_cap20 start: adarnn0 seed 0 (PID 3776), joint cap 20.

- **2026-11-21 13:10:07** — ablation_cap20: first epoch adarnn0 seed 0: 1.2 s.

- **2026-11-21 13:10:07** — ablation_cap20: first epoch adarnn10 seed 1: 1.2 s.

- **2026-11-21 13:10:07** — ablation_cap20: first epoch adarnn10 seed 0: 1.2 s.

- **2026-11-21 13:10:07** — ablation_cap20: first epoch of (c) adarnn10 seed 0 = 1.2 s (no projection; 6 short runs).

- **2026-11-21 13:11:55** — ablation_cap20 done: adarnn10 seed 0: src-val MSE 0.07078, best epoch 12/15, target test MAE 1.6949 RMSE 2.2336, 115 s.

- **2026-11-21 13:12:03** — ablation_cap20 start: adarnn0 seed 1 (PID 7804), joint cap 20.

- **2026-11-21 13:12:15** — ablation_cap20 done: adarnn10 seed 1: src-val MSE 0.06997, best epoch 14/17, target test MAE 1.6810 RMSE 2.2176, 134 s.

- **2026-11-21 13:12:17** — ablation_cap20: first epoch adarnn0 seed 1: 6.5 s.

- **2026-11-21 13:12:23** — ablation_cap20 start: adarnn10 seed 2 (PID 34956), joint cap 20.

- **2026-11-21 13:12:25** — ablation_cap20 done: adarnn0 seed 0: src-val MSE 0.07042, best epoch 16/19, target test MAE 1.6695 RMSE 2.2027, 144 s.

- **2026-11-21 13:12:26** — ablation_cap20: first epoch adarnn10 seed 2: 1.2 s.

- **2026-11-21 13:12:29** — ablation_cap20 start: adarnn0 seed 2 (PID 33964), joint cap 20.

- **2026-11-21 13:12:32** — ablation_cap20: first epoch adarnn0 seed 2: 1.1 s.

- **2026-11-21 13:13:25** — ablation_cap20 done: adarnn0 seed 1: src-val MSE 0.07006, best epoch 14/17, target test MAE 1.6813 RMSE 2.2177, 81 s.

- **2026-11-21 13:14:39** — ablation_cap20 done: adarnn10 seed 2: src-val MSE 0.06968, best epoch 23/23, target test MAE 1.6515 RMSE 2.1816, 136 s.

- **2026-11-21 13:15:07** — ablation_cap20 done: adarnn0 seed 2: src-val MSE 0.06954, best epoch 23/23, target test MAE 1.6502 RMSE 2.1784, 158 s.

- **2026-11-21 13:15:09** — ablation_cap20 queue finished: {'adarnn10 s0': 'done', 'adarnn10 s1': 'done', 'adarnn0 s0': 'done', 'adarnn0 s1': 'done', 'adarnn10 s2': 'done', 'adarnn0 s2': 'done'}

- **2026-11-21 13:15:47** — ablation_cap20 STEP 3-4: all 6 runs done 13:09-13:15 (no crash, no epoch > 300 s, max 14.2 s; runs 81-158 s wall vs 28-45 s originally, cause not investigated). Every cap-20 run matches its cap-60 run exactly (same val curve, restored epoch, source-val MSE and target-test MAE/RMSE; diff 0.00000). Pre-registered reading: the cap did not matter, for (c) and (d) on all seeds. (c) cap 20 vs (a): MAE d=-0.0017, RMSE d=+0.0002, both 'within both'; criterion not met. Not separated: an old-style 17-joint-epoch cap with step matching (seed 2) was not run. Write-up results_clean/novelty2/ablation_cap20.md. Stopped here per task.
