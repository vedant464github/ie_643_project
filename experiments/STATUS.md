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
