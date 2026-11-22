# Ablation: AdaRNN step-matched at joint-epoch cap 20 (climate Shimla → Delhi, GRU)

Pre-registered in CLAUDE.md Amendments 2026-11-21 (k). Same code path as `experiments/novelty_fix` (`src.adarnn_fix.run.run()` called unchanged by `src/adarnn_fix/ablation_cap20.py`, which only sets the output dir, the joint cap 20 and the hard stop). Target test set (610 windows, the same as results_clean/canonical.csv), target-train-stat normalisation, MAE/RMSE in °C via `src.evaluate.evaluate_model`. Std is the sample std (ddof=1) over seeds 0, 1, 2. Target labels, target val and target test were not used in training or selection.

## (i) Epoch counting

`src/adarnn_full/adarnn.py::fit_adarnn_full` uses a single counter `for epoch in range(1, epochs + 1)`. Epochs 1–3 are the MSE-only pretrain phase and later epochs are joint. `best_epoch` and `stopped_epoch` come from that counter, so **the best/stopped epochs reported for (c) and (d) in STATUS.md and summary.md include the 3 pretrain epochs**. Early stopping is checked only from the first joint epoch. `run.py` passes `epochs = 3 + JOINT_CAP`, so the cap counts joint epochs only. Every JSON's `training.phase` list shows 3 `pretrain` entries, then the joint entries. (a) has no pretrain, so its epochs are joint epochs. Note: the JSON field `joint_epochs_run` is stopped − 3, the number of joint epochs trained, not the best joint epoch.

| run | best (total) | stopped (total) | best (joint) | stopped (joint) | joint cap |
|---|---|---|---|---|---|
| (c) AdaRNN step-matched λ=10, cap 60 seed 0 | 12 | 15 | 9 | 12 | 60 |
| (c) AdaRNN step-matched λ=10, cap 60 seed 1 | 14 | 17 | 11 | 14 | 60 |
| (c) AdaRNN step-matched λ=10, cap 60 seed 2 | 23 | 26 | 20 | 23 | 60 |
| (c) AdaRNN step-matched λ=10, cap 20 seed 0 | 12 | 15 | 9 | 12 | 20 |
| (c) AdaRNN step-matched λ=10, cap 20 seed 1 | 14 | 17 | 11 | 14 | 20 |
| (c) AdaRNN step-matched λ=10, cap 20 seed 2 | 23 | 23 | 20 | 20 | 20 |
| (d) AdaRNN step-matched λ=0, cap 60 seed 0 | 16 | 19 | 13 | 16 | 60 |
| (d) AdaRNN step-matched λ=0, cap 60 seed 1 | 14 | 17 | 11 | 14 | 60 |
| (d) AdaRNN step-matched λ=0, cap 60 seed 2 | 23 | 26 | 20 | 23 | 60 |
| (d) AdaRNN step-matched λ=0, cap 20 seed 0 | 16 | 19 | 13 | 16 | 20 |
| (d) AdaRNN step-matched λ=0, cap 20 seed 1 | 14 | 17 | 11 | 14 | 20 |
| (d) AdaRNN step-matched λ=0, cap 20 seed 2 | 23 | 23 | 20 | 20 | 20 |

## (ii) Target test, °C

| row | n | MAE s0; s1; s2 | MAE mean ± std | RMSE s0; s1; s2 | RMSE mean ± std |
|---|---|---|---|---|---|
| baseline: naive train-mean (target-train stats) | – | | 6.137 | | 6.964 |
| baseline: persistence (last observed value repeated) | – | | 1.733 | | 2.350 |
| (a) GRU source-only, target-train stats | 3 | 1.6672; 1.6937; 1.6717 | 1.6776 ± 0.0142 | 2.2037; 2.2191; 2.2093 | 2.2107 ± 0.0078 |
| (c) AdaRNN step-matched λ=10, cap 60 | 3 | 1.6949; 1.6810; 1.6515 | 1.6758 ± 0.0222 | 2.2336; 2.2176; 2.1816 | 2.2109 ± 0.0266 |
| (c) AdaRNN step-matched λ=10, cap 20 | 3 | 1.6949; 1.6810; 1.6515 | 1.6758 ± 0.0222 | 2.2336; 2.2176; 2.1816 | 2.2109 ± 0.0266 |
| (d) AdaRNN step-matched λ=0, cap 60 | 3 | 1.6695; 1.6813; 1.6502 | 1.6670 ± 0.0157 | 2.2027; 2.2177; 2.1784 | 2.1996 ± 0.0198 |
| (d) AdaRNN step-matched λ=0, cap 20 | 3 | 1.6695; 1.6813; 1.6502 | 1.6670 ± 0.0157 | 2.2027; 2.2177; 2.1784 | 2.1996 ± 0.0198 |

## (iii) Paired per-seed differences, cap 20 − cap 60 (same method, same seed)

| method | metric | seed 0 | seed 1 | seed 2 | max abs | identical weights? |
|---|---|---|---|---|---|---|
| adarnn10 | MAE | +0.00000 | +0.00000 | +0.00000 | 0.00000 | yes; yes; yes |
| adarnn10 | RMSE | +0.00000 | +0.00000 | +0.00000 | 0.00000 | yes; yes; yes |
| adarnn0 | MAE | +0.00000 | +0.00000 | +0.00000 | 0.00000 | yes; yes; yes |
| adarnn0 | RMSE | +0.00000 | +0.00000 | +0.00000 | 0.00000 | yes; yes; yes |

'Identical weights' = the per-epoch source-val MSE curve matches the cap-60 run's curve over the same epochs, and the restored epoch and final source-val MSE are the same (exact float equality).

Pre-registered interpretation: the cap did not matter if every seed's cap-20 MAE is within 0.005 °C of its cap-60 MAE.

- adarnn10: all three seeds within 0.005 °C MAE → **the cap did not matter**.
- adarnn0: all three seeds within 0.005 °C MAE → **the cap did not matter**.

## (iv) Noise rule (results_clean/noise_rule.md): (c) cap 20 vs (a)

| metric | mean(M) | mean(R) | d | paired diffs (s0; s1; s2) | s_R | s_M | direction | verdict |
|---|---|---|---|---|---|---|---|---|
| MAE | 1.6758 | 1.6776 | -0.0017 | +0.0277; -0.0127; -0.0202 | 0.0142 | 0.0222 | lower error | within both |
| RMSE | 2.2109 | 2.2107 | +0.0002 | +0.0298; -0.0016; -0.0277 | 0.0078 | 0.0266 | higher error | within both |

Counted as a gain only if (c) has lower error and the verdict is 'exceeds both stds' (STATUS.md 2026-11-20 03:46:51).

## (v) Schedule or cap?

At this cap the higher cap explains none of the change. All six cap-20 runs exactly match their cap-60 runs in every way logged: the same validation curve over the shared epochs, the same restored epoch, the same source-val MSE, and target-test MAE/RMSE differences of 0.00000. So the pre-registered rule reads "the cap did not matter" for both (c) and (d) on every seed. Seeds 0 and 1 stopped early (joint epochs 12–16) before either cap. Seed 2 is the only case where the caps could differ. Its best epoch, joint epoch 20, is the last epoch the cap-20 run is allowed, and the cap-60 run trained 3 more joint epochs without improving and restored the same epoch-20 weights. So the earlier gap (faithful AdaRNN 1.792 °C MAE vs 1.676 now) comes from the step-matched schedule, not from going from a 20- to a 60-joint-epoch cap. Two limits remain. First, the old 20-epoch protocol counted the 3 pretrain epochs in its cap (CLAUDE.md 2026-11-15 (h)), so it allowed only 17 joint epochs. This ablation tested a joint cap of 20, not the old 17. Seeds 0 and 1 stopped before joint epoch 17 in both methods, so a 17-joint cap would not have affected them. Seed 2's best epoch (joint 20) is past 17, so a 17-joint cap with step matching was not run, and its effect on seed 2 is unknown. Second, "step-matched" changes three things together: optimizer steps per epoch (47 vs 15), source windows seen per epoch (about 3× a baseline epoch), and therefore total updates before early stopping. This ablation cannot separate these from each other. It does not show that period balancing or the matching term helps; the old protocol also used period-balanced batches, and (c) vs (d) stays "within both" seed stds. Under the noise rule, (c) at cap 20 vs (a) is "within both" on MAE (d = −0.0017) and RMSE (d = +0.0002), the same as at cap 60, so the pre-registered success criterion is still not met. Nothing was left unrun: all 6 runs finished by 13:15 on 2026-11-21 with no crash, no epoch over 300 s (max 14.2 s) and no setting changed. Wall-clock per run was 81–158 s vs 28–45 s for the original runs; the cause was not investigated, and it does not affect the numbers, which are bit-identical. This is climate (Shimla → Delhi) only, GRU only, with three seeds.
