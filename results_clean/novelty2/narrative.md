## What this shows

**The schedule explains the documented failure; the matching term adds nothing measurable; the pre-registered success criterion is not met.**

Step matching gives 47 steps per epoch instead of 15 and lifts the cap to 60 joint epochs. With that change, AdaRNN λ=10 falls from 1.792 to 1.676 ± 0.022 °C MAE, and the λ=0 control from 1.786 (seed 0 only) to 1.667 ± 0.016. The earlier +0.114 °C gap to the GRU source-only baseline (a) is gone. Both AdaRNN rows now beat persistence (1.733), and (c) is "within both" seed stds of (a) on MAE (−0.002) and on RMSE (+0.000). So (c) does not beat (a): it ties it. The criterion is not met under any reading of the noise rule.

The fix changed two things at once, the step count and the cap, so they are not separated cleanly. The restored epochs point mostly at the step count, though. They are 12, 14 and 23 counting the 3 pretrain epochs, which is 9, 11 and 20 joint epochs. Under the old 20-epoch cap that counted pretrain, seeds 0 and 1 would have stopped where they did here, and only seed 2 needed more. Each step-matched AdaRNN step still draws 64 windows from each of the 3 periods, so an AdaRNN epoch sees about 3× the source windows of a baseline epoch. The comparison is step-matched, not sample-matched.

**Does matching add anything beyond (d)? No.** (c) vs (d) is +0.009 °C MAE and +0.011 RMSE, "within both", with paired differences of +0.025, −0.000 and +0.001. At the restored epoch, λ·L_tdm is 0.45–0.60% of the training MSE. The validation curves of (c) and (d) lie almost on top of each other. Whatever AdaRNN gains here comes from period-balanced batches and more training, not from the distribution matching.

(d) shows lower RMSE than (a): −0.011, "exceeds the target-stats std only", driven mostly by seed 2. That falls short of "exceeds both stds", and (d) is a control, not a pre-registered method, so it is not claimed as a gain. CORAL (b) is again indistinguishable from (a).

Rows (a) and (b) are identical to the 20-epoch reference rows. Every one of those runs either early-stopped before epoch 20 or, for source-only seed 0 (best epoch 19), restores the same epoch-19 weights after stopping at 22. Training is deterministic per seed, so the cap change does not alter them.

**Not run:** nothing. All 4 methods × seeds 0–2 finished at 03:48 on 2026-11-20. No crash, no epoch over 300 s and no sleep event occurred, and no setting was changed. The runs are much faster than the earlier ones (about 1–2 s per epoch).
