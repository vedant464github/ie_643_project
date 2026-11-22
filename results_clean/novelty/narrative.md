## What was tried and what was found

**Tried.** A faithful AdaRNN (Du et al. 2021, Sec. 4.2, Eqs. 3, 5, 6, 7, Algorithm 1) on the project's GRU
(2 layers, hidden 32; 9,861 params, 10,488 for energy): TDC periods reused from `experiments/adarnn/tdc.json`,
CORAL distance between the last-layer hidden states at **every** input time step for all 6 pairs of
{P1, P2, P3, target-train inputs}, per-pair importance weights α ∈ R^V updated by the paper's boosting rule after
each joint epoch, 3 MSE-only pretrain epochs, early stopping on source validation. It was compared against the same
GRU trained source-only (source stats and target-train stats) and with plain CORAL on the final hidden state, with λ
chosen per domain and method by the TCN rule on seed-0 source validation (λ=10 everywhere). All settings were fixed
in CLAUDE.md 2026-11-15 (h) before any run, and nothing was changed after seeing results.

**Found.** On **climate** the faithful AdaRNN is *worse* than the plain GRU with target-train statistics:
1.792 ± 0.006 vs 1.678 ± 0.014 °C MAE (+0.114 °C, "exceeds both stds", worse on every seed), and also worse than
persistence (1.733 °C). This deficit is already there at λ=0 (seed-0 grid run 1.786 °C), so it comes from the
AdaRNN training schedule rather than from the distribution matching. Period-balanced steps give 15 optimiser steps
per epoch instead of 47, and every climate AdaRNN run stopped at the 20-epoch cap while still improving (source-val
MSE 0.077 vs 0.071 for source-only). The matching itself barely acts. The per-step CORAL term is 1e-4 to 2e-4, so
λ·L_tdm ≈ 1e-3 against an MSE of about 0.09, and changing λ from 0 to 10 moves source-val MSE by 0.3%. The boosting
weights do move away from uniform (largest α 0.08 to 0.10 vs 1/V = 0.033 at the restored epoch). Plain GRU + CORAL
is indistinguishable from source-only on climate (+0.004 °C MAE, within the target-stats std). On **finance** every
GRU row sits at 99.9 to 100.1% of the naive train-mean MAE, the same negative-control picture as the TCN. CORAL
reproduces source-only exactly, and AdaRNN differs by one rounding step (1e-5); the "exceeds both stds" label there
only reflects zero seed spread at 5-decimal rounding. On **finance_vol** AdaRNN has lower target-test error than the
GRU baseline (MAE 0.005073 vs 0.005433, RMSE 0.006327 vs 0.006553, both "exceeds both stds"). This should not be
claimed as a gain: finance_vol is a negative control by its source-validation result, and AdaRNN's own source-val MSE
(0.823 to 0.827) is worse than both source-only (0.804 to 0.811) and the train-mean (0.805). So the selection split
gives no reason to prefer it, and its MAE is still above the mean-of-last-5 baseline (0.00497). On **energy** only one
AdaRNN seed finished (seed 2: 700.51 MW MAE, 953.23 MW RMSE, early-stopped at epoch 11). There is no same-backbone
baseline to compare it with, so no energy conclusion is drawn; the TCN rows (about 555 to 562 MW) are shown only as a
reference. **Net:** in this project's setting the faithful AdaRNN adds no measurable benefit over the same GRU with
target-train normalisation, and its period-balanced schedule costs accuracy on climate under the shared epoch cap.

## What was not run, and why

- **Energy, everything except AdaRNN seed 2.** The pre-registered rule "if an energy epoch is over 300 s, stop
  energy" fired at 23:09 on 2026-11-15. AdaRNN seeds 0 and 1 logged 805 and 806 s for epoch 14, and source-only seed
  2 logged 812 s for epoch 2. The earlier epochs had taken about 40 s. The Windows System log (Kernel-Power events 506
  and 507) shows the laptop was in Modern Standby from 22:56:03 to 23:08:54 (771 s). The over-limit epochs therefore
  measure sleep time, not training speed. The rule was applied as written; no override was made. AdaRNN seeds 0 and 1
  were still improving at the stop (source-val MSE 0.111) and saved nothing.
- **Energy protocol.** The probe chose the full protocol (first epochs 48.8 / 43.0 / 43.9 s for 3 parallel
  processes, limit 150 s), so the cheap-protocol switch was never needed.
- **Orchestration bug (fixed, nothing rerun).** After the stop, `run.py`'s refusal to start further energy jobs
  exited with code 1, and the queue driver counted that as a crash. The finance and finance_vol finals queued behind
  it therefore did not start. The refusal now exits 0. Those finals had not started, so they were run once afterwards
  with unchanged settings (logged in experiments/STATUS.md).
- Nothing else is missing. Climate, finance and finance_vol have all four GRU methods for seeds 0 to 2. Seed 0 of
  CORAL and AdaRNN is the seed-0 λ-grid run at the chosen λ (same protocol), as for the TCN CORAL runs.
