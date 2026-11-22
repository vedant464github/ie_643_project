# Model architecture diagrams

`IE643_Model_Architectures.pptx` is a 16:9 deck with 8 slides. It is separate from `slides/IE643_Project_Slides*.pptx`
and contains only architecture diagrams.

- Every element is a native, editable PowerPoint shape (rectangle, rounded rectangle, oval, connector or text box) in
  Calibri. There are no images.
- `png/slide1.png` … `png/slide8.png` are 200-dpi renders (2666 × 1500 px) of the same specification. LibreOffice is
  not installed, so they are drawn with matplotlib rather than exported from PowerPoint.
- Verification results are in [CHECKS.md](CHECKS.md).

Notation on every slide:

- **D** = daily setting (finance, finance_vol, climate): L = 30, H = 5.
- **E** = energy setting: L = 168, H = 24.
- **B** = batch size: 64 daily, 128 energy.

Parameter counts are written as D / E.

| Slide | Diagram | Drawn from (file · function) | Shapes shown | Parameter counts shown |
|---|---|---|---|---|
| 1 | Four backbones at block level, get_features() tap on each, legend, 50k-parameter cap | `src/models/backbone.py` · `build_model`, `GRUForecaster`, `DLinear`, `TCNForecaster`, `PatchTSTForecaster`, `count_params` | x D (B, 30) · E (B, 168); features (B, 32) (DLinear 2×(B, H) → cat (B, 2H)); ŷ D (B, 5) · E (B, 24) | GRU 9,861 / 10,488 (encoder 9,696, head 165 / 792); DLinear 310 / 8,112; TCN 34,501 / 35,128 (encoder 34,336, head 165 / 792); PatchTST 17,797 / 19,896 (encoder 17,632 / 19,104, head 165 / 792) |
| 2 | GRU: input reshape, 2 GRU layers, final hidden state, head, plus an unrolled view | `src/models/backbone.py` · `GRUForecaster.__init__`, `forward`, `get_features` | (B, L) → unsqueeze(-1) → D (B, 30, 1) · E (B, 168, 1); sequence D (B, 30, 32) · E (B, 168, 32); h_n (2, B, 32); h_n[-1] (B, 32); ŷ D (B, 5) · E (B, 24) | layer 1 3,360; layer 2 6,336; head 165 / 792; total 9,861 / 10,488 |
| 3 | DLinear: moving-average decomposition (kernel 25, replicate-pad 12 \| 12), trend and remainder branches, two linear maps, sum, get_features() | `src/models/backbone.py` · `DLinear._components`, `forward`, `get_features` | x / trend / remainder D (B, 30) · E (B, 168); padded D (B, 54) · E (B, 192); branch outputs D (B, 5) · E (B, 24); get_features D (B, 10) · E (B, 48) | trend 155 / 4,056; seasonal 155 / 4,056; total 310 / 8,112 |
| 4 | TCN (chosen backbone): 6 TemporalBlocks with dilations 1–32, zoom on one block (causal left-padding 2d, two Conv1d k = 3, ReLU, residual), last-step get_features(), head, receptive field | `src/models/backbone.py` · `TemporalBlock.__init__`, `forward`; `TCNForecaster.__init__`, `forward`, `get_features` | x (B, L) → unsqueeze(1) → D (B, 1, 30) · E (B, 1, 168); blocks D (B, 32, 30) · E (B, 32, 168); last step (B, 32); ŷ D (B, 5) · E (B, 24); receptive field 253 | block 1 3,296 (conv1 128, conv2 3,104, 1×1 residual 64); blocks 2–6 6,208 each; encoder 34,336; head 165 / 792; total 34,501 / 35,128 |
| 5 | PatchTST-style: end-pad by one stride, unfold (patch 6, stride 3), patch embedding, learned positions, 2 post-norm encoder layers (zoom), mean-pool get_features(), head, dashed RevIN option | `src/models/backbone.py` · `PatchTSTForecaster.__init__`, `_encode`, `forward`, `get_features`; `src/models/revin.py` · `RevIN` | D (B, 30) → (B, 33) → (B, 10, 6) → (B, 10, 32); E (B, 168) → (B, 171) → (B, 56, 6) → (B, 56, 32); FFN D (B, 10, 64) · E (B, 56, 64); pooled (B, 32); ŷ D (B, 5) · E (B, 24); N = 10 (D), 56 (E) | embed 224; positions 320 / 1,792; per layer 8,544 (attention 4,224, linear1 2,112, linear2 2,080, 2 × LayerNorm 64); head 165 / 792; total 17,797 / 19,896; RevIN +2 if on (off in all runs) |
| 6 | CORAL training: source and target-train streams through one shared TCN, C_S and C_T, loss, λ = 10 rule, "target labels never enter" | `src/adaptation/coral.py` · `fit_coral`, `coral_loss`; `src/run_coral.py` · `run_coral`, lambda rule; `experiments/coral/lambda_selection.json` | x_s, x_t D (B, 30) · E (B, 168); f_s, f_t (B, 32); C_S, C_T (32, 32); ŷ_s D (B, 5) · E (B, 24) | TCN encoder 34,336 (shared) |
| 7 | AdaRNN-style training: 3-epoch MSE-only pretrain then joint phase; P1–P3 plus target-train T through the shared TCN; mean of 6 pairwise CORAL terms; MSE on the P1–P3 labels; TDC split table; dashed "not in our version" box | `src/adaptation/adarnn.py` · `tdc_split`, `period_distance`, `fit_adarnn`; `src/run_adarnn.py` · `compute_tdc`, `LAMBDA`; `experiments/adarnn/tdc.json` | each group D (B, 30) · E (B, 168); features (B, 32) each; cat (3B, 32); ŷ D (3B, 5) · E (3B, 24); split points finance / finance_vol 60/80 %, climate 40/60 %, energy 20/80 % | TCN encoder 34,336 (shared) |
| 8 | Training setups behind the results rows: (a) source-only with source stats, (b) source-only with target-train stats, (c) CORAL, (d) AdaRNN-style, (e) target-only (red: uses target labels) | `src/experiment.py` · `prepare_pair`, `run_transfer`; `src/evaluate.py` · `evaluate_model`, `denorm`; `src/run_coral.py`, `src/run_adarnn.py` (evaluation with target-train stats) | x (B, L), ŷ (B, H), with D and E expanded in the subtitle | — |

## Rebuilding

The build scripts are in `_build/`. Run them from the repo root:

```
python architecture/_build/verify_models.py        # instantiate models, write verify.json and the log
python architecture/_build/verify_tdc_lambda.py    # recompute TDC splits and the lambda rule
python architecture/_build/build.py --all          # layout checks, PNGs, PPTX
python architecture/_build/write_checks.py         # CHECKS.md
```

`render.py` imports python-pptx from `slides/_lib` (read-only). The slide geometry and text are in
`_build/slides.py`, and `_build/spec.py` holds the shared primitives, the palette and the checks.
