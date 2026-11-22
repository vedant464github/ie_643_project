# CHECKS: verification of the architecture deck

All checks were run with the project venv on CPU, from the repo root:

- `python architecture/_build/verify_models.py`
- `python architecture/_build/verify_tdc_lambda.py`
- `python architecture/_build/build.py --all`

No training was run. Every model was built with `build_model(name, L, H)` and no keyword arguments, which is what
every run used: `model_kwargs = {}` in `experiments/bakeoff/runs.json` and in the final, CORAL and AdaRNN-style
checkpoints and results.

## 1. Backbones: forward pass on a random (B, L) tensor, B = 4

| Backbone | output D / E | get_features() D / E | computed params D / E | expected D / E | match |
|---|---|---|---|---|---|
| GRU | (4, 5) / (4, 24) | (4, 32) / (4, 32) | 9,861 / 10,488 | 9,861 / 10,488 | yes |
| DLinear | (4, 5) / (4, 24) | (4, 10) / (4, 48) | 310 / 8,112 | 310 / 8,112 | yes |
| TCN | (4, 5) / (4, 24) | (4, 32) / (4, 32) | 34,501 / 35,128 | 34,501 / 35,128 | yes |
| PatchTST | (4, 5) / (4, 24) | (4, 32) / (4, 32) | 17,797 / 19,896 | 17,797 / 19,896 | yes |

All four totals match the expected counts, so no count on the slides needed a flag. For GRU, TCN and PatchTST,
`head(get_features(x))` equals `forward(x)` (checked with `torch.allclose`). DLinear has no `head`.

Full output, with the parameter count per module:

```
gru       daily  x(4, 30) -> y(4, 5)  get_features(4, 32)  params 9,861 (expected 9,861: OK)  head(feat)==fwd: True
      gru                             9,696
      head                              165
gru       energy x(4, 168) -> y(4, 24)  get_features(4, 32)  params 10,488 (expected 10,488: OK)  head(feat)==fwd: True
      gru                             9,696
      head                              792
dlinear   daily  x(4, 30) -> y(4, 5)  get_features(4, 10)  params 310 (expected 310: OK)  head(feat)==fwd: None
      seasonal                          155
      trend                             155
dlinear   energy x(4, 168) -> y(4, 24)  get_features(4, 48)  params 8,112 (expected 8,112: OK)  head(feat)==fwd: None
      seasonal                        4,056
      trend                           4,056
tcn       daily  x(4, 30) -> y(4, 5)  get_features(4, 32)  params 34,501 (expected 34,501: OK)  head(feat)==fwd: True
      tcn                            34,336
      head                              165
      tcn.0                           3,296
        tcn.0.conv1                     128
        tcn.0.conv2                   3,104
        tcn.0.residual                   64
      tcn.1                           6,208
        tcn.1.conv1                   3,104
        tcn.1.conv2                   3,104
        tcn.1.residual                    0
      tcn.2                           6,208
        tcn.2.conv1                   3,104
        tcn.2.conv2                   3,104
        tcn.2.residual                    0
      tcn.3                           6,208
        tcn.3.conv1                   3,104
        tcn.3.conv2                   3,104
        tcn.3.residual                    0
      tcn.4                           6,208
        tcn.4.conv1                   3,104
        tcn.4.conv2                   3,104
        tcn.4.residual                    0
      tcn.5                           6,208
        tcn.5.conv1                   3,104
        tcn.5.conv2                   3,104
        tcn.5.residual                    0
tcn       energy x(4, 168) -> y(4, 24)  get_features(4, 32)  params 35,128 (expected 35,128: OK)  head(feat)==fwd: True
      tcn                            34,336
      head                              792
      tcn.0                           3,296
        tcn.0.conv1                     128
        tcn.0.conv2                   3,104
        tcn.0.residual                   64
      tcn.1                           6,208
        tcn.1.conv1                   3,104
        tcn.1.conv2                   3,104
        tcn.1.residual                    0
      tcn.2                           6,208
        tcn.2.conv1                   3,104
        tcn.2.conv2                   3,104
        tcn.2.residual                    0
      tcn.3                           6,208
        tcn.3.conv1                   3,104
        tcn.3.conv2                   3,104
        tcn.3.residual                    0
      tcn.4                           6,208
        tcn.4.conv1                   3,104
        tcn.4.conv2                   3,104
        tcn.4.residual                    0
      tcn.5                           6,208
        tcn.5.conv1                   3,104
        tcn.5.conv2                   3,104
        tcn.5.residual                    0
patchtst  daily  x(4, 30) -> y(4, 5)  get_features(4, 32)  params 17,797 (expected 17,797: OK)  head(feat)==fwd: True
      embed                             224
      encoder                        17,088
      head                              165
      pos (nn.Parameter)                320
      encoder.layers.0                8,544
        self_attn                     4,224
        linear1                       2,112
        linear2                       2,080
        norm1+norm2                     128
patchtst  energy x(4, 168) -> y(4, 24)  get_features(4, 32)  params 19,896 (expected 19,896: OK)  head(feat)==fwd: True
      embed                             224
      encoder                        17,088
      head                              792
      pos (nn.Parameter)              1,792
      encoder.layers.0                8,544
        self_attn                     4,224
        linear1                       2,112
        linear2                       2,080
        norm1+norm2                     128
TCN receptive field: formula 1 + sum_blocks 2*(k-1)*d = 253; empirical (nonzero grad span) = 253
```

Further details read from the instantiated models:

- GRU parameters by layer, from the `nn.GRU` parameter names: layer 1 = 3,360, layer 2 =
  6,336.
- PatchTST padded length: 33 (D) / 171 (E).
- PatchTST patch tensor (B = 4): (4, 10, 6) (D) / (4, 56, 6) (E).
- PatchTST encoder layer: `norm_first = False` (post-norm), ReLU activation, dropout
  0.0, RevIN off.

## 2. TCN receptive field

Each block has two convolutions, and each convolution adds (k − 1)·d steps. From the code, the receptive field is
1 + Σ 2·(3 − 1)·d over d = 1, 2, 4, 8, 16, 32, which gives **253**.

Empirical check: on a 400-step input, the non-zero gradients of the last-step features span **253**
input steps.

The slides and the `TCNForecaster` docstring both say 253: confirmed.

## 3. TDC split points and λ

TDC was recomputed from the source-train input windows with `src.adaptation.adarnn.tdc_split`. This loads and windows
the data only; nothing is trained. λ was recomputed from the table in `experiments/coral/lambda_selection.json`:

```
TDC finance: recomputed [0.6, 0.8] idx [1196, 1594] of 1993 | json [0.6, 0.8] idx [1196, 1594] | MATCH
TDC finance_vol: recomputed [0.6, 0.8] idx [1196, 1594] of 1993 | json [0.6, 0.8] idx [1196, 1594] | MATCH
TDC climate: recomputed [0.4, 0.6] idx [1188, 1781] of 2969 | json [0.4, 0.6] idx [1188, 1781] | MATCH
TDC energy: recomputed [0.2, 0.8] idx [9272, 37089] of 46361 | json [0.2, 0.8] idx [9272, 37089] | MATCH
lambda rule recomputed: qualifying [1.0, 10.0], chosen 10.0 | json chosen 10.0 | run_adarnn.LAMBDA = 10.0
energy lambda=0.1 ratio 1.0500075517966792
```

- All four TDC splits match `experiments/adarnn/tdc.json`.
- The λ rule gives qualifying λ ∈ {1, 10} and chosen λ = 10. This matches `lambda_selection.json` and
  `LAMBDA = 10.0` in `src/run_adarnn.py`.
- λ = 0.1 fails only on energy, and only just (ratio 1.0500076 > 1.05).

## 4. Slide layout: text fit and overlaps

LibreOffice (`soffice`) is not installed, so the deck was not converted to PDF. Instead, the deck and the PNGs are
drawn from the same specification (`architecture/_build/slides.py`), and every slide was checked as follows:

- **Text fit.** Every text line was measured with the Calibri TTF metrics (`C:/Windows/Fonts/calibri.ttf` and
  `calibrib.ttf`) at its point size. The measurement adds the box's internal margins (0.07 in left and right, 0.03 in
  top and bottom) and uses a line height of 1.2 × the font size. Lines are set explicitly, so nothing wraps
  automatically, and free labels have word-wrap turned off in the PPTX.
- **Overlaps.** Every pair of shapes and text boxes was checked. Container frames are allowed only to contain shapes
  completely.
- **Arrows.** No arrow segment may pass through any shape or label other than its endpoints, and no two arrows may
  cross.
- **Safe area.** Every element stays at least 0.3 in from the left and right edges and 0.25 in from the top and
  bottom.
- **Font size.** Every label is at least 12 pt.

| Slide | Title | shapes + text boxes | arrows | issues | tightest width fit |
|---|---|---|---|---|---|
| 1 | Four backbones at block level | 71 | 12 (12 segments) | 0 | label: D (B, 5): 0.04 in spare |
| 2 | GRU | 44 | 18 (18 segments) | 0 | label: (B, 1): 0.03 in spare |
| 3 | DLinear | 25 | 9 (12 segments) | 0 | Subtitle: 0.05 in spare |
| 4 | TCN (chosen backbone) | 45 | 18 (20 segments) | 0 | Subtitle: 0.05 in spare |
| 5 | PatchTST-style encoder | 57 | 17 (21 segments) | 0 | label: skip: 0.03 in spare |
| 6 | CORAL training (shared TCN) | 28 | 12 (12 segments) | 0 | label: D (B, 5): 0.04 in spare |
| 7 | AdaRNN-style training (simplified) | 36 | 16 (16 segments) | 0 | label: scalar: 0.04 in spare |
| 8 | Training setups behind the results rows | 51 | 15 (15 segments) | 0 | label: fit: (B, L): 0.04 in spare |

Results:

- **0 layout issues** on all 8 slides.
- I inspected all 8 PNGs and found no visible overflow or overlap.
- `scripts/office/validate.py` from the pptx skill reported "All validations PASSED!".
- The deck contains no pictures. Every element is a native rectangle, rounded rectangle, oval, connector or text box.

Caveats:

- The PNGs are matplotlib renders of the same specification, not PowerPoint exports. Rounded corners and arrowheads
  are drawn slightly differently from PowerPoint.
- The tightest labels have only 0.03–0.05 in of spare width, so the fit estimate depends on the Calibri metrics
  matching PowerPoint's. Free labels do not wrap, so a small metric difference would push text a few points past the
  edge of an invisible text box rather than break a line.

## 5. Conventions in the deck that depart from the brief

- **Slide 8:** the arrows carry symbolic shapes: `fit: (B, L)`, `test: (B, L)` and `ŷ (B, H)`. The two settings are
  expanded once in the subtitle (x: D (B, 30) · E (B, 168); ŷ: D (B, 5) · E (B, 24)), because each of the five rows has
  three arrows.
- **Slide 5:** the two residual connections in the encoder layer are labelled only "skip". Each carries the same tensor
  as the main-path arrow beside it, D (B,10,32) · E (B,56,32).
- **Slide 4:** the residual path's input shape is written inside the residual box (block 1: 1 → 32 channels). Its
  output shape is labelled next to its last segment.
- **Slide 2, unrolled panel:** shapes are written once, as "(B, 32)" and "(B, 1)", because they are the same for D and
  E. A caption says so.
- **Slide 7:** the arrow between the two phase boxes marks a point in time ("after epoch 3"), not a tensor.
- **Titles** are 28 pt, smaller than usual for slide titles, to leave room for the diagrams. All other text is
  12–16 pt.

## 6. Discrepancies between the code and the main deck, the report or CLAUDE.md

Sources compared with the code:

- Main deck: `slides/IE643_Project_Slides_v2.pptx`, the newest version. `IE643_Project_Slides.pptx` makes the same
  architecture statements.
- Report: `report/overleaf/report.tex`.

The diagrams follow the code in every case.

1. **get_features() pooling.** CLAUDE.md, under Decisions, says "mean-pooled get_features()" for the backbones. In the
   code, only PatchTST mean-pools (`_encode(...).mean(dim=1)`). The others do not:
   - TCN takes the last time step, `self.tcn(x.unsqueeze(1))[:, :, -1]`, which gives (B, 32).
   - GRU returns the top layer's final hidden state, `h_n[-1]`, which gives (B, 32).
   - DLinear returns its two component forecasts concatenated, `cat(seasonal_out, trend_out)`, which gives (B, 2H):
     (B, 10) daily and (B, 48) energy.

   The report already discloses this for the TCN (report.tex ll. 306–307 and l. 570). The deck does not say how the
   features are pooled.
2. **CORAL and AdaRNN-style batches.** Deck v2 slide 11, in both the text box and the speaker notes, says "CORAL and
   AdaRNN-style mix both scalings in one batch". In the code they are not in one batch:
   - In `fit_coral` and `fit_adarnn`, source batches (source stats) and the target batch (target-train stats) go
     through separate `get_features()` calls and give separate covariance matrices. They meet only in the loss of the
     same optimizer step.
   - In AdaRNN-style, only the three source-period feature batches are concatenated, and only for the head and the
     MSE.

   Suggested wording: "in one training step".
3. **CORAL batch size.** The report's CORAL objective (report.tex ll. 320–330) assumes a source batch and a target
   batch of the same size B. The code differs at the end of each epoch:
   - `fit_coral`'s source loader has no `drop_last`, so the last source batch of each epoch is smaller than B.
   - A final source batch of size 1 contributes only its MSE; `coral_loss` is skipped for it.
   - Target batches use `drop_last=True`.

   The AdaRNN-style period and target loaders all use `drop_last=True`, as the report implies.
4. **PatchTST details missing from the report and the deck.** These are omissions, not contradictions. The report's
   backbone list (report.tex l. 301) gives patch 6, stride 3, 2 layers, d_model 32, 4 heads and feed-forward width 64.
   It does not say that:
   - the input is replicate-padded at the end by one stride before unfolding, which is why N = 10 (daily) and 56
     (energy) rather than 9 and 55;
   - the positional encoding is a learned `nn.Parameter` (320 / 1,792 parameters, the only part besides the head that
     grows with L);
   - the encoder layer is post-norm (`norm_first=False`).
5. **DLinear branch names.** The deck (slide 7) says "trend + remainder linear maps", and the report (l. 297) says
   "trend and seasonal parts". In the code, the remainder branch is `self.seasonal`, and its input is `x - trend`. The
   diagram labels it "x − trend (remainder)" feeding `self.seasonal`. Only the names differ; the meaning is the same.
6. **Why DLinear was not used with CORAL.** The deck (slide 7) and the report (l. 503) say DLinear "has no learned
   feature layer". The code gives a more concrete reason: `fit_coral` and `fit_adarnn` call `model.head(...)`, and
   DLinear does not define `head`, so it cannot be used as is. DLinear does define `get_features()`, which returns the
   component forecasts (see item 1). This is not a contradiction; it records the reason in the code.

No discrepancy was found for:

- the parameter counts (deck slides 7 and 13; report ll. 295–302 and the table at ll. 486–489);
- the TCN: 6 blocks, 32 channels, kernel 3, dilations 1–32, receptive field 253;
- PatchTST: 2 layers, d_model 32, RevIN flag off;
- the CORAL loss ||C_S − C_T||²_F / (4d²), with unbiased covariance and d = 32;
- λ = 10 and the rule that chose it;
- TDC: the grid, the ≥ 20% period constraint and the split points (deck slide 9: 60/80, 40/60, 20/80; report l. 533);
- the 3 pretrain epochs and the joint epoch of ⌊n_src / 3B⌋ steps;
- the mean over the six pairwise CORAL terms;
- the normalisation used for each results row (deck slides 5, 11 and 12; report ll. 309–318).
