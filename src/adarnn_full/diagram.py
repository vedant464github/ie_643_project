"""
Architecture diagram of the final GRU + faithful AdaRNN model, drawn from the instantiated module (layer types,
sizes and parameter counts are read from the code, not typed in). Writes results_clean/novelty/architecture.png.

  python -m src.adarnn_full.diagram
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.adarnn_full.adarnn import PAIR_NAMES
from src.adarnn_full.model import build_gru
from src.models.backbone import count_params
from src.run_experiments import DOMAIN_CONFIGS

OUT = Path("results_clean/novelty/architecture.png")
INK, GREY, BLUE, ORANGE, GREEN, PURPLE = "#111111", "#6f6e69", "#0072B2", "#E69F00", "#009E73", "#CC79A7"


def box(ax, x, y, w, h, text, color, fs=12, fc=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", lw=1.8,
                                edgecolor=color, facecolor=fc or "white"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK)


def arrow(ax, a, b, color=INK, ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=16, lw=1.6, color=color, linestyle=ls))


def main():
    cfg = {d: (DOMAIN_CONFIGS[d]["input_len"], DOMAIN_CONFIGS[d]["horizon"]) for d in ("climate", "energy")}
    m = build_gru(*cfg["climate"])
    gru, head = m.gru, m.head
    n_gru = sum(p.numel() for p in gru.parameters())
    n_head = {d: build_gru(*cfg[d]).head.weight.numel() + build_gru(*cfg[d]).head.bias.numel() for d in cfg}
    total = {d: count_params(build_gru(*cfg[d])) for d in cfg}

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16); ax.set_ylim(0, 9); ax.axis("off")
    ax.text(0.2, 8.6, "GRU + faithful AdaRNN (Du et al. 2021): shared encoder, per-step temporal distribution matching",
            fontsize=16, color=INK, weight="bold")

    # four input streams sharing the encoder
    streams = [("P1: source period 1", BLUE), ("P2: source period 2", BLUE), ("P3: source period 3", BLUE),
               ("target-train inputs (no labels)", ORANGE)]
    for i, (lab, c) in enumerate(streams):
        y = 6.9 - i * 1.25
        box(ax, 0.2, y, 3.0, 0.9, lab + f"\nx: B × V,  V = {cfg['climate'][0]} / {cfg['energy'][0]}", c, fs=10)
        arrow(ax, (3.2, y + 0.45), (3.8, 4.9 + 0.2 * (1.5 - i)), c)
    ax.text(0.2, 2.8, "TDC periods:\nexperiments/adarnn/tdc.json\n(K=3 split of source-train\nwindows, inputs only)",
            fontsize=10, color=GREY, va="top")

    # encoder
    box(ax, 3.8, 3.9, 2.9, 2.2,
        f"{type(gru).__name__} (shared weights)\ninput size {gru.input_size}, hidden {gru.hidden_size}\n"
        f"{gru.num_layers} layers\n{n_gru:,} params", INK, fs=11, fc="#f4f4f2")
    arrow(ax, (6.7, 5.0), (7.3, 5.0))
    box(ax, 7.3, 4.25, 2.7, 1.5, f"last-layer hidden states\nh¹ … h^V\n(B × V × {gru.hidden_size})", INK, fs=10.5)

    # prediction head
    arrow(ax, (8.65, 5.75), (8.65, 6.6))
    box(ax, 7.3, 6.6, 2.7, 1.2, f"{type(head).__name__} on h^V\n{head.in_features} → H (5 / 24)\n"
        f"{n_head['climate']} / {n_head['energy']} params", INK, fs=10.5)
    arrow(ax, (10.0, 7.2), (10.6, 7.2))
    box(ax, 10.6, 6.6, 5.1, 1.2, "ŷ (B × H)\nL_pred = MSE on P1, P2, P3 (Eq. 2; source labels only)", GREEN, fs=10)

    # TDM
    arrow(ax, (10.0, 5.0), (10.6, 5.0), PURPLE)
    box(ax, 10.6, 3.7, 5.1, 2.6,
        "TDM (Eq. 3, 5), at every time step t = 1..V:\n"
        "d_ij^t = CORAL(h_i^t, h_j^t)\n"
        f"         = ‖C_i^t − C_j^t‖²_F / (4·{gru.hidden_size}²)\n"
        f"pairs: {', '.join(PAIR_NAMES)}\n"
        "L_tdm = mean over the 6 pairs of Σ_t α_ij^t d_ij^t", PURPLE, fs=10)
    box(ax, 10.6, 0.6, 5.1, 2.5,
        "Boosting importance α_ij ∈ R^V (Eq. 6, 7):\ninit α = 1/V; after each joint epoch n ≥ 2,\n"
        "where d^t(n) ≥ d^t(n−1):\n  α^t ← α^t · (1 + σ(d^t(n) − d^t(n−1)))\nthen α ← α / Σ_t α^t", PURPLE, fs=10)
    arrow(ax, (13.15, 3.1), (13.15, 3.7), PURPLE, ls="--")

    box(ax, 3.8, 0.6, 6.2, 2.9,
        "Training (Algorithm 1):\n"
        "3 pretrain epochs with λ = 0 (step 2), then\n"
        "L = L_pred + λ · L_tdm, one batch each of\n"
        "P1, P2, P3 and target per step;\n"
        "early stopping on source val MSE (patience 3).\n"
        f"Total params: climate {total['climate']:,}, energy {total['energy']:,} (cap 50k).\n"
        "Inference: target inputs, target-train stats → ŷ", GREY, fs=10)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
