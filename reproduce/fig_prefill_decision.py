#!/usr/bin/env python3
"""Reproduce Figure ``fig:prefill_decision`` — three-panel prefill-decision plot.

Panels (shared x = normalized prefill position 0-100%, 21 k-shot steps):
  1. Fisher Discriminant   — last-layer ``norm_fisher`` separability per model,
                             harmful (solid, "X") vs benign-macro (dashed, "o")
                             over the surviving benign sub-sources; own
                             autoscaled non-negative y-axis (no 0.5 chance line).
  2. AUROC                 — 5-fold-CV LogReg AUROC with 95% CI band per model.
  3. Balanced Accuracy     — same, BAcc.
Panels 2-3 share y 0.47-1.0 with a 0.5 chance line.

The benign-macro averages only the benign sub-sources that survive the bestpipe
MIN_POS rule (PHTest dropped for Qwen3-8B & Olmo-3-7B-Think, kept for the other
two); benign_kept is read straight from the bestpipe JSON so the rule is honoured
exactly.

Inputs :
  artifacts/experiments/refusal_cliff/results/kshop_K20_user_content/norm_fisher_<model>.json
  artifacts/experiments/refusal_cliff/results/kshop_K20_user_content/bestpipe_pca100_c003.json
Output : figures/fig_prefill_decision.{pdf,png}

Original research script: scripts/neurips_final/plot_kshop_fisher_auroc_bacc_sidebyside.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from lrm_safety_deliberation import paths
from lrm_safety_deliberation.io import read_json
from lrm_safety_deliberation.models import PRIMARY
from lrm_safety_deliberation.plotting import MODEL_COLORS, MODEL_LABELS, register_fonts

MODELS = list(PRIMARY)
COLORS = MODEL_COLORS

POOL_MARKERS = {"harmful": "X", "benign": "o"}
POOL_LS = {"harmful": "-", "benign": (0, (4, 2))}
POOL_LABELS = {"harmful": "Harmful", "benign": "Benign"}

# Text scaled 1.5x (base was 15, 19, 18, 13, 12).
FONT_SIZE, TITLE_SIZE, LABEL_SIZE, LEGEND_SIZE, TICK_SIZE = 22.5, 28.5, 27, 19.5, 18
# Markers: 2x bigger (was 5) and a bit transparent (lines stay opaque).
MS, MFC_ALPHA, LINE_ALPHA = 10, 0.5, 0.85

# Benign sub-source -> norm_fisher pool-label mapping.
SUBMAP = {"benign_main": "benign_main_only", "phtest": "phtest", "orfuzz": "orfuzz"}

_FW, _FH = 2.8 * 5, 4.5
ORIG_BOX_W = ((0.998 - 0.045) * _FW) / (5 + 4 * 0.06)
ORIG_BOX_H = ((0.86 - 0.13) * _FH) / (2 + 1 * 0.10)
BOX_W, BOX_H = ORIG_BOX_W * 2.3, ORIG_BOX_H * 2.0


def _kshop_dir() -> Path:
    return paths.artifact("experiments", "refusal_cliff", "results",
                          "kshop_K20_user_content")


def benign_kept(data, model, K):
    """Surviving benign sub-sources (stable across k; assert uniqueness)."""
    sets = {tuple(data[model][str(k)]["benign_kept"]) for k in range(K)}
    assert len(sets) == 1, f"{model}: benign_kept varies across k: {sets}"
    return list(next(iter(sets)))


def fisher_curves(model, kept, K):
    pools = read_json(_kshop_dir() / f"norm_fisher_{model}.json")["pools"]
    harmful = np.array(pools["harmful"]["norm_fisher_lastlayer"])[:K]
    subs = [np.array(pools[SUBMAP[s]]["norm_fisher_lastlayer"])[:K] for s in kept]
    benign = np.nanmean(np.vstack(subs), axis=0)
    return harmful, benign


def main() -> None:
    register_fonts()
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["TeX Gyre Termes"],
        "font.size": FONT_SIZE,
        "axes.titlesize": TITLE_SIZE, "axes.labelsize": LABEL_SIZE,
        "legend.fontsize": LEGEND_SIZE, "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
    })

    data = read_json(_kshop_dir() / "bestpipe_pca100_c003.json")
    K = 21
    x = np.arange(K)

    WSPACE = 0.10
    LEFT_IN, RIGHT_IN, TOP_IN, BOTTOM_IN = 0.85, 0.18, 0.50, 0.85
    arw = 3 * BOX_W + 2 * WSPACE * BOX_W
    fw, fh = LEFT_IN + arw + RIGHT_IN, TOP_IN + BOX_H + BOTTOM_IN
    fig, axes = plt.subplots(1, 3, figsize=(fw, fh),
                             gridspec_kw={"wspace": WSPACE})
    fig.subplots_adjust(left=LEFT_IN / fw, right=1 - RIGHT_IN / fw,
                        top=1 - TOP_IN / fh, bottom=BOTTOM_IN / fh)

    # ---- Panel 1: Fisher ----
    ax = axes[0]
    fmax = 0.0
    for model in MODELS:
        kept = benign_kept(data, model, K)
        harmful, benign = fisher_curves(model, kept, K)
        fmax = max(fmax, np.nanmax(harmful), np.nanmax(benign))
        color = COLORS[model]
        lc = mpl.colors.to_rgba(color, LINE_ALPHA)
        mfc = mpl.colors.to_rgba(color, MFC_ALPHA)
        ax.plot(x, harmful, color=lc, marker=POOL_MARKERS["harmful"],
                linestyle=POOL_LS["harmful"], markersize=MS, linewidth=2.0,
                markerfacecolor=mfc, markeredgecolor=lc, markeredgewidth=0.05)
        ax.plot(x, benign, color=lc, marker=POOL_MARKERS["benign"],
                linestyle=POOL_LS["benign"], markersize=MS, linewidth=2.0,
                markerfacecolor=mfc, markeredgecolor=lc, markeredgewidth=0.05)
    ax.grid(True, which="major", linestyle=":", alpha=0.45)
    ax.grid(True, which="minor", linestyle=":", alpha=0.20)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_xticklabels(["0", "25", "50", "75", "100"])
    ax.tick_params(axis="both", length=3, pad=2)
    ax.tick_params(which="minor", length=1.5)
    ax.set_title("Fisher Discriminant", pad=6)
    ax.set_xlim(-0.25, K - 1 + 0.25)
    ytop = np.ceil(fmax / 0.02) * 0.02 + 0.02
    ax.set_ylim(0.0, ytop)
    ax.set_yticks(np.arange(0.0, ytop + 1e-9, 0.04))
    ax.set_yticks(np.arange(0.0, ytop + 1e-9, 0.02), minor=True)

    # ---- Panels 2 & 3: AUROC, BAcc ----
    metrics = [("auroc", "AUROC"), ("bacc", "Balanced Accuracy (BAcc)")]
    for ci, (mkey, title) in enumerate(metrics):
        ax = axes[ci + 1]
        for model in MODELS:
            d = data.get(model)
            if d is None:
                continue
            color = COLORS[model]
            for pool, marker in POOL_MARKERS.items():
                y = np.array([d[str(k)][pool][mkey] for k in range(K)])
                lo = np.array([d[str(k)][pool][mkey + "_lo"] for k in range(K)])
                hi = np.array([d[str(k)][pool][mkey + "_hi"] for k in range(K)])
                v = ~np.isnan(lo) & ~np.isnan(hi)
                if v.any():
                    ax.fill_between(x[v], lo[v], hi[v], color=color, alpha=0.12, linewidth=0)
                ax.plot(x, y, color=mpl.colors.to_rgba(color, LINE_ALPHA), marker=marker,
                        linestyle=POOL_LS[pool], markersize=MS, linewidth=2.0,
                        markerfacecolor=mpl.colors.to_rgba(color, MFC_ALPHA),
                        markeredgecolor=mpl.colors.to_rgba(color, LINE_ALPHA), markeredgewidth=0.05)
        ax.grid(True, which="major", linestyle=":", alpha=0.45)
        ax.grid(True, which="minor", linestyle=":", alpha=0.20)
        ax.axhline(0.5, color="0.35", linestyle=":", linewidth=1.3, alpha=0.9, zorder=1.5)
        ax.set_yticks(np.arange(0.4, 1.01, 0.1))
        ax.set_yticks(np.arange(0.4, 1.01, 0.025), minor=True)
        ax.set_xticks([0, 5, 10, 15, 20])
        ax.set_xticklabels(["0", "25", "50", "75", "100"])
        ax.tick_params(axis="both", length=3, pad=2)
        ax.tick_params(which="minor", length=1.5)
        ax.set_title(title, pad=6)
        ax.tick_params(labelleft=True)
        ax.set_xlim(-0.25, K - 1 + 0.25)
        ax.set_ylim(0.47, 1.0)

    mh = [Line2D([0], [0], color=COLORS[m], linewidth=2.6, label=MODEL_LABELS[m]) for m in MODELS]
    ph = [Line2D([0], [0], color="#4d4d4d", marker=POOL_MARKERS[p], linestyle=POOL_LS[p],
                 markersize=12, linewidth=1.6, markerfacecolor=mpl.colors.to_rgba("#4d4d4d", MFC_ALPHA),
                 markeredgecolor="#4d4d4d", markeredgewidth=0.05, label=POOL_LABELS[p]) for p in POOL_MARKERS]
    chance = [Line2D([0], [0], color="0.35", linestyle=":", linewidth=1.3, label="Random")]
    leg = axes[0].legend(handles=mh + ph + chance, loc="upper left", ncol=1, frameon=True,
                         handletextpad=0.5, handlelength=1.8, borderpad=0.5,
                         labelspacing=0.35, fontsize=15)
    leg.get_frame().set_alpha(0.9)
    leg.get_frame().set_edgecolor("#bbbbbb")
    leg.get_frame().set_linewidth(0.6)
    fig.supxlabel("Normalized Prefill Position (%)", fontsize=LABEL_SIZE, y=0.04)

    for ext in ("pdf", "png"):
        out = paths.figure_out(f"fig_prefill_decision.{ext}")
        fig.savefig(out, dpi=220, bbox_inches="tight", pad_inches=0.02)
        print("wrote", out)
    plt.close(fig)
    print(f"Fisher y-range: [0.0, {ytop:.3f}]")


if __name__ == "__main__":
    main()
