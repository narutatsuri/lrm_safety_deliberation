#!/usr/bin/env python3
"""Reproduce Figure ``fig:refusal_valley_supplementary`` — the appendix refusal
valley for the five supplementary reasoning models.

Identical layout / numeric logic to ``fig_refusal_valley.py`` (last-layer
``norm_fisher`` separability vs normalized thinking position, harmful row + benign
row, endpoint-extended fills, per-panel N annotation), but covering the
supplementary model set instead of the four headline models:
Qwen3-32B, Phi-4-Reasoning-Plus, Qwen3.5-9B, DeepSeek-R1-Distill-Llama-8B,
DeepSeek-R1-0528-Qwen3-8B.

Inputs : artifacts/eval_results/neurips_final/<model>/valley/explore_{harmful,benign_combined}.pt
Output : figures/fig_refusal_valley_supplementary.{pdf,png}

Original research script: scripts/plotting/plot_refusal_valley_supplementary.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter, MultipleLocator

from lrm_safety_deliberation import paths
from lrm_safety_deliberation.models import SUPPLEMENTARY
from lrm_safety_deliberation.plotting import SEMANTIC_COLORS, register_fonts

METRIC_KEY = "norm_fisher"

# Paper display names for the supplementary models (original hardcoded these).
SUPP_LABELS = {
    "Qwen3-32B": "Qwen3-32B",
    "Phi-4-reasoning-plus": "Phi-4-Reasoning-Plus",
    "Qwen3.5-9B": "Qwen3.5-9B",
    "DeepSeek-R1-Distill-Llama-8B": "DeepSeek-R1-Distill-Llama-8B",
    "DeepSeek-R1-0528-Qwen3-8B": "DeepSeek-R1-0528-Qwen3-8B",
}
# Original figure column order.
SUPP_ORDER = [
    "Qwen3-32B", "Phi-4-reasoning-plus", "Qwen3.5-9B",
    "DeepSeek-R1-Distill-Llama-8B", "DeepSeek-R1-0528-Qwen3-8B",
]

ROWS = [
    ("harmful", "Harmful", SEMANTIC_COLORS["harmful"], SEMANTIC_COLORS["harmful_fill"]),
    ("benign_combined", "Benign", SEMANTIC_COLORS["benign"], SEMANTIC_COLORS["benign_fill"]),
]


def _style() -> None:
    register_fonts()
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["TeX Gyre Termes", "Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
    })


def _annotate_n(ax, n_total, n_pos) -> None:
    ax.text(0.5, 0.97, f"N={n_total:,}  ({n_pos:,} ref)",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=6.5, color="#333333",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                      edgecolor="none", alpha=0.75))


def _fmt_yticks(ax, ymax) -> None:
    if ymax <= 0:
        ax.set_yticks([0])
        return
    exp = math.floor(math.log10(ymax))
    base = ymax / (10 ** exp)
    if base <= 1.2:
        step = 2e-1 * 10 ** exp
    elif base <= 2.5:
        step = 5e-1 * 10 ** exp
    elif base <= 5:
        step = 1.0 * 10 ** exp
    else:
        step = 2.0 * 10 ** exp
    top = math.ceil(ymax / step) * step
    ax.set_yticks(np.arange(0, top + step * 0.5, step))
    ax.set_ylim(0, top)
    if top >= 0.1:
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    elif top >= 0.01:
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    else:
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1e"))


def load(model: str, intent: str):
    p = paths.eval_results("neurips_final", model, "valley", f"explore_{intent}.pt")
    if not p.exists():
        return None
    return torch.load(p, map_location="cpu", weights_only=False)


def main() -> None:
    _style()
    models = [(m, SUPP_LABELS[m]) for m in SUPP_ORDER]
    assert set(SUPP_ORDER) == set(SUPPLEMENTARY), \
        f"supplementary set mismatch: {sorted(SUPP_ORDER)} vs {sorted(SUPPLEMENTARY)}"
    ncols, nrows = len(models), 2
    fig_w = ncols * 2.1 + 1.0
    fig_h = nrows * 1.6 + 0.3
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(fig_w, fig_h),
        sharex=True, sharey=False,
        gridspec_kw={"hspace": 0.28, "wspace": 0.32},
    )
    L_EXT, R_EXT = 2, 3
    x_ext = np.arange(-L_EXT, 100 + R_EXT)

    for col, (key, display) in enumerate(models):
        for row, (intent, _row_label, line_c, fill_c) in enumerate(ROWS):
            ax = axes[row, col]
            blob = load(key, intent)
            if blob is None or blob.get(METRIC_KEY) is None:
                ax.text(0.5, 0.5, "N/A", transform=ax.transAxes,
                        ha="center", va="center", fontsize=10,
                        color="#888888", fontstyle="italic")
                ax.set_xlim(-2, 102)
                ax.xaxis.set_major_locator(MultipleLocator(20))
                ax.xaxis.set_minor_locator(MultipleLocator(10))
                ax.set_yticks([])
                continue

            arr = np.asarray(blob[METRIC_KEY], dtype=float)
            y_true = np.asarray(blob["y"]).astype(np.int64)
            N = int(len(y_true))
            n_pos = int((y_true == 1).sum())

            arr_ext = np.concatenate([[arr[0]] * L_EXT, arr, [arr[-1]] * R_EXT])
            ax.fill_between(x_ext, 0, arr_ext, color=fill_c, alpha=0.75)
            ax.plot(x_ext, arr_ext, color=line_c, linewidth=1.1)
            ax.set_xlim(-2, 102)
            ax.xaxis.set_major_locator(MultipleLocator(20))
            ax.xaxis.set_minor_locator(MultipleLocator(10))
            _fmt_yticks(ax, float(np.nanmax(arr)))
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            ax.grid(True, which="major", linewidth=0.3,
                    linestyle=":", color="#bbbbbb", alpha=0.9)
            ax.grid(True, which="minor", linewidth=0.2,
                    linestyle=":", color="#dddddd", alpha=0.7)
            ax.set_axisbelow(True)

            if row == 0:
                ax.set_title(display, fontsize=9, fontweight="bold", pad=4)

            _annotate_n(ax, N, n_pos)

    axes[0, 0].set_ylabel("Harmful Input", fontsize=9.5, fontweight="bold",
                          color=ROWS[0][2], labelpad=8)
    axes[1, 0].set_ylabel("Benign Input", fontsize=9.5, fontweight="bold",
                          color=ROWS[1][2], labelpad=8)
    fig.supxlabel("Normalized Thinking Position (%)", fontsize=10,
                  fontweight="bold", y=0.02)

    for ext in ("pdf", "png"):
        out = paths.figure_out(f"fig_refusal_valley_supplementary.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.06)
        print("wrote", out)
    plt.close(fig)


if __name__ == "__main__":
    main()
