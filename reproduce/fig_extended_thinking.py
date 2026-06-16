#!/usr/bin/env python3
"""Reproduce Figure ``fig:asr_orr_extended_thinking`` — ASR/ORR vs thinking amount (K=32).

Two-panel ASR (Harmful Pool) / ORR (Benign Pool) curves as a function of the
percentage of the natural thinking trace that is forced as a prefix, over the
full K=32 pools (ASR=2500 / ORR=2885 prompts). The 100->200% region is the
forced-extension regime (r=0.2 continuation), shaded to flag it.

For the Benign pool the aggregator's stored ``point_pct`` IS the ORR; for the
Harmful pool ASR = 100 - point_pct (% comply). ``half_pct`` is the 95% bootstrap
CI half-width. A grey "Mean across models" curve overlays the per-model lines.

Inputs : artifacts/experiments/nested_branching_variance/k32_full/analysis/data/asr_orr_within_by_cut.json
Output : figures/fig_asr_orr_vs_thinking_k32full.{pdf,png}

Original research script:
  experiments/nested_branching_variance/k32_full/analysis/plot_asr_orr_vs_thinking_k32full.py
(self-contained in the original; reads only the data JSON.)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from lrm_safety_deliberation import paths
from lrm_safety_deliberation.models import PRIMARY
from lrm_safety_deliberation.plotting import MODEL_COLORS, MODEL_LABELS, register_fonts

CUTS = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]

FONT_SIZE = 15
TITLE_SIZE = 21
LABEL_SIZE = 21
LEGEND_SIZE = 11.5
TICK_SIZE = 12

MODEL_LINE_ALPHA = 0.75
MODEL_BAND_ALPHA = 0.14
MEAN_COLOR = "#4d4d4d"
MEAN_BAND_ALPHA = 0.22

HSQUISH = 0.95
BOX_ASPECT_W_OVER_H = (1114.0 / 727.0) * HSQUISH


def data_json() -> Path:
    return paths.artifact(
        "experiments", "nested_branching_variance", "k32_full",
        "analysis", "data", "asr_orr_within_by_cut.json",
    )


def plot_panel(ax, data: dict, pool: str, present_models: list[str]) -> None:
    """Per-prompt soft refusal_pct. Benign pool: this IS ORR. Harmful: ASR = 100 - pct."""
    flip_to_asr = (pool == "ASR")
    xs = np.array(CUTS, dtype=float)
    per_cut_y: dict[int, list[float]] = {c: [] for c in CUTS}
    per_cut_half: dict[int, list[float]] = {c: [] for c in CUTS}
    for m in present_models:
        cells = data[m][pool]
        raw = np.array([cells[str(c)]["point_pct"] for c in CUTS])
        halves = np.array([cells[str(c)]["half_pct"] for c in CUTS])
        ys = (100.0 - raw) if flip_to_asr else raw
        lo = ys - halves
        hi = ys + halves
        color = MODEL_COLORS[m]
        ax.fill_between(xs, lo, hi, color=color, alpha=MODEL_BAND_ALPHA, zorder=2)
        ax.plot(xs, ys, "o-", color=color, lw=1.8, ms=8.5,
                markeredgecolor="black", markeredgewidth=0.5,
                alpha=MODEL_LINE_ALPHA,
                label=MODEL_LABELS[m], zorder=3)
        for i, c in enumerate(CUTS):
            per_cut_y[c].append(float(ys[i]))
            per_cut_half[c].append(float(halves[i]))

    means = np.array([np.mean(per_cut_y[c]) for c in CUTS])
    half = np.array([np.mean(per_cut_half[c]) for c in CUTS])
    ax.fill_between(xs, means - half, means + half,
                    facecolor=MEAN_COLOR, alpha=MEAN_BAND_ALPHA,
                    hatch="....", edgecolor=MEAN_COLOR, linewidth=0.0,
                    zorder=5)
    ax.plot(xs, means, marker="s", linestyle=":", color=MEAN_COLOR,
            lw=2.6, ms=9.0,
            markerfacecolor=MEAN_COLOR, markeredgecolor=MEAN_COLOR,
            label="Mean across models", zorder=6)

    ax.axvspan(100, 200, color="#cccccc", alpha=0.25, zorder=0)
    ax.axvline(100, color="black", lw=0.6, ls="--", alpha=0.5, zorder=1)


def main() -> None:
    register_fonts()
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": FONT_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "legend.fontsize": LEGEND_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
    })

    with data_json().open() as f:
        results = json.load(f)["per_cell"]

    present_models = []
    for m in PRIMARY:
        cells = results.get(m, {})
        if all(str(c) in cells.get("ASR", {}) and str(c) in cells.get("ORR", {})
               for c in CUTS):
            present_models.append(m)
    print(f"Models with full data: {present_models}")

    # Echo per-model percentages for numeric verification.
    for pool in ("ASR", "ORR"):
        flip = (pool == "ASR")
        print(f"\n[{pool}] per-model {'ASR' if flip else 'ORR'} (%) at each cut:")
        for m in present_models:
            cells = results[m][pool]
            ys = [(100.0 - cells[str(c)]["point_pct"]) if flip else cells[str(c)]["point_pct"]
                  for c in CUTS]
            print(f"  {m:18s} " + " ".join(f"{y:6.2f}" for y in ys))

    fig, (ax_asr, ax_orr) = plt.subplots(
        1, 2, figsize=(14.5 * HSQUISH, 5.5), sharex=True,
    )
    plot_panel(ax_asr, results, "ASR", present_models)
    plot_panel(ax_orr, results, "ORR", present_models)

    ax_asr.set_title(r"Harmful Pool $\downarrow$")
    ax_orr.set_title(r"Benign Pool $\downarrow$")
    ax_asr.set_ylabel("ASR (%)")
    ax_orr.set_ylabel("ORR (%)")

    for ax in (ax_asr, ax_orr):
        ax.set_xticks(CUTS)
        ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
        ax.tick_params(axis="x", which="minor", length=0)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(10))
        ax.yaxis.set_minor_locator(mticker.MultipleLocator(5))
        ax.grid(True, which="major", ls=":", alpha=0.45)
        ax.grid(True, which="minor", ls=":", alpha=0.20)

    ymin = 0
    ymax = max(ax_asr.get_ylim()[1], ax_orr.get_ylim()[1])
    ymax = 10 * np.ceil(ymax / 10)
    for ax in (ax_asr, ax_orr):
        ax.set_ylim(ymin, ymax)
        ax.set_box_aspect(1.0 / BOX_ASPECT_W_OVER_H)

    handles, labels = ax_asr.get_legend_handles_labels()
    ax_asr.legend(handles, labels, loc="upper right", framealpha=0.92)

    fig.supxlabel("Percentage of Natural Thinking Trace (%)",
                  fontsize=LABEL_SIZE, y=0.06)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = paths.figure_out(f"fig_asr_orr_vs_thinking_k32full.{ext}")
        fig.savefig(p, dpi=220, bbox_inches="tight")
        print(f"wrote {p}", flush=True)


if __name__ == "__main__":
    main()
