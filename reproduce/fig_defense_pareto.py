#!/usr/bin/env python3
"""Reproduce Figure ``fig:defense_pareto`` — the ASR-vs-ORR Pareto plane.

Every cell (Base + 6 training + 3 inference defenses, x 4 base models) is a point
on the ASR-ORR plane; both axes are "lower is better", so Pareto-optimal cells
sit toward the bottom-left.  Per-base-model Pareto frontiers are drawn in the
model's colour.

Training + Base points use the canonical 4-guardrail soft vote from the
``__fullpipe__`` summaries; the 3 inference-defense points are transcribed from
the paper's evaluation table (no standalone summary exists for them).

Output: figures/fig_defense_pareto.{pdf,png,data.json}

Original research script: neurips/plot_defense_pareto.py
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
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

from lrm_safety_deliberation import paths
from lrm_safety_deliberation.behavioral import cell_asr_orr
from lrm_safety_deliberation.models import PRIMARY
from lrm_safety_deliberation.plotting import MODEL_COLORS, MODEL_LABELS, register_fonts

TRAINING = ["STAR1", "SafeKey", "R1ACT", "ThinkSafe", "STAIR", "RAPO"]
INFERENCE = ["PSR", "SafeRemind", "SafePath-ZS"]
METHOD_MARKERS = {
    "Base": "*",
    "PSR": "X", "SafeRemind": "p", "SafePath-ZS": "H",
    "STAR1": "o", "SafeKey": "s", "R1ACT": "D", "ThinkSafe": "^", "STAIR": "v", "RAPO": "P",
}
# Inference-defense (ASR, ORR) in percent, from the paper evaluation table.
INF_DATA = {
    "PSR":         {"Qwen3-8B": (86.90, 6.10),  "Olmo-3-7B-Think": (38.00, 43.50), "Phi-4-reasoning": (32.70, 70.60), "GPT-OSS-20B": (29.80, 42.70)},
    "SafeRemind":  {"Qwen3-8B": (80.80, 11.00), "Olmo-3-7B-Think": (32.50, 44.90), "Phi-4-reasoning": (33.30, 70.40), "GPT-OSS-20B": (21.30, 49.20)},
    "SafePath-ZS": {"Qwen3-8B": (68.20, 13.50), "Olmo-3-7B-Think": (30.70, 49.10), "Phi-4-reasoning": (31.90, 71.90), "GPT-OSS-20B": (22.50, 48.80)},
}


def load_points():
    pts = []
    for method in ["Base"] + TRAINING:
        for m in PRIMARY:
            asr, orr = cell_asr_orr(method, m, vote="soft")
            pts.append(dict(method=method, model=m, asr=asr * 100, orr=orr * 100,
                            kind=("base" if method == "Base" else "training")))
    for method in INFERENCE:
        for m in PRIMARY:
            asr, orr = INF_DATA[method][m]
            pts.append(dict(method=method, model=m, asr=asr, orr=orr, kind="inference"))
    return pts


def pareto_front(pts):
    front = [p for p in pts if not any(
        (q["asr"] <= p["asr"] and q["orr"] <= p["orr"]) and (q["asr"] < p["asr"] or q["orr"] < p["orr"])
        for q in pts)]
    return sorted(front, key=lambda z: z["asr"])


def main() -> None:
    register_fonts()
    FS = 1.0
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["TeX Gyre Termes", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": FS * 15, "axes.labelsize": FS * 17,
        "xtick.labelsize": FS * 13.5, "ytick.labelsize": FS * 13.5, "legend.fontsize": FS * 12,
    })
    pts = load_points()
    fig, ax = plt.subplots(figsize=(9.2, 6.0))

    for m in PRIMARY:
        fr = pareto_front([p for p in pts if p["model"] == m])
        if len(fr) >= 2:
            ax.plot([p["asr"] for p in fr], [p["orr"] for p in fr],
                    color=MODEL_COLORS[m], lw=1.3, ls="--", alpha=0.55, zorder=1)

    for p in pts:
        is_base = p["method"] == "Base"
        ax.scatter(p["asr"], p["orr"], s=(430 if is_base else 150),
                   color=MODEL_COLORS[p["model"]], marker=METHOD_MARKERS[p["method"]],
                   edgecolor="none", linewidth=0,
                   alpha=(0.95 if is_base else 0.78), zorder=(4 if is_base else 3))

    ax.set_xlabel("Attack Success Rate (%)  ↓")
    ax.set_ylabel("Over-Refusal Rate (%)  ↓")
    ax.set_xlim(0, 100); ax.set_ylim(0, 73)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.yaxis.set_major_locator(MultipleLocator(20))
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_minor_locator(MultipleLocator(5))
    ax.tick_params(which="minor", length=2.5)
    ax.tick_params(which="major", length=4.5)
    ax.grid(True, which="major", ls=":", color="0.55", alpha=0.85)
    ax.grid(True, which="minor", ls=":", color="0.80", alpha=0.55)
    ax.spines[["top", "right"]].set_visible(False)

    def swatch(color):
        return Line2D([], [], marker="o", color="none", markerfacecolor=color,
                      markeredgecolor="none", markersize=9, lw=0)

    def mark(mm, big=False):
        return Line2D([], [], marker=METHOD_MARKERS[mm], color="none", markerfacecolor="0.55",
                      markeredgecolor="none", markersize=(13 if big else 9), lw=0)

    blank = Line2D([], [], color="none")
    handles, labels = [blank], ["$\\bf{Model}$"]
    for m in PRIMARY:
        handles.append(swatch(MODEL_COLORS[m])); labels.append(MODEL_LABELS[m])
    handles.append(mark("Base", big=True)); labels.append("Base")
    handles += [blank, blank]; labels += ["", "$\\bf{Inference}$"]
    for mm in INFERENCE:
        handles.append(mark(mm)); labels.append(mm)
    handles += [blank, blank]; labels += ["", "$\\bf{Training}$"]
    for mm in TRAINING:
        handles.append(mark(mm)); labels.append(mm)
    ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False,
              fontsize=FS * 11 * 1.2, handlelength=1.0, handletextpad=0.4,
              borderpad=0.2, labelspacing=0.34)

    fig.tight_layout(pad=0.3)
    for ext in ("pdf", "png"):
        fig.savefig(paths.figure_out(f"fig_defense_pareto.{ext}"), dpi=240,
                    bbox_inches="tight", pad_inches=0.03)
    paths.figure_out("fig_defense_pareto.data.json").write_text(json.dumps(pts, indent=2))
    print("wrote", paths.figure_out("fig_defense_pareto.{pdf,png,data.json}"))


if __name__ == "__main__":
    main()
