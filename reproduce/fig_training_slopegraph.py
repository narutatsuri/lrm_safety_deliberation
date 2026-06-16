#!/usr/bin/env python3
"""Reproduce Figure ``fig:training_slopegraph`` — the defense-mechanism slopegraph.

For every (base model, training defense) cell we plot the Base->Trained shift in
three first-token / oscillation diagnostics, split into two rows by trace
provenance (On-Policy vs Off-Policy defenses; R1ACT dropped as a third mechanism
type).  Each panel column is one diagnostic:

  * First-token AUROC  — how separable refuse-vs-comply is at the first thinking
    token (pooled over folds), from ``defense_first_token_auroc_v2.json``.
  * Oscillations / trace — mean number of *salient* stance transitions per trace
    (a salient transition = adjacent opposite-sign carry-forward segments, each
    at least length 1), counted from the M=4 majority-3 chunk-stance labels.
  * Meaningful Oscillations / trace — the causal-oscillation rate (published
    per-cell constants; these require the causal-cut rollouts to recompute and
    are transcribed here, as in the original generator).

Lines are coloured by each cell's ΔASR (blue = ASR fell / safer, red = ASR rose
/ backfired); per-row grey-dotted mean trajectories (here coloured by the group's
mean ΔASR) and ±1 SE bands summarise each policy group.

Inputs (resolved under ``LRM_SAFETY_ARTIFACTS``; mirror the original tree):
  neurips/defense_first_token_auroc_v2.json
  neurips/defense_asr_orr_table_soft.json          (ASR/ORR per cell, soft vote)
  experiments/causal_cuts/rebuttal/m4_majority3.jsonl                 (base osc)
  experiments/causal_cuts/rebuttal/m4_training_defense_majority3.jsonl (trained osc)

Output: figures/N4_slopegraph_byASR_wide_mean_policy_band_meancolor.{pdf,png}

Original research script: neurips/proposals_n1_rows.py
  (variant: ``fig_byasr_3type`` with classify=_policy, mean_band, mean_color_by_dasr)
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, MaxNLocator

from lrm_safety_deliberation import paths
from lrm_safety_deliberation.io import iter_jsonl, read_json
from lrm_safety_deliberation.models import PRIMARY, TRAINING_METHODS
from lrm_safety_deliberation.plotting import MODEL_COLORS, MODEL_LABELS, register_fonts

MODELS = list(PRIMARY)                      # Qwen3-8B, Olmo-3-7B-Think, Phi-4-reasoning, GPT-OSS-20B
METHODS = list(TRAINING_METHODS)            # STAR1, SafeKey, R1ACT, ThinkSafe, STAIR, RAPO
COL = dict(MODEL_COLORS)
MLAB = dict(MODEL_LABELS)
MARK = {"STAR1": "o", "SafeKey": "s", "R1ACT": "D", "ThinkSafe": "^", "STAIR": "v", "RAPO": "P"}
A2M = {"Olmo3-7B-Think": "Olmo-3-7B-Think"}  # base-cell name -> registry model name
LOOP = 120                                   # drop traces longer than this many chunks
MIN_TRACES = 50                              # need >=50 traces per cell to plot

# Three metric columns: (key, title, y-limits).
COLS = [("auroc", "First Token AUROC", (0.58, 1.0)),
        ("osc", "Oscillations", (0, 1.55)),
        ("caus", "Meaningful Oscillation", (0, 0.32))]

# Published per-cell causal-oscillation rate ("meaningful osc."): requires the
# causal-cut rollouts to recompute, transcribed verbatim from the original.
CAUS = {
    ("GPT-OSS-20B", "Base"): .092, ("GPT-OSS-20B", "STAR1"): .162, ("GPT-OSS-20B", "SafeKey"): .097,
    ("GPT-OSS-20B", "R1ACT"): .022, ("GPT-OSS-20B", "ThinkSafe"): .053, ("GPT-OSS-20B", "STAIR"): .250,
    ("Qwen3-8B", "Base"): .096, ("Qwen3-8B", "STAR1"): .159, ("Qwen3-8B", "SafeKey"): .131,
    ("Qwen3-8B", "R1ACT"): .022, ("Qwen3-8B", "ThinkSafe"): .004, ("Qwen3-8B", "STAIR"): .093,
    ("Qwen3-8B", "RAPO"): .079,
    ("Olmo-3-7B-Think", "Base"): .189, ("Olmo-3-7B-Think", "STAR1"): .099, ("Olmo-3-7B-Think", "SafeKey"): .101,
    ("Olmo-3-7B-Think", "R1ACT"): .035, ("Olmo-3-7B-Think", "ThinkSafe"): .081, ("Olmo-3-7B-Think", "STAIR"): .121,
    ("Olmo-3-7B-Think", "RAPO"): .036,
    ("Phi-4-reasoning", "Base"): .140, ("Phi-4-reasoning", "STAR1"): .292, ("Phi-4-reasoning", "SafeKey"): .152,
    ("Phi-4-reasoning", "R1ACT"): .033, ("Phi-4-reasoning", "ThinkSafe"): .075, ("Phi-4-reasoning", "STAIR"): .078,
    ("Phi-4-reasoning", "RAPO"): .006,
}


# --------------------------------------------------------------------------- #
# Salient-transition counting (inlined from experiments/causal_cuts/generate_cuts.py)
# --------------------------------------------------------------------------- #
def segments_carry_forward(labels: list[int]) -> list[tuple[int, int, int, int]]:
    """(sign, start_idx, end_idx, length) segments after forward-filling 0s."""
    n = len(labels)
    if n == 0:
        return []
    filled, last = list(labels), 0
    for i in range(n):
        if filled[i] == 0:
            filled[i] = last
        else:
            last = filled[i]
    segs, cur_sign, cur_start, cur_len = [], filled[0], 0, 1
    for i in range(1, n):
        if filled[i] == cur_sign:
            cur_len += 1
        else:
            segs.append((cur_sign, cur_start, i - 1, cur_len))
            cur_sign, cur_start, cur_len = filled[i], i, 1
    segs.append((cur_sign, cur_start, n - 1, cur_len))
    return segs


def n_salient_transitions(labels: list[int], l_pre: int = 1, l_min: int = 1) -> int:
    """Count adjacent opposite-sign nonzero segment pairs meeting length bounds."""
    segs = segments_carry_forward(labels)
    n = 0
    for a, b in zip(segs, segs[1:]):
        if a[0] == 0 or b[0] == 0 or a[0] * b[0] >= 0:
            continue
        if a[3] < l_pre or b[3] < l_min:
            continue
        n += 1
    return n


# --------------------------------------------------------------------------- #
# Build the per-cell metric table
# --------------------------------------------------------------------------- #
def load_cells() -> dict:
    au = read_json(paths.artifact("neurips", "defense_first_token_auroc_v2.json"))
    asror = read_json(paths.artifact("neurips", "defense_asr_orr_table_soft.json"))

    osc = defaultdict(list)

    def feed(path: Path, is_base: bool):
        for d in iter_jsonl(path):
            lab = d.get("labels", [])
            if not lab or any(x is None for x in lab) or int(d.get("n_chunks", len(lab))) > LOOP:
                continue
            cell = d["cell"]
            if is_base:
                m, method = A2M.get(cell, cell), "Base"
            else:
                method = next((x for x in METHODS if cell.startswith(x + "-")), None)
                if not method:
                    continue
                m = cell[len(method) + 1:]
            if m in MODELS:
                osc[(m, method)].append(n_salient_transitions(lab))

    feed(paths.artifact("experiments", "causal_cuts", "rebuttal", "m4_majority3.jsonl"), True)
    feed(paths.artifact("experiments", "causal_cuts", "rebuttal", "m4_training_defense_majority3.jsonl"), False)

    def auroc(m, c):
        k = f"{c}|{m}"
        return au[k]["pooled"]["auroc"] if k in au else None

    cells = {}
    for m in MODELS:
        for c in ["Base"] + METHODS:
            if (m, c) in osc and len(osc[(m, c)]) >= MIN_TRACES and auroc(m, c) is not None:
                v = asror["Base"][m] if c == "Base" else asror.get(c, {}).get(m)
                cells[(m, c)] = dict(
                    auroc=auroc(m, c), osc=float(np.mean(osc[(m, c)])), caus=CAUS.get((m, c)),
                    asr=(v["asr"] * 100 if v else None), orr=(v["orr"] * 100 if v else None))
    return cells


# --------------------------------------------------------------------------- #
# Figure: 2 rows (On-/Off-Policy) x 3 metric columns, ΔASR-coloured lines,
# per-group mean trajectory + ±1 SE band coloured by the group's mean ΔASR.
# --------------------------------------------------------------------------- #
def make_figure(cells: dict) -> None:
    FS = 1.2
    tscale, title_scale, mean_lw, ylabel_scale, legend_scale = 1.5, 1.148, 4.6, 1.2, 1.2
    register_fonts()
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["TeX Gyre Termes", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": FS * 12.5, "axes.labelsize": FS * 12.5,
        "xtick.labelsize": FS * 11, "ytick.labelsize": FS * 10 * 1.5,
    })

    policy = lambda c: ("drop" if c == "R1ACT" else ("off" if c in ("STAR1", "SafeKey") else "on"))
    ROWS = [("on", "On-Policy"), ("off", "Off-Policy")]

    dasr = {(m, c): cells[(m, c)]["asr"] - cells[(m, "Base")]["asr"]
            for m in MODELS for c in METHODS if (m, c) in cells}
    _c = plt.cm.RdBu_r(np.linspace(0, 1, 256))
    _x = np.linspace(0, 1, 256)
    _c[_x > 0.5, :3] = 0.75 * _c[_x > 0.5, :3] + 0.25          # tone the red end down
    cmap = ListedColormap(_c)
    norm = TwoSlopeNorm(vmin=-60, vcenter=0, vmax=85)

    fig, axes = plt.subplots(len(ROWS), len(COLS), figsize=(14.9, 7.56), constrained_layout=True)
    axes = np.atleast_2d(axes)
    for ri, (pk, plab) in enumerate(ROWS):
        for ci, (key, _t, ylim) in enumerate(COLS):
            ax = axes[ri][ci]
            bv, tv, dv = [], [], []
            for m in MODELS:
                if (m, "Base") not in cells:
                    continue
                ms = [c for c in METHODS if (m, c) in cells and policy(c) == pk]
                if not ms:
                    continue
                b = cells[(m, "Base")][key]
                ax.scatter(0, b, s=300, color=COL[m], marker="*", edgecolor="black",
                           lw=1.0, alpha=0.8, zorder=6)
                for c in ms:
                    v = cells[(m, c)][key]
                    ax.plot([0, 1], [b, v], color=cmap(norm(dasr[(m, c)])), lw=1.9, alpha=0.4, zorder=2)
                    ax.scatter(1, v, s=80, color=COL[m], marker=MARK[c], edgecolor="black",
                               lw=0.5, alpha=0.8, zorder=5)
                    bv.append(b); tv.append(v); dv.append(dasr[(m, c)])
            if bv:
                mb, mt = float(np.mean(bv)), float(np.mean(tv))
                mcol = cmap(norm(float(np.mean(dv))))               # mean line ~ group's mean ΔASR
                if len(bv) > 1:                                     # ±1 SE band
                    seb = float(np.std(bv, ddof=1) / np.sqrt(len(bv)))
                    set_ = float(np.std(tv, ddof=1) / np.sqrt(len(tv)))
                    ax.fill_between([0, 1], [mb - seb, mt - set_], [mb + seb, mt + set_],
                                    color=mcol, alpha=0.30, lw=0, zorder=3)
                ax.plot([0, 1], [mb, mt], color=mcol, lw=mean_lw, ls=":", zorder=8,
                        dash_capstyle="round",
                        path_effects=[pe.Stroke(linewidth=mean_lw + 1.8, foreground="0.2"), pe.Normal()])
                ax.scatter([0, 1], [mb, mt], color=mcol, s=84, zorder=9, edgecolor="0.2", lw=0.9)
            ax.set_xlim(-0.15, 1.15)
            ax.set_xticks([0, 1])
            ax.set_xticklabels((["Base", "Trained"] if ri == len(ROWS) - 1 else []), fontsize=FS * 11 * tscale)
            ax.set_ylim(*ylim)
            if ri == 0:
                ax.set_title(_t, fontsize=FS * 11 * tscale * title_scale, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(plab, fontsize=FS * 11 * tscale * ylabel_scale, fontweight="bold")
            ax.yaxis.set_major_locator(MaxNLocator(6))
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            ax.grid(True, axis="y", which="major", ls=":", color="0.6", alpha=0.7)
            ax.grid(True, axis="y", which="minor", ls=":", color="0.8", alpha=0.5)
            ax.spines[["top", "right"]].set_visible(False)

    axes[0][0].legend([Line2D([], [], color="0.3", lw=mean_lw, ls=":")], ["mean"], loc="lower left",
                      frameon=False, fontsize=FS * 11 * tscale * 0.85 * legend_scale, handlelength=2.0)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axes, fraction=0.03, pad=0.02)
    cb.set_ticks([])
    cb.ax.text(0.5, 1.01, "↑ASR\n↓ORR", transform=cb.ax.transAxes, ha="center", va="bottom", fontsize=FS * 8.5 * tscale)
    cb.ax.text(0.5, -0.01, "↓ASR\n↑ORR", transform=cb.ax.transAxes, ha="center", va="top", fontsize=FS * 8.5 * tscale)

    present = [c for c in METHODS if policy(c) in ("on", "off")]
    mh = [Line2D([], [], marker="o", color="white", markerfacecolor=COL[m], markeredgecolor="black",
                 markersize=12, lw=0) for m in MODELS]
    kh = [Line2D([], [], marker="*", color="white", markerfacecolor="0.5", markeredgecolor="black",
                 markersize=17, lw=0)] + \
         [Line2D([], [], marker=MARK[c], color="white", markerfacecolor="0.5", markeredgecolor="black",
                 markersize=13, lw=0) for c in present]
    fig.legend(mh, [MLAB[m] for m in MODELS], loc="upper center", bbox_to_anchor=(0.5, 0.05),
               ncol=4, frameon=False, fontsize=FS * 15.5 * legend_scale, columnspacing=1.2, handletextpad=0.3)
    fig.legend(kh, ["Base"] + present, loc="upper center", bbox_to_anchor=(0.5, 0.012),
               ncol=len(present) + 1, frameon=False, fontsize=FS * 15.5 * legend_scale,
               columnspacing=0.9, handletextpad=0.35)

    for ext in ("pdf", "png"):
        fig.savefig(paths.figure_out(f"N4_slopegraph_byASR_wide_mean_policy_band_meancolor.{ext}"),
                    dpi=240, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def main() -> None:
    cells = load_cells()

    # ---- console summary: per-cell metrics + per-group mean slopes (for verification) ----
    policy = lambda c: ("drop" if c == "R1ACT" else ("off" if c in ("STAR1", "SafeKey") else "on"))
    print(f"[cells] {len(cells)} (model, method) cells with >= {MIN_TRACES} traces")
    for m in MODELS:
        for c in ["Base"] + METHODS:
            if (m, c) in cells:
                d = cells[(m, c)]
                print(f"  {c:9s}|{m:16s} auroc={d['auroc']:.4f} osc={d['osc']:.4f} "
                      f"caus={d['caus']} asr={d['asr']:.2f} orr={d['orr']:.2f}")
    for grp in ("on", "off"):
        print(f"[mean slope] {grp}-policy")
        for key, _t, _y in COLS:
            bv, tv, dv = [], [], []
            for m in MODELS:
                ms = [c for c in METHODS if (m, c) in cells and policy(c) == grp]
                if not ms or (m, "Base") not in cells:
                    continue
                b = cells[(m, "Base")][key]
                for c in ms:
                    bv.append(b); tv.append(cells[(m, c)][key])
                    dv.append(cells[(m, c)]["asr"] - cells[(m, "Base")]["asr"])
            if bv:
                print(f"  {key:6s}: mb={np.mean(bv):.6f} mt={np.mean(tv):.6f} "
                      f"mean_dasr={np.mean(dv):.6f} n={len(bv)}")

    make_figure(cells)
    print("wrote", paths.figure_out("N4_slopegraph_byASR_wide_mean_policy_band_meancolor.{pdf,png}"))


if __name__ == "__main__":
    main()
