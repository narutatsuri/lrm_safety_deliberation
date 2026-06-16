#!/usr/bin/env python3
"""Reproduce Figures ``fig:thinking_causal`` and ``fig:thinking_causal_bysplit``.

The "exploded stacked column" figures: per model, a 100%-of-rollouts overview
column (no-oscillation vs >=1 oscillation) whose oscillation block is *exploded*
into a magnified column of the four meaningfulness categories.  The headline
result is that most detected CoT oscillations are PERFORMATIVE (the text wavers
but the K=100 resampled decision does not move); only a small fraction are
statistically causal.

  figures/m4_cut_exploded_pooled.{pdf,png}  -> fig:thinking_causal          (pooled splits)
  figures/m4_cut_exploded.{pdf,png}         -> fig:thinking_causal_bysplit   (harmful/benign)

Meaningfulness 2x2 per cut:
  * Locked  = comply_pre <=0.05 or >=0.95 (decision saturated on entry to the segment).
  * Significant = the cut's Holm-corrected (across the model's pooled cuts)
    two-sided Fisher-exact p < 0.05.
  Categories (theater -> noise -> soft causal -> hard flip):
    (Locked,nosig)  (Wavering,nosig)  (Wavering,sig)  (Locked,sig)
  "Causal" = the two Significant cells.

Inputs (original research-repo layout under LRM_SAFETY_ARTIFACTS):
  experiments/causal_cuts/rebuttal/m4_allcuts.jsonl      (small per-cut cache:
      comply_pre/post + raw comply/refuse counts for the new M=4 cut-rollouts)
  experiments/causal_cuts/rebuttal/sig_cuts_enriched.jsonl  (rollout-0 cuts with
      comply rates + fisher_p; also the rollout-0 oscillation membership)
  experiments/causal_cuts/rebuttal/m4_majority3.jsonl    (audited M=4 denominator)
  experiments/causal_cuts/rebuttal/m4_cuts.jsonl         (which new rollouts have cuts)

Outputs: figures/m4_cut_exploded_pooled.{pdf,png}, figures/m4_cut_exploded.{pdf,png}

Original research script: experiments/causal_cuts/plots/plot_m4_cut_meaningfulness.py
  (its ``exploded_stacked`` / ``exploded_stacked_pooled``, which import
  ``plot2_data`` / ``plot2_data_bysplit`` from plot_m4_oscillation.py — inlined here).
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, PathPatch
from matplotlib.path import Path as MplPath
from scipy.stats import fisher_exact

from lrm_safety_deliberation import paths
from lrm_safety_deliberation.io import iter_jsonl
from lrm_safety_deliberation.models import PRIMARY
from lrm_safety_deliberation.plotting import MODEL_LABELS, register_fonts
from lrm_safety_deliberation.stats import holm_significant

# Figure model order (matches the paper's exploded columns).
MODELS = PRIMARY[:4]  # Qwen3-8B, Olmo-3-7B-Think, Phi-4-reasoning, GPT-OSS-20B
# The audit files label OLMo "Olmo3-7B-Think"; canonicalize to the registry name.
AUDIT_TO_MODEL = {"Olmo3-7B-Think": "Olmo-3-7B-Think"}

# meaningfulness order: theater -> noise -> soft causal -> hard flip
ORDER = [("locked", "nosig"), ("wav", "nosig"), ("wav", "sig"), ("locked", "sig")]
LABELS = ["Locked, No Significance", "Not Locked, No Significance",
          "Not Locked, Significant", "Locked, Significant"]


def _reb(name: str) -> Path:
    return paths.artifact("experiments", "causal_cuts", "rebuttal", name)


def is_locked(pre: float, post: float) -> bool:
    ex = lambda v: v <= 0.05 or v >= 0.95
    return ex(pre)


# --------------------------------------------------------------------------- #
# Per-cut 2x2 counts (Locked x Significant), pooled over M=4
# --------------------------------------------------------------------------- #
def _load_allcuts() -> list[dict]:
    return list(iter_jsonl(_reb("m4_allcuts.jsonl")))


def _new_cuts_per_cell(allcuts: list[dict], model: str, split: str):
    """New-M=4 cut comply rates + raw counts for one (model, split); None if none."""
    rows = [r for r in allcuts
            if r["source"] == "new" and r["model"] == model and r["split"] == split]
    if not rows:
        return None
    return [dict(comply_pre=r["comply_pre"], comply_post=r["comply_post"],
                 n_comply_pre=r["n_comply_pre"], n_pre=r["n_pre"],
                 n_comply_post=r["n_comply_post"], n_post=r["n_post"]) for r in rows]


def _rollout0_cuts():
    """{model -> [cut]} and {(model,split) -> [cut]} from sig_cuts_enriched
    (rollout-0 cuts already carry comply rates + fisher_p)."""
    pooled = defaultdict(list)
    bysplit = defaultdict(list)
    for d in iter_jsonl(_reb("sig_cuts_enriched.jsonl")):
        cut = dict(comply_pre=d["comply_rate_pre"], comply_post=d["comply_rate_post"],
                   fisher_p=d.get("fisher_p", 1.0))
        pooled[d["model"]].append(cut)
        bysplit[(d["model"], d["split"])].append(cut)
    return pooled, bysplit


def _count_2x2(cuts: list[dict]) -> tuple[dict, int]:
    """Holm-correct the pooled cuts' Fisher p-values, then bin into the 2x2."""
    ps = [c["fisher_p"] for c in cuts]
    sig = holm_significant(ps, alpha=0.05)
    cm = {("locked", "sig"): 0, ("locked", "nosig"): 0,
          ("wav", "sig"): 0, ("wav", "nosig"): 0}
    for c, s in zip(cuts, sig):
        lk = "locked" if is_locked(c["comply_pre"], c["comply_post"]) else "wav"
        cm[(lk, "sig" if s else "nosig")] += 1
    return cm, len(cuts)


def _new_cut_pvals(nc: list[dict]) -> list[dict]:
    """Append a fresh two-sided Fisher p to each new cut (comply/refuse pre vs post)."""
    out = []
    for c in nc:
        table = [[c["n_comply_pre"], c["n_pre"] - c["n_comply_pre"]],
                 [c["n_comply_post"], c["n_post"] - c["n_comply_post"]]]
        try:
            _, p = fisher_exact(table)
        except Exception:
            p = 1.0
        out.append(dict(comply_pre=c["comply_pre"], comply_post=c["comply_post"], fisher_p=p))
    return out


def plot2_data(allcuts, r0_pooled):
    """Per model: (2x2 counts, n_cuts) pooled over M=4. Skips models without K=100."""
    per_model = {}
    for m in MODELS:
        cuts = list(r0_pooled.get(m, []))
        ok = True
        for split in ("harmful", "benign"):
            nc = _new_cuts_per_cell(allcuts, m, split)
            if nc is None:
                ok = False
                break
            cuts.extend(_new_cut_pvals(nc))
        if not ok:
            continue
        per_model[m] = _count_2x2(cuts)
    return per_model


def plot2_data_bysplit(allcuts, r0_bysplit):
    """Per (model, split): (2x2 counts, n_cuts). Holm per (model, split)."""
    per = {}
    for m in MODELS:
        for split in ("harmful", "benign"):
            nc = _new_cuts_per_cell(allcuts, m, split)
            if nc is None:
                continue
            cuts = list(r0_bysplit.get((m, split), []))
            cuts.extend(_new_cut_pvals(nc))
            per[(m, split)] = _count_2x2(cuts)
    return per


# --------------------------------------------------------------------------- #
# Per-rollout oscillation fraction (overview "Oscillation" share)
# --------------------------------------------------------------------------- #
def per_rollout_osc_fraction(split=None):
    """Fraction of M=4 rollouts that contain >=1 oscillation, per model, over
    M=4-complete prompts (1 orig rollout-0 + 3 audited new rollouts).
    split=None pools harmful+benign; else restrict to one split."""
    r0 = set()
    for d in iter_jsonl(_reb("sig_cuts_enriched.jsonl")):
        r0.add((d["model"], d["split"], int(d["global_id"])))
    new_aud, new_cut = defaultdict(set), defaultdict(set)
    for d in iter_jsonl(_reb("m4_majority3.jsonl")):
        m = AUDIT_TO_MODEL.get(d["cell"], d["cell"])
        new_aud[(m, d["split"], int(d["global_id"]))].add(int(d["rollout_idx"]))
    for d in iter_jsonl(_reb("m4_cuts.jsonl")):
        m = AUDIT_TO_MODEL.get(d["cell"], d["cell"])
        new_cut[(m, d["split"], int(d["global_id"]))].add(int(d["rollout_idx"]))
    out = {}
    for m in MODELS:
        keys = [k for k in new_aud
                if k[0] == m and len(new_aud[k]) == 3 and (split is None or k[1] == split)]
        osc = sum((1 if k in r0 else 0)
                  + sum(1 for ri in new_aud[k] if ri in new_cut.get(k, set()))
                  for k in keys)
        out[m] = osc / (4 * len(keys)) if keys else float("nan")
    return out


# --------------------------------------------------------------------------- #
# Plotting (faithful copies of exploded_stacked / exploded_stacked_pooled)
# --------------------------------------------------------------------------- #
NO_OSC, HAS_OSC = "#f0f0f0", "#b8d8e8"
CAT_COLS = ["#ededed", "#d3d8dc", "#f8c890", "#f4a6a6"]  # theater -> causal (pastel)
CAT_TXT = {c: "black" for c in CAT_COLS}
CAT_HATCH = ["+++", "+++", "\\\\", "\\\\"]               # performative (grid) vs meaningful (\\)
HATCH_C = "#9a9a9a"
LABEL = MODEL_LABELS  # standardizes OLMo label


def _legend_handles():
    return [Patch(facecolor=NO_OSC, ec="black", lw=0.6, label="No oscillation"),
            Patch(facecolor=HAS_OSC, ec="black", lw=0.6, label="Oscillation"),
            Patch(facecolor=CAT_COLS[0], ec="black", lw=0.6, label=LABELS[0]),
            Patch(facecolor=CAT_COLS[1], ec="black", lw=0.6, label=LABELS[1]),
            Patch(facecolor=CAT_COLS[2], ec="black", lw=0.6, label=LABELS[2]),
            Patch(facecolor=CAT_COLS[3], ec="black", lw=0.6, label=LABELS[3]),
            Patch(facecolor="white", ec=HATCH_C, lw=0.6, hatch="+++", label="Performative"),
            Patch(facecolor="white", ec=HATCH_C, lw=0.6, hatch="\\\\", label="Meaningful")]


def _draw_pair(ax, ov_x, has, no, cm, n, IN, BW):
    ex_x = ov_x + IN
    x0, x1, mid = ov_x + BW / 2, ex_x - BW / 2, (ov_x + ex_x) / 2
    fs = round(20.0 * BW / 0.78, 1)   # scale in-bar number size to bar width (pooled 20.0 -> bysplit ~12.3)
    verts = [(x0, 100), (mid, 100), (mid, 100), (x1, 100), (x1, 0),
             (mid, 0), (mid, no), (x0, no), (x0, 100)]
    codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.LINETO,
             MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=HAS_OSC, alpha=0.16,
                           edgecolor="none", zorder=1))
    LEAD = "#7ba4c4"
    gp = MplPath([(x0, no), (mid, no), (mid, 0), (x1, 0)],
                 [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
    ax.add_patch(PathPatch(gp, facecolor="none", edgecolor=LEAD, lw=1.0, alpha=0.7, zorder=2))
    tp = MplPath([(x0, 99.6), (mid, 99.6), (mid, 99.6), (x1, 99.6)],
                 [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
    ax.add_patch(PathPatch(tp, facecolor="none", edgecolor=LEAD, lw=1.0, alpha=0.7, zorder=2))
    ax.bar(ov_x, no, BW, color=NO_OSC, edgecolor="black", lw=0.4, zorder=3)
    ax.bar(ov_x, has, BW, bottom=no, color=HAS_OSC, edgecolor="black", lw=0.4, zorder=3)
    ax.text(ov_x, no / 2, f"{no:.0f}%", ha="center", va="center", fontsize=fs,
            color="black", fontweight="bold", zorder=5)
    if has >= 8:
        ax.text(ov_x, no + has / 2, f"{has:.0f}%", ha="center", va="center", fontsize=fs,
                color="black", fontweight="bold", zorder=5)
    bottom = 0
    for col, key, hh in zip(CAT_COLS, ORDER, CAT_HATCH):
        w = 100 * cm[key] / n
        ax.bar(ex_x, w, BW, bottom=bottom, color=col, edgecolor=HATCH_C, lw=0.0, hatch=hh, zorder=4)
        ax.bar(ex_x, w, BW, bottom=bottom, color="none", edgecolor="black", lw=0.4, zorder=4)
        if w >= 6:
            ax.text(ex_x, bottom + w / 2, f"{w:.0f}%", ha="center", va="center", fontsize=fs,
                    color=CAT_TXT[col], fontweight="bold", zorder=6)
        bottom += w
    return ex_x


def _style_axes(ax, centers, pad):
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))
    ax.set_yticks(range(0, 101, 10), minor=True)
    ax.set_yticklabels([f"{t}%" for t in range(0, 101, 20)])
    ax.set_ylabel("Percentage of Segments (%)", fontsize=19.8)
    ax.set_xticks(centers)
    ax.set_xticklabels([LABEL[m] for m in MODELS], fontsize=21.384, fontweight="bold")
    ax.tick_params(axis="y", which="major", labelsize=16.2, length=3.5, width=0.8, color="black")
    ax.tick_params(axis="y", which="minor", length=2, width=0.6, color="black")
    ax.tick_params(axis="x", length=0, color="black", pad=pad)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for s in ("bottom", "left"):
        ax.spines[s].set_visible(True)
        ax.spines[s].set_color("black")
        ax.spines[s].set_linewidth(0.9)
    ax.spines["left"].set_position(("outward", 8))
    ax.legend(handles=_legend_handles(), ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.01),
              frameon=False, fontsize=18.576, columnspacing=1.3, handlelength=1.3,
              handletextpad=0.45, labelspacing=0.45)


def exploded_pooled(data_pooled, roll):
    plt.rcParams["hatch.linewidth"] = 0.7
    BW, IN, GAP = 0.78, 1.05, 1.90
    GROUP = IN + GAP
    fig, ax = plt.subplots(figsize=(13.0, 5.33))
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, which="major", color="#f1f1f1", lw=0.8, zorder=0)
    for i, m in enumerate(MODELS):
        base = i * GROUP
        _draw_pair(ax, base, 100 * roll[m], 100 - 100 * roll[m], *data_pooled[m], IN, BW)
    centers = [i * GROUP + IN / 2 for i in range(len(MODELS))]
    ax.set_xlim(-0.45, (len(MODELS) - 1) * GROUP + IN + 0.45)
    _style_axes(ax, centers, pad=10)
    fig.tight_layout(pad=0.4)
    for ext in ("pdf", "png"):
        fig.savefig(paths.figure_out(f"m4_cut_exploded_pooled.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", paths.figure_out("m4_cut_exploded_pooled.{pdf,png}"))


def exploded_bysplit(data_bysplit, roll_by):
    plt.rcParams["hatch.linewidth"] = 0.7
    BW, IN, PAIR, GAP = 0.48, 0.72, 0.64, 1.14
    GROUP = 2 * IN + PAIR + GAP
    fig, ax = plt.subplots(figsize=(13.0, 5.33))
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, which="major", color="#f1f1f1", lw=0.8, zorder=0)
    for i, m in enumerate(MODELS):
        base = i * GROUP
        h_ov = base
        h_ex = _draw_pair(ax, h_ov, 100 * roll_by["harmful"][m],
                          100 - 100 * roll_by["harmful"][m], *data_bysplit[(m, "harmful")], IN, BW)
        b_ov = base + IN + PAIR
        b_ex = _draw_pair(ax, b_ov, 100 * roll_by["benign"][m],
                          100 - 100 * roll_by["benign"][m], *data_bysplit[(m, "benign")], IN, BW)
        ax.text((h_ov + h_ex) / 2, -2.5, "Harmful", ha="center", va="top", fontsize=16.2, color="black")
        ax.text((b_ov + b_ex) / 2, -2.5, "Benign", ha="center", va="top", fontsize=16.2, color="black")
    centers = [i * GROUP + IN + PAIR / 2 for i in range(len(MODELS))]
    ax.set_xlim(-0.45, (len(MODELS) - 1) * GROUP + 2 * IN + PAIR + 0.45)
    _style_axes(ax, centers, pad=30)
    fig.tight_layout(pad=0.4)
    for ext in ("pdf", "png"):
        fig.savefig(paths.figure_out(f"m4_cut_exploded.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", paths.figure_out("m4_cut_exploded.{pdf,png}"))


def main() -> None:
    register_fonts()
    plt.rcParams.update({"font.family": "serif", "font.serif": ["TeX Gyre Termes"],
                         "mathtext.fontset": "stix", "axes.linewidth": 0.8, "font.size": 13.2})
    allcuts = _load_allcuts()
    r0_pooled, r0_bysplit = _rollout0_cuts()

    data_pooled = plot2_data(allcuts, r0_pooled)
    data_bysplit = plot2_data_bysplit(allcuts, r0_bysplit)
    roll_pooled = per_rollout_osc_fraction()
    roll_by = {sp: per_rollout_osc_fraction(sp) for sp in ("harmful", "benign")}

    # Echo the underlying counts so the figure is verifiable against the original.
    print("\n=== Pooled 2x2 (locked x sig), pooled over M=4 ===")
    for m in MODELS:
        cm, n = data_pooled[m]
        print(f"  {m:<18} n_cuts={n}  locked_sig={cm[('locked','sig')]} "
              f"locked_nosig={cm[('locked','nosig')]} wav_sig={cm[('wav','sig')]} "
              f"wav_nosig={cm[('wav','nosig')]}  | osc_frac={roll_pooled[m]:.4f}")
    print("\n=== By-split 2x2 ===")
    for m in MODELS:
        for sp in ("harmful", "benign"):
            cm, n = data_bysplit[(m, sp)]
            print(f"  {m:<18} {sp:<8} n_cuts={n}  L+sig={cm[('locked','sig')]} "
                  f"L+nosig={cm[('locked','nosig')]} W+sig={cm[('wav','sig')]} "
                  f"W+nosig={cm[('wav','nosig')]}  | osc_frac={roll_by[sp][m]:.4f}")

    exploded_pooled(data_pooled, roll_pooled)
    exploded_bysplit(data_bysplit, roll_by)


if __name__ == "__main__":
    main()
