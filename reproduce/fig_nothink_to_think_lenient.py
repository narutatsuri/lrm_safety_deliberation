#!/usr/bin/env python3
"""Reproduce Figure ``fig:no_think_to_think_lenient`` — lenient no-think -> think (K=32).

Two side-by-side panels over the four primary models x {ASR, ORR}:

Left  (delta-sign, majority): for each prompt take the signed change in the
  majority-refuse rate between modes (a rollout is majority-refuse iff >=3/4
  guardrails vote refusal; ASR uses q_think-q_nothink, ORR its negation) and
  count it Good / Bad / Unchanged by sign. Identical to the right panel of
  fig:no_think_to_think.
Right (mean signed delta +/- std, soft): per prompt take the SOFT refusal rate
  in each mode (mean over K=32 rollouts of the mean 4-guardrail vote), form a
  signed pp change where positive = safer/better, and plot mean +/- sample std
  (ddof=1) over prompts per cell.

Inputs : artifacts/eval_results/k32_full_pool/<model>/{think,nothink}/{asr,orr}/
           classifications_<guardrail>.{jsonl,audit.json}
Output : figures/fig_delta_sign_mean_std_sidebyside_k32.{pdf,png,data.json}

Original research script:
  experiments/nested_branching_variance/k32_full/analysis/plot_delta_sign_mean_std_sidebyside_k32.py
Inlines logic from siblings transition_table_k32.py (cell_is_ready,
per_prompt_mean_refuse), plot_flip_summary_maj_k32_combined.py
(per_prompt_maj_refuse_rate), plot_delta_sign_summary_maj_k32.py
(delta_sign_summary), plot_good_delta_mean_std_k32.py (signed_good_deltas), and
plot_flip_summary_maj_k32_sidebyside.py (draw_panel).
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from lrm_safety_deliberation import paths
from lrm_safety_deliberation.guardrails import GUARDRAIL_ORDER
from lrm_safety_deliberation.models import PRIMARY
from lrm_safety_deliberation.plotting import MODEL_LABELS, register_fonts

MODELS = PRIMARY
SPLITS = {"ASR": "asr", "ORR": "orr"}

SIZE_SCALE = 1.2
GOOD_COLOR = "#1b9e77"
BAD_COLOR = "#e7298a"
PENDING_COLOR = "#bbbbbb"
ERR_COLOR = "#555555"
GRID_COLOR = "#d8d8d8"
GROUP_BG = "#f3f3f3"


# --------------------------------------------------------------------------- #
# Data aggregation (inlined from the sibling analysis scripts)
# --------------------------------------------------------------------------- #
def _cell_dir(model: str, mode: str, split_lower: str) -> Path:
    return paths.eval_results("k32_full_pool", model, mode, split_lower)


def cell_is_ready(model: str, mode: str, split_lower: str) -> tuple[bool, int]:
    cell = _cell_dir(model, mode, split_lower)
    counts = []
    for g in GUARDRAIL_ORDER:
        ap = cell / f"classifications_{g}.audit.json"
        if not ap.exists():
            return False, 0
        try:
            a = json.load(open(ap))
        except Exception:
            return False, 0
        counts.append(a.get("n_classified", 0))
    if len(set(counts)) != 1 or counts[0] == 0:
        return False, counts[0] if counts else 0
    return True, counts[0]


def per_prompt_maj_refuse_rate(model: str, mode: str, split_lower: str) -> dict[int, float]:
    """{gid: fraction of K=32 rollouts where >=3 of 4 guardrails voted refusal}."""
    votes_by_key: dict[tuple[int, int], list[bool]] = defaultdict(list)
    cell = _cell_dir(model, mode, split_lower)
    for g in GUARDRAIL_ORDER:
        p = cell / f"classifications_{g}.jsonl"
        with open(p) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                key = (r["global_pool_idx"], r["rollout_idx"])
                votes_by_key[key].append(bool(r["vote_refusal"]))
    by_gid: dict[int, list[bool]] = defaultdict(list)
    for (gid, _ri), votes in votes_by_key.items():
        if len(votes) != 4:
            raise RuntimeError(
                f"Bad vote count for {(gid, _ri)} in {model}/{mode}/{split_lower}: {len(votes)}"
            )
        by_gid[gid].append(sum(votes) >= 3)
    return {g: sum(votes) / len(votes) for g, votes in by_gid.items()}


def per_prompt_mean_refuse(model: str, mode: str, split_lower: str) -> dict[int, float]:
    """{gid: mean over K=32 rollouts of the mean 4-guardrail (soft) refusal vote}."""
    votes_by_key: dict[tuple[int, int], list[bool]] = defaultdict(list)
    cell = _cell_dir(model, mode, split_lower)
    for g in GUARDRAIL_ORDER:
        p = cell / f"classifications_{g}.jsonl"
        with open(p) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                key = (r["global_pool_idx"], r["rollout_idx"])
                votes_by_key[key].append(bool(r["vote_refusal"]))
    by_gid: dict[int, list[float]] = defaultdict(list)
    for (gid, _ri), votes in votes_by_key.items():
        if len(votes) != 4:
            raise RuntimeError(
                f"Bad vote count for {(gid, _ri)} in {model}/{mode}/{split_lower}: {len(votes)}"
            )
        by_gid[gid].append(sum(votes) / 4.0)
    return {g: mean(scores) for g, scores in by_gid.items()}


def delta_sign_summary(model: str, pool: str) -> dict | None:
    split_lower = SPLITS[pool]
    nt_ready, _ = cell_is_ready(model, "nothink", split_lower)
    th_ready, _ = cell_is_ready(model, "think", split_lower)
    if not (nt_ready and th_ready):
        return None
    q_nothink = per_prompt_maj_refuse_rate(model, "nothink", split_lower)
    q_think = per_prompt_maj_refuse_rate(model, "think", split_lower)
    common = sorted(set(q_nothink) & set(q_think))
    counts = {"good": 0, "bad": 0, "unchanged": 0}
    deltas = []
    for gid in common:
        raw_delta = q_think[gid] - q_nothink[gid]
        signed_delta = raw_delta if pool == "ASR" else -raw_delta
        deltas.append(signed_delta)
        if signed_delta > 0:
            counts["good"] += 1
        elif signed_delta < 0:
            counts["bad"] += 1
        else:
            counts["unchanged"] += 1
    n = len(common)
    pct = {k: 100.0 * v / max(1, n) for k, v in counts.items()}
    return {"n_prompts": n, "counts": counts, "pct": pct,
            "mean_good_delta_pp": float(np.mean(deltas) * 100.0) if deltas else 0.0}


def signed_good_deltas(model: str, pool: str) -> dict | None:
    split_lower = SPLITS[pool]
    nt_ready, _ = cell_is_ready(model, "nothink", split_lower)
    th_ready, _ = cell_is_ready(model, "think", split_lower)
    if not (nt_ready and th_ready):
        return None
    q_nothink = per_prompt_mean_refuse(model, "nothink", split_lower)
    q_think = per_prompt_mean_refuse(model, "think", split_lower)
    common = sorted(set(q_nothink) & set(q_think))
    if not common:
        return None
    nt = np.array([q_nothink[g] for g in common], dtype=float)
    th = np.array([q_think[g] for g in common], dtype=float)
    raw_delta_pp = 100.0 * (th - nt)
    good_delta_pp = raw_delta_pp if pool == "ASR" else -raw_delta_pp
    return {
        "n_prompts": int(good_delta_pp.size),
        "mean_pp": float(good_delta_pp.mean()),
        "std_pp": float(good_delta_pp.std(ddof=1)),
        "median_pp": float(np.median(good_delta_pp)),
        "p25_pp": float(np.percentile(good_delta_pp, 25)),
        "p75_pp": float(np.percentile(good_delta_pp, 75)),
    }


# --------------------------------------------------------------------------- #
# Plotting (inlined panels)
# --------------------------------------------------------------------------- #
def draw_panel(ax, values, *, xmax, show_model_labels, legend_good, legend_bad, xlabel):
    bar_h = 0.50
    intra = 0.62
    inter = 1.55
    y_pos: dict[tuple[str, str], float] = {}
    group_top: dict[str, float] = {}
    group_bot: dict[str, float] = {}
    for i, model in enumerate(MODELS):
        y_asr = i * inter
        y_orr = y_asr + intra
        y_pos[(model, "ASR")] = y_asr
        y_pos[(model, "ORR")] = y_orr
        group_top[model] = y_asr
        group_bot[model] = y_orr
    total_y_max = (len(MODELS) - 1) * inter + intra + 0.5

    for model in MODELS:
        ytop = group_top[model] - bar_h * 0.60
        ybot = group_bot[model] + bar_h * 0.60
        ax.axhspan(ytop, ybot, color=GROUP_BG, zorder=0)
        if show_model_labels:
            ymid = (group_top[model] + group_bot[model]) / 2.0
            ax.text(-xmax - 0.6, ymid, MODEL_LABELS[model], ha="right", va="center",
                    fontsize=16 * SIZE_SCALE, fontweight="bold", color="#202020")

    legend_added = {"good": False, "bad": False}
    for model in MODELS:
        for pool in ("ASR", "ORR"):
            y = y_pos[(model, pool)]
            r = values[model][pool]
            if r is None:
                ax.barh(y, xmax * 1.92, height=bar_h, left=-xmax * 0.96,
                        color=PENDING_COLOR, alpha=0.18, edgecolor="none", zorder=1)
                ax.text(0.0, y, "(pending)", va="center", ha="center",
                        fontsize=10.5 * SIZE_SCALE, color="#666666", style="italic",
                        zorder=3)
                continue
            good = r["better"]
            bad = r["worse"]
            unchanged = r["unchanged"]
            label_good = legend_good if not legend_added["good"] else None
            label_bad = legend_bad if not legend_added["bad"] else None
            ax.barh(y, good, height=bar_h, left=0, color=GOOD_COLOR,
                    edgecolor="black", linewidth=0.4, label=label_good, zorder=2)
            ax.barh(y, -bad, height=bar_h, left=0, color=BAD_COLOR,
                    edgecolor="black", linewidth=0.4, label=label_bad, zorder=2)
            if label_good:
                legend_added["good"] = True
            if label_bad:
                legend_added["bad"] = True
            in_bar_thresh = 5.0 if xmax <= 40 else 12.0
            if good >= in_bar_thresh:
                ax.text(good / 2.0, y, f"{good:.1f}%", va="center", ha="center",
                        fontsize=14 * SIZE_SCALE, color="white", fontweight="bold")
            else:
                ax.text(good + 0.4, y, f"{good:.1f}%", va="center", ha="left",
                        fontsize=14 * SIZE_SCALE, color=GOOD_COLOR, fontweight="bold")
            if bad >= in_bar_thresh:
                ax.text(-bad / 2.0, y, f"{bad:.1f}%", va="center", ha="center",
                        fontsize=14 * SIZE_SCALE, color="white", fontweight="bold")
            else:
                ax.text(-bad - 0.4, y, f"{bad:.1f}%", va="center", ha="right",
                        fontsize=14 * SIZE_SCALE, color=BAD_COLOR, fontweight="bold")
            ax.text(xmax * 0.98, y, f"{unchanged:.0f}% Unchanged", va="center",
                    ha="right",
                    fontsize=11.5 * SIZE_SCALE if xmax > 40 else 13.5 * SIZE_SCALE,
                    color="black", fontweight="bold")

    ax.set_yticks([])
    for model in MODELS:
        for pool in ("ASR", "ORR"):
            ax.text(-xmax + 0.5, y_pos[(model, pool)], pool, ha="left", va="center",
                    fontsize=14 * SIZE_SCALE, color="#202020", fontweight="bold")

    ax.axvline(0, color="black", lw=1.0)
    ax.set_xlim(-xmax, xmax)
    tick_step = 10.0 if xmax <= 50 else 20.0
    tick_max = np.floor(xmax / tick_step) * tick_step
    ticks = np.arange(-tick_max, tick_max + 0.1, tick_step)
    ticks = [t for t in ticks if abs(t) > 1e-9]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{abs(int(t))}%" for t in ticks])
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(1 if xmax <= 50 else 2))
    ax.set_xlabel(xlabel, fontsize=16 * SIZE_SCALE)
    ax.axvspan(-1.0, 1.0, color="#dddddd", alpha=0.40, zorder=0)
    ax.grid(True, axis="x", which="major", ls=":", alpha=0.45)
    ax.grid(True, axis="x", which="minor", ls=":", alpha=0.15)
    ax.set_ylim(-1.0, total_y_max)
    ax.invert_yaxis()

    handles, labels = ax.get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: 0 if labels[i] == legend_bad else 1)
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=2, frameon=False,
              fontsize=14 * SIZE_SCALE, handletextpad=0.5, columnspacing=1.5)


def draw_mean_std_panel(ax, stats, *, xmax, show_model_labels):
    bar_h = 0.50
    intra = 0.62
    inter = 1.55
    y_pos: dict[tuple[str, str], float] = {}
    group_top: dict[str, float] = {}
    group_bot: dict[str, float] = {}
    for i, model in enumerate(MODELS):
        y_asr = i * inter
        y_orr = y_asr + intra
        y_pos[(model, "ASR")] = y_asr
        y_pos[(model, "ORR")] = y_orr
        group_top[model] = y_asr
        group_bot[model] = y_orr
    total_y_max = (len(MODELS) - 1) * inter + intra + 0.5

    ax.axvspan(0, xmax, color=GOOD_COLOR, alpha=0.055, zorder=0)
    ax.axvspan(-xmax, 0, color=BAD_COLOR, alpha=0.055, zorder=0)

    for model in MODELS:
        ytop = group_top[model] - bar_h * 0.60
        ybot = group_bot[model] + bar_h * 0.60
        ax.axhspan(ytop, ybot, color=GROUP_BG, zorder=0)
        if show_model_labels:
            ymid = (group_top[model] + group_bot[model]) / 2.0
            ax.text(-xmax - 0.6, ymid, MODEL_LABELS[model], ha="right", va="center",
                    fontsize=16 * SIZE_SCALE, fontweight="bold", color="#202020")

    for model in MODELS:
        for pool in ("ASR", "ORR"):
            y = y_pos[(model, pool)]
            s = stats[model][pool]
            if s is None:
                ax.text(0, y, "pending", ha="center", va="center", color="#666666")
                continue
            mn = s["mean_pp"]
            std = s["std_pp"]
            color = GOOD_COLOR if mn >= 0 else BAD_COLOR
            ax.errorbar(mn, y, xerr=std, fmt="o", markersize=7.8, color=color,
                        ecolor=ERR_COLOR, elinewidth=1.8, capsize=4, capthick=1.4,
                        zorder=3)
            label_y = y - 0.22 if pool == "ASR" else y + 0.22
            if mn < 0:
                label_x = min(mn - 2.0, -8.0)
                label_ha = "right"
            else:
                label_x = max(mn + 2.0, 8.0)
                label_ha = "left"
            ax.text(label_x, label_y, f"{mn:+.1f} +/- {std:.1f} pp", ha=label_ha,
                    va="center", fontsize=10.7 * SIZE_SCALE, color=color,
                    fontweight="bold")

    ax.set_yticks([])
    for model in MODELS:
        for pool in ("ASR", "ORR"):
            ax.text(-xmax + 0.8, y_pos[(model, pool)], pool, ha="left", va="center",
                    fontsize=14 * SIZE_SCALE, color="#202020", fontweight="bold")

    ax.axvline(0, color="black", lw=1.1, zorder=1)
    ax.set_xlim(-xmax, xmax)
    tick_step = 20.0
    tick_max = np.floor(xmax / tick_step) * tick_step
    ticks = np.arange(-tick_max, tick_max + 0.1, tick_step)
    ticks = [t for t in ticks if abs(t) > 1e-9]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{abs(int(t))}" for t in ticks])
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(2))
    ax.set_xlabel(
        r"$\leftarrow$ Undesired          Mean $\Delta$ Refusal Rate (pp)          Desired $\rightarrow$",
        fontsize=14.5 * SIZE_SCALE,
    )
    ax.grid(True, axis="x", which="major", ls=":", color=GRID_COLOR, alpha=0.9)
    ax.grid(True, axis="x", which="minor", ls=":", color=GRID_COLOR, alpha=0.18)
    ax.set_axisbelow(True)
    ax.set_ylim(-1.0, total_y_max)
    ax.invert_yaxis()


def main() -> None:
    register_fonts()

    delta_values: dict[str, dict[str, dict | None]] = {}
    mean_values: dict[str, dict[str, dict | None]] = {}
    for model in MODELS:
        delta_values[model] = {}
        mean_values[model] = {}
        for pool in ("ASR", "ORR"):
            d = delta_sign_summary(model, pool)
            delta_values[model][pool] = None if d is None else {
                "better": d["pct"]["good"],
                "worse": d["pct"]["bad"],
                "unchanged": d["pct"]["unchanged"],
            }
            mean_values[model][pool] = signed_good_deltas(model, pool)

    paths.figure_out("fig_delta_sign_mean_std_sidebyside_k32.data.json").write_text(
        json.dumps({"delta_sign": delta_values, "mean_std": mean_values}, indent=2)
    )

    # Echo for verification.
    print("\n[delta sign] good / bad / unchanged (% of prompts):")
    for m in MODELS:
        for pool in ("ASR", "ORR"):
            v = delta_values[m][pool]
            if v is None:
                print(f"  {m:18s} {pool}  pending")
            else:
                print(f"  {m:18s} {pool}  good={v['better']:6.3f}  "
                      f"bad={v['worse']:6.3f}  unchanged={v['unchanged']:6.2f}")
    print("\n[mean +/- std signed delta (pp)]:")
    for m in MODELS:
        for pool in ("ASR", "ORR"):
            s = mean_values[m][pool]
            if s is None:
                print(f"  {m:18s} {pool}  pending")
            else:
                print(f"  {m:18s} {pool}  mean={s['mean_pp']:+7.3f}  "
                      f"std={s['std_pp']:7.3f}  n={s['n_prompts']}")

    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["TeX Gyre Termes"],
        "mathtext.fontset": "stix",
        "font.size": 15 * SIZE_SCALE,
        "axes.labelsize": 21 * SIZE_SCALE,
        "legend.fontsize": 11.5 * SIZE_SCALE,
        "xtick.labelsize": 12 * SIZE_SCALE,
        "ytick.labelsize": 12 * SIZE_SCALE,
    })

    fig, axes = plt.subplots(1, 2, figsize=(10.5 * 0.85 * 2.0 * 0.9, 4.6))
    draw_panel(axes[0], delta_values, xmax=90.0, show_model_labels=True,
               legend_good="Good Change", legend_bad="Bad Change",
               xlabel=r"$\leftarrow$ Bad Change          % of Prompts          Good Change $\rightarrow$")
    draw_mean_std_panel(axes[1], mean_values, xmax=80.0, show_model_labels=False)

    fig.subplots_adjust(wspace=0.16)
    fig.tight_layout()

    for ext in ("pdf", "png"):
        p = paths.figure_out(f"fig_delta_sign_mean_std_sidebyside_k32.{ext}")
        fig.savefig(p, dpi=220, bbox_inches="tight")
        print(f"wrote {p}", flush=True)


if __name__ == "__main__":
    main()
