#!/usr/bin/env python3
"""Reproduce Figure ``fig:no_think_to_think`` — no-think -> think flip summary (K=32).

Two side-by-side diverging-bar panels over the four primary models x {ASR, ORR}:

Left  (strict, majority-label flips): each prompt is labelled refuse/comply in
  each mode (refuse iff the K=32 majority-refuse rate > 0.5, where a rollout is a
  majority-refuse iff >=3/4 guardrails vote refusal). Bars show the % of prompts
  whose label flips to the safer side ("Flipped to Better", green) vs the worse
  side ("Flipped to Worse", red); the safer/worse direction is pool-specific.
Right (permissive, delta-sign): no per-mode 0.5 threshold. For each prompt take
  the signed change in majority-refuse rate (ASR: q_think-q_nothink; ORR: the
  negation); count it as a Good / Bad / Unchanged change by its sign.

Inputs : artifacts/eval_results/k32_full_pool/<model>/{think,nothink}/{asr,orr}/
           classifications_<guardrail>.{jsonl,audit.json}
Output : figures/fig_flip_summary_intro_maj_k32_sidebyside.{pdf,png,data.json}

Original research script:
  experiments/nested_branching_variance/k32_full/analysis/plot_flip_summary_maj_k32_sidebyside.py
Inlines logic from its siblings transition_table_k32.py (cell_is_ready),
plot_flip_summary_maj_k32_combined.py (per_prompt_maj_refuse_rate, transitions),
and plot_delta_sign_summary_maj_k32.py (delta_sign_summary).
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

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
GROUP_BG = "#f3f3f3"


# --------------------------------------------------------------------------- #
# Data aggregation (inlined from the sibling analysis scripts)
# --------------------------------------------------------------------------- #
def _cell_dir(model: str, mode: str, split_lower: str) -> Path:
    return paths.eval_results("k32_full_pool", model, mode, split_lower)


def cell_is_ready(model: str, mode: str, split_lower: str) -> tuple[bool, int]:
    """Ready iff all 4 guardrails present + agree on n_classified > 0."""
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
        by_gid[gid].append(sum(votes) >= 3)   # majority refuse, tie -> comply
    return {g: sum(votes) / len(votes) for g, votes in by_gid.items()}


def _classify(refuse_rate: float) -> str:
    return "refuse" if refuse_rate > 0.5 else "comply"


def transitions(model: str, pool: str) -> dict | None:
    split_lower = SPLITS[pool]
    nt_ready, _ = cell_is_ready(model, "nothink", split_lower)
    th_ready, _ = cell_is_ready(model, "think", split_lower)
    if not (nt_ready and th_ready):
        return None
    nt = per_prompt_maj_refuse_rate(model, "nothink", split_lower)
    th = per_prompt_maj_refuse_rate(model, "think", split_lower)
    common = sorted(set(nt) & set(th))
    counts = {("refuse", "refuse"): 0, ("refuse", "comply"): 0,
              ("comply", "refuse"): 0, ("comply", "comply"): 0}
    for g in common:
        counts[(_classify(nt[g]), _classify(th[g]))] += 1
    n = len(common)
    return {
        "n_prompts": n,
        "counts": {f"{a}_to_{b}": v for (a, b), v in counts.items()},
        "pct": {f"{a}_to_{b}": 100 * v / max(1, n) for (a, b), v in counts.items()},
    }


def majority_flip_summary(model: str, pool: str) -> dict | None:
    r = transitions(model, pool)
    if r is None:
        return None
    p = r["pct"]
    if pool == "ASR":
        better, worse = p["comply_to_refuse"], p["refuse_to_comply"]
    else:
        better, worse = p["refuse_to_comply"], p["comply_to_refuse"]
    unchanged = p["refuse_to_refuse"] + p["comply_to_comply"]
    return {"n_prompts": r["n_prompts"], "better": better, "worse": worse,
            "unchanged": unchanged}


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


# --------------------------------------------------------------------------- #
# Plotting (inlined diverging-bar panel)
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


def main() -> None:
    register_fonts()

    majority_values: dict[str, dict[str, dict | None]] = {}
    delta_values: dict[str, dict[str, dict | None]] = {}
    for model in MODELS:
        majority_values[model] = {}
        delta_values[model] = {}
        for pool in ("ASR", "ORR"):
            majority_values[model][pool] = majority_flip_summary(model, pool)
            d = delta_sign_summary(model, pool)
            delta_values[model][pool] = None if d is None else {
                "better": d["pct"]["good"],
                "worse": d["pct"]["bad"],
                "unchanged": d["pct"]["unchanged"],
            }

    # Echo + write data.json for numeric verification.
    print("\n[majority flip] better / worse / unchanged (% of prompts):")
    for m in MODELS:
        for pool in ("ASR", "ORR"):
            v = majority_values[m][pool]
            if v is None:
                print(f"  {m:18s} {pool}  pending")
            else:
                print(f"  {m:18s} {pool}  better={v['better']:6.3f}  "
                      f"worse={v['worse']:6.3f}  unchanged={v['unchanged']:6.2f}  "
                      f"n={v['n_prompts']}")
    print("\n[delta sign] good / bad / unchanged (% of prompts):")
    for m in MODELS:
        for pool in ("ASR", "ORR"):
            v = delta_values[m][pool]
            if v is None:
                print(f"  {m:18s} {pool}  pending")
            else:
                print(f"  {m:18s} {pool}  good={v['better']:6.3f}  "
                      f"bad={v['worse']:6.3f}  unchanged={v['unchanged']:6.2f}")

    paths.figure_out("fig_flip_summary_intro_maj_k32_sidebyside.data.json").write_text(
        json.dumps({"majority_flip": majority_values, "delta_sign": delta_values},
                   indent=2)
    )

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

    fig, axes = plt.subplots(1, 2, figsize=(10.5 * 0.85 * 2.0, 4.6))
    draw_panel(axes[0], majority_values, xmax=35.0, show_model_labels=True,
               legend_good="Flipped to Better", legend_bad="Flipped to Worse",
               xlabel=r"$\leftarrow$ Worse Flip          % of Prompts          Better Flip $\rightarrow$")
    draw_panel(axes[1], delta_values, xmax=90.0, show_model_labels=False,
               legend_good="Good Change", legend_bad="Bad Change",
               xlabel=r"$\leftarrow$ Bad Change          % of Prompts          Good Change $\rightarrow$")
    fig.subplots_adjust(wspace=0.16)
    fig.tight_layout()

    for ext in ("pdf", "png"):
        p = paths.figure_out(f"fig_flip_summary_intro_maj_k32_sidebyside.{ext}")
        fig.savefig(p, dpi=220, bbox_inches="tight")
        print(f"wrote {p}", flush=True)


if __name__ == "__main__":
    main()
