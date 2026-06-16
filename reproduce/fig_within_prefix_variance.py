#!/usr/bin/env python3
"""Reproduce Figure ``fig:within_prefix_variance`` — within-prefix continuation variance (K=32).

ContinuationVariance(B) = E_m[4 q_{m,B}(1-q_{m,B})], where q_{m,B} is the refusal
rate over the N continuations sampled from prefix m at budget B%. Plotted vs cut%
for B in {20,40,60,80,100}, averaged across the ASR and ORR pools, plus a B=0
"unconditioned" point: fix NO prefix and resample the entire thinking trace
(the K=32 ``think`` full-trace pool), treating each prompt as a single empty
prefix with K=32 continuations:
    p_hat(prompt) = fraction of the K rollouts that majority-refuse (>=3/4 guards)
    var(prompt)   = 4 * p_hat * (1 - p_hat)
    B=0 variance  = mean over prompts, averaged across the ASR and ORR pools.
The dashed "Random" line is 4*(N-1)/N*0.25 = 0.875 (N=8).

B=0 CIs use the ORIGINAL Python ``random.Random`` percentile bootstrap (n_boot=1000,
seed=20260518+0) inlined verbatim so the numbers match bit-for-bit; the library's
numpy bootstrap uses a different RNG and would shift the CI half-widths.

Inputs :
  - artifacts/experiments/nested_branching_variance/k32_full/analysis/data/asr_orr_within_by_cut.json  (B>0)
  - artifacts/eval_results/k32_full_pool/<model>/think/<split>/classifications_<g>.jsonl  (B=0)
Output : figures/fig_within_pooled_k32full_full_noext_with_b0.{pdf,png,data.json}

Original research script:
  experiments/nested_branching_variance/k32_full/analysis/plot_within_pooled_k32full_full_noext_with_b0.py
(self-contained in the original.)
"""
from __future__ import annotations

import json
import random
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
from lrm_safety_deliberation.plotting import MODEL_COLORS, MODEL_LABELS, register_fonts

MODELS = PRIMARY
CUTS = [20, 40, 60, 80, 100]   # B > 100% dropped (original)
B0 = 0                          # leftmost (unconditioned) point

N_PER_PREFIX = 8
WITHIN_BASELINE = 4.0 * (N_PER_PREFIX - 1) / N_PER_PREFIX * 0.25  # = 0.875

# --- B=0 computation constants (match aggregate_k32_nb.py / the original) ---
N_BOOT = 1000
SEED = 20260518


def data_json() -> Path:
    return paths.artifact(
        "experiments", "nested_branching_variance", "k32_full",
        "analysis", "data", "asr_orr_within_by_cut.json",
    )


def _cell_dir(model: str, split: str) -> Path:
    return paths.eval_results("k32_full_pool", model, "think", split)


def _load_votes_k32(cell_dir: Path) -> dict[tuple[int, int], list[bool]]:
    """Read 4-guardrail bool votes per (gid, rollout_idx). {} on incomplete cell."""
    votes: dict[tuple[int, int], list[bool]] = defaultdict(list)
    for g in GUARDRAIL_ORDER:
        p = cell_dir / f"classifications_{g}.jsonl"
        if not p.exists():
            return {}
        with open(p) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                key = (r["global_pool_idx"], r["rollout_idx"])
                votes[key].append(bool(r["vote_refusal"]))
    return votes


def _majority_refuse(votes: list[bool]) -> bool | None:
    if len(votes) != 4:
        return None
    return sum(votes) >= 3   # >=3/4 = majority (tie -> comply)


def _bootstrap_mean_ci(xs: list[float], seed: int) -> tuple[float, float]:
    """(point, half_width) of percentile-95% CI via Python random.Random bootstrap.

    Inlined verbatim from the original generator so the B=0 CI half-widths match
    exactly (the library's numpy bootstrap is a different RNG)."""
    if not xs:
        return (float("nan"), 0.0)
    rng = random.Random(seed)
    n = len(xs)
    pt = sum(xs) / n
    samples = []
    for _ in range(N_BOOT):
        s = [xs[rng.randrange(n)] for _ in range(n)]
        samples.append(sum(s) / n)
    samples.sort()
    lo = samples[int(0.025 * N_BOOT)]
    hi = samples[int(0.975 * N_BOOT)]
    return (pt, (hi - lo) / 2.0)


def _b0_pool_var(model: str, split: str) -> tuple[float, float] | None:
    """B=0 unconditioned variance for one (model, pool)."""
    votes = _load_votes_k32(_cell_dir(model, split))
    if not votes:
        return None
    by_gid: dict[int, list[bool]] = defaultdict(list)
    for (gid, _ri), v in votes.items():
        mr = _majority_refuse(v)
        if mr is not None:
            by_gid[gid].append(mr)
    phats = [sum(refs) / len(refs) for refs in by_gid.values() if refs]
    if not phats:
        return None
    wvars = [4.0 * p * (1.0 - p) for p in phats]
    return _bootstrap_mean_ci(wvars, seed=SEED + B0)


def load_pooled(model: str, data: dict) -> dict | None:
    """For each cut, average within-prefix variance across (asr, orr); prepend B=0."""
    out = {"cuts": [], "w": [], "wlo": [], "whi": []}

    # --- B=0 point (from raw think rollouts) ---
    per_pool_b0: list[float] = []
    per_pool_b0_half: list[float] = []
    for sp in ("asr", "orr"):
        res = _b0_pool_var(model, sp)
        if res is None:
            continue
        per_pool_b0.append(res[0])
        per_pool_b0_half.append(res[1])
    if per_pool_b0:
        out["cuts"].append(B0)
        m0 = float(np.mean(per_pool_b0))
        h0 = float(np.mean(per_pool_b0_half))
        out["w"].append(m0)
        out["wlo"].append(m0 - h0)
        out["whi"].append(m0 + h0)

    # --- B>0 points (identical to original) ---
    cells = data.get(model, {})
    if not cells:
        return out if out["cuts"] else None
    for c in CUTS:
        per_pool = []
        per_pool_half = []
        for pool in ("ASR", "ORR"):
            v = cells.get(pool, {}).get(str(c))
            if v is None or v.get("within_var") is None:
                continue
            per_pool.append(v["within_var"])
            per_pool_half.append(v["within_half"])
        if not per_pool:
            continue
        out["cuts"].append(c)
        m = float(np.mean(per_pool))
        h = float(np.mean(per_pool_half))
        out["w"].append(m)
        out["wlo"].append(m - h)
        out["whi"].append(m + h)
    return out if out["cuts"] else None


def plot_lines(ax, series: dict[str, dict]) -> None:
    for m in MODELS:
        s = series.get(m)
        if not s or not s["cuts"]:
            continue
        xs = np.array(s["cuts"])
        ys = np.array(s["w"])
        lo = np.array(s["wlo"])
        hi = np.array(s["whi"])
        color = MODEL_COLORS[m]
        ax.fill_between(xs, lo, hi, color=color, alpha=0.18)
        ax.plot(xs, ys, "o-", color=color, lw=1.7, ms=11.0,
                markeredgewidth=0, alpha=0.75,
                label=MODEL_LABELS[m], zorder=3)
        y_err = np.vstack([ys - lo, hi - ys])
        ax.errorbar(xs, ys, yerr=y_err, fmt="none",
                    ecolor=color, elinewidth=0.9, capsize=3.5, capthick=0.9,
                    alpha=0.85, zorder=2)


def main() -> None:
    register_fonts()
    mpl.rcParams.update({
        "font.family":     "serif",
        "font.serif":      ["TeX Gyre Termes"],
        "mathtext.fontset": "stix",
        "font.size":       9   * 1.4 * 1.2,
        "axes.titlesize":  10  * 1.4 * 1.2,
        "axes.labelsize":  10  * 1.4 * 1.2,
        "legend.fontsize": 8.5 * 1.2 * 1.2,
        "xtick.labelsize": 8.5 * 1.4 * 1.2,
        "ytick.labelsize": 8.5 * 1.4 * 1.2,
    })
    with data_json().open() as f:
        data = json.load(f)["per_cell"]
    series = {m: load_pooled(m, data) for m in MODELS}

    fig, ax = plt.subplots(figsize=(5.775, 4.608))
    plot_lines(ax, series)

    ax.axhline(WITHIN_BASELINE, color="black", lw=1.1, ls="--", alpha=0.75,
               zorder=1, label="Random")

    ax.set_ylim(-0.03, 1.0)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.05))

    xticks = [B0] + CUTS
    ax.set_xticks(xticks)
    ax.set_xlim(xticks[0] - 5, xticks[-1] + 5)
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax.tick_params(axis="x", which="minor", length=0)
    ax.tick_params(axis="x", which="major", pad=1.5)
    ax.grid(True, which="major", ls=":", alpha=0.45)
    ax.grid(True, which="minor", ls=":", alpha=0.20)

    ax.set_ylabel("Variance across Continuations", fontsize=11 * 1.4 * 1.2)
    ax.set_xlabel("Percentage of Natural Thinking Trace (%)", fontsize=11 * 1.4 * 1.2)

    ax.legend(loc="center right", framealpha=0.92,
              fontsize=8.5 * 1.2 * 1.2, title_fontsize=9 * 1.2 * 1.2)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = paths.figure_out(f"fig_within_pooled_k32full_full_noext_with_b0.{ext}")
        fig.savefig(p, dpi=220, bbox_inches="tight")
        print(f"wrote {p}", flush=True)

    # Echo + persist the pooled series (the figure encodes these numbers).
    print("\nB=0 vs B=20 pooled continuation variance:")
    for m in MODELS:
        s = series.get(m)
        if not s:
            continue
        cuts = s["cuts"]
        b0 = s["w"][cuts.index(0)] if 0 in cuts else float("nan")
        b20 = s["w"][cuts.index(20)] if 20 in cuts else float("nan")
        print(f"  {m:18s} B=0={b0:.4f}  B=20={b20:.4f}")

    paths.figure_out("fig_within_pooled_k32full_full_noext_with_b0.data.json").write_text(
        json.dumps(series, indent=2)
    )


if __name__ == "__main__":
    main()
