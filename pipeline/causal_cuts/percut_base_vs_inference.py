#!/usr/bin/env python3
"""PER-CUT base-vs-inference comparison, in the SAME unit as the m4_cut_meaning
waffle (denominator = detected cuts, not traces). The waffle reports base
per-cut "causal" %: GPT-OSS 29, Qwen 8, Olmo 13, Phi-4 20.

My analyze_m4_inference_defenses budge/sig are PER-TRACE (frac of traces with >=1
cut that have >=1 wavering/sig cut) — a looser denominator. This script recomputes
the inference-vs-base comparison per CUT so it is directly comparable to the waffle.

For each unit we report (pooling both splits):
  n_cuts        — detected cuts (each is one row)
  wav%          — Holm-FREE: frac of cuts that waver (not locked); locked = both
                  comply_pre and comply_post in {<=0.05 or >=0.95}
  rawsig%       — Holm-FREE: frac of cuts with Fisher p < 0.05 (uncorrected)
  holmsig%      — Holm across the unit's cuts (matches the waffle "causal" metric)

Holm stringency scales with n_cuts, and base units (32 trace-rollouts) are far
larger than inference units (M=4), which biases base holmsig% DOWN — so the
Holm-free wav%/rawsig% are the cleanest base-vs-inference comparison.
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.stats import fisher_exact

# Provenance: copied verbatim (logic unchanged) from the original research repo
#   experiments/causal_cuts/rebuttal/percut_base_vs_inference.py
# load_cut_comply_rates is imported from the original analyze_m4_inference_defenses
# via ROOT; the 4-guardrail vote + model registry now also live in the shared
# `lrm_safety_deliberation` library (lrm_safety_deliberation.guardrails / lrm_safety_deliberation.models). Set
# LRM_SAFETY_ARTIFACTS to relocate the artifact tree.
ROOT = Path(os.environ.get("LRM_SAFETY_ARTIFACTS", Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(ROOT))
from experiments.causal_cuts.rebuttal.analyze_m4_inference_defenses import load_cut_comply_rates  # noqa

BASE_CUTS = ROOT / "experiments/causal_cuts/eval_results/causal_cuts"
INF_CUTS = ROOT / "experiments/causal_cuts/eval_results/causal_cuts_inference"
OUT = ROOT / "experiments/causal_cuts/rebuttal/m4_percut_base_vs_inference.md"

BASE_MODELS = ["GPT-OSS-20B", "Qwen3-8B", "Olmo3-7B-Think", "Phi-4-reasoning"]
INF_MODELS = ["GPT-OSS-20B", "Qwen3-8B", "Olmo-3-7B-Think", "Phi-4-reasoning"]
DEFENSES = ["safepath_zs", "saferemind", "psr"]
SPLITS = ["benign", "harmful"]


def is_locked(pre, post):
    ex = lambda v: v <= 0.05 or v >= 0.95
    return ex(pre) and ex(post)


def collect_cuts(cuts_dir, cell, in_name, suffix):
    """Pool per-cut entries over both splits for one unit; return list of dicts."""
    out = []
    for split in SPLITS:
        pc = load_cut_comply_rates(cuts_dir, cell, split, in_name, suffix)
        if not pc:
            continue
        for v in pc.values():
            out.append(v)
    return out


def summarize(cuts):
    n = len(cuts)
    if n == 0:
        return dict(n_cuts=0, wav=None, rawsig=None, holmsig=None)
    ps = np.empty(n)
    wav = 0
    for i, c in enumerate(cuts):
        table = [[c["n_comp_pre"], c["n_ref_pre"]], [c["n_comp_post"], c["n_ref_post"]]]
        try:
            _, p = fisher_exact(table)
        except Exception:
            p = 1.0
        ps[i] = p
        if not is_locked(c["comply_pre"], c["comply_post"]):
            wav += 1
    rawsig = int((ps < 0.05).sum())
    # Holm across the unit
    order = np.argsort(ps)
    sig = np.zeros(n, bool)
    for rank, idx in enumerate(order):
        if ps[idx] <= 0.05 / (n - rank):
            sig[idx] = True
        else:
            break
    return dict(n_cuts=n, wav=wav / n, rawsig=rawsig / n, holmsig=int(sig.sum()) / n)


def fmt(v, p=3):
    return "—" if v is None else f"{v:.{p}f}"


def main():
    rows = []  # (label, summary)
    # base per model
    print("[base] computing per-model per-cut ...", flush=True)
    for m in BASE_MODELS:
        s = summarize(collect_cuts(BASE_CUTS, m, "rollouts_segment_m4.jsonl", "m4"))
        rows.append((f"base/{m}", s))
        print(f"  base/{m}: {s}", flush=True)
    # inference per defense x model, and pooled per model (across defenses)
    print("[inference] computing ...", flush=True)
    for m in INF_MODELS:
        pooled = []
        for d in DEFENSES:
            cell = f"{d}-{m}"
            cuts = collect_cuts(INF_CUTS, cell, "rollouts_segment_repro.jsonl", "repro")
            pooled += cuts
            s = summarize(cuts)
            rows.append((f"inf/{d}-{m}", s))
            print(f"  inf/{cell}: {s}", flush=True)
        sp = summarize(pooled)
        rows.append((f"inf/POOLED-{m}", sp))
        print(f"  inf/POOLED-{m}: {sp}", flush=True)

    lines = ["# Per-CUT base-vs-inference (same unit as the m4_cut_meaning waffle)\n",
             "Denominator = detected cuts (NOT traces). wav%/rawsig% are Holm-free; "
             "holmsig% applies Holm across the unit (matches the waffle 'causal' metric). "
             "Base units pool 32 trace-rollouts so their Holm is far more stringent than "
             "inference (M=4) — compare the Holm-free wav%/rawsig% for a fair read.\n",
             "Waffle reference (base, per-cut causal): GPT-OSS 0.29, Qwen 0.08, Olmo 0.13, Phi-4 0.20.\n",
             "| unit | n_cuts | wav% (Holm-free) | rawsig% (p<.05) | holmsig% (waffle-style) |",
             "|---|--:|--:|--:|--:|"]
    for label, s in rows:
        lines.append(f"| {label} | {s['n_cuts']} | {fmt(s['wav'])} | {fmt(s['rawsig'])} | {fmt(s['holmsig'])} |")
    OUT.write_text("\n".join(lines) + "\n")
    print(f"[write] {OUT}")


if __name__ == "__main__":
    main()
