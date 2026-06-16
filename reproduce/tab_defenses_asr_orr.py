#!/usr/bin/env python3
"""Reproduce Table ``tab:defenses_asr_orr`` — ASR / ORR for every defense cell
relative to its base model.

ASR = comply rate on harmful prompts (lower is safer); ORR = refusal rate on
benign prompts (lower is more useful).  Emits Markdown + LaTeX + JSON; the JSON
is also consumed by ``fig_defense_pareto.py``.

  --vote majority  (default)  : 4-guardrail majority vote, ties -> comply
  --vote soft                 : mean refusal_votes / 4

Inputs : artifacts/eval_results/{neurips_final,defenses_eval}/.../__fullpipe__/<asr|orr>/.../summary.json
Output : figures/defense_asr_orr_table[suffix].{md,tex,json}

Original research script: scripts/defenses_eval/aggregate_asr_orr_table.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lrm_safety_deliberation import paths
from lrm_safety_deliberation.behavioral import cell_asr_orr
from lrm_safety_deliberation.models import PRIMARY

# Paper presentation order (Base first, then the six training defenses).
METHODS = ["Base", "STAR1", "SafeKey", "R1ACT", "ThinkSafe", "RAPO", "STAIR"]


def _fmt(x, base_x=None):
    if x is None:
        return "—"
    s = f"{x * 100:.1f}"
    if base_x is None or base_x == x:
        return s
    delta = (x - base_x) * 100
    return f"{s} ({'↓' if delta < 0 else '↑'}{abs(delta):.1f})"


def build(vote: str):
    cell = {m: {} for m in METHODS}
    for method in METHODS:
        for base in PRIMARY:
            asr, orr = cell_asr_orr(method, base, vote=vote)
            cell[method][base] = {"asr": asr, "orr": orr}

    rule = ("4-guardrail **soft** vote: score = refusal_votes/4 ∈ {0,.25,.5,.75,1}, mean across pool."
            if vote == "soft" else
            "4-guardrail **majority** vote, ties → comply.")
    md = ["# Defense ASR vs ORR — comparison to base model\n",
          "**ASR** = comply rate on harmful prompts. Lower is safer.\n",
          "**ORR** = refusal rate on benign prompts. Lower is more useful.\n",
          "Δ vs Base shown in parens (↓ = improvement for ASR; ↑ = regression for ORR). " + rule + "\n"]
    for split_name, getter in (("ASR (harmful)", "asr"), ("ORR (benign)", "orr")):
        md.append(f"\n## {split_name}\n")
        md.append("| Method | " + " | ".join(PRIMARY) + " |")
        md.append("|---|" + "---|" * len(PRIMARY))
        for method in METHODS:
            row = [method]
            for base in PRIMARY:
                v = cell[method][base][getter]
                bv = cell["Base"][base][getter]
                row.append(_fmt(v) if method == "Base" else _fmt(v, bv))
            md.append("| " + " | ".join(row) + " |")

    tex = []
    for split_name, getter in (("ASR", "asr"), ("ORR", "orr")):
        tex.append(f"% {split_name}")
        tex.append(r"\begin{tabular}{l" + "c" * len(PRIMARY) + r"}")
        tex.append(r"\toprule")
        tex.append("Method & " + " & ".join(PRIMARY) + r" \\")
        tex.append(r"\midrule")
        for method in METHODS:
            cells = []
            for base in PRIMARY:
                v = cell[method][base][getter]
                bv = cell["Base"][base][getter]
                if v is None:
                    cells.append("---")
                elif method == "Base":
                    cells.append(f"{v * 100:.1f}")
                else:
                    delta = (v - bv) * 100 if bv is not None else None
                    if delta is None:
                        cells.append(f"{v * 100:.1f}")
                    else:
                        sign = r"$\downarrow$" if delta < 0 else r"$\uparrow$"
                        cells.append(f"{v * 100:.1f} ({sign}{abs(delta):.1f})")
            tex.append(f"{method} & " + " & ".join(cells) + r" \\")
        tex.append(r"\bottomrule")
        tex.append(r"\end{tabular}")
        tex.append("")
    return "\n".join(md) + "\n", "\n".join(tex), cell


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vote", choices=["majority", "soft"], default="majority")
    ap.add_argument("--suffix", default=None)
    args = ap.parse_args()
    suffix = args.suffix if args.suffix is not None else ("" if args.vote == "majority" else "_soft")

    md, tex, cell = build(args.vote)
    base = f"defense_asr_orr_table{suffix}"
    paths.figure_out(f"{base}.md").write_text(md)
    paths.figure_out(f"{base}.tex").write_text(tex)
    paths.figure_out(f"{base}.json").write_text(json.dumps(cell, indent=2))
    print("wrote", paths.figure_out(f"{base}.{{md,tex,json}}"))


if __name__ == "__main__":
    main()
