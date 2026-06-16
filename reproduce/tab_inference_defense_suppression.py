#!/usr/bin/env python3
"""Reproduce Table ``tab:inference_defense_suppression`` — inference-defense block.

Per (model, method) we report the two columns the paper shows:
  * Osc.       = average number of stance oscillations per reasoning trace
                 (mean over audited traces of ``len(salient_transitions(labels,1,1))``;
                 traces with >120 chunks are excluded as degenerate loops).
  * Meaningful = average number of oscillations that *significantly* shift the
                 final-answer comply rate (Holm-corrected Fisher exact). This is
                 ``Osc. x meaningful_share``, where the per-cell meaningful share
                 comes from the K=100 cut-replay significance cache.

Defense cells are shown as the value with the % change versus Base
(down = suppressed). This reproduces the Base row + the inference-defense block
(PSR, SafeRemind, SafePath-ZS) of the paper table; the training-defense block of
the combined table is produced by the analogous computation over
``m4_training_defense_majority3.jsonl`` + the training K=100 cut data (full tier).

Inputs : artifacts/experiments/causal_cuts/rebuttal/{m4_majority3,m4_inference_defense_majority3}.jsonl
         artifacts/experiments/causal_cuts/plots/finding4/f4_metrics.json   (meaningful share)
Output : figures/tab_inference_defense_suppression.tex

Original research scripts: experiments/causal_cuts/plots/{plot_finding4.py,_dump_f4_decomp_values.py}
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from lrm_safety_deliberation import paths
from lrm_safety_deliberation.io import iter_jsonl, read_json
from lrm_safety_deliberation.models import PRIMARY
from lrm_safety_deliberation.plotting import MODEL_LABELS

REB = paths.artifact("experiments", "causal_cuts", "rebuttal")
F4_METRICS = paths.artifact("experiments", "causal_cuts", "plots", "finding4", "f4_metrics.json")

METHODS = ["PSR", "SafeRemind", "SafePath-ZS"]          # paper row order
DEF2KEY = {"SafePath-ZS": "safepath_zs", "SafeRemind": "saferemind", "PSR": "psr"}
SHARE_COND = {"SafePath-ZS": "SafePath", "SafeRemind": "SafeRemind", "PSR": "PSR"}  # f4_metrics cond names
A2M = {"Olmo3-7B-Think": "Olmo-3-7B-Think"}            # base-cell -> registry name
LOOP = 120


# --- stance-oscillation counting (verbatim from generate_cuts.py) -----------
def _segments_carry_forward(labels):
    n = len(labels)
    if n == 0:
        return []
    filled, last = list(labels), 0
    for i in range(n):
        if filled[i] == 0:
            filled[i] = last
        else:
            last = filled[i]
    segs, sign, start, length = [], filled[0], 0, 1
    for i in range(1, n):
        if filled[i] == sign:
            length += 1
        else:
            segs.append((sign, start, i - 1, length))
            sign, start, length = filled[i], i, 1
    segs.append((sign, start, n - 1, length))
    return segs


def _salient_transitions(labels, l_pre=1, l_min=1):
    segs = _segments_carry_forward(labels)
    out = []
    for i in range(len(segs) - 1):
        a, b = segs[i], segs[i + 1]
        if a[0] == 0 or b[0] == 0 or a[0] * b[0] >= 0:
            continue
        if a[3] < l_pre or b[3] < l_min:
            continue
        out.append(i)
    return out


def _model_of(cell):
    for method, dk in DEF2KEY.items():
        if cell.startswith(dk + "-"):
            return method, cell[len(dk) + 1:]
    return None, cell


def avg_osc_per_cell():
    """Mean # salient oscillations per audited trace, for Base + inference cells."""
    osc = defaultdict(list)

    def feed(path, is_base):
        for d in iter_jsonl(path):
            lab = d.get("labels", [])
            if not lab or any(x is None for x in lab):
                continue
            if int(d.get("n_chunks", len(lab))) > LOOP:
                continue
            if is_base:
                model, cond = A2M.get(d["cell"], d["cell"]), "Base"
            else:
                method, mm = _model_of(d["cell"])
                if not method:
                    continue
                # key by the f4_metrics cond name so osc and share align
                model, cond = A2M.get(mm, mm), SHARE_COND[method]
            osc[(model, cond)].append(len(_salient_transitions(lab)))

    feed(REB / "m4_majority3.jsonl", True)
    feed(REB / "m4_inference_defense_majority3.jsonl", False)
    return {k: float(np.mean(v)) for k, v in osc.items()}


# --- table formatting (matches the \fall/\rise macro spec) ------------------
def _delta_cell(value, base):
    """\\fall{i}{v}{p%} (down) or \\rise{i}{v}{p%} (up) vs Base.

    p is the integer % change; the colour-shade index i = max(8, round(0.5*p))
    is computed from that integer (matching the original macro spec).
    """
    pct = 100 * (value - base) / base if base else 0.0
    p = round(abs(pct))
    i = max(8, round(0.5 * p))
    macro = "rise" if value >= base else "fall"
    return f"\\{macro}{{{i}}}{{{value:.2f}}}{{{p}\\%}}"


def main() -> None:
    osc = avg_osc_per_cell()
    share = {tuple(k.split("|", 1)): v["share"] for k, v in read_json(F4_METRICS).items()}

    def good(model, cond):  # Meaningful = avg_osc * meaningful_share
        return osc.get((model, cond), 0.0) * (share.get((model, cond), 0.0) / 100.0)

    L = [r"\begin{NiceTabular}[color-inside]{clcccccccc}", r"  \toprule",
         "  & & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{\textbf{{{MODEL_LABELS[m]}}}}}" for m in PRIMARY) + r" \\",
         "  " + " ".join(rf"\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}" for i in range(len(PRIMARY))),
         "  & \\textbf{Method} & " + " & ".join("Osc. & Meaningful" for _ in PRIMARY) + r" \\",
         r"  \midrule"]

    base_cells = []
    for m in PRIMARY:
        base_cells.append(f"\\basecell{{{osc.get((m, 'Base'), 0.0):.2f}}}")
        base_cells.append(f"\\basecell{{{good(m, 'Base'):.2f}}}")
    L.append(r"  & \cellcolor{baseGray}Base & " + " & ".join(base_cells) + r" \\")
    L.append(r"  \midrule")

    L.append(r"  \Block[fill=gray!8]{3-1}{\rotatebox{90}{\textbf{Inf.}}}")
    for method in METHODS:
        cond = SHARE_COND[method]
        cells = []
        for m in PRIMARY:
            cells.append(_delta_cell(osc.get((m, cond), 0.0), osc.get((m, "Base"), 0.0)))
            cells.append(_delta_cell(good(m, cond), good(m, "Base")))
        L.append(f"    & {method} & " + " & ".join(cells) + r" \\")
    L += [r"  \bottomrule", r"\end{NiceTabular}"]

    out = paths.figure_out("tab_inference_defense_suppression.tex")
    out.write_text("\n".join(L) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
