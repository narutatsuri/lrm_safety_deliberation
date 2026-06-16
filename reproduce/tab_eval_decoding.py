#!/usr/bin/env python3
"""Reproduce Table ``tab:eval_decoding`` — per-model sampling configuration.

Emitted directly from the model registry (``lrm_safety_deliberation.models``): temperature,
top-p, top-k, min-p, and the thinking-token budget for each of the four primary
models.  This is the single source of truth used by every generation step.

Output: figures/tab_eval_decoding.tex
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lrm_safety_deliberation import paths
from lrm_safety_deliberation.models import PRIMARY, get
from lrm_safety_deliberation.plotting import MODEL_LABELS


def _fmt(v) -> str:
    return "--" if v is None else (f"{v:g}")


def main() -> None:
    rows = []
    for name in PRIMARY:
        s = get(name)
        d = s.decoding
        rows.append((MODEL_LABELS[name], _fmt(d.temperature), _fmt(d.top_p),
                     _fmt(d.top_k), _fmt(d.min_p), str(s.max_new_tokens)))

    L = [r"\begin{tabular}{lccccc}", r"\toprule",
         r"Model & Temperature & Top-$p$ & Top-$k$ & Min-$p$ & Thinking budget \\",
         r"\midrule"]
    L += [f"{m} & {t} & {tp} & {tk} & {mp} & {mb} \\\\" for (m, t, tp, tk, mp, mb) in rows]
    L += [r"\bottomrule", r"\end{tabular}"]

    out = paths.figure_out("tab_eval_decoding.tex")
    out.write_text("\n".join(L) + "\n")
    print("wrote", out)
    print("\n".join(L))


if __name__ == "__main__":
    main()
