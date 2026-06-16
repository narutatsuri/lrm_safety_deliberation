#!/usr/bin/env python3
"""Reproduce Table ``tab:eval_datasets`` — the main evaluation benchmark suite.

The main body suite: ASR = WildJailbreak (2,000 adversarial-harmful) + FORTRESS
(500); ORR = OR-Bench-Hard (1,319) + FalseReject (1,187) + CoCoNot (379), plus
the over-refusal supplementary sets PHTest (2,077) + ORFuzzSet (1,788).

The five main-suite counts are computed from the shipped ``data/`` manifests; the
two supplementary counts are HuggingFace-hosted (not redistributed) and are taken
from those datasets' fixed sizes.

Output: figures/tab_eval_datasets.tex
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lrm_safety_deliberation import paths
from lrm_safety_deliberation.data import MAIN_ASR, MAIN_ORR, split_counts

LABELS = {
    "wildjailbreak": "WildJailbreak", "fortress": "FORTRESS",
    "or_bench": "OR-Bench-Hard", "falsereject": "FalseReject", "coconot": "CoCoNot",
}
# Over-refusal supplementary sets (HuggingFace-hosted; fixed sizes).
SUPP_ORR = [("PHTest", 2077), ("ORFuzzSet", 1788)]


def _fmt(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


def main() -> None:
    asr = split_counts("asr")
    orr = split_counts("orr")

    L = [r"\begin{NiceTabular}[color-inside]{clc}", r"\toprule",
         r"& Dataset & \# Examples \\", r"\midrule",
         r"\Block[fill=gray!8]{%d-1}{\textbf{ASR}}" % len(MAIN_ASR)]
    for b in MAIN_ASR:
        L.append(f"& {LABELS[b]} & {_fmt(asr[b])} \\\\")
    L.append(r"\midrule")
    n_orr = len(MAIN_ORR) + len(SUPP_ORR)
    L.append(r"\Block[fill=gray!8]{%d-1}{\textbf{ORR}}" % n_orr)
    for b in MAIN_ORR:
        L.append(f"& {LABELS[b]} & {_fmt(orr[b])} \\\\")
    L.append(r"\cmidrule(lr){2-3}")
    for name, n in SUPP_ORR:
        L.append(f"& {name} & {_fmt(n)} \\\\")
    L += [r"\bottomrule", r"\end{NiceTabular}"]

    out = paths.figure_out("tab_eval_datasets.tex")
    out.write_text("\n".join(L) + "\n")
    print("wrote", out, f"(ASR {sum(asr.values()):,} / ORR-main {sum(orr.values()):,})")


if __name__ == "__main__":
    main()
