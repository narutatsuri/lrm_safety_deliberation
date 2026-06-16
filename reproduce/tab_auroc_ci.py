#!/usr/bin/env python3
"""Reproduce Table ``tab:auroc_ci`` — first-token probe AUROC / BAcc by probe type.

Rep. rows: the first-token h_0 probe (StandardScaler -> PCA(100) -> LogReg
C=0.03, balanced; 5-fold OOF; per-benchmark held-out Youden for BAcc), read at
k=20 from the bestpipe cache (the same artifact that backs fig:prefill_decision).
The ± is the half-width of the 95% bootstrap CI stored in the cache.

Text rows: the first-thinking-token TF-IDF control. As in the original table
generator, these near-chance control values are carried verbatim (they are a
baseline, not a headline result; recompute with the per-class-match script if
desired).

Inputs : artifacts/experiments/refusal_cliff/results/kshop_K20_user_content/bestpipe_pca100_c003.json
Output : figures/tab_auroc_ci.tex

Original research script: scripts/neurips_final/build_kshop_bestpipe_table.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lrm_safety_deliberation import paths
from lrm_safety_deliberation.io import read_json
from lrm_safety_deliberation.models import PRIMARY
from lrm_safety_deliberation.plotting import MODEL_LABELS

BESTPIPE = paths.artifact(
    "experiments", "refusal_cliff", "results", "kshop_K20_user_content",
    "bestpipe_pca100_c003.json",
)

# TF-IDF first-thinking-token control rows (verbatim, as in the original generator).
TEXT_ROWS = {
    "Qwen3-8B":        r"& Text & $0.492\,{\scriptstyle \pm\,0.042}$ & $0.502\,{\scriptstyle \pm\,0.005}$ & $0.502\,{\scriptstyle \pm\,0.019}$ & $0.502\,{\scriptstyle \pm\,0.002}$ & $0.499\,{\scriptstyle \pm\,0.017}$ & $0.502\,{\scriptstyle \pm\,0.002}$ \\",
    "Olmo-3-7B-Think": r"& Text & $0.500\,{\scriptstyle \pm\,0.000}$ & $0.500\,{\scriptstyle \pm\,0.000}$ & $0.500\,{\scriptstyle \pm\,0.000}$ & $0.500\,{\scriptstyle \pm\,0.000}$ & $0.500\,{\scriptstyle \pm\,0.000}$ & $0.500\,{\scriptstyle \pm\,0.000}$ \\",
    "Phi-4-reasoning": r"& Text & $0.561\,{\scriptstyle \pm\,0.022}$ & $0.537\,{\scriptstyle \pm\,0.011}$ & $0.502\,{\scriptstyle \pm\,0.014}$ & $0.516\,{\scriptstyle \pm\,0.004}$ & $0.515\,{\scriptstyle \pm\,0.013}$ & $0.518\,{\scriptstyle \pm\,0.004}$ \\",
    "GPT-OSS-20B":     r"& Text & $0.678\,{\scriptstyle \pm\,0.031}$ & $0.671\,{\scriptstyle \pm\,0.025}$ & $0.619\,{\scriptstyle \pm\,0.013}$ & $0.597\,{\scriptstyle \pm\,0.009}$ & $0.601\,{\scriptstyle \pm\,0.012}$ & $0.592\,{\scriptstyle \pm\,0.008}$ \\",
}

HEADER = r"""\begin{table}[t]
\centering
\small
\caption{
  First-token probe AUROC and balanced accuracy
  (BAcc) values decomposed by probe type.
  Rep.\ uses $\mathbf{h}_0$, and Text uses the first thinking-token TF-IDF feature.
  A linear probe on $\mathbf{h}_0$ predicts the final refusal/compliance outcome, while text-only controls have far less predictive power.
}
\vspace{1mm}
\label{tab:auroc_ci}
\setlength{\tabcolsep}{3.5pt}
\resizebox{\linewidth}{!}{%
\begin{NiceTabular}{llcccccc}
\toprule
\Block{2-1}{\textbf{Model}}
& \Block{2-1}{\textbf{Probe}}
& \Block{1-2}{\textbf{Harmful}}
&
& \Block{1-2}{\textbf{Benign}}
&
& \Block{1-2}{\textbf{Pooled}}
& \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}
&
& \textbf{AUROC} & \textbf{BAcc}
& \textbf{AUROC} & \textbf{BAcc}
& \textbf{AUROC} & \textbf{BAcc} \\
\midrule"""

FOOTER = r"""\bottomrule
\end{NiceTabular}%
}
\end{table}"""


def _pm(mean: float, lo: float, hi: float) -> str:
    return f"${mean:.3f}\\,{{\\scriptstyle \\pm\\,{(hi - lo) / 2:.3f}}}$"


def main() -> None:
    data = read_json(BESTPIPE)
    rows = []
    for i, key in enumerate(PRIMARY):
        cell = data[key]["20"]  # k=20 == first thinking-token readout
        vals = []
        for grp in ("harmful", "benign", "pooled"):
            g = cell[grp]
            vals.append(_pm(g["auroc"], g["auroc_lo"], g["auroc_hi"]))
            vals.append(_pm(g["bacc"], g["bacc_lo"], g["bacc_hi"]))
        rows.append(f"\\Block{{2-1}}{{{MODEL_LABELS[key]}}}\n& Rep. & " + " & ".join(vals) + r" \\")
        rows.append(TEXT_ROWS[key])
        if i < len(PRIMARY) - 1:
            rows.append(r"\midrule")

    out = paths.figure_out("tab_auroc_ci.tex")
    out.write_text(HEADER + "\n" + "\n".join(rows) + "\n" + FOOTER + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
