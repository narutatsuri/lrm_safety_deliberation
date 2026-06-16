#!/usr/bin/env python3
"""Reproduce Table ``tab:base_asr_orr`` — base-model per-benchmark ASR / ORR.

Point estimates are the fractional (soft) four-guardrail success rates averaged
over the four rollouts (``soft_asr`` / ``soft_orr`` from each cell's summary);
``±`` is a 95% bootstrap CI half-width over the benchmark's prompts (1000
resamples, seed 42 — the soft per-prompt score is not Bernoulli, so a binomial
CI is invalid).  Emits the NiceTabular used in the paper appendix.

Inputs  : artifacts/eval_results/asr_orr_16k_K4/{asr_full,orr_full}/<model>/{summary.json,results.jsonl}
Output  : figures/tab_base_asr_orr.tex

Original research script: neurips/make_base_asr_orr_table_k4_soft.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from lrm_safety_deliberation import paths
from lrm_safety_deliberation.data import ASR_FULL, ORR_FULL
from lrm_safety_deliberation.io import read_concat_json, read_json
from lrm_safety_deliberation.models import PRIMARY
from lrm_safety_deliberation.plotting import MODEL_LABELS
from lrm_safety_deliberation.stats import bootstrap_mean_ci

TAG = "asr_orr_16k_K4"
N_BOOT = 1000
SEED = 42

# Paper display names per benchmark.
BENCH_LABELS = {
    "advbench": "AdvBench", "harmbench": "HarmBench", "strongreject": "StrongREJECT",
    "sorrybench": "SORRY-Bench", "jailbreakbench": "JailbreakBench",
    "wildjailbreak": "WildJailbreak", "fortress": "FORTRESS", "hexphi": "HEx-PHI",
    "or_bench": "OR-Bench", "falsereject": "FalseReject", "coconot": "CoCoNot",
    "xstest_safe": "XSTest",
}
BLOCKS = [("ASR", "asr_full", ASR_FULL), ("ORR", "orr_full", ORR_FULL)]


def _cell_path(split: str, model: str, name: str) -> Path:
    return paths.eval_results(TAG, split, model, name)


def per_prompt_soft(rows: list[dict], bench: str | None = None) -> dict:
    """gid -> mean soft_refusal_fraction over rollouts (optionally one benchmark)."""
    acc: dict = {}
    for r in rows:
        if bench is not None and r["benchmark"] != bench:
            continue
        acc.setdefault(r["global_id"], []).append(r["soft_refusal_fraction"])
    return {g: float(np.mean(v)) for g, v in acc.items()}


def boot_halfwidth_pct(values) -> float:
    """95% bootstrap CI half-width (in percent) for the mean of per-prompt metrics."""
    _, lo, hi = bootstrap_mean_ci(list(values), n_boot=N_BOOT, seed=SEED, alpha=0.05)
    return (hi - lo) / 2 * 100


def heat(v: float) -> int:
    return min(62, round(v * 0.6))


def _cell(mean: float, hw: float, is_min: bool) -> str:
    head = f"\\textbf{{{mean:.1f}}}" if is_min else f"{mean:.1f}"
    return f"{head}{{\\scriptsize$\\,\\pm${hw:.1f}}}"


def main() -> None:
    hdr = [MODEL_LABELS[m] for m in PRIMARY]
    body_rows: list[tuple] = []
    color_cmds: list[str] = []
    row_idx = 1

    for bi, (label, split, benches) in enumerate(BLOCKS):
        is_asr = label == "ASR"
        key = "soft_asr" if is_asr else "soft_orr"
        S = {m: read_json(_cell_path(split, m, "summary.json")) for m in PRIMARY}
        R = {m: read_concat_json(_cell_path(split, m, "results.jsonl")) for m in PRIMARY}

        rows_here = []
        for bk in benches:
            vals = []
            for m in PRIMARY:
                pt = S[m]["per_benchmark"][bk][key] * 100
                metric = [(1 - s) if is_asr else s for s in per_prompt_soft(R[m], bk).values()]
                vals.append((pt, boot_halfwidth_pct(metric)))
            rows_here.append((BENCH_LABELS[bk], vals))
        ov = []
        for m in PRIMARY:
            pt = S[m][key] * 100
            metric = [(1 - s) if is_asr else s for s in per_prompt_soft(R[m]).values()]
            ov.append((pt, boot_halfwidth_pct(metric)))
        rows_here.append(("Overall", ov))

        for j, (bn, vals) in enumerate(rows_here):
            row_idx += 1
            mn = min(v for v, _ in vals)
            cells = [(v, hw, v == mn) for v, hw in vals]
            for ci, (v, _hw, _m) in enumerate(cells):
                if heat(v) > 0:
                    color_cmds.append(f"\\cellcolor{{red!{heat(v)}}}{{{row_idx}-{ci + 3}}}")
            blk = (f"\\Block[fill=gray!8]{{{len(rows_here)}-1}}"
                   f"{{\\rotatebox{{90}}{{\\textbf{{{label}}}\\,$\\downarrow$}}}}"
                   if j == 0 else None)
            body_rows.append((bn, cells, blk))
        body_rows.append(("__MIDRULE__" if bi == 0 else "__END__", None, None))

    cb = ["    " + " ".join(color_cmds[k:k + 6]) for k in range(0, len(color_cmds), 6)]
    L = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\caption{Base model ASR and ORR per benchmark.",
         r"Lower is better for both.",
         r"Values are success rates averaged over the four rollouts under the fractional four-guardrail vote; $\pm$ denotes a $95\%$ bootstrap CI over the benchmark's prompts.",
         r"Safest model per benchmark in bold.",
         r"``Overall'' is pooled over all prompts in the block.}",
         r"\vspace{1mm}", r"\label{tab:base_asr_orr}", r"\setlength{\tabcolsep}{8.5pt}",
         r"\begin{NiceTabular}{cl cccc}[code-before = {%"]
    L += cb
    L += [r"}]", r"    \toprule",
          r"    & \textbf{Benchmark} & " + " & ".join(f"\\textbf{{{h}}}" for h in hdr) + r" \\",
          r"    \midrule"]
    for bn, cells, blk in body_rows:
        if bn == "__MIDRULE__":
            L.append(r"    \midrule")
            continue
        if bn == "__END__":
            continue
        if blk:
            L.append("    " + blk)
        if bn == "Overall":
            L.append(r"    \cmidrule(l){2-6}")
        L.append(f"    & {bn} & " + " & ".join(_cell(v, hw, m) for v, hw, m in cells) + r" \\")
    L += [r"    \bottomrule", r"\end{NiceTabular}", r"\end{table}"]

    out = paths.figure_out("tab_base_asr_orr.tex")
    out.write_text("\n".join(L) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
