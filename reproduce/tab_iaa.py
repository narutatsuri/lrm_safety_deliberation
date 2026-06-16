#!/usr/bin/env python3
"""Reproduce Table ``tab:iaa`` — inter-annotator agreement for the CoT-stance audit.

The downstream majority panel is three LLM annotators — GPT-5.4, Gemini-3-Pro,
and Sonnet-4.6 — labelling each reasoning chunk with a stance in ``{-1, 0, 1}``.
We report Fleiss' kappa for the 3-annotator panel (with a trace-clustered
bootstrap CI), the three pairwise Cohen kappas (also trace-clustered CIs), and
the 3-way concordance (3/3, 2/3, full-disagreement counts).  Opus-4.7 is carried
as a 4th reference annotator on the all-4 common set.

Agreement is computed per chunk over the set of traces that *all* panel members
labelled with a matching chunk count; CIs resample whole traces (clusters of
per-chunk rater tuples) so chunks within a trace are not treated as independent.

Inputs (resolved under ``LRM_SAFETY_ARTIFACTS``; mirror the original tree):
  experiments/cot_audit/data/sample_n500_traces.jsonl   (trace metadata)
  experiments/cot_audit/outputs/iaa_n500_new/gpt54.jsonl
  experiments/cot_audit/outputs/iaa_n500_new/gemini.jsonl
  experiments/cot_audit/outputs/iaa_n500_new/sonnet46.jsonl
  experiments/cot_audit/outputs/iaa_n500_new/opus47.jsonl

  Note: GPT-5.4 labels are read from the canonical ``gpt54.jsonl`` annotation
  file (the source the original generator uses), NOT the ``gpt54_labels`` field
  embedded in ``sample_n500_traces.jsonl`` — the two disagree on 358/500 traces
  and only ``gpt54.jsonl`` reproduces the published numbers.

Output: figures/tab_iaa.tex  (+ key numbers printed to stdout)

Original research script: experiments/cot_audit/compute_iaa_sonnet_table.py
"""
from __future__ import annotations

import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lrm_safety_deliberation import paths
from lrm_safety_deliberation.io import iter_jsonl
from lrm_safety_deliberation.stats import cluster_bootstrap_ci, cohen_kappa, fleiss_kappa

# 3-annotator downstream majority panel; Opus is the 4th reference annotator.
MAIN = ["GPT-5.4", "Gemini-3-Pro", "Sonnet-4.6"]
REFERENCE = "Opus-4.7"
N_BOOT = 2000
SEED = 20260610

_IAA = ("experiments", "cot_audit", "outputs", "iaa_n500_new")
PATHS = {
    "GPT-5.4":      paths.artifact(*_IAA, "gpt54.jsonl"),
    "Gemini-3-Pro": paths.artifact(*_IAA, "gemini.jsonl"),
    "Sonnet-4.6":   paths.artifact(*_IAA, "sonnet46.jsonl"),
    "Opus-4.7":     paths.artifact(*_IAA, "opus47.jsonl"),
}
SAMPLE = paths.artifact("experiments", "cot_audit", "data", "sample_n500_traces.jsonl")


def load_labels(path: Path) -> dict:
    """key (cell, split, global_id) -> {labels, n_chunks} for successful rows."""
    out = {}
    for r in iter_jsonl(path):
        if "labels" not in r:
            continue
        out[(r["cell"], r["split"], int(r["global_id"]))] = {
            "labels": r["labels"], "n_chunks": int(r["n_chunks"]),
        }
    return out


def common_table(raw: dict, meta: dict, names: list[str]):
    """Per-chunk rows over traces every annotator in `names` labelled with a
    matching chunk count.  Returns (rows, n_traces, n_len_mismatch).
    Each row carries the trace key + per-annotator integer label."""
    keys = set.intersection(*(set(raw[n].keys()) for n in names))
    rows, n_traces, n_mismatch = [], 0, 0
    for k in sorted(keys):
        recs = [raw[n][k] for n in names]
        nc = recs[0]["n_chunks"]
        if any(r["n_chunks"] != nc or len(r["labels"]) != nc for r in recs):
            n_mismatch += 1
            continue
        if k not in meta:
            continue
        n_traces += 1
        for i in range(nc):
            row = {"key": k, **meta[k]}
            for n, r in zip(names, recs):
                row[n] = int(r["labels"][i])
            rows.append(row)
    return rows, n_traces, n_mismatch


def trace_clusters(rows: list[dict], cols: list[str]) -> list[list[tuple]]:
    """Group per-chunk rater tuples by trace (sorted-key order) for the cluster
    bootstrap, so resampling draws whole traces."""
    by_trace: dict = {}
    for r in rows:
        by_trace.setdefault(r["key"], []).append(tuple(r[c] for c in cols))
    return [by_trace[k] for k in by_trace]


def concordance(rows: list[dict], panel: list[str]) -> dict:
    triples = [tuple(r[p] for p in panel) for r in rows]
    sizes = Counter()
    for t in triples:
        sizes[max(Counter(t).values())] += 1
    n = len(triples)
    return {"agree_3of3": sizes[3], "agree_2of3": sizes[2], "disagree": sizes[1],
            "n": n, "pct_3of3": 100 * sizes[3] / n, "pct_disagree": 100 * sizes[1] / n}


def fmt_ci(lo: float, hi: float) -> str:
    return f"[{lo:.3f}, {hi:.3f}]"


def main() -> None:
    # Trace metadata (base vs defense stratum, split) from the sample manifest.
    meta = {}
    for r in iter_jsonl(SAMPLE):
        key = (r["cell"], r["split"], int(r["global_id"]))
        meta[key] = {"base_model": r["base_model"],
                     "stratum": "base" if r["cell"] == r["base_model"] else "defense",
                     "split": r["split"]}

    raw = {m: load_labels(p) for m, p in PATHS.items()}

    # --- Main panel (GPT-5.4 + Gemini-3-Pro + Sonnet-4.6) on its own common set ---
    rows, n_traces, n_mis = common_table(raw, meta, MAIN)

    fleiss_pt, fleiss_lo, fleiss_hi = cluster_bootstrap_ci(
        trace_clusters(rows, MAIN), fleiss_kappa, n_boot=N_BOOT, seed=SEED, alpha=0.05)

    pairwise = {}
    for a, b in combinations(MAIN, 2):
        pt, lo, hi = cluster_bootstrap_ci(
            trace_clusters(rows, [a, b]), cohen_kappa, n_boot=N_BOOT, seed=SEED, alpha=0.05)
        raw_agree = 100 * sum(1 for r in rows if r[a] == r[b]) / len(rows)
        pairwise[(a, b)] = {"kappa": pt, "lo": lo, "hi": hi, "raw_agree": raw_agree}

    conc = concordance(rows, MAIN)

    # --- Opus reference: head-to-head Fleiss on the all-4 common set ---
    rows4, n_traces4, n_mis4 = common_table(raw, meta, MAIN + [REFERENCE])
    fleiss_sonnet_panel = fleiss_kappa([tuple(r[p] for p in MAIN) for r in rows4])
    fleiss_opus_panel = fleiss_kappa(
        [(r["GPT-5.4"], r["Gemini-3-Pro"], r[REFERENCE]) for r in rows4])

    # ---- console summary ----
    print(f"[main panel] {' + '.join(MAIN)}")
    print(f"  n_traces={n_traces}  n_chunks={len(rows)}  len_mismatch_dropped={n_mis}")
    print(f"  Fleiss kappa = {fleiss_pt:.4f}  95% CI {fmt_ci(fleiss_lo, fleiss_hi)}")
    print("  Pairwise Cohen kappa:")
    for (a, b), v in pairwise.items():
        print(f"    {a} vs {b:13s}: kappa={v['kappa']:.4f}  CI {fmt_ci(v['lo'], v['hi'])}"
              f"  raw_agree={v['raw_agree']:.2f}%")
    print(f"  Concordance: 3/3={conc['agree_3of3']} ({conc['pct_3of3']:.2f}%)  "
          f"2/3={conc['agree_2of3']}  disagree={conc['disagree']} ({conc['pct_disagree']:.2f}%)  "
          f"n={conc['n']}")
    print(f"[all-4 set] n_traces={n_traces4}  n_chunks={len(rows4)}")
    print(f"  Fleiss (GPT+Gemini+Sonnet) = {fleiss_sonnet_panel:.4f}")
    print(f"  Fleiss (GPT+Gemini+Opus)   = {fleiss_opus_panel:.4f}")

    # ---- LaTeX table ----
    short = {"GPT-5.4": "GPT-5.4", "Gemini-3-Pro": "Gemini-3-Pro", "Sonnet-4.6": "Sonnet-4.6"}
    L = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\caption{Inter-annotator agreement for the CoT-stance audit. "
        r"The downstream majority panel (GPT-5.4 + Gemini-3-Pro + Sonnet-4.6) "
        r"labels each reasoning chunk with a stance in $\{-1,0,1\}$. "
        r"Fleiss' $\kappa$ and pairwise Cohen's $\kappa$ are reported with "
        r"trace-clustered bootstrap 95\% CIs "
        rf"($n={n_traces}$ traces, {len(rows)} chunks). "
        r"Opus-4.7 is a held-out reference annotator.}",
        r"\label{tab:iaa}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"\textbf{Agreement measure} & \boldmath$\kappa$ & \textbf{95\% CI} \\",
        r"\midrule",
        r"\multicolumn{3}{l}{\textit{Panel (3 annotators)}} \\",
        rf"\quad Fleiss' $\kappa$ & {fleiss_pt:.3f} & {fmt_ci(fleiss_lo, fleiss_hi)} \\",
        r"\midrule",
        r"\multicolumn{3}{l}{\textit{Pairwise Cohen's $\kappa$}} \\",
    ]
    for (a, b), v in pairwise.items():
        L.append(rf"\quad {short[a]} -- {short[b]} & {v['kappa']:.3f} & {fmt_ci(v['lo'], v['hi'])} \\")
    L += [
        r"\midrule",
        r"\multicolumn{3}{l}{\textit{3-way concordance (chunks)}} \\",
        rf"\quad Unanimous (3/3) & \multicolumn{{2}}{{c}}{{{conc['agree_3of3']} "
        rf"({conc['pct_3of3']:.1f}\%)}} \\",
        rf"\quad Majority (2/3) & \multicolumn{{2}}{{c}}{{{conc['agree_2of3']} "
        rf"({100 * conc['agree_2of3'] / conc['n']:.1f}\%)}} \\",
        rf"\quad Full disagreement & \multicolumn{{2}}{{c}}{{{conc['disagree']} "
        rf"({conc['pct_disagree']:.1f}\%)}} \\",
        r"\midrule",
        rf"\multicolumn{{3}}{{l}}{{\textit{{Reference (all-4 common set, {len(rows4)} chunks)}}}} \\",
        rf"\quad Fleiss GPT+Gemini+Sonnet & \multicolumn{{2}}{{c}}{{{fleiss_sonnet_panel:.3f}}} \\",
        rf"\quad Fleiss GPT+Gemini+Opus & \multicolumn{{2}}{{c}}{{{fleiss_opus_panel:.3f}}} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    out = paths.figure_out("tab_iaa.tex")
    out.write_text("\n".join(L) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
