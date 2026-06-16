#!/usr/bin/env python3
"""Compute every cell of the Sonnet-as-3rd-annotator IAA table for the paper.

Main panel = GPT-5.4 + Gemini-3-Pro + Sonnet-4.6 (the downstream majority panel).
Opus-4.7 included as a 4th reference annotator.

All numbers from experiments/cot_audit/outputs/iaa_n500_new/{gpt54,gemini,sonnet46,opus47}.jsonl.
Read-only; writes a JSON summary only (no paper files touched).
"""
from __future__ import annotations
import json, os, sys
from collections import Counter
from itertools import combinations
from pathlib import Path
import numpy as np

# Provenance: copied verbatim (logic unchanged) from the original research repo
#   experiments/cot_audit/compute_iaa_sonnet_table.py
# The κ helpers (imported from the original iaa_n500_compare via ROOT) are
# superseded by the shared `lrm_safety_deliberation` library (`lrm_safety_deliberation.stats`); the paper's
# final IAA table is produced by camera_ready/reproduce/tab_iaa.py. Set
# LRM_SAFETY_ARTIFACTS to relocate the artifact tree.
ROOT = Path(os.environ.get("LRM_SAFETY_ARTIFACTS", Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(ROOT))
from experiments.cot_audit.iaa_n500_compare import (
    load_jsonl_labels, fleiss_kappa, cohen_kappa,
)

OUT = ROOT / "experiments/cot_audit/outputs/iaa_n500_new"
SAMPLE = ROOT / "experiments/cot_audit/data/sample_n500_traces.jsonl"
CATS = (-1, 0, 1)

# ---- trace-level metadata (base vs defense, base_model) ----
META = {}
for ln in open(SAMPLE):
    r = json.loads(ln)
    key = (r["cell"], r["split"], int(r["global_id"]))
    is_base = (r["cell"] == r["base_model"])
    META[key] = {"base_model": r["base_model"],
                 "stratum": "base" if is_base else "defense",
                 "split": r["split"]}

PATHS = {
    "GPT-5.4":      OUT / "gpt54.jsonl",
    "Gemini-3-Pro": OUT / "gemini.jsonl",
    "Sonnet-4.6":   OUT / "sonnet46.jsonl",
    "Opus-4.7":     OUT / "opus47.jsonl",
}
RAW = {m: load_jsonl_labels(p) for m, p in PATHS.items()}

# coverage / refusal (full 500)
def coverage(path):
    ok = err = 0
    for ln in open(path):
        d = json.loads(ln)
        if "labels" in d: ok += 1
        else: err += 1
    return ok, err
COV = {m: coverage(p) for m, p in PATHS.items()}


def common_table(names):
    """Per-chunk rows over traces where ALL `names` succeeded w/ matching length.
    Each row: {key, base_model, stratum, split, <name>:label...}."""
    keys = set.intersection(*(set(RAW[n].keys()) for n in names))
    rows, n_traces, n_mismatch = [], 0, 0
    for k in sorted(keys):
        recs = [RAW[n][k] for n in names]
        nc = recs[0]["n_chunks"]
        if any(r["n_chunks"] != nc or len(r["labels"]) != nc for r in recs):
            n_mismatch += 1
            continue
        if k not in META:
            continue
        n_traces += 1
        for i in range(nc):
            row = {"key": k, **META[k]}
            for n, r in zip(names, recs):
                row[n] = int(r["labels"][i])
            rows.append(row)
    return rows, n_traces, n_mismatch


def marginals(rows, name):
    c = Counter(r[name] for r in rows)
    n = len(rows)
    return {lab: 100 * c[lab] / n for lab in CATS}


def majority(vals):
    c = Counter(vals)
    top, cnt = c.most_common(1)[0]
    return top, cnt  # label, agreement count


def centrality(rows, panel, who):
    """match rate of `who` vs majority-of-`panel` (only where a majority exists)."""
    num = den = 0
    for r in rows:
        maj_lab, cnt = majority([r[p] for p in panel])
        if cnt == 1:  # full 3-way disagreement -> no majority
            continue
        den += 1
        if r[who] == maj_lab:
            num += 1
    return 100 * num / den, den


def boot_ci_fleiss(rows, panel, B=2000, seed=20260610):
    """trace-clustered bootstrap CI for Fleiss kappa of `panel`."""
    rng = np.random.default_rng(seed)
    by_trace = {}
    for r in rows:
        by_trace.setdefault(r["key"], []).append(tuple(r[p] for p in panel))
    keys = list(by_trace.keys())
    ests = []
    for _ in range(B):
        samp = rng.choice(len(keys), size=len(keys), replace=True)
        triples = []
        for idx in samp:
            triples.extend(by_trace[keys[idx]])
        ests.append(fleiss_kappa(triples))
    return float(np.percentile(ests, 2.5)), float(np.percentile(ests, 97.5))


def boot_ci_cohen(rows, a, b, B=2000, seed=20260610):
    rng = np.random.default_rng(seed)
    by_trace = {}
    for r in rows:
        by_trace.setdefault(r["key"], []).append((r[a], r[b]))
    keys = list(by_trace.keys())
    ests = []
    for _ in range(B):
        samp = rng.choice(len(keys), size=len(keys), replace=True)
        pairs = []
        for idx in samp:
            pairs.extend(by_trace[keys[idx]])
        ests.append(cohen_kappa(pairs))
    return float(np.percentile(ests, 2.5)), float(np.percentile(ests, 97.5))


MAIN = ["GPT-5.4", "Gemini-3-Pro", "Sonnet-4.6"]

# ===== A. Main panel on its OWN common set =====
mrows, m_traces, m_mis = common_table(MAIN)
res = {"coverage": {m: {"ok": COV[m][0], "err": COV[m][1],
                        "refusal_pct": 100*COV[m][1]/sum(COV[m])} for m in PATHS}}
res["main_panel"] = {
    "panel": MAIN,
    "n_traces": m_traces, "n_chunks": len(mrows), "len_mismatch_dropped": m_mis,
    "fleiss_kappa": fleiss_kappa([tuple(r[p] for p in MAIN) for r in mrows]),
}
res["main_panel"]["fleiss_ci"] = boot_ci_fleiss(mrows, MAIN)

# marginals (main set)
res["marginals_mainset"] = {m: marginals(mrows, m) for m in MAIN}

# centrality (main set)
res["centrality_mainset"] = {m: centrality(mrows, MAIN, m) for m in MAIN}

# pairwise among main (main set) + CI
res["pairwise_main"] = {}
for a, b in combinations(MAIN, 2):
    pairs = [(r[a], r[b]) for r in mrows]
    res["pairwise_main"][f"{a} vs {b}"] = {
        "kappa": cohen_kappa(pairs),
        "ci": boot_ci_cohen(mrows, a, b),
        "raw_agree": 100*sum(1 for x,y in pairs if x==y)/len(pairs),
    }

# 3-way concordance on main set
triples = [tuple(r[p] for p in MAIN) for r in mrows]
maj_sizes = Counter()
for t in triples:
    maj_sizes[max(Counter(t).values())] += 1
res["concordance_main"] = {
    "agree_3of3": maj_sizes[3], "agree_2of3": maj_sizes[2], "disagree": maj_sizes[1],
    "n": len(triples),
    "pct_3of3": 100*maj_sizes[3]/len(triples),
    "pct_disagree": 100*maj_sizes[1]/len(triples),
}

# per-stratum & per-model Fleiss (main set)
def fleiss_subset(rows, panel, pred):
    sub = [tuple(r[p] for p in panel) for r in rows if pred(r)]
    return fleiss_kappa(sub), len(sub)
res["fleiss_strata_main"] = {
    "base":   fleiss_subset(mrows, MAIN, lambda r: r["stratum"]=="base"),
    "defense":fleiss_subset(mrows, MAIN, lambda r: r["stratum"]=="defense"),
    "harmful":fleiss_subset(mrows, MAIN, lambda r: r["split"]=="harmful"),
    "benign": fleiss_subset(mrows, MAIN, lambda r: r["split"]=="benign"),
}
for bm in ("Qwen3-8B","Olmo-3-7B-Think","Phi-4-reasoning","GPT-OSS-20B"):
    res["fleiss_strata_main"][bm] = fleiss_subset(mrows, MAIN, lambda r,bm=bm: r["base_model"]==bm)

# ===== B. Opus reference on all-4 common set =====
allrows, a_traces, a_mis = common_table(["GPT-5.4","Gemini-3-Pro","Sonnet-4.6","Opus-4.7"])
res["all4"] = {"n_traces": a_traces, "n_chunks": len(allrows), "len_mismatch_dropped": a_mis}
# Sonnet panel vs Opus panel on the SAME chunks (head-to-head)
res["headtohead_all4set"] = {
    "fleiss_GPT_Gemini_Sonnet": fleiss_kappa([(r["GPT-5.4"],r["Gemini-3-Pro"],r["Sonnet-4.6"]) for r in allrows]),
    "fleiss_GPT_Gemini_Opus":   fleiss_kappa([(r["GPT-5.4"],r["Gemini-3-Pro"],r["Opus-4.7"]) for r in allrows]),
}
# pairwise of Opus vs each main (all-4 set) + Sonnet vs each main (all-4 set)
res["pairwise_all4"] = {}
for a,b in combinations(["GPT-5.4","Gemini-3-Pro","Sonnet-4.6","Opus-4.7"], 2):
    pairs=[(r[a],r[b]) for r in allrows]
    res["pairwise_all4"][f"{a} vs {b}"]={"kappa":cohen_kappa(pairs),
        "raw_agree":100*sum(1 for x,y in pairs if x==y)/len(pairs)}
# marginals + centrality of all 4 on the all-4 set (panel=main 3)
res["marginals_all4set"] = {m: marginals(allrows, m) for m in PATHS}
res["centrality_all4set"] = {m: centrality(allrows, MAIN, m) for m in PATHS}

print(json.dumps(res, indent=2))
(OUT / "sonnet_table_full.json").write_text(json.dumps(res, indent=2))
print("\n[wrote]", (OUT/"sonnet_table_full.json").relative_to(ROOT), file=sys.stderr)
