#!/usr/bin/env python3
"""IAA n=500 analysis under the NEW protocol + comparison to OLD-protocol same-model labels.

Reports
-------
A. NEW-protocol IAA across the three annotators (Fleiss kappa + per-pair Cohen kappa
   + per-chunk majority size).
B. Per-model old-vs-new per-chunk agreement (chunk-level, micro-averaged).
   Also: split-conditional and label-conditional confusion matrices.

Inputs:
  - OLD GPT-5.4: experiments/cot_audit/data/sample_n500_traces.jsonl  (gpt54_labels in row)
  - OLD Gemini:  experiments/cot_audit/outputs/gemini3pro_n500_batched.jsonl
  - OLD Opus:    experiments/cot_audit/outputs/opus47_n500_openrouter.jsonl
  - NEW (3):     experiments/cot_audit/outputs/iaa_n500_new/{gpt54,gemini,opus47}.jsonl

Output:
  - Prints to stdout
  - Writes experiments/cot_audit/outputs/iaa_n500_new/iaa_report.json
"""
from __future__ import annotations
import json
import os
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

# Provenance: copied verbatim (logic unchanged) from the original research repo
#   experiments/cot_audit/iaa_n500_compare.py
# The Fleiss/Cohen κ helpers here are superseded by the shared `lrm_safety_deliberation`
# library (`lrm_safety_deliberation.stats`); the paper's final IAA table is produced by
# camera_ready/reproduce/tab_iaa.py using lrm_safety_deliberation.stats κ. Set
# LRM_SAFETY_ARTIFACTS to relocate the artifact tree.
ROOT = Path(os.environ.get("LRM_SAFETY_ARTIFACTS", Path(__file__).resolve().parents[3]))

# ---------- I/O helpers ----------

def load_old_gpt():
    """OLD GPT-5.4 labels are embedded in sample file under key gpt54_labels."""
    out = {}
    with open(ROOT / "experiments/cot_audit/data/sample_n500_traces.jsonl") as f:
        for ln in f:
            r = json.loads(ln)
            out[(r["cell"], r["split"], int(r["global_id"]))] = {
                "labels": r["gpt54_labels"],
                "n_chunks": int(r["n_chunks"]),
                "split": r["split"],
            }
    return out


def load_jsonl_labels(path):
    out = {}
    with open(path) as f:
        for ln in f:
            r = json.loads(ln)
            if "labels" not in r:
                continue
            out[(r["cell"], r["split"], int(r["global_id"]))] = {
                "labels": r["labels"],
                "n_chunks": int(r["n_chunks"]),
                "split": r["split"],
            }
    return out


# ---------- IAA stats ----------

def fleiss_kappa(triples):
    """triples: iterable of (l1, l2, l3) for each chunk. Labels in {-1,0,1}."""
    items = list(triples)
    if not items:
        return float("nan")
    N = len(items)
    n = len(items[0])  # raters per item, here 3
    cats = (-1, 0, 1)
    # Per-category total
    cat_total = Counter()
    P_i = []
    for row in items:
        cnt = Counter(row)
        # P_i = sum_j n_ij(n_ij - 1) / (n(n-1))
        s = sum(cnt[c] * (cnt[c] - 1) for c in cats)
        P_i.append(s / (n * (n - 1)))
        for c in cats:
            cat_total[c] += cnt[c]
    P_bar = sum(P_i) / N
    p_j = {c: cat_total[c] / (N * n) for c in cats}
    P_e = sum(p_j[c] ** 2 for c in cats)
    if P_e == 1.0:
        return 1.0
    return (P_bar - P_e) / (1 - P_e)


def cohen_kappa(pairs):
    """pairs: iterable of (l_a, l_b). Labels in {-1,0,1}. Standard linear kappa."""
    items = list(pairs)
    if not items:
        return float("nan")
    N = len(items)
    cats = (-1, 0, 1)
    obs_agree = sum(1 for a, b in items if a == b) / N
    a_count = Counter(a for a, _ in items)
    b_count = Counter(b for _, b in items)
    pe = sum((a_count[c] / N) * (b_count[c] / N) for c in cats)
    if pe == 1.0:
        return 1.0
    return (obs_agree - pe) / (1 - pe)


# ---------- Build per-chunk views ----------

def build_chunk_table(label_dicts, names):
    """
    label_dicts: list of {key -> {labels, n_chunks, split}} for each annotator.
    names: list of annotator names.

    Returns list of dicts:
        {key, chunk_idx, split, lbl[name1], lbl[name2], ...}
    only for chunks present in ALL annotators with matching n_chunks.
    """
    keys = set.intersection(*(set(d.keys()) for d in label_dicts))
    out = []
    n_skipped_keys = 0
    n_len_mismatch = 0
    for k in sorted(keys):
        rows = [d[k] for d in label_dicts]
        nc = rows[0]["n_chunks"]
        if any(r["n_chunks"] != nc or len(r["labels"]) != nc for r in rows):
            n_len_mismatch += 1
            continue
        for i in range(nc):
            entry = {"key": k, "chunk_idx": i, "split": rows[0]["split"]}
            for n, r in zip(names, rows):
                entry[n] = int(r["labels"][i])
            out.append(entry)
    return out, len(keys), n_len_mismatch


# ---------- Reporting ----------

def print_confusion(cm, label_axis=("old", "new")):
    cats = (-1, 0, 1)
    print(f"        {label_axis[1]}=-1  {label_axis[1]}= 0  {label_axis[1]}=+1")
    for o in cats:
        row = [cm[(o, n)] for n in cats]
        print(f"  {label_axis[0]}={o:+d}  {row[0]:>6}  {row[1]:>6}  {row[2]:>6}")


def main():
    out_dir = ROOT / "experiments/cot_audit/outputs/iaa_n500_new"
    new_paths = {
        "GPT-5.4":      out_dir / "gpt54.jsonl",
        "Gemini-3-Pro": out_dir / "gemini.jsonl",
        "Opus-4.7":     out_dir / "opus47.jsonl",
    }
    missing = [m for m, p in new_paths.items() if not p.exists()]
    if missing:
        sys.exit(f"NEW label files missing: {missing}")

    old_labels = {
        "GPT-5.4":      load_old_gpt(),
        "Gemini-3-Pro": load_jsonl_labels(ROOT / "experiments/cot_audit/outputs/gemini3pro_n500_batched.jsonl"),
        "Opus-4.7":     load_jsonl_labels(ROOT / "experiments/cot_audit/outputs/opus47_n500_openrouter.jsonl"),
    }
    new_labels = {m: load_jsonl_labels(p) for m, p in new_paths.items()}

    print("=" * 80)
    print("Coverage")
    print("=" * 80)
    print(f"{'Model':<14} {'OLD':>8} {'NEW':>8}")
    for m in old_labels:
        print(f"{m:<14} {len(old_labels[m]):>8} {len(new_labels[m]):>8}")

    report = {"coverage": {m: {"old": len(old_labels[m]), "new": len(new_labels[m])} for m in old_labels}}

    # ---------- A. NEW-protocol IAA ----------
    print("\n" + "=" * 80)
    print("A. NEW-PROTOCOL IAA (3 annotators agreeing per chunk, all on common keys)")
    print("=" * 80)
    chunk_table, n_common_traces, n_len_mismatch = build_chunk_table(
        [new_labels["GPT-5.4"], new_labels["Gemini-3-Pro"], new_labels["Opus-4.7"]],
        ["GPT-5.4", "Gemini-3-Pro", "Opus-4.7"],
    )
    print(f"Common traces: {n_common_traces}, length-mismatch dropped: {n_len_mismatch}")
    print(f"Chunks usable: {len(chunk_table)}")

    triples = [(c["GPT-5.4"], c["Gemini-3-Pro"], c["Opus-4.7"]) for c in chunk_table]
    fk = fleiss_kappa(triples)
    print(f"\nFleiss kappa (3 raters, all chunks): {fk:.4f}")

    # Split-conditional
    for sp in ("benign", "harmful"):
        sub = [(c["GPT-5.4"], c["Gemini-3-Pro"], c["Opus-4.7"]) for c in chunk_table if c["split"] == sp]
        fk_s = fleiss_kappa(sub)
        print(f"  Fleiss kappa (split={sp}, n={len(sub)}): {fk_s:.4f}")

    # Pairwise Cohen kappa
    print("\nCohen kappa (pairwise, all chunks):")
    pair_kappas = {}
    for a, b in combinations(["GPT-5.4", "Gemini-3-Pro", "Opus-4.7"], 2):
        pairs = [(c[a], c[b]) for c in chunk_table]
        ck = cohen_kappa(pairs)
        agree = sum(1 for x, y in pairs if x == y) / len(pairs)
        pair_kappas[f"{a} vs {b}"] = ck
        print(f"  {a:<14} vs {b:<14}: kappa={ck:.4f}  raw_agree={100*agree:.1f}%")

    # Per-chunk majority size distribution
    maj_sizes = Counter()
    for t in triples:
        c = Counter(t)
        maj_sizes[max(c.values())] += 1
    print(f"\nMajority size distribution across {len(triples)} chunks:")
    for k, v in sorted(maj_sizes.items()):
        print(f"  {k}/3 agree: {v} chunks ({100*v/len(triples):.1f}%)")

    report["new_iaa"] = {
        "n_common_traces": n_common_traces,
        "n_chunks": len(chunk_table),
        "fleiss_kappa_all": fk,
        "fleiss_kappa_benign": fleiss_kappa([(c["GPT-5.4"], c["Gemini-3-Pro"], c["Opus-4.7"]) for c in chunk_table if c["split"]=="benign"]),
        "fleiss_kappa_harmful": fleiss_kappa([(c["GPT-5.4"], c["Gemini-3-Pro"], c["Opus-4.7"]) for c in chunk_table if c["split"]=="harmful"]),
        "cohen_kappa_pairs": pair_kappas,
        "majority_sizes": {f"{k}/3": v for k, v in maj_sizes.items()},
    }

    # ---------- B. Old-vs-new same-model agreement ----------
    print("\n" + "=" * 80)
    print("B. SAME-MODEL: OLD vs NEW per-chunk agreement")
    print("=" * 80)
    same_model = {}
    for m in old_labels:
        chunk_pairs, n_traces, n_skip = build_chunk_table(
            [old_labels[m], new_labels[m]], ["old", "new"]
        )
        n_chunks = len(chunk_pairs)
        if n_chunks == 0:
            print(f"\n{m}: no overlap")
            continue
        agree = sum(1 for c in chunk_pairs if c["old"] == c["new"])
        agree_pct = 100 * agree / n_chunks
        cm = Counter()
        for c in chunk_pairs:
            cm[(c["old"], c["new"])] += 1
        # Cohen kappa for same-model old-vs-new
        ck = cohen_kappa([(c["old"], c["new"]) for c in chunk_pairs])
        # split-conditional
        agree_by_split = {}
        for sp in ("benign", "harmful"):
            sub = [c for c in chunk_pairs if c["split"] == sp]
            if sub:
                a_sp = sum(1 for c in sub if c["old"] == c["new"])
                agree_by_split[sp] = (a_sp, len(sub))

        print(f"\n--- {m} ---")
        print(f"  Common traces: {n_traces}  Length-mismatch dropped: {n_skip}")
        print(f"  Total chunks: {n_chunks}")
        print(f"  Raw agreement: {agree}/{n_chunks} = {agree_pct:.2f}%")
        print(f"  Cohen kappa (old vs new, same model): {ck:.4f}")
        for sp, (a_sp, n_sp) in agree_by_split.items():
            print(f"    {sp}: {a_sp}/{n_sp} = {100*a_sp/n_sp:.2f}%")
        print(f"  Confusion (old -> new):")
        print_confusion(cm)

        # Label distribution
        old_dist = Counter(c["old"] for c in chunk_pairs)
        new_dist = Counter(c["new"] for c in chunk_pairs)
        n = n_chunks
        print(f"  Label distribution (-1/0/+1):")
        print(f"    old: {old_dist[-1]}/{old_dist[0]}/{old_dist[1]}  ({100*old_dist[-1]/n:.0f}/{100*old_dist[0]/n:.0f}/{100*old_dist[1]/n:.0f}%)")
        print(f"    new: {new_dist[-1]}/{new_dist[0]}/{new_dist[1]}  ({100*new_dist[-1]/n:.0f}/{100*new_dist[0]/n:.0f}/{100*new_dist[1]/n:.0f}%)")

        same_model[m] = {
            "n_traces": n_traces,
            "n_len_mismatch": n_skip,
            "n_chunks": n_chunks,
            "raw_agreement": agree_pct / 100,
            "cohen_kappa_old_vs_new": ck,
            "agreement_by_split": {sp: a/n for sp, (a, n) in agree_by_split.items()},
            "confusion_old_new": {f"{o}_{n}": cm[(o, n)] for o in (-1, 0, 1) for n in (-1, 0, 1)},
            "old_dist": {str(k): old_dist[k] for k in (-1, 0, 1)},
            "new_dist": {str(k): new_dist[k] for k in (-1, 0, 1)},
        }
    report["same_model_old_vs_new"] = same_model

    out_path = out_dir / "iaa_report.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\n[wrote] {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
