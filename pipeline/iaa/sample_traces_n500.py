#!/usr/bin/env python3
"""Sample 500 traces stratified by (group x base x split) for the FU-1 IAA audit.

Pulls from the combined GPT-5.4 labeled pool:
  - experiments/cot_audit/eval_results/stance_audit_full/labeled_traces.jsonl   (4 base x 2 splits)
  - experiments/cot_audit/eval_results/defenses_stance_v2/labeled_traces.jsonl  (19 defense cells)

For each sampled trace, re-derives the chunk text from the source
generations.jsonl using the same chunker / extract_thinking as the
production v2 audit, and emits one row per trace containing the chunks
plus the existing GPT-5.4 labels. Output is the input to the
full-trace-batched Gemini-3-Pro and Opus 4.7 annotation scripts.

Stratification:
  group   in {base, defense}                            (2)
  base    in {Qwen3-8B, Olmo-3-7B-Think, Phi-4-reasoning, GPT-OSS-20B} (4)
  split   in {harmful, benign}                          (2)
  total: 16 strata

For the defense group, only the four "non-trivial" defense families
(STAR1, SafeKey, STAIR, RAPO) contribute; cells whose label distribution
collapses under the base intent (e.g. R1ACT, ThinkSafe) are excluded so
each stratum spans real stance variation.

Seed = 42, reproducible.
"""
from __future__ import annotations
import argparse
import json
import os
import random
import sys
from collections import defaultdict, Counter
from pathlib import Path

# Provenance: copied verbatim (logic unchanged) from the original research repo
#   experiments/cot_audit/sample_traces_n500.py
# chunk_trace / extract_thinking are imported from the original cot_audit helper
# via ROOT; the model registry + 4-guardrail vote now also live in the shared
# `lrm_safety_deliberation` library (lrm_safety_deliberation.models / lrm_safety_deliberation.guardrails), and the final
# κ table lives in lrm_safety_deliberation.stats. Set LRM_SAFETY_ARTIFACTS to relocate.
ROOT = Path(os.environ.get("LRM_SAFETY_ARTIFACTS", Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(ROOT))
from experiments.cot_audit.label_defense_stances import chunk_trace, extract_thinking  # noqa: E402

BASE_FILE = ROOT / "experiments/cot_audit/eval_results/stance_audit_full/labeled_traces.jsonl"
DEF_FILE = ROOT / "experiments/cot_audit/eval_results/defenses_stance_v2/labeled_traces.jsonl"

GENERATIONS_DIR = ROOT / "experiments/cot_audit/eval_results/generations"
# Local copies of source thinking-trace generations live under generations/{base,defense}/<cell>/<split>.jsonl.
# {split} placeholder kept for backwards compat with the legacy upstream-path template, but the
# resolver below builds the new layout directly.
BASE_GEN_PATHS = {
    "Qwen3-8B":        GENERATIONS_DIR / "base/Qwen3-8B/{split}.jsonl",
    "Olmo-3-7B-Think": GENERATIONS_DIR / "base/Olmo-3-7B-Think/{split}.jsonl",
    "Phi-4-reasoning": GENERATIONS_DIR / "base/Phi-4-reasoning/{split}.jsonl",
    "GPT-OSS-20B":     GENERATIONS_DIR / "base/GPT-OSS-20B/{split}.jsonl",
}
DEF_GEN_BASE = GENERATIONS_DIR / "defense"
SPLIT_TO_FULLPIPE = {"harmful": "asr", "benign": "orr"}

# Display name shown in stance_audit_full -> canonical base used for stratification.
BASE_CANON = {
    "Qwen3-8B": "Qwen3-8B",
    "Olmo-3-7B-Think": "Olmo-3-7B-Think",
    "Olmo3-7B-Think":  "Olmo-3-7B-Think",
    "Phi-4-reasoning": "Phi-4-reasoning",
    "GPT-OSS-20B":     "GPT-OSS-20B",
}

NONTRIVIAL_DEFENSE_PREFIXES = ("SafeKey", "STAIR", "RAPO", "STAR1")


def load_rows(path: Path):
    rows = []
    with open(path) as f:
        for ln in f:
            try:
                d = json.loads(ln)
            except Exception:
                continue
            if "labels" not in d:
                continue
            rows.append(d)
    return rows


def load_gen_index(path: Path, is_channel: bool) -> dict:
    """Return {gid: thinking_trace}."""
    idx = {}
    if not path.exists():
        return idx
    with open(path) as f:
        for ln in f:
            try:
                d = json.loads(ln)
            except Exception:
                continue
            gid = d.get("global_id")
            if gid is None:
                continue
            tt = (d.get("thinking_trace") or "").strip()
            if not tt:
                tt = extract_thinking(
                    d.get("raw_response", "") or "",
                    is_channel=is_channel,
                    response=d.get("response", "") or "",
                )
            if tt:
                idx[int(gid)] = tt
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_total", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(ROOT / "experiments/cot_audit/data/sample_n500_traces.jsonl"))
    args = ap.parse_args()

    rng = random.Random(args.seed)

    base_rows = load_rows(BASE_FILE)
    def_rows = [d for d in load_rows(DEF_FILE)
                if d["cell"].startswith(NONTRIVIAL_DEFENSE_PREFIXES)]
    print(f"[load] base_rows={len(base_rows)}  def_rows(non-trivial)={len(def_rows)}")

    # Tag each row with (group, canonical_base, cell, split, gid, labels, prompt, benchmark)
    pool = []
    for d in base_rows:
        canon = BASE_CANON.get(d["model"])
        if canon is None:
            continue
        pool.append({
            "group": "base", "base": canon, "cell": d["model"],
            "split": d["split"], "gid": int(d["global_id"]),
            "benchmark": d.get("benchmark", ""), "prompt": d.get("prompt", "")[:200],
            "labels": list(d["labels"]), "source_dataset": "stance_audit_full",
        })
    for d in def_rows:
        canon = BASE_CANON.get(d["base_model"])
        if canon is None:
            continue
        pool.append({
            "group": "defense", "base": canon, "cell": d["cell"],
            "split": d["split"], "gid": int(d["global_id"]),
            "benchmark": d.get("benchmark", ""), "prompt": d.get("prompt", "")[:200],
            "labels": list(d["labels"]), "source_dataset": "defenses_stance_v2",
        })
    print(f"[pool] total candidate traces: {len(pool)}")

    # Stratify by (group x base x split)
    strata = defaultdict(list)
    for r in pool:
        strata[(r["group"], r["base"], r["split"])].append(r)
    keys = sorted(strata.keys())
    print(f"[strata] {len(keys)} cells:")
    for k in keys:
        print(f"  {k}: {len(strata[k])}")

    per_strat = args.n_total // len(keys)         # 500 // 16 = 31
    remainder = args.n_total - per_strat * len(keys)  # 4 leftover

    picked = []
    for k in keys:
        items = strata[k][:]
        rng.shuffle(items)
        picked.extend(items[:per_strat])

    # Distribute remainder among the largest strata first.
    if remainder:
        leftovers = []
        for k in keys:
            items = strata[k][per_strat:]
            rng.shuffle(items)
            leftovers.extend(items)
        rng.shuffle(leftovers)
        picked.extend(leftovers[:remainder])

    rng.shuffle(picked)
    print(f"[sample] {len(picked)} traces selected")

    # Resolve thinking_trace + re-chunk.
    gen_cache: dict = {}

    def get_gens(group: str, cell: str, base: str, split: str) -> dict:
        key = (group, cell, split)
        if key in gen_cache:
            return gen_cache[key]
        is_channel = "GPT-OSS" in (cell or "") or "GPT-OSS" in (base or "")
        if group == "base":
            tpl = BASE_GEN_PATHS.get(BASE_CANON.get(base) or base)
            if tpl is None:
                gen_cache[key] = {}
                return {}
            # Olmo path uses Olmo-3-7B-Think directory name.
            path = Path(str(tpl).replace("{split}", split))  # local copy uses literal harmful/benign
        else:
            path = DEF_GEN_BASE / cell / f"{split}.jsonl"
        idx = load_gen_index(path, is_channel)
        gen_cache[key] = idx
        print(f"  [gens] {key}: {len(idx)} traces from {path.relative_to(ROOT)}")
        return idx

    # Resolve picked traces; topping up from per-stratum reserves when a
    # trace can't be re-chunked (source gen missing for that gid).
    picked_keys = {(r["cell"], r["gid"], r["split"]) for r in picked}
    reserves = {}
    for k in keys:
        rem = [r for r in strata[k]
               if (r["cell"], r["gid"], r["split"]) not in picked_keys]
        rng.shuffle(rem)
        reserves[k] = rem

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    n_miss = 0
    n_mismatch = 0
    label_dist = Counter()
    strata_dist = Counter()
    target_per_strat = {k: per_strat for k in keys}
    # Account for remainder distribution: pick which strata got the remainders.
    # Realized counts will be checked after the first pass; reserves drain
    # to fill any deficit per stratum.

    fout = open(out_path, "w")

    def emit(r):
        nonlocal n_ok, n_mismatch
        idx = get_gens(r["group"], r["cell"], r["base"], r["split"])
        tt = idx.get(r["gid"])
        if not tt:
            return False
        chunks = chunk_trace(tt)
        if len(chunks) != len(r["labels"]):
            n_mismatch += 1
        row = {
            "cell": r["cell"], "base_model": r["base"],
            "split": r["split"], "global_id": r["gid"],
            "benchmark": r["benchmark"], "prompt": r["prompt"],
            "n_chunks": len(chunks),
            "chunks": chunks,
            "gpt54_labels": r["labels"],
            "n_gpt54_labels": len(r["labels"]),
            "group": r["group"],
            "source_dataset": r["source_dataset"],
        }
        fout.write(json.dumps(row) + "\n")
        n_ok += 1
        strata_dist[(r["group"], r["base"], r["split"])] += 1
        for lab in r["labels"]:
            label_dist[int(lab)] += 1
        return True

    # First pass: try to emit each picked trace; failures count toward deficit.
    deficits = Counter()
    for r in picked:
        ok = emit(r)
        if not ok:
            n_miss += 1
            deficits[(r["group"], r["base"], r["split"])] += 1

    # Second pass: for each stratum with deficit, pull from reserves until
    # filled or reserves exhausted.
    for k, defc in deficits.items():
        while defc > 0 and reserves[k]:
            r = reserves[k].pop()
            if emit(r):
                defc -= 1
            else:
                n_miss += 1
        if defc > 0:
            print(f"  [deficit] {k}: still short {defc}, reserves exhausted")

    fout.close()

    print(f"\n[write] {n_ok} traces -> {out_path.relative_to(ROOT)}")
    print(f"  missed (source-not-found): {n_miss}")
    print(f"  length-mismatch chunks vs labels (flagged, kept): {n_mismatch}")

    print(f"\n[strata realized]")
    for k in sorted(strata_dist):
        print(f"  {k}: {strata_dist[k]}")
    print(f"\n[gpt54 label distribution across all chunks in sample]")
    for k in sorted(label_dist):
        print(f"  {k:+d}: {label_dist[k]}")


if __name__ == "__main__":
    main()
