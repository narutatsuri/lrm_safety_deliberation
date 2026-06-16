#!/usr/bin/env python3
"""Build stratified n=2400 (300/cell × 8 cells) subsample of the rebuttal audit pool.

Source: audit_chunked_n3885.jsonl (3,470 reconstructable audit traces with chunks).
Stratification: by (base_model, split). 300 per cell where available, else all.
Seed: 42 (project convention).

Output: experiments/causal_cuts/rebuttal/sample_n2400.jsonl
  Same schema as experiments/cot_audit/data/sample_n500_traces.jsonl so the
  annotator scripts can be reused unchanged.
"""
from __future__ import annotations
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

# Provenance: copied verbatim (logic unchanged) from the original research repo
#   experiments/causal_cuts/rebuttal/sample_n2400.py
# The model registry + 4-guardrail vote now also live in the shared `lrm_safety_deliberation`
# library (lrm_safety_deliberation.models / lrm_safety_deliberation.guardrails). Set LRM_SAFETY_ARTIFACTS
# to relocate the artifact tree.
ROOT = Path(os.environ.get("LRM_SAFETY_ARTIFACTS", Path(__file__).resolve().parents[3]))
SRC = ROOT / "experiments/causal_cuts/eval_results/stance_audit_full/audit_chunked_n3885.jsonl"
OUT = ROOT / "experiments/causal_cuts/rebuttal/sample_n2400.jsonl"
SEED = 42
PER_CELL = 300


def main():
    rng = random.Random(SEED)
    by_cell: dict = defaultdict(list)
    with open(SRC) as f:
        for ln in f:
            d = json.loads(ln)
            by_cell[(d["base_model"], d["split"])].append(d)
    print(f"[load] cells × counts:")
    for k, v in sorted(by_cell.items()):
        print(f"  {k}: {len(v)}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    n_total = 0
    realized = Counter()
    with open(OUT, "w") as f:
        for k, items in sorted(by_cell.items()):
            rng.shuffle(items)
            picked = items[:PER_CELL]
            for r in picked:
                f.write(json.dumps(r) + "\n")
            realized[k] = len(picked)
            n_total += len(picked)

    print()
    print(f"[write] {n_total} traces -> {OUT.relative_to(ROOT)}")
    print(f"[realized strata]")
    for k in sorted(realized):
        print(f"  {k}: {realized[k]}")


if __name__ == "__main__":
    main()
