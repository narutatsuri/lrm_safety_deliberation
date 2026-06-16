#!/usr/bin/env python3
"""Build the M=4 inference-defense stance-audit manifest.

Reads the M=4 defense generations at
  results/full_splits_v2/m4/<defense>/<split>/<model>__r<idx>.jsonl
extracts + chunks each thinking trace, and writes one manifest row per
(cell, split, gid, rollout_idx) mirroring the base-model m4_sample.jsonl schema
so the SAME annotator + analysis pipeline applies unchanged.

Manifest row schema (superset of base m4_sample.jsonl):
  cell        : "<defense>-<model>"  (e.g. "safepath_zs-Qwen3-8B")
  base_model  : audit display name   (Olmo-3-7B-Think -> "Olmo3-7B-Think")
  defense     : "<defense>"
  split       : "harmful" | "benign"
  global_id   : int
  rollout_idx : int
  benchmark   : str
  prompt      : str  (original goal)
  chunks      : list[str]
  n_chunks    : int
  force_closed: bool

Usage:
  .venv/bin/python experiments/causal_cuts/rebuttal/m4_inference_defense_sample.py
"""
from __future__ import annotations
import json
import os
import sys
from collections import Counter
from pathlib import Path

# Provenance: copied verbatim (logic unchanged) from the original research repo
#   experiments/causal_cuts/rebuttal/m4_inference_defense_sample.py
# chunk_trace is imported from the original cot_audit helper via ROOT; the model
# registry + 4-guardrail vote now also live in the shared `lrm_safety_deliberation` library
# (lrm_safety_deliberation.models / lrm_safety_deliberation.guardrails). Set LRM_SAFETY_ARTIFACTS to
# relocate the artifact tree.
ROOT = Path(os.environ.get("LRM_SAFETY_ARTIFACTS", Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(ROOT))
from experiments.cot_audit.label_defense_stances import chunk_trace  # noqa: E402

M4_BASE = ROOT / "results/full_splits_v2/m4"
OUT = ROOT / "experiments/causal_cuts/rebuttal/m4_inference_defense_sample.jsonl"
META = ROOT / "experiments/causal_cuts/rebuttal/m4_inference_defense_sample_meta.json"

DEFENSES = ["safepath_zs", "saferemind", "psr"]
MODELS = ["Qwen3-8B", "Olmo-3-7B-Think", "Phi-4-reasoning", "GPT-OSS-20B"]
SPLITS = ["asr", "orr"]
SPLIT_TO_AUDIT = {"asr": "harmful", "orr": "benign"}
MODEL_TO_AUDIT = {"Olmo-3-7B-Think": "Olmo3-7B-Think"}


def main():
    rows = []
    cov = Counter()       # (cell, split) -> usable trace count
    n_nothink = Counter()  # (cell, split) -> dropped (<2 chunks / empty)
    n_files = 0
    for defense in DEFENSES:
        for model in MODELS:
            cell = f"{defense}-{model}"
            audit_model = MODEL_TO_AUDIT.get(model, model)
            for split in SPLITS:
                audit_split = SPLIT_TO_AUDIT[split]
                d = M4_BASE / defense / split
                for f in sorted(d.glob(f"{model}__r*.jsonl")):
                    # parse rollout idx from filename suffix
                    try:
                        ridx = int(f.stem.split("__r")[1])
                    except Exception:
                        continue
                    n_files += 1
                    for ln in open(f):
                        ln = ln.strip()
                        if not ln:
                            continue
                        rec = json.loads(ln)
                        if rec.get("error"):
                            continue
                        think = (rec.get("target_thinking_trace") or "").strip()
                        if not think:
                            n_nothink[(cell, audit_split)] += 1
                            continue
                        chunks = chunk_trace(think)
                        if len(chunks) < 2:
                            n_nothink[(cell, audit_split)] += 1
                            continue
                        gid = int(rec.get("global_id", rec.get("sample_id")))
                        rows.append({
                            "cell": cell,
                            "base_model": audit_model,
                            "defense": defense,
                            "split": audit_split,
                            "global_id": gid,
                            "rollout_idx": ridx,
                            "benchmark": rec.get("benchmark", ""),
                            "prompt": rec.get("goal", ""),
                            "chunks": chunks,
                            "n_chunks": len(chunks),
                            "force_closed": bool(rec.get("force_closed", False)),
                        })
                        cov[(cell, audit_split)] += 1

    with open(OUT, "w") as fout:
        for r in rows:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")

    meta = {
        "n_rows": len(rows),
        "n_files_read": n_files,
        "coverage": {f"{c}|{s}": n for (c, s), n in sorted(cov.items())},
        "dropped_no_usable_thinking": {f"{c}|{s}": n for (c, s), n in sorted(n_nothink.items())},
    }
    META.write_text(json.dumps(meta, indent=2))
    print(f"[done] wrote {len(rows)} manifest rows from {n_files} files -> {OUT}")
    print(f"[meta] {META}")
    # quick coverage print
    for (c, s), n in sorted(cov.items()):
        drop = n_nothink.get((c, s), 0)
        print(f"  {c:28s} {s:8s}  usable={n:5d}  dropped={drop}")


if __name__ == "__main__":
    main()
