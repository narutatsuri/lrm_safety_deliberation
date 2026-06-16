#!/usr/bin/env python3
"""4-guardrail classifier for the M=4 inference-defense cut rollouts.

Adapts experiments/causal_cuts/classify_segment_repro.py to the inference cell
layout:
  in : experiments/causal_cuts/eval_results/causal_cuts_inference/<CELL>/<SPLIT>/rollouts_segment_repro.jsonl
  out: experiments/causal_cuts/eval_results/causal_cuts_inference/<CELL>/<SPLIT>/guardrails_segment_repro__<GR>.jsonl

Prompts are loaded from the canonical eval datasets by global_id
(harmful->asr.jsonl, benign->orr.jsonl) since the original goal is identical
across defenses. Guardrail models + classification reuse the base helpers.
"""
from __future__ import annotations
import argparse
import gc
import json
import os
import sys
from pathlib import Path

# Provenance: copied verbatim (logic unchanged) from the original research repo
#   experiments/causal_cuts/rebuttal/classify_cuts_inference.py
# GUARDRAIL_ORDER (imported from scripts.neurips_s16.common) + the 4-guardrail
# majority vote are superseded by the shared `lrm_safety_deliberation` library
# (`lrm_safety_deliberation.guardrails.GUARDRAIL_ORDER` / `.guardrail_vote`); this driver still
# imports the original repo modules via ROOT. Set LRM_SAFETY_ARTIFACTS to relocate.
PROJECT_ROOT = Path(os.environ.get("LRM_SAFETY_ARTIFACTS", Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.neurips_s16.common import GUARDRAIL_ORDER
from experiments.causal_cuts.classify_segment_repro import (
    build_llm, classify_one, read_jsonl, write_jsonl,
)

CUTS_ROOT = PROJECT_ROOT / "experiments/causal_cuts/eval_results/causal_cuts_inference"
ASR = PROJECT_ROOT / "evaluation/full_eval_pipeline/data/asr.jsonl"
ORR = PROJECT_ROOT / "evaluation/full_eval_pipeline/data/orr.jsonl"
SPLIT_DATASET = {"harmful": ASR, "benign": ORR}


def load_full_prompts(split):
    prompts = {}
    with open(SPLIT_DATASET[split]) as f:
        for line in f:
            d = json.loads(line)
            prompts[int(d["global_id"])] = d["prompt"]
    return prompts


def rollout_path(cell, split):
    return CUTS_ROOT / cell / split / "rollouts_segment_repro.jsonl"


def guardrail_path(cell, split, gr):
    return CUTS_ROOT / cell / split / f"guardrails_segment_repro__{gr}.jsonl"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cell", required=True, help="e.g. safepath_zs-Qwen3-8B")
    p.add_argument("--splits", nargs="+", default=["harmful", "benign"], choices=["harmful", "benign"])
    p.add_argument("--guardrails", nargs="+", default=list(GUARDRAIL_ORDER), choices=list(GUARDRAIL_ORDER))
    args = p.parse_args()
    cell = args.cell

    per_gr = {gr: [] for gr in args.guardrails}
    for split in args.splits:
        rp = rollout_path(cell, split)
        if not rp.exists():
            print(f"[{cell}/{split}] {rp.name} not found — skipping")
            continue
        n_rows = sum(1 for _ in rp.open() if _.strip())
        for gr in args.guardrails:
            out_p = guardrail_path(cell, split, gr)
            if out_p.exists() and sum(1 for _ in out_p.open() if _.strip()) == n_rows:
                print(f"[{cell}/{split}] {gr}: complete; skip")
                continue
            per_gr[gr].append(split)

    todo = [(gr, splits) for gr, splits in per_gr.items() if splits]
    if not todo:
        print(f"[{cell}] nothing to do."); return

    for gr, splits in todo:
        print(f"[{cell}] loading {gr} for {splits}", flush=True)
        llm = build_llm(gr)
        try:
            for split in splits:
                rows = read_jsonl(rollout_path(cell, split))
                full_prompts = load_full_prompts(split)
                non_skip = [i for i, r in enumerate(rows) if not r.get("skipped")]
                prompts = [full_prompts.get(int(rows[i]["global_id"]), "") for i in non_skip]
                responses = [rows[i].get("response", "") for i in non_skip]
                outputs = classify_one(gr, llm, prompts, responses) if prompts else []
                out_rows = []
                oi = 0
                for r in rows:
                    out = {
                        "global_id": r["global_id"],
                        "trace_rollout_idx": r["trace_rollout_idx"],
                        "transition_idx": r.get("transition_idx"),
                        "cut_kind": r.get("cut_kind"),
                        "rollout_idx": r["rollout_idx"],
                        "skipped": bool(r.get("skipped", False)),
                    }
                    if not r.get("skipped"):
                        out[gr] = outputs[oi]; oi += 1
                    out_rows.append(out)
                write_jsonl(guardrail_path(cell, split, gr), out_rows)
                print(f"[{cell}/{split}] {gr}: wrote {len(out_rows)}", flush=True)
        finally:
            del llm
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available(): torch.cuda.empty_cache()
            except Exception:
                pass


if __name__ == "__main__":
    main()
