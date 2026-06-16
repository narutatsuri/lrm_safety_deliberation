#!/usr/bin/env python3
"""4-guardrail classifier for the reproducibility experiment's K=32 rollouts.

Differs from classify_segment.py by including trace_rollout_idx in the join
key (so the same (gid, transition_idx, cut_kind, rollout_idx) for two different
natural traces of the same prompt doesn't collide).

Inputs:
  experiments/causal_cuts/eval_results/causal_cuts/<MODEL>/<SPLIT>/rollouts_segment_repro.jsonl
Outputs:
  experiments/causal_cuts/eval_results/causal_cuts/<MODEL>/<SPLIT>/guardrails_segment_repro__<GR>.jsonl
"""
from __future__ import annotations
import argparse
import gc
import json
import os
import sys
from pathlib import Path

# Provenance: copied verbatim (logic unchanged) from the original research repo
#   experiments/causal_cuts/classify_segment_repro.py
# The 4-guardrail constants + majority vote imported here from
# scripts.neurips_s16.common (GUARDRAIL_ORDER, OR_BENCH_REFUSAL_RUBRIC) are
# superseded by the shared `lrm_safety_deliberation` library
# (`lrm_safety_deliberation.guardrails.GUARDRAIL_ORDER` / `.OR_BENCH_REFUSAL_RUBRIC` /
# `.guardrail_vote`); this driver still imports the original repo modules via
# ROOT so it runs standalone. Set LRM_SAFETY_ARTIFACTS to relocate the tree.
PROJECT_ROOT = Path(os.environ.get("LRM_SAFETY_ARTIFACTS", Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(PROJECT_ROOT))

from vllm import LLM

from evaluation.classifiers import (
    classify_with_granite_guardian,
    classify_with_oss_safeguard,
    classify_with_qwen3guard,
    classify_with_wildguard,
)
from scripts.neurips_s16.common import GUARDRAIL_ORDER, OR_BENCH_REFUSAL_RUBRIC
from scripts.neurips_final.common import MODELS

CUTS_ROOT = PROJECT_ROOT / "experiments" / "causal_cuts" / "eval_results" / "causal_cuts"
BASE_ROOT = PROJECT_ROOT / "eval_results" / "neurips_final"
SUPP_ROOT = PROJECT_ROOT / "eval_results" / "neurips_final_supplementary"


IN_NAME = "rollouts_segment_repro.jsonl"
OUT_SUFFIX = "repro"


def rollout_path(model, split):
    return CUTS_ROOT / model / split / IN_NAME


def guardrail_path(model, split, gr):
    return CUTS_ROOT / model / split / f"guardrails_segment_{OUT_SUFFIX}__{gr}.jsonl"


def load_full_prompts(model_key, split):
    prompts = {}
    base_path = BASE_ROOT / model_key / "generations" / "base" / f"{split}.jsonl"
    if base_path.exists():
        with open(base_path) as f:
            for line in f:
                d = json.loads(line)
                prompts[d["global_id"]] = d["prompt"]
    if model_key == "Qwen3-8B" and split == "benign":
        supp_path = SUPP_ROOT / model_key / "generations" / "base" / "orfuzz.jsonl"
        if supp_path.exists():
            with open(supp_path) as f:
                for line in f:
                    d = json.loads(line)
                    prompts.setdefault(d["global_id"], d["prompt"])
    return prompts


def read_jsonl(p):
    out = []
    for line in open(p):
        line = line.strip()
        if not line: continue
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


def write_jsonl(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def build_llm(name):
    paths = {
        "wildguard": ("models/wildguard", 0.9),
        "qwen3guard": ("models/Qwen3Guard-Gen-8B", 0.9),
        "granite_guardian": ("models/granite-guardian-3.3-8b", 0.88),
        "oss_safeguard": ("/scratch/gpfs/ARORA/nr3764/.hf_models/gpt-oss-safeguard-20b", 0.9),
    }
    model_path, util = paths[name]
    return LLM(model=model_path, dtype="bfloat16", gpu_memory_utilization=util,
               trust_remote_code=True, enforce_eager=True, max_model_len=8192)


def classify_one(name, llm, prompts, responses):
    if name == "wildguard":
        return classify_with_wildguard(prompts, responses, llm=llm)
    if name == "qwen3guard":
        return classify_with_qwen3guard(prompts, responses, llm=llm)
    if name == "granite_guardian":
        return classify_with_granite_guardian(prompts, responses,
                                              [OR_BENCH_REFUSAL_RUBRIC] * len(prompts), llm=llm)
    if name == "oss_safeguard":
        return classify_with_oss_safeguard(prompts, responses, llm=llm)
    raise KeyError(name)


def main():
    global IN_NAME, OUT_SUFFIX, BASE_ROOT
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=sorted(MODELS), metavar="MODEL_KEY",
                   help="base model OR defense cell key in common.py MODELS")
    p.add_argument("--prompts-root", default=str(BASE_ROOT),
                   help="root holding <model>/generations/base/<split>.jsonl for prompt lookup "
                        "(defense cells: eval_results/defenses_eval)")
    p.add_argument("--splits", nargs="+", default=["harmful", "benign"], choices=["harmful", "benign"])
    p.add_argument("--guardrails", nargs="+", default=list(GUARDRAIL_ORDER),
                   choices=list(GUARDRAIL_ORDER))
    p.add_argument("--in-name", default=IN_NAME, help="rollouts filename to classify")
    p.add_argument("--out-suffix", default=OUT_SUFFIX, help="guardrails_segment_<suffix>__<gr>.jsonl")
    p.add_argument("--batch-rows", type=int, default=20000,
                   help="classify+append this many rollout rows per checkpoint (incremental writing)")
    args = p.parse_args()

    IN_NAME = args.in_name
    OUT_SUFFIX = args.out_suffix
    BASE_ROOT = Path(args.prompts_root)

    per_gr_cells = {gr: [] for gr in args.guardrails}
    for split in args.splits:
        rp = rollout_path(args.model, split)
        if not rp.exists():
            print(f"[{args.model}/{split}] {rp.name} not found — skipping")
            continue
        n_rows = sum(1 for _ in rp.open() if _.strip())
        for gr in args.guardrails:
            out_p = guardrail_path(args.model, split, gr)
            if out_p.exists():
                n_done = sum(1 for _ in out_p.open() if _.strip())
                if n_done == n_rows:
                    print(f"[{args.model}/{split}] {gr}: complete; skip")
                    continue
            per_gr_cells[gr].append(split)

    todo = [(gr, splits) for gr, splits in per_gr_cells.items() if splits]
    if not todo:
        print(f"[{args.model}] nothing to do."); return

    for gr, splits in todo:
        print(f"[{args.model}] loading {gr} for {splits}", flush=True)
        llm = build_llm(gr)
        try:
            for split in splits:
                rows = read_jsonl(rollout_path(args.model, split))
                full_prompts = load_full_prompts(args.model, split)
                out_p = guardrail_path(args.model, split, gr)
                out_p.parent.mkdir(parents=True, exist_ok=True)
                # Resume: output rows are 1:1 with rollout rows, in order. Keep only
                # complete JSON lines already on disk (drops any torn trailing line),
                # rewrite that clean prefix, then append remaining batches.
                start = 0
                if out_p.exists():
                    valid = []
                    for line in out_p.open():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            json.loads(line); valid.append(line)
                        except Exception:
                            break
                    start = len(valid)
                    with open(out_p, "w") as f:
                        for line in valid:
                            f.write(line + "\n")
                if start >= len(rows):
                    print(f"[{args.model}/{split}] {gr}: already complete ({start} rows)", flush=True)
                    continue
                BATCH = args.batch_rows
                with open(out_p, "a") as fout:
                    for bstart in range(start, len(rows), BATCH):
                        bend = min(bstart + BATCH, len(rows))
                        batch = rows[bstart:bend]
                        nsk = [j for j, r in enumerate(batch) if not r.get("skipped")]
                        prompts = [full_prompts.get(batch[j]["global_id"], "") for j in nsk]
                        responses = [batch[j].get("response", "") for j in nsk]
                        outputs = classify_one(gr, llm, prompts, responses) if prompts else []
                        oi = 0
                        for r in batch:
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
                            fout.write(json.dumps(out) + "\n")
                        fout.flush()
                        print(f"[{args.model}/{split}] {gr}: rows {bstart}-{bend}/{len(rows)} written",
                              flush=True)
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
