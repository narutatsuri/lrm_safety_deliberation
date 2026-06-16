#!/usr/bin/env python3
"""K=100 cut-rollout generator for the M=4 INFERENCE-DEFENSE oscillation audit.

Mirrors experiments/causal_cuts/generate_cuts_repro.py (the base-model Stage-5),
reading cuts + chunks from m4_inference_defense_majority3.jsonl. For each salient
refuse<->comply transition it forks the trace at the pre- and post-cut chunk,
closes </think>, and resamples K final responses; the response stance shift Δq
is the causal "budge" measured downstream by classify_segment_repro.py.

INTERVENTION REPLAY (the one structural difference vs base):
  The fork closes thinking at the cut and resamples only the *response* — the
  defense interventions act during *thinking*, so their effect on chunks[:cut]
  is already baked into the saved trace for SafeRemind (entropy reminders) and
  PSR (committed post-reflection text). The lone exception is SafePath-ZS, whose
  safety primer is *prefilled* at thinking start and NOT stored in the saved
  thinking trace; we re-prepend it to the prefix so the resampled response sees
  the same in-context framing the original generation had.

Response sampling uses MODELS[model]["sampling"] (== each model's recommended
defaults), identical to the base-model Stage-5, so defense Δq is comparable.

Output:
  experiments/causal_cuts/eval_results/causal_cuts_inference/<DEFENSE>-<MODEL>/<SPLIT>/rollouts_segment_repro.jsonl
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

# Provenance: copied verbatim (logic unchanged) from the original research repo
#   experiments/causal_cuts/rebuttal/generate_cuts_repro_inference.py
# The model registry (scripts.neurips_final.common.MODELS) and the 4-guardrail
# majority vote now also live in the `lrm_safety_deliberation` library (lrm_safety_deliberation.models /
# lrm_safety_deliberation.guardrails); this driver still imports the original repo modules via
# ROOT. Set LRM_SAFETY_ARTIFACTS to relocate the artifact tree.
PROJECT_ROOT = Path(os.environ.get("LRM_SAFETY_ARTIFACTS", Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(PROJECT_ROOT))

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams, TokensPrompt

from scripts.neurips_final.common import MODELS
from experiments.causal_cuts.generate_cuts import (
    BASE_SEED,
    salient_transitions,
    tok_ids,
    apply_chat_template,
    open_think_ids,
    close_think_ids,
    gptoss_analysis_ids,
    gptoss_final_ids,
    prompt_has_open_think,
)

ROOT = PROJECT_ROOT
AUDIT = ROOT / "experiments/causal_cuts/rebuttal/m4_inference_defense_majority3.jsonl"
OUT_ROOT = ROOT / "experiments/causal_cuts/eval_results/causal_cuts_inference"

# SafePath-ZS prefills this primer at thinking start; not in the saved trace.
SAFEPATH_PRIMER = "Let's think about safety first."
DEFENSES = ["safepath_zs", "saferemind", "psr"]


def run(args):
    model_key = args.model
    defense = args.defense
    split = args.split
    cell = f"{defense}-{model_key}"

    spec = MODELS[model_key]
    model_path = str(ROOT / spec["model_path"])
    max_model_len = args.max_model_len or spec.get("max_model_len", 8192)
    gpu_util = spec.get("gpu_memory_utilization", 0.90)
    sampling_kw = spec.get("sampling", {"temperature": 0.6, "top_p": 0.95})
    is_channel = bool(spec.get("is_channel", False))
    max_new = args.max_new
    prompt_budget = max_model_len - max_new - 64
    primer = SAFEPATH_PRIMER + " " if defense == "safepath_zs" else ""

    out_dir = OUT_ROOT / cell / split
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "rollouts_segment_repro.jsonl"

    done = set()
    if out_path.exists():
        for line in open(out_path):
            try:
                d = json.loads(line)
                if d.get("skipped"):
                    continue
                done.add((d["global_id"], d["trace_rollout_idx"],
                          d["transition_idx"], d["cut_kind"], d["rollout_idx"]))
            except Exception:
                pass
    print(f"[{cell}/{split}] already done: {len(done)} rollouts", flush=True)

    audit_rows = []
    with open(AUDIT) as f:
        for ln in f:
            d = json.loads(ln)
            if d.get("cell") != cell:
                continue
            if d.get("split") != split:
                continue
            if any(x is None for x in d.get("labels", [])):
                continue  # drop traces with unresolved chunks
            audit_rows.append(d)
    print(f"[{cell}/{split}] audit rows: {len(audit_rows)}", flush=True)

    tasks = []
    for d in audit_rows:
        gid = int(d["global_id"])
        trace_rollout_idx = int(d["rollout_idx"])
        labels = d["labels"]
        chunks = d["chunks"]
        n_chunks_total = len(chunks)
        if not chunks or not labels:
            continue
        transitions = salient_transitions(labels, args.L_pre, args.L_min)
        for (t_idx, s_pre_sign, s_pre_end, s_pre_len,
             s_post_sign, s_post_end, s_post_len) in transitions:
            pre_cut = s_pre_end + 1
            post_cut = s_post_end + 1
            if pre_cut <= 0 or post_cut <= pre_cut:
                continue
            if pre_cut > n_chunks_total or post_cut > n_chunks_total:
                continue
            for kind, cut_chunk_idx in (("pre", pre_cut), ("post", post_cut)):
                tasks.append(dict(
                    gid=gid, trace_rollout_idx=trace_rollout_idx,
                    transition_idx=t_idx, cut_kind=kind, cut_chunk_idx=cut_chunk_idx,
                    s_pre_sign=int(s_pre_sign), s_pre_len=s_pre_len, s_pre_end=s_pre_end,
                    s_post_sign=int(s_post_sign), s_post_len=s_post_len, s_post_end=s_post_end,
                    chunks=chunks, prompt_text=d.get("prompt", ""),
                    n_chunks_total=n_chunks_total, benchmark=d.get("benchmark", ""),
                ))

    print(f"[{cell}/{split}] cuts to generate: {len(tasks)}", flush=True)
    if not tasks:
        print("Nothing to do.")
        return
    if args.dry_run:
        traces = len({(t["gid"], t["trace_rollout_idx"]) for t in tasks})
        print(f"[{cell}/{split}] DRY: {len(tasks)} cuts across {traces} (gid,rollout) pairs; "
              f"{len(tasks) * args.K} rollouts at K={args.K}")
        return

    print(f"Loading {model_key} ...", flush=True)
    llm = LLM(model=model_path, max_model_len=max_model_len,
              gpu_memory_utilization=gpu_util, dtype="bfloat16",
              trust_remote_code=True, enforce_eager=False)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if is_channel:
        open_ids = gptoss_analysis_ids(tokenizer)
        close_ids = gptoss_final_ids(tokenizer)
    else:
        open_ids = open_think_ids(tokenizer)
        close_ids = close_think_ids(tokenizer)

    sampling = SamplingParams(n=args.K, seed=BASE_SEED, max_tokens=max_new, **sampling_kw)

    prompts, metas, skipped_rows = [], [], []
    for task in tasks:
        chunks = task["chunks"]
        cut_chunk_idx = task["cut_chunk_idx"]
        # Prepend the safepath primer (intervention replay); empty for others.
        prefix_chunks_text = primer + " ".join(chunks[:cut_chunk_idx])
        prefix_ids = tok_ids(tokenizer, prefix_chunks_text)
        prompt_rendered = apply_chat_template(tokenizer, task["prompt_text"])
        prompt_ids = tok_ids(tokenizer, prompt_rendered)
        if is_channel:
            full_ids = list(prompt_ids) + open_ids + prefix_ids + close_ids
        elif prompt_has_open_think(prompt_rendered):
            full_ids = list(prompt_ids) + prefix_ids + close_ids
        else:
            full_ids = list(prompt_ids) + open_ids + prefix_ids + close_ids

        common = dict(
            cell=cell, defense=defense,
            global_id=task["gid"], trace_rollout_idx=task["trace_rollout_idx"],
            split=split, n_chunks_total=task["n_chunks_total"], benchmark=task["benchmark"],
            prefix_n_chunks=cut_chunk_idx, prefix_n_tokens=len(prefix_ids),
            prompt_token_len=len(full_ids),
            transition_idx=task["transition_idx"], cut_kind=task["cut_kind"],
            cut_chunk_idx=task["cut_chunk_idx"],
            s_pre_sign=task["s_pre_sign"], s_pre_len_chunks=task["s_pre_len"], s_pre_end_chunk=task["s_pre_end"],
            s_post_sign=task["s_post_sign"], s_post_len_chunks=task["s_post_len"], s_post_end_chunk=task["s_post_end"],
            L_pre_used=args.L_pre, L_min_used=args.L_min,
            primer_prepended=bool(primer),
        )

        if len(full_ids) > prompt_budget:
            for r in range(args.K):
                row = dict(common)
                row.update(rollout_idx=r, response="", raw_response="",
                           thinking_prefix=prefix_chunks_text,
                           seed=BASE_SEED, skipped=True, skip_reason="overlong_prefix")
                skipped_rows.append(row)
            continue

        existing_r_idx = {r for (g, t_r, t, k, r) in done
                          if g == task["gid"] and t_r == task["trace_rollout_idx"]
                          and t == task["transition_idx"] and k == task["cut_kind"]}
        if len(existing_r_idx) >= args.K:
            continue
        prompts.append(TokensPrompt(prompt_token_ids=full_ids))
        metas.append({"common": common, "prefix_text": prefix_chunks_text,
                      "existing_r_idx": existing_r_idx})

    print(f"[{cell}/{split}] prompts={len(prompts)} skipped={len(skipped_rows)//max(1,args.K)}", flush=True)
    if skipped_rows:
        with open(out_path, "a") as fout:
            for row in skipped_rows:
                fout.write(json.dumps(row) + "\n")

    BATCH = max(1, args.batch_size)
    n_done = 0
    for start in range(0, len(prompts), BATCH):
        end = min(start + BATCH, len(prompts))
        bp, bm = prompts[start:end], metas[start:end]
        if not bp:
            continue
        outputs = llm.generate(bp, sampling)
        with open(out_path, "a") as fout:
            for m, out in zip(bm, outputs):
                common = m["common"]
                prefix_text = m["prefix_text"]
                existing = m.get("existing_r_idx", set())
                for r_idx, comp in enumerate(out.outputs):
                    if r_idx in existing:
                        continue
                    if is_channel:
                        raw = (f"<|channel|>analysis<|message|>{prefix_text}"
                               f"<|end|><|start|>assistant<|channel|>final<|message|>{comp.text}")
                    else:
                        raw = f"<think>\n{prefix_text}\n</think>\n\n{comp.text}"
                    row = dict(common)
                    row.update(rollout_idx=r_idx, response=comp.text.strip(),
                               raw_response=raw, thinking_prefix=prefix_text,
                               seed=BASE_SEED, skipped=False, skip_reason="")
                    fout.write(json.dumps(row) + "\n")
        n_done += len(bp)
        print(f"[{cell}/{split}] batch {start}-{end} ({n_done}/{len(prompts)})", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--defense", required=True, choices=DEFENSES)
    p.add_argument("--model", required=True,
                   choices=["Qwen3-8B", "Olmo-3-7B-Think", "Phi-4-reasoning", "GPT-OSS-20B"])
    p.add_argument("--split", required=True, choices=["harmful", "benign"])
    p.add_argument("--L-pre", type=int, default=1)
    p.add_argument("--L-min", type=int, default=1)
    p.add_argument("--K", type=int, default=100)
    p.add_argument("--max-new", type=int, default=2048)
    p.add_argument("--max-model-len", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
