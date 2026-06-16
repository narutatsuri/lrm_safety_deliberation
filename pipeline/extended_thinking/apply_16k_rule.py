#!/usr/bin/env python3
"""Apply the 2026-05-16 universal 16K-think + force-close protocol to one K=32 shard.

For each of the K=32 rollout slots per prompt, classify and act:
  - KEEP if existing rollout has finish_reason=='stop' and n_tokens <= 16384.
  - REPROCESS if existing rollout has finish_reason != 'stop' OR n_tokens > 16384:
      Truncate raw_response to first min(n_tokens, 16384) tokens, append the
      closure marker, run Phase 2 (max_tokens=1024, temp=0.0) to generate the
      final answer, then stitch.
  - UNFILLED (no existing rollout): standard new-rule generation —
      Phase 1 with model's sampling config and max_tokens=16384;
      then Phase 2 force-close + final answer for rows that didn't close.

Closure markers (matching evaluation/full_eval_pipeline/pipeline.py):
  - Non-channel: "\\n</think>\\n"
  - Channel (GPT-OSS): "<|end|><|start|>assistant<|channel|>final<|message|>"

Outputs preserve the existing JSONL schema. Atomic chunked writes for resume.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

# Provenance: copied verbatim (logic unchanged) from the original research repo
#   experiments/nested_branching_variance/k32_full/apply_16k_rule.py
# The model registry (scripts.neurips_final.common.MODELS) and the 4-guardrail
# majority vote now live in the `lrm_safety_deliberation` library (lrm_safety_deliberation.models /
# lrm_safety_deliberation.guardrails); this driver still imports the original modules via ROOT.
ROOT = Path(os.environ.get("LRM_SAFETY_ARTIFACTS", Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(ROOT))

from scripts.neurips_final.common import MODELS
from evaluation.full_eval_pipeline.datasets import read_jsonl
from evaluation.full_eval_pipeline.pipeline import (
    _build_chat_prompts,
    _byte_level_decode_if_needed,
    _strip_response_channel,
)
from evaluation.base import strip_thinking

# Universal 2026-05-16 protocol.
THINK_BUDGET = 16384            # max thinking tokens (Phase 1)
ANSWER_BUDGET = 1024            # max final-answer tokens (Phase 2)
PHASE2_TEMPERATURE = 0.0        # greedy Phase 2

# Closure markers (must match pipeline.py).
CLOSE_NONCHANNEL = "\n</think>\n"
CLOSE_CHANNEL = "<|end|><|start|>assistant<|channel|>final<|message|>"

CHUNK_PROMPTS = int(os.environ.get("K32_CHUNK_PROMPTS", "32"))
# Total max_model_len budget: prompt + 16K think + closure + 1K answer + headroom.
# Use the same per-model max_model_len that already worked.
PER_MODEL_MAXLEN = {
    "Qwen3-8B":        32768,
    "Olmo-3-7B-Think": 32768,
    "Phi-4-reasoning": 32768,
    "GPT-OSS-20B":     32768,
}


def shard_slice(n_total: int, shard_idx: int, n_shards: int) -> tuple[int, int]:
    chunk = (n_total + n_shards - 1) // n_shards
    start = shard_idx * chunk
    end = min(n_total, start + chunk)
    return start, end


def atomic_write_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def init_row(rec: dict, global_id: int, shard_idx: int, prompt_token_len: int, K: int) -> dict:
    row = dict(rec)
    row["global_pool_idx"] = global_id
    row["shard_idx"] = shard_idx
    row["prompt_token_len"] = prompt_token_len
    row["rollouts"] = [None] * K
    row["finish_reasons"] = [None] * K
    row["truncated_any"] = False
    row["over_budget"] = False
    return row


def mark_over_budget(rec: dict, global_id: int, shard_idx: int, prompt_token_len: int) -> dict:
    row = dict(rec)
    row["global_pool_idx"] = global_id
    row["shard_idx"] = shard_idx
    row["prompt_token_len"] = prompt_token_len
    row["rollouts"] = []
    row["finish_reasons"] = []
    row["truncated_any"] = False
    row["over_budget"] = True
    return row


def categorize_rollout(rollout: dict | None) -> str:
    """Return 'KEEP' | 'REPROCESS' | 'UNFILLED' for one rollout slot."""
    if rollout is None:
        return "UNFILLED"
    fr = rollout.get("finish_reason")
    n_tok = rollout.get("n_tokens", 0)
    if fr == "stop" and n_tok <= THINK_BUDGET:
        return "KEEP"
    return "REPROCESS"


def truncate_raw_to_budget(raw: str, n_tokens: int, tokenizer) -> str:
    """Truncate raw to <= THINK_BUDGET tokens; return decoded text.

    If n_tokens <= THINK_BUDGET we keep the original string (avoids decode roundtrip).
    """
    if n_tokens <= THINK_BUDGET:
        return raw
    tok_ids = tokenizer(raw, add_special_tokens=False).input_ids
    tok_ids = tok_ids[:THINK_BUDGET]
    return tokenizer.decode(tok_ids, skip_special_tokens=False)


def build_decoded(raw: str, no_think: bool, is_channel: bool) -> str:
    if no_think:
        return _strip_response_channel(raw) if is_channel else raw.strip()
    return _strip_response_channel(raw) if is_channel else strip_thinking(raw)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--split", required=True, choices=["asr", "orr"])
    p.add_argument("--mode", required=True, choices=["think", "nothink"])
    p.add_argument("--shard_idx", type=int, required=True)
    p.add_argument("--n_shards", type=int, required=True)
    p.add_argument("--K", type=int, default=32)
    p.add_argument("--output_root", default=str(ROOT / "eval_results" / "k32_full_pool"))
    args = p.parse_args()

    assert args.model in MODELS, f"Unknown model {args.model}"
    assert 0 <= args.shard_idx < args.n_shards

    model_cfg = MODELS[args.model]
    is_channel = bool(model_cfg.get("is_channel", False))
    use_plain_prompt = bool(model_cfg.get("is_pretrained", False))
    no_think = (args.mode == "nothink")
    closure = CLOSE_CHANNEL if is_channel else CLOSE_NONCHANNEL

    out_dir = Path(args.output_root) / args.model / args.mode / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"shard_{args.shard_idx:03d}_of_{args.n_shards:03d}"
    gen_path = out_dir / f"{tag}.jsonl"
    audit_path = out_dir / f"{tag}.audit.json"

    data_path = ROOT / "evaluation" / "full_eval_pipeline" / "data" / f"{args.split}.jsonl"
    records = read_jsonl(data_path)
    n_total = len(records)
    start, end = shard_slice(n_total, args.shard_idx, args.n_shards)
    my_records = records[start:end]

    print(
        f"[shard] {args.model} {args.mode} {args.split} {tag}: "
        f"prompts {start}..{end} (n={len(my_records)}); "
        f"THINK_BUDGET={THINK_BUDGET} ANSWER_BUDGET={ANSWER_BUDGET} "
        f"CHUNK={CHUNK_PROMPTS} is_channel={is_channel}",
        flush=True,
    )

    # Load existing rows (if any)
    existing: dict[int, dict] = {}
    if gen_path.exists():
        with open(gen_path) as f:
            for line in f:
                row = json.loads(line)
                local_idx = row["global_pool_idx"] - start
                if not row.get("over_budget"):
                    while len(row.setdefault("finish_reasons", [])) < args.K:
                        row["finish_reasons"].append(None)
                    while len(row.setdefault("rollouts", [])) < args.K:
                        row["rollouts"].append(None)
                existing[local_idx] = row
        print(f"[resume] Loaded {len(existing)} existing rows", flush=True)

    # Spin up vLLM
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    model_path = str(ROOT / model_cfg["model_path"])
    eff_max_model_len = PER_MODEL_MAXLEN.get(args.model, model_cfg["max_model_len"])
    print(f"[vllm] Loading {model_path}  max_model_len={eff_max_model_len}", flush=True)
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        gpu_memory_utilization=model_cfg["gpu_memory_utilization"],
        max_model_len=eff_max_model_len,
        trust_remote_code=True,
        enable_prefix_caching=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    texts = _build_chat_prompts(
        tokenizer, my_records, no_think=no_think, use_plain_prompt=use_plain_prompt,
    )
    budget = eff_max_model_len - THINK_BUDGET - ANSWER_BUDGET - 64
    prompt_lens = [len(tokenizer(t, add_special_tokens=False).input_ids) for t in texts]
    over_set = set(i for i, n in enumerate(prompt_lens) if n > budget)
    if over_set:
        print(f"[over-budget] {len(over_set)}/{len(texts)} prompts exceed budget={budget}; marked over_budget", flush=True)

    # Normalize rows
    for local_i in range(len(my_records)):
        if local_i in over_set:
            existing[local_i] = mark_over_budget(
                my_records[local_i], start + local_i, args.shard_idx, prompt_lens[local_i],
            )
            continue
        if local_i not in existing:
            existing[local_i] = init_row(
                my_records[local_i], start + local_i, args.shard_idx, prompt_lens[local_i], args.K,
            )
        else:
            # Existing row from prior run may have over_budget=True under a tighter
            # old-rule budget. Under the looser new-rule budget, this prompt is no
            # longer over_budget — fresh-init so its rollouts list is K-sized.
            row = existing[local_i]
            if row.get("over_budget"):
                existing[local_i] = init_row(
                    my_records[local_i], start + local_i, args.shard_idx,
                    prompt_lens[local_i], args.K,
                )

    # Base sampling params for Phase 1 (matches K=32 production sampling).
    samp = model_cfg.get("sampling", {})
    sp1_base: dict = dict(max_tokens=THINK_BUDGET, skip_special_tokens=not is_channel)
    for k in ("temperature", "top_p", "top_k", "min_p"):
        if k in samp:
            sp1_base[k] = samp[k]
    sp1_base.setdefault("temperature", 0.7)
    sp2_kwargs = dict(max_tokens=ANSWER_BUDGET, temperature=PHASE2_TEMPERATURE,
                      skip_special_tokens=not is_channel, n=1)
    print(f"[sampling] phase1={sp1_base}  phase2={sp2_kwargs}", flush=True)

    def make_sp1(n: int, seed: int | None) -> SamplingParams:
        kw = dict(sp1_base); kw["n"] = n
        if seed is not None:
            kw["seed"] = seed
        return SamplingParams(**kw)

    def make_sp2() -> SamplingParams:
        return SamplingParams(**sp2_kwargs)

    def rollout_from(out_text: str, finish_reason: str, n_tokens: int) -> dict:
        raw = _byte_level_decode_if_needed(out_text)
        resp = build_decoded(raw, no_think, is_channel)
        return {
            "raw_response": raw,
            "response": resp,
            "n_tokens": n_tokens,
            "finish_reason": finish_reason,
        }

    def stitch_rollout(existing_raw: str, existing_n_tokens: int, phase2_text: str,
                       phase2_n_tokens: int, phase2_finish_reason: str) -> dict:
        """Truncate existing raw to <= THINK_BUDGET tokens, append closure + Phase 2 text."""
        trunc_raw = truncate_raw_to_budget(existing_raw, existing_n_tokens, tokenizer)
        trunc_n = min(existing_n_tokens, THINK_BUDGET)
        full_raw = trunc_raw + closure + phase2_text
        resp = build_decoded(full_raw, no_think, is_channel)
        return {
            "raw_response": full_raw,
            "response": resp,
            "n_tokens": trunc_n + len(tokenizer(closure, add_special_tokens=False).input_ids) + phase2_n_tokens,
            "finish_reason": phase2_finish_reason if phase2_finish_reason == "stop" else "stop",
        }

    t0 = time.time()
    n_kept = n_reproc = n_unfilled = 0
    n_phase2_calls = 0

    for chunk_start in range(0, len(my_records), CHUNK_PROMPTS):
        chunk_end = min(chunk_start + CHUNK_PROMPTS, len(my_records))

        # Categorize each slot in this chunk
        phase1_work: list[tuple[int, list[int]]] = []  # (li, positions_needing_fresh_gen)
        reproc_items: list[tuple[int, int, str, int]] = []  # (li, pos, existing_raw, existing_n_tokens)
        for li in range(chunk_start, chunk_end):
            if li in over_set:
                continue
            row = existing[li]
            unfilled_positions = []
            for pos in range(args.K):
                cat = categorize_rollout(row["rollouts"][pos])
                if cat == "KEEP":
                    pass
                elif cat == "REPROCESS":
                    r = row["rollouts"][pos]
                    reproc_items.append((li, pos, r["raw_response"], r.get("n_tokens", 0)))
                else:  # UNFILLED
                    unfilled_positions.append(pos)
            if unfilled_positions:
                phase1_work.append((li, unfilled_positions))

        # Phase 1 (only for UNFILLED): batched sampling
        # Each entry of phase1_work: prompt li gets n=len(positions) fresh samples.
        phase1_results: list[tuple[int, int, dict, bool]] = []  # (li, pos, rollout_dict, needs_close)
        if phase1_work:
            prompts_to_run = [texts[li] for li, _ in phase1_work]
            sps = []
            base_seed = args.shard_idx * 10**7 + chunk_start * 1000
            for k_idx, (li, positions) in enumerate(phase1_work):
                sps.append(make_sp1(n=len(positions), seed=base_seed + k_idx))
            outs = llm.generate(prompts_to_run, sps, use_tqdm=False)
            for (li, positions), out in zip(phase1_work, outs):
                for pi, pos in enumerate(positions):
                    r = out.outputs[pi]
                    raw = _byte_level_decode_if_needed(r.text)
                    n_tok = len(r.token_ids)
                    fr = r.finish_reason
                    # Decide: does this need Phase 2 force-close?
                    needs_close = False
                    if is_channel:
                        needs_close = "<|channel|>final<|message|>" not in raw
                    else:
                        # If <think> in template prefix but no </think> in raw → needs close
                        # Heuristic: if finish_reason == "length" AND no </think> in raw
                        needs_close = (fr == "length") and ("</think>" not in raw)
                    rollout = rollout_from(raw, fr, n_tok)
                    phase1_results.append((li, pos, rollout, needs_close))

        # Build Phase 2 batch: (REPROCESS items + UNFILLED items that need close)
        phase2_prompts: list[str] = []
        phase2_meta: list[tuple[int, int, str, int]] = []  # (li, pos, existing_raw, existing_n_tokens)
        for li, pos, raw, n_tok in reproc_items:
            trunc_raw = truncate_raw_to_budget(raw, n_tok, tokenizer)
            phase2_prompts.append(texts[li] + trunc_raw + closure)
            phase2_meta.append((li, pos, raw, n_tok))
        for li, pos, rollout, needs_close in phase1_results:
            if needs_close:
                raw = rollout["raw_response"]
                n_tok = rollout["n_tokens"]
                trunc_raw = truncate_raw_to_budget(raw, n_tok, tokenizer)
                phase2_prompts.append(texts[li] + trunc_raw + closure)
                phase2_meta.append((li, pos, raw, n_tok))

        phase2_results: dict[tuple[int, int], tuple[str, int, str]] = {}
        if phase2_prompts:
            sp2 = make_sp2()
            outs2 = llm.generate(phase2_prompts, sp2, use_tqdm=False)
            n_phase2_calls += 1
            for (li, pos, existing_raw, existing_n), o in zip(phase2_meta, outs2):
                r = o.outputs[0]
                p2_text = _byte_level_decode_if_needed(r.text)
                phase2_results[(li, pos)] = (p2_text, len(r.token_ids), r.finish_reason)

        # Apply results to existing rows
        for li, pos, rollout, needs_close in phase1_results:
            row = existing[li]
            if needs_close and (li, pos) in phase2_results:
                p2_text, p2_n, p2_fr = phase2_results[(li, pos)]
                stitched = stitch_rollout(rollout["raw_response"], rollout["n_tokens"], p2_text, p2_n, p2_fr)
                row["rollouts"][pos] = stitched
                row["finish_reasons"][pos] = stitched["finish_reason"]
            else:
                row["rollouts"][pos] = rollout
                row["finish_reasons"][pos] = rollout["finish_reason"]
            n_unfilled += 1
        for li, pos, existing_raw, existing_n in reproc_items:
            row = existing[li]
            if (li, pos) in phase2_results:
                p2_text, p2_n, p2_fr = phase2_results[(li, pos)]
                stitched = stitch_rollout(existing_raw, existing_n, p2_text, p2_n, p2_fr)
                row["rollouts"][pos] = stitched
                row["finish_reasons"][pos] = stitched["finish_reason"]
                n_reproc += 1

        # truncated_any flag refresh
        for li in range(chunk_start, chunk_end):
            if li in over_set:
                continue
            row = existing[li]
            row["truncated_any"] = any(fr is not None and fr != "stop"
                                       for fr in row["finish_reasons"])

        # Atomic chunk write
        rows_to_write = [existing[i] for i in range(len(my_records))]
        atomic_write_jsonl(gen_path, rows_to_write)
        print(
            f"[chunk {chunk_start}-{chunk_end}] phase1={len(phase1_work)} prompts "
            f"reproc={len(reproc_items)} phase2={len(phase2_prompts)} "
            f"elapsed={time.time()-t0:.0f}s",
            flush=True,
        )

    # Count KEEP at end (didn't track in-loop)
    for row in existing.values():
        if row.get("over_budget"):
            continue
        for r in row.get("rollouts", []):
            if r is None:
                continue
            if r.get("finish_reason") == "stop" and r.get("n_tokens", 0) <= THINK_BUDGET:
                # heuristic: stitched rollouts have finish_reason=='stop' too but their
                # n_tokens might exceed THINK_BUDGET (because they include closure + phase2)
                # we can't perfectly distinguish; rely on raw_response containing closure
                if not (closure in r.get("raw_response", "")):
                    n_kept += 1

    dt = time.time() - t0
    # Audit
    n_truncated_final = sum(
        sum(1 for fr in r.get("finish_reasons", []) if fr is not None and fr != "stop")
        for r in existing.values()
    )
    n_rollouts_total = sum(
        sum(1 for fr in r.get("finish_reasons", []) if fr is not None)
        for r in existing.values()
    )
    n_over_budget = len(over_set)
    n_expected = (len(my_records) - n_over_budget) * args.K

    audit = {
        "model": args.model, "mode": args.mode, "split": args.split,
        "shard_idx": args.shard_idx, "n_shards": args.n_shards, "K": args.K,
        "max_new_tokens": THINK_BUDGET, "answer_budget": ANSWER_BUDGET,
        "max_model_len": eff_max_model_len,
        "n_prompts_in_shard": len(my_records),
        "n_prompts_over_budget": n_over_budget,
        "n_rollouts_total": n_rollouts_total,
        "n_rollouts_expected": n_expected,
        "n_truncated": n_truncated_final,
        "n_kept": n_kept, "n_reprocessed": n_reproc, "n_unfilled": n_unfilled,
        "n_phase2_calls": n_phase2_calls,
        "trunc_rate": n_truncated_final / max(1, n_rollouts_total),
        "wall_seconds": dt,
        "tps_observed": (n_rollouts_total * 1000) / max(1.0, dt) if dt else 0.0,
        "protocol": "universal_2026-05-16: think_budget=16384 + force-close + answer_budget=1024",
        "chunk_prompts": CHUNK_PROMPTS,
        "sampling_phase1": sp1_base, "sampling_phase2": sp2_kwargs,
        "model_path": model_path,
        "prompt_data_path": str(data_path),
    }
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    print(
        f"[summary] kept~{n_kept} reproc={n_reproc} unfilled={n_unfilled} "
        f"truncated_final={n_truncated_final}/{n_rollouts_total} "
        f"wall={dt:.0f}s",
        flush=True,
    )

    del llm, tokenizer
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    if n_truncated_final > 0:
        # Under new rule, any remaining trunc means Phase 2 failed (shouldn't happen given temp=0 + 1K budget).
        print(f"WARN: {n_truncated_final} rollouts still malformed after Phase 2.", flush=True)
    print("OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
